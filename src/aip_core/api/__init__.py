"""FastAPI 应用 - 课程生成 API

T1.5 需求解析 API 对接前端：
- POST /api/course/create  创建课程生成会话
- POST /api/course/{session_id}/message  发送对话消息
- POST /api/course/{session_id}/hitl/action  HITL 确认操作
- GET  /api/course/{session_id}/progress  轮询进度
- GET  /api/course/{session_id}/resume  恢复会话
- GET  /api/course/sessions  列出用户会话
"""

import json
import uuid
import time
import sqlite3
import threading
from typing import Optional
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from ..graph import CourseState, build_course_graph, create_sqlite_checkpointer
from ..graph.state import HITL_DEFINITIONS, ALL_NODES
from ..config import MAX_CONCURRENT_SESSIONS, SQLITE_DB_PATH
from ..content_safety import check_user_input, check_ai_output
from ..i18n import to_traditional
from ..digital_human.runninghub_client import RunningHubClient
from fastapi import Request, Response

# ============================================================
# Pydantic 模型
# ============================================================

class CreateCourseRequest(BaseModel):
    initial_message: str = Field(..., description="用户初始需求描述")
    user_id: str = Field(default="anonymous", description="用户 ID")

class CreateCourseResponse(BaseModel):
    session_id: str
    stage: str
    agent_message: Optional[str] = None
    profile: Optional[dict] = None
    hitl: Optional[dict] = None
    need_followup: bool = False

class MessageRequest(BaseModel):
    message: str = Field(..., description="用户回复消息")

class HitlActionRequest(BaseModel):
    hitl_id: str = Field(..., description="HITL 确认点 ID，如 HITL-1")
    action: str = Field(..., description="操作：confirm/skip/regenerate/edit/skip_all")
    edits: Optional[dict] = Field(default=None, description="编辑内容(action=edit时)")

class ProgressResponse(BaseModel):
    session_id: str
    stages: list[dict]
    current_hitl: Optional[dict] = None
    estimated_remaining_seconds: Optional[int] = None
    is_complete: bool = False

# ============================================================
# 应用初始化
# ============================================================

app = FastAPI(
    title="文经客 AIP OPC 核心 API",
    description="分层多Agent协作系统 - 课程生成引擎（MiniMax 语音克隆 + Duix-Avatar 数字人）",
    version="0.3.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# 繁體中文響應中間件：將所有 JSON 回應中的中文字串轉為繁體中文（OpenCC s2t 兜底）。
# 與各 Agent 系統提示詞中的「繁體中文」指令雙重保障，確保前端看到的輸出皆為繁體。
# ============================================================
@app.middleware("http")
async def _traditional_chinese_response(request: Request, call_next):
    response = await call_next(request)
    if "application/json" not in response.headers.get("content-type", ""):
        return response
    body = b""
    async for chunk in response.body_iterator:
        body += chunk if isinstance(chunk, bytes) else chunk.encode("utf-8")
    try:
        data = json.loads(body.decode("utf-8"))
        data = to_traditional(data)
        new_body = json.dumps(data, ensure_ascii=False).encode("utf-8")
    except Exception:
        new_body = body
    headers = {k: v for k, v in response.headers.items() if k.lower() != "content-length"}
    return Response(content=new_body, status_code=response.status_code,
                    headers=headers, media_type="application/json")


# ============================================================
# 全局异常处理：绝不让外部服务（硅基流动 / DeepSeek 等）的原始报错
# （如 "input length too long"）裸漏到前端。真实错误仅记录到服务端日志，
# 前端收到统一的中文友好提示。
# ============================================================
import logging
from fastapi import Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError

logger = logging.getLogger("aip_core.api")

# 单条用户输入最大字符数：超过则友好拒绝，避免把超长文本直接丢给外部 LLM/TTS
MAX_INPUT_CHARS = 20000


def _friendly_error(message: str, error: str = "internal_error", status: int = 422):
    return JSONResponse(
        status_code=status,
        content={"detail": {"error": error, "message": message}},
    )


@app.exception_handler(Exception)
async def _global_exception_handler(request: Request, exc: Exception):
    # 记录真实错误（含外部 API 原始报文），便于事后定位根因
    logger.error(
        "未捕获异常 %s %s: %r",
        request.method, request.url.path, exc, exc_info=True,
    )
    # HTTPException 透传，但把外部 400 的裸文本转成友好提示
    if isinstance(exc, HTTPException):
        if exc.status_code == 400:
            return _friendly_error("请求无法处理，请检查输入后重试。", "bad_request", 400)
        detail = exc.detail
        if isinstance(detail, str) and "input length" in detail.lower():
            detail = "输入内容过长，请精简后重试。"
        return JSONResponse(status_code=exc.status_code, content={"detail": detail})
    return _friendly_error(
        "服务暂时不可用，请稍后重试。若问题持续，请联系管理员。",
        "internal_error", 422,
    )


@app.exception_handler(RequestValidationError)
async def _validation_exception_handler(request: Request, exc: RequestValidationError):
    return _friendly_error("请求参数格式有误，请检查后重试。", "validation_error", 422)


def _guard_input(text: str, field: str = "输入") -> None:
    """对用户输入做长度护栏：超长直接友好拒绝，不触发外部 API 400。"""
    if text and len(text) > MAX_INPUT_CHARS:
        raise HTTPException(
            status_code=422,
            detail={"error": "input_too_long",
                    "message": f"{field}过长（{len(text)} 字），请精简到 {MAX_INPUT_CHARS} 字以内后重试。"},
        )


# 全局：编译后的 Graph + Checkpointer（启动时初始化）
_graph = None
_checkpointer = None

# ============================================================
# SQLite 会话持久化层
# ============================================================

class SessionStore:
    """SQLite 会话持久化存储

    替代内存 _session_cache，解决服务重启丢数据问题。
    线程安全（每线程一个连接），自动建表，支持 JSON 序列化。
    """

    def __init__(self, db_path: str = None):
        self._db_path = db_path or SQLITE_DB_PATH
        self._local = threading.local()
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        """获取当前线程的数据库连接（自动创建）"""
        if not hasattr(self._local, "conn") or self._local.conn is None:
            Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(self._db_path, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            self._local.conn = conn
        return self._local.conn

    def _init_db(self):
        """初始化数据库表"""
        conn = self._get_conn()
        conn.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                state_json TEXT NOT NULL DEFAULT '{}',
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                completed INTEGER NOT NULL DEFAULT 0
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_sessions_user_id
            ON sessions(user_id)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_sessions_created_at
            ON sessions(created_at)
        """)
        conn.commit()

    def save(self, session_id: str, state: dict, user_id: str = "anonymous",
             completed: bool = False, created_at: float = None):
        """保存或更新会话（UPSERT）"""
        now = time.time()
        state_json = json.dumps(state, ensure_ascii=False, default=str)
        conn = self._get_conn()
        conn.execute("""
            INSERT INTO sessions (session_id, user_id, state_json, created_at, updated_at, completed)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(session_id) DO UPDATE SET
                state_json = excluded.state_json,
                updated_at = excluded.updated_at,
                completed = excluded.completed
        """, (
            session_id,
            user_id,
            state_json,
            created_at or now,
            now,
            1 if completed else 0,
        ))
        conn.commit()

    def get(self, session_id: str) -> Optional[dict]:
        """获取单个会话"""
        conn = self._get_conn()
        row = conn.execute(
            "SELECT * FROM sessions WHERE session_id = ?", (session_id,)
        ).fetchone()
        if row is None:
            return None
        return {
            "state": json.loads(row["state_json"]),
            "user_id": row["user_id"],
            "created_at": row["created_at"],
            "completed": bool(row["completed"]),
        }

    def list_by_user(self, user_id: str) -> list[dict]:
        """列出用户的所有会话"""
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM sessions WHERE user_id = ? ORDER BY created_at DESC",
            (user_id,)
        ).fetchall()
        return [
            {
                "state": json.loads(r["state_json"]),
                "user_id": r["user_id"],
                "created_at": r["created_at"],
                "completed": bool(r["completed"]),
            }
            for r in rows
        ]

    def count_active_by_user(self, user_id: str) -> int:
        """统计用户活跃会话数（未完成）"""
        conn = self._get_conn()
        row = conn.execute(
            "SELECT COUNT(*) as cnt FROM sessions WHERE user_id = ? AND completed = 0",
            (user_id,)
        ).fetchone()
        return row["cnt"]

    def total_sessions(self) -> int:
        """总会话数"""
        conn = self._get_conn()
        return conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]

    def active_sessions(self) -> int:
        """活跃会话数"""
        conn = self._get_conn()
        return conn.execute(
            "SELECT COUNT(*) FROM sessions WHERE completed = 0"
        ).fetchone()[0]

    def cleanup_expired(self, max_age_seconds: float = 86400 * 7):
        """清理过期会话（默认 7 天）"""
        cutoff = time.time() - max_age_seconds
        conn = self._get_conn()
        conn.execute(
            "DELETE FROM sessions WHERE updated_at < ?", (cutoff,)
        )
        conn.commit()


# 全局 SessionStore 实例
_session_store = SessionStore()


# ============================================================
# 并发控制：真正"正在生成中"的会话计数（按用户）
# 与 DB 中的 completed 状态解耦，避免被中断/放弃/旧 schema 的
# 僵尸会话永久占用并发名额。每次同步生成请求结束必定释放名额。
# ============================================================
_running_lock = threading.Lock()
_running_by_user: dict = {}


def _try_acquire_slot(user_id: str) -> bool:
    """尝试占用一个并发生成名额；成功返回 True，已达上限返回 False。"""
    with _running_lock:
        current = _running_by_user.get(user_id, 0)
        if current >= MAX_CONCURRENT_SESSIONS:
            return False
        _running_by_user[user_id] = current + 1
        return True


def _release_slot(user_id: str) -> None:
    """释放一个并发生成名额。"""
    with _running_lock:
        _running_by_user[user_id] = max(0, _running_by_user.get(user_id, 0) - 1)


def get_graph():
    """懒加载 Graph"""
    global _graph, _checkpointer
    if _graph is None:
        _checkpointer = create_sqlite_checkpointer()
        _graph = build_course_graph(checkpointer=_checkpointer)
    return _graph, _checkpointer


# ============================================================
# 会话管理
# ============================================================

def _create_initial_state(session_id: str, user_id: str, initial_message: str) -> CourseState:
    """创建初始 CourseState"""
    return CourseState(
        session_id=session_id,
        user_id=user_id,
        current_node="",
        user_profile=None,
        requirement_completeness=0,
        followup_rounds=0,
        ip_positioning=None,
        course_outline=None,
        scripts=None,
        slides=None,
        cases=None,
        marketing_copy=None,
        pricing_plan=None,
        review_score=None,
        review_detail=None,
        review_round=0,
        hitl_status={hid: "pending" for hid in HITL_DEFINITIONS},
        skip_all_hitl=False,
        errors=[],
        node_history=[],
        messages=[{"role": "user", "content": initial_message, "timestamp": time.time()}],
    )


def _get_node_order(node_name: str) -> int:
    """获取节点在流程中的顺序"""
    try:
        return ALL_NODES.index(node_name)
    except ValueError:
        return 99


# ============================================================
# API 路由
# ============================================================

@app.post("/api/course/create", response_model=CreateCourseResponse)
async def create_course(request: CreateCourseRequest):
    """创建课程生成会话 - 发起需求解析"""
    user_id = request.user_id

    # 输入长度护栏（防止超长文本触发外部 LLM/TTS 的 "input length too long"）
    _guard_input(request.initial_message, "初始需求描述")

    # 内容安全审核：用户输入
    safety_result = check_user_input(request.initial_message)
    if not safety_result.passed:
        raise HTTPException(
            status_code=422,
            detail={"error": "content_safety_violation", "message": safety_result.message}
        )

    # 检查并发限制：统计"正在生成中"的请求数（而非未完成的会话数），
    # 避免被僵尸/中断会话永久占用名额。
    if not _try_acquire_slot(user_id):
        raise HTTPException(
            status_code=429,
            detail=f"同时进行的课程生成已达上限 ({MAX_CONCURRENT_SESSIONS})，请等待现有任务完成"
        )

    session_id = str(uuid.uuid4())[:8]
    state = _create_initial_state(session_id, user_id, request.initial_message)

    # 运行需求解析
    from ..agents.requirement_agent import run_requirement_analysis
    try:
        state = run_requirement_analysis(state)
    finally:
        _release_slot(user_id)

    # 持久化到 SQLite
    _session_store.save(session_id, state, user_id=user_id, completed=False)

    # 构建响应
    messages = state.get("messages", [])
    last_assistant_msg = ""
    need_followup = False
    for msg in reversed(messages):
        if isinstance(msg, dict) and msg.get("role") == "assistant":
            last_assistant_msg = msg.get("content", "")
            need_followup = msg.get("type") == "followup"
            break

    hitl_status = state.get("hitl_status", {}).get("HITL-1")
    hitl = None
    if hitl_status == "pending":
        hitl = {"hitl_id": "HITL-1", "label": "需求解析确认", "status": "pending"}

    return CreateCourseResponse(
        session_id=session_id,
        stage="requirement_analysis" if not need_followup else "asking",
        agent_message=last_assistant_msg,
        profile=state.get("user_profile"),
        hitl=hitl,
        need_followup=need_followup,
    )


@app.post("/api/course/{session_id}/message")
async def send_message(session_id: str, request: MessageRequest):
    """发送对话消息 - 用于追问回复或继续流程"""
    # 输入长度护栏
    _guard_input(request.message, "回复消息")

    # 内容安全审核：用户消息
    safety_result = check_user_input(request.message)
    if not safety_result.passed:
        raise HTTPException(
            status_code=422,
            detail={"error": "content_safety_violation", "message": safety_result.message}
        )

    session = _session_store.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="会话不存在或已过期")

    state = session["state"]
    user_id = session.get("user_id", "anonymous")

    # 追加用户消息
    state["messages"] = state.get("messages", []) + [
        {"role": "user", "content": request.message, "timestamp": time.time()}
    ]

    current_node = state.get("current_node", "")

    # 需求解析阶段（追问中 或 已汇总待确认）都允许继续对话补充信息：
    # - asking：继续追问下一缺失字段
    # - pending：用户已提供足够信息、系统已生成确认摘要，但用户可能还想补充，
    #            此时重跑需求解析吸纳新信息并刷新确认摘要，而不是静默丢弃输入。
    hitl1 = state.get("hitl_status", {}).get("HITL-1")
    if hitl1 in ("asking", "pending"):
        from ..agents.requirement_agent import run_requirement_analysis
        state = run_requirement_analysis(state)

        messages = state.get("messages", [])
        last_assistant_msg = ""
        need_followup = False
        for msg in reversed(messages):
            if isinstance(msg, dict) and msg.get("role") == "assistant":
                last_assistant_msg = msg.get("content", "")
                need_followup = msg.get("type") == "followup"
                break

        _session_store.save(session_id, state, user_id=user_id, completed=False)

        hitl_status = state.get("hitl_status", {}).get("HITL-1")
        hitl = None
        if hitl_status == "pending":
            hitl = {"hitl_id": "HITL-1", "label": "需求解析确认", "status": "pending"}

        return CreateCourseResponse(
            session_id=session_id,
            stage="requirement_analysis" if not need_followup else "asking",
            agent_message=last_assistant_msg,
            profile=state.get("user_profile"),
            hitl=hitl,
            need_followup=need_followup,
        )

    return {"session_id": session_id, "status": "ok", "message": "消息已接收"}


@app.post("/api/course/{session_id}/hitl/action")
async def hitl_action(session_id: str, request: HitlActionRequest):
    """HITL 确认点操作"""
    session = _session_store.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="会话不存在或已过期")

    state = session["state"]
    user_id = session.get("user_id", "anonymous")
    hitl_id = request.hitl_id

    if hitl_id not in HITL_DEFINITIONS:
        raise HTTPException(status_code=400, detail=f"无效的 HITL ID: {hitl_id}")

    # 处理用户操作
    if request.action == "skip_all":
        state["skip_all_hitl"] = True

    # 更新 HITL 状态
    if request.action == "edit" and request.edits:
        # 应用编辑
        node_name = HITL_DEFINITIONS[hitl_id]["node"]
        if node_name == "requirement_analysis" and "user_profile" in request.edits:
            state["user_profile"] = request.edits["user_profile"]
        elif node_name == "ip_positioning" and "ip_positioning" in request.edits:
            state["ip_positioning"] = request.edits["ip_positioning"]
        elif node_name == "course_architecture" and "course_outline" in request.edits:
            state["course_outline"] = request.edits["course_outline"]

    # 标记 HITL 状态
    status_map = {
        "confirm": "confirmed",
        "skip": "skipped",
        "regenerate": "regenerating",
        "edit": "confirmed",
        "skip_all": "skipped",
    }
    state["hitl_status"][hitl_id] = status_map.get(request.action, "confirmed")

    # 推进到下一个节点
    hitl_order = HITL_DEFINITIONS[hitl_id]["order"]

    if request.action == "regenerate":
        # 回到当前 HITL 对应的节点重新生成
        node_name = HITL_DEFINITIONS[hitl_id]["node"]
        if not _try_acquire_slot(user_id):
            raise HTTPException(
                status_code=429,
                detail=f"同时进行的课程生成已达上限 ({MAX_CONCURRENT_SESSIONS})，请等待现有任务完成"
            )
        try:
            _run_node(state, node_name)
        finally:
            _release_slot(user_id)
    else:
        # 推进到下一个节点
        if not _try_acquire_slot(user_id):
            raise HTTPException(
                status_code=429,
                detail=f"同时进行的课程生成已达上限 ({MAX_CONCURRENT_SESSIONS})，请等待现有任务完成"
            )
        try:
            _advance_to_next(state, hitl_order)
        finally:
            _release_slot(user_id)

    # 检查是否全部完成
    is_complete = state.get("current_node") == "packaging"

    _session_store.save(session_id, state, user_id=user_id, completed=is_complete)

    return {
        "session_id": session_id,
        "action": request.action,
        "current_stage": state.get("current_node", ""),
        "is_complete": is_complete,
        "next_hitl": _get_next_pending_hitl(state),
    }


@app.get("/api/course/{session_id}/progress", response_model=ProgressResponse)
async def get_progress(session_id: str):
    """轮询课程生成进度"""
    session = _session_store.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="会话不存在或已过期")

    state = session["state"]
    user_id = state.get("user_id", "anonymous")
    current_node = state.get("current_node", "")
    hitl_status = state.get("hitl_status", {})

    # v15 安全网：修复旧会话 current_node 卡在已执行完成节点上的问题
    # 当 current_node 指向一个已在 node_history 中标记为完成的节点时，
    # 自动推进到下一个未完成的节点（或 packaging）
    if current_node:
        completed_nodes = {
            h["node"] for h in state.get("node_history", [])
            if h.get("status") in ("ok", "partial")
        }
        if current_node in completed_nodes:
            node_to_hitl = {hdef["node"]: hid for hid, hdef in HITL_DEFINITIONS.items()}

            # v17(修正): 同步确认「节点已执行完成但 HITL 仍 pending」的旧会话 HITL。
            # 但【排除当前节点本身】——它正处于等待用户确认的状态，不能替用户越权确认，
            # 否则会丢失用户应有的确认环节（如质量审核 HITL-7）。
            # 这样 progress 返回的 current_hitl 不会再指向已执行节点的过期确认点
            # （例如已到质量审核阶段却仍弹出「语音合成确认」按钮）。
            hitl_synced = False
            for node_name in completed_nodes:
                if node_name == current_node:
                    continue
                hid = node_to_hitl.get(node_name)
                if hid and state.get("hitl_status", {}).get(hid) == "pending":
                    state["hitl_status"][hid] = "confirmed"
                    hitl_synced = True
            if hitl_synced:
                # 同步结果落盘：否则后续 HITL 确认动作调用 _get_next_pending_hitl
                # 时会再次读到过期的 pending HITL，导致状态回跳。
                _session_store.save(session_id, state, user_id=user_id, completed=False)

            # 终态：所有生产节点完成 且 全部 HITL 已确认/跳过，但 packaging 尚未执行
            # → 补执行打包（v16）。仅当用户已确认全部环节才自动打包，避免跳过质量审核。
            hitl_all_done = all(
                state.get("hitl_status", {}).get(hid) in ("confirmed", "skipped")
                for hid in HITL_DEFINITIONS
            )
            if hitl_all_done and "packaging" not in completed_nodes:
                try:
                    from ..agents import run_packaging
                    run_packaging(state)
                    current_node = "packaging"
                    state["current_node"] = "packaging"   # 同步 state，确保 is_complete 正确
                    # 持久化（确保 _package_path 保存到 SQLite）
                    _session_store.save(session_id, state, user_id=user_id, completed=True)
                except Exception as e:
                    # 打包失败仍标记为 packaging（让前端显示完成），但记录错误
                    current_node = "packaging"
                    state["current_node"] = "packaging"
                    state["errors"] = state.get("errors", []) + [{
                        "node": "packaging",
                        "error": f"safety-net packaging failed: {e}",
                        "time": time.time(),
                    }]
            elif "packaging" in completed_nodes:
                current_node = "packaging"
                state["current_node"] = "packaging"

    # 构建阶段状态
    stages = []
    current_found = False
    for node_name in ALL_NODES:
        node_order = _get_node_order(node_name)
        current_order = _get_node_order(current_node)

        if node_order < current_order:
            status = "completed"
        elif node_order == current_order:
            status = "running"
            current_found = True
        else:
            status = "pending" if current_found else "pending"

        stages.append({
            "name": node_name,
            "label": _node_label(node_name),
            "status": status,
        })

    # 当前 HITL
    current_hitl = _get_next_pending_hitl(state)

    # 估算剩余时间
    remaining_nodes = len([s for s in stages if s["status"] == "pending"])
    estimated_remaining = remaining_nodes * 45  # 每节点约 45 秒

    is_complete = state.get("current_node") == "packaging"
    if is_complete:
        _session_store.save(session_id, state, user_id=user_id, completed=True)

    return ProgressResponse(
        session_id=session_id,
        stages=stages,
        current_hitl=current_hitl,
        estimated_remaining_seconds=estimated_remaining if not is_complete else 0,
        is_complete=is_complete,
    )


@app.get("/api/course/sessions")
async def list_sessions(user_id: str = Query(..., description="用户 ID")):
    """列出用户的所有会话"""
    sessions = _session_store.list_by_user(user_id)
    user_sessions = []
    for s in sessions:
        state = s["state"]
        outline = state.get("course_outline") or {}
        user_sessions.append({
            "session_id": state.get("session_id", ""),
            "stage": state.get("current_node", "init"),
            "created_at": s.get("created_at"),
            "completed": s.get("completed", False),
            "course_title": outline.get("course_title", ""),
        })
    return {"sessions": user_sessions}


@app.get("/api/course/{session_id}/resume")
async def resume_session(session_id: str):
    """恢复会话"""
    session = _session_store.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="会话不存在或已过期")

    state = session["state"]
    return {
        "session_id": session_id,
        "stage": state.get("current_node", ""),
        "next_hitl": _get_next_pending_hitl(state),
        "is_complete": state.get("current_node") == "packaging",
    }


# ============================================================
# 辅助函数
# ============================================================

def _node_label(node_name: str) -> str:
    """节点中文标签"""
    labels = {
        "requirement_analysis": "需求解析",
        "ip_positioning": "IP定位",
        "course_architecture": "课程架构",
        "content_production_parallel": "内容生产(讲稿/课件/案例)",
        "content_production_serial": "内容生产(营销/定价)",
        "quality_review": "质量审核",
        "packaging": "打包交付",
    }
    return labels.get(node_name, node_name)


def _run_node(state: CourseState, node_name: str):
    """执行指定节点"""
    from ..agents import (
        run_requirement_analysis, run_ip_positioning,
        run_course_architecture, run_content_parallel,
        run_content_serial,
        run_quality_review, run_packaging,
    )
    node_funcs = {
        "requirement_analysis": run_requirement_analysis,
        "ip_positioning": run_ip_positioning,
        "course_architecture": run_course_architecture,
        "content_production_parallel": run_content_parallel,
        "content_production_serial": run_content_serial,
        "quality_review": run_quality_review,
        "packaging": run_packaging,
    }
    func = node_funcs.get(node_name)
    if func:
        func(state)


def _advance_to_next(state: CourseState, current_hitl_order: int):
    """推进到下一个 HITL 对应的节点

    节点执行顺序与 LangGraph 编排边一致：
    requirement_analysis → ip_positioning → course_architecture
    → content_production_parallel → content_production_serial
    → quality_review → packaging
    """
    # 完整的节点执行序列（与 LangGraph StateGraph 边一致）
    full_sequence = [
        "ip_positioning",
        "course_architecture",
        "content_production_parallel",
        "content_production_serial",
        "quality_review",
        "packaging",
    ]

    # 计算从当前 HITL 之后应该执行哪些节点
    # HITL-1 对应 requirement_analysis 之后，从 ip_positioning 开始
    # HITL-2 对应 ip_positioning 之后，从 course_architecture 开始
    # 以此类推...
    # current_hitl_order 从 1 开始，执行 full_sequence 中索引 >= current_hitl_order 的节点
    # 但要注意：HITL-1 的 order=1，对应 full_sequence[0]（ip_positioning）

    start_idx = current_hitl_order  # HITL-1 → idx=1 → full_sequence[1]="course_architecture"
                                     # 但我们想从 ip_positioning 开始，即 full_sequence[0]
    # 修正：HITL-N 确认后，应执行 full_sequence[N-1] 开始的节点
    start_idx = current_hitl_order - 1

    # 如果不是 skip_all，持续推进直到遇到需要等待的 HITL
    if not state.get("skip_all_hitl"):
        # v14: 循环执行节点，直到遇到真正需要用户确认的 HITL
        # v15: 跳过已在 node_history 中完成的节点
        completed_nodes = {h.get("node") for h in state.get("node_history", []) if h.get("status") == "ok"}

        while start_idx < len(full_sequence):
            if state.get("current_node") == full_sequence[start_idx]:
                start_idx += 1; continue
            if full_sequence[start_idx] in completed_nodes:
                start_idx += 1; continue

            node_name = full_sequence[start_idx]
            _run_node(state, node_name)
            state["current_node"] = node_name

            # 注意：此处【不】自动确认节点对应的 HITL。
            # 每个生产节点跑完后应当停留在它自己的 HITL 上等待用户确认
            # （content_production_serial→HITL-4, quality_review→HITL-7），
            # 若在此自动确认会把「下一个确认点」提前到尚未生产的节点，造成错位。

            next_hitl = _get_next_pending_hitl(state)
            if next_hitl:
                break
            start_idx += 1
        return

    # skip_all：依次执行所有剩余节点，每个节点的输出会传递到下一个
    for i in range(start_idx, len(full_sequence)):
        node_name = full_sequence[i]
        try:
            _run_node(state, node_name)
            state["current_node"] = node_name
        except Exception as e:
            # 单节点失败不阻断后续节点
            state["errors"] = state.get("errors", []) + [{
                "node": node_name,
                "error": str(e),
            }]


def _get_next_pending_hitl(state: CourseState) -> Optional[dict]:
    """获取下一个待处理的 HITL"""
    hitl_status = state.get("hitl_status", {})
    if state.get("skip_all_hitl"):
        return None

    # 需求解析追问阶段（HITL-1 处于 "asking"）尚未到达任何真实 HITL 确认点。
    # 若在此返回后续 HITL（HITL-2~7 在初始化时即被置为 "pending"），前端进度轮询会把
    # pendingHitl 覆盖成后续确认点，与 sendMessage 设置的状态冲突，导致输入框竞态消失。
    if hitl_status.get("HITL-1") == "asking":
        return None

    for hid in sorted(HITL_DEFINITIONS.keys(), key=lambda h: HITL_DEFINITIONS[h]["order"]):
        if hitl_status.get(hid) == "pending":
            return {
                "hitl_id": hid,
                "label": HITL_DEFINITIONS[hid]["label"],
                "status": "pending",
            }
    return None


# ============================================================
# M5: 打包下载
# ============================================================

@app.get("/api/course/{session_id}/download")
async def download_package(session_id: str):
    """下载课程 ZIP 包"""
    from fastapi.responses import FileResponse
    from pathlib import Path

    session = _session_store.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="会话不存在或已过期")

    state = session.get("state", {})
    package_path = state.get("_package_path")
    # v20: 用户直接点「下载课程包」时，若尚未打包或包文件丢失，先 lazy 补打一次。
    # 此前打包只在 progress 轮询的 safety-net 中触发，download 端点本身不重试，
    # 导致「永远返回『课程包尚未生成』」的死结。取值已全面兜底（or {} / or []），
    # 即使个别节点内容缺失也会生成骨架包，run_packaging 不再崩溃。
    if not package_path or not Path(package_path).exists():
        try:
            from ..agents import run_packaging
            run_packaging(state)
            _session_store.save(session_id, state, user_id=state.get("user_id"), completed=True)
            package_path = state.get("_package_path")
        except Exception as e:
            raise HTTPException(status_code=404, detail=f"课程包生成失败: {e}")

    if not package_path or not Path(package_path).exists():
        raise HTTPException(status_code=404, detail="课程包尚未生成，请稍后重试")

    outline = state.get("course_outline") or {}
    title = outline.get("course_title", "课程包")
    safe_title = title.replace(" ", "_").replace("/", "_")[:50]

    return FileResponse(
        path=package_path,
        filename=f"{safe_title}.zip",
        media_type="application/zip",
    )


# ============================================================
# M5: MCP Server 管理
# ============================================================

@app.get("/api/mcp/status")
async def mcp_status():
    """获取所有 MCP Server 状态"""
    from ..mcp_servers import mcp_registry
    return {"servers": mcp_registry.get_all_status()}


@app.post("/api/mcp/health-check")
async def mcp_health_check():
    """对所有 MCP Server 执行健康检查"""
    from ..mcp_servers import mcp_registry
    results = await mcp_registry.health_check_all()
    return {"results": results, "servers": mcp_registry.get_all_status()}


# ============================================================
# Voice: 语音合成
# ============================================================

@app.get("/api/voices")
async def list_voices():
    """获取可用 TTS 语音列表（自动检测 CosyVoice2 或 Edge TTS）"""
    import os
    from ..tools.tts_factory import get_available_voices
    api_key = os.getenv("SILICONFLOW_API_KEY", "")
    return {"voices": get_available_voices(api_key=api_key)}


class VoiceCloneRequest(BaseModel):
    audio_base64: str = Field(..., description="参考音频 base64 编码")
    audio_name: str = Field(default="reference.mp3", description="音频文件名")
    text: str = Field(default="", description="参考音频对应文本")


class VoiceTestRequest(BaseModel):
    voice: str = Field(..., description="音色 ID")
    text: str = Field(..., description="试听文本")


@app.post("/api/voice/clone")
async def voice_clone(request: VoiceCloneRequest):
    """语音克隆 — 上传参考音频创建克隆音色"""
    import os, base64, tempfile
    from pathlib import Path
    from ..tools.tts_factory import SiliconFlowTTS

    api_key = os.getenv("SILICONFLOW_API_KEY", "")
    if not api_key:
        raise HTTPException(status_code=400, detail="未配置 SILICONFLOW_API_KEY，语音克隆不可用")

    # 参考文本必填：缺少它克隆出的音色在合成时会报错(50507)或回吐参考音频，
    # 表现为「试听内容与输入完全无关、且时长异常」。从源头拦截残缺克隆。
    if not request.text or not request.text.strip():
        raise HTTPException(
            status_code=400,
            detail="请先填写『参考音频对应文本』（即参考音频中实际念出的文字）再克隆。"
                   "缺少参考文本会导致克隆音色无法正常合成，试听会出现内容/时长都不对的情况。",
        )

    # 将 base64 写入临时文件
    try:
        audio_bytes = base64.b64decode(request.audio_base64)
    except Exception:
        raise HTTPException(status_code=400, detail="无效的音频 base64 编码")

    suffix = Path(request.audio_name).suffix or ".mp3"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        f.write(audio_bytes)
        tmp_path = f.name

    try:
        engine = SiliconFlowTTS(
            output_dir=Path("/tmp"),
            api_key=api_key,
            clone_audio_path=tmp_path,
            clone_text=request.text,
        )
        voice_uri = engine.clone_voice()
        return {"voice_uri": voice_uri, "message": "克隆成功"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"语音克隆失败: {str(e)}")
    finally:
        try:
            Path(tmp_path).unlink()
        except Exception:
            pass


@app.post("/api/voice/test")
async def voice_test(request: VoiceTestRequest):
    """试听合成 — 用指定音色合成文本并返回音频"""
    import os, tempfile
    from pathlib import Path
    from fastapi.responses import Response
    from ..tools.tts_factory import create_tts_engine

    api_key = os.getenv("SILICONFLOW_API_KEY", "")

    # 参数校验：拒绝无效/占位音色，给出明确提示而非底层报错
    if not request.voice or request.voice == "__clone__":
        raise HTTPException(
            status_code=400,
            detail="请选择有效音色：先上传参考音频完成克隆，或直接选择 CosyVoice2 预置音色",
        )
    if not request.text or not request.text.strip():
        raise HTTPException(status_code=400, detail="请输入试听文本")

    with tempfile.TemporaryDirectory() as tmpdir:
        engine = create_tts_engine(
            output_dir=Path(tmpdir),
            siliconflow_api_key=api_key,
            voice=request.voice,
        )
        try:
            mp3_path = engine.synthesize(request.text, "test_preview")
            with open(mp3_path, "rb") as f:
                audio_data = f.read()
            return Response(content=audio_data, media_type="audio/mpeg")
        except Exception as e:
            # 前端已统一加「试听合成失败: 」前缀，这里只透传内部错误明细
            raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# F5: 数字人视频生成（RunningHub digital_customize，纯云端 API）
# ============================================================

_dh_client = RunningHubClient()


@app.post("/api/digital-human/generate")
async def digital_human_generate(
    image: UploadFile = File(..., description="肖像照（数字人面部）"),
    audio: UploadFile = File(..., description="旁白音频（CosyVoice2 合成，mp3）"),
    resolution: str = Form("1280x720", description="输出分辨率，如 1280x720"),
    prompt: str = Form("", description="视觉风格提示词（可选）"),
):
    """提交数字人口播视频生成任务：肖像图 + 旁白音频 -> task_id

    前端契约（DigitalHumanPage.tsx）：
      FormData: image(文件) / audio(blob,mp3) / resolution / prompt
      返回: { task_id }
    """
    if not image or not getattr(image, "filename", None):
        raise HTTPException(status_code=400, detail="请上传肖像照（数字人面部）")
    if not audio or not getattr(audio, "filename", None):
        raise HTTPException(status_code=400, detail="旁白音频缺失，请先完成语音合成")

    try:
        image_bytes = await image.read()
        audio_bytes = await audio.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"读取上传文件失败: {str(e)}")
    finally:
        try:
            await image.close()
            await audio.close()
        except Exception:
            pass

    if not image_bytes:
        raise HTTPException(status_code=400, detail="肖像图内容为空")
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="旁白音频内容为空")

    try:
        task_id = _dh_client.submit(image_bytes, audio_bytes, resolution, prompt)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"提交数字人生成任务失败: {str(e)}")

    return {"task_id": task_id}


@app.get("/api/digital-human/task/{task_id}")
async def digital_human_task(task_id: str):
    """轮询数字人生成任务状态

    前端契约：返回 { status: 'done'|'failed'|'processing', video_url, error }
    非 done/failed 的状态视为「继续轮询」。
    """
    try:
        result = _dh_client.get_status(task_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"查询任务状态失败: {str(e)}")
    return result


# ============================================================
# M6: 课程内容查询
# ============================================================

@app.get("/api/course/{session_id}/content/{content_type}")
async def get_course_content(session_id: str, content_type: str):
    """获取课程特定内容类型（供前端预览）

    content_type: outline|ip_report|scripts|slides|cases|marketing|pricing|review
    """
    session = _session_store.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="会话不存在或已过期")

    state = session["state"]
    content_map = {
        "outline": state.get("course_outline"),
        "ip_report": state.get("ip_positioning"),
        "scripts": state.get("scripts"),
        "slides": state.get("slides"),
        "cases": state.get("cases"),
        "marketing": state.get("marketing_copy"),
        "pricing": state.get("pricing_plan"),
        "review": state.get("review_detail"),
    }

    if content_type not in content_map:
        raise HTTPException(status_code=400, detail=f"无效的内容类型: {content_type}")

    return {
        "session_id": session_id,
        "content_type": content_type,
        "data": content_map[content_type],
    }


# ============================================================
# M7: 系统统计
# ============================================================

@app.get("/api/stats")
async def system_stats():
    """系统统计信息"""
    total_sessions = _session_store.total_sessions()
    active_sessions = _session_store.active_sessions()
    completed_sessions = total_sessions - active_sessions

    return {
        "total_sessions": total_sessions,
        "active_sessions": active_sessions,
        "completed_sessions": completed_sessions,
        "version": "0.3.0",
    }


# ============================================================
# 静态前端 + 根路径入口
# ============================================================

import os as _os
_STATIC_DIR = _os.path.join(_os.path.dirname(__file__), "..", "static")
_STATIC_DIR = _os.path.abspath(_STATIC_DIR)

# 挂载静态文件目录（/static/*）
app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")

# 根路径直接返回前端页面
@app.get("/")
async def root():
    return FileResponse(_os.path.join(_STATIC_DIR, "index.html"))


# ============================================================
# 健康检查
# ============================================================

@app.get("/health")
async def health():
    return {"status": "ok", "version": "0.3.0", "sessions": _session_store.total_sessions()}
