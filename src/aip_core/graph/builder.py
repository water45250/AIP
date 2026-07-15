"""LangGraph 主状态图构建

基于 LangGraph 1.0+ 构建课程生成主图:
- 9 个核心节点（含数字人视频合成）
- 7 个 HITL 确认点 (基于 interrupt + Command)
- SQLite Checkpoint 持久化

路由策略:
- HITL 节点通过 Command(goto=...) 直接跳转到目标节点
- 审核节点通过 conditional_edges 决定修正/HITL-7/打包
"""

import sqlite3
import time
from typing import Literal
from pathlib import Path

from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command, interrupt

from .state import (
    CourseState,
    NODE_REQUIREMENT_ANALYSIS,
    NODE_IP_POSITIONING,
    NODE_COURSE_ARCHITECTURE,
    NODE_CONTENT_PARALLEL,
    NODE_CONTENT_SERIAL,
    NODE_VOICE_TTS,
    NODE_DIGITAL_HUMAN,
    NODE_QUALITY_REVIEW,
    NODE_PACKAGING,
    HITL_DEFINITIONS,
)
from ..config import SQLITE_DB_PATH, REVIEW_AUTO_SKIP_THRESHOLD, REVIEW_PASS_THRESHOLD

# ============================================================
# 节点实现 (占位 - 后续里程碑逐一替换为真实 Agent)
# ============================================================

def requirement_analysis_node(state: CourseState) -> CourseState:
    """需求解析节点 - Agent 1"""
    from ..agents.requirement_agent import run_requirement_analysis
    return run_requirement_analysis(state)


def ip_positioning_node(state: CourseState) -> CourseState:
    """IP 定位节点 - Agent 2"""
    from ..agents.ip_agent import run_ip_positioning
    return run_ip_positioning(state)


def course_architecture_node(state: CourseState) -> CourseState:
    """课程架构节点 - Agent 3"""
    from ..agents.course_architect_agent import run_course_architecture
    return run_course_architecture(state)


def content_parallel_node(state: CourseState) -> CourseState:
    """内容生产并行节点 - Agent 4-6"""
    from ..agents.content_agents import run_content_parallel
    return run_content_parallel(state)


def content_serial_node(state: CourseState) -> CourseState:
    """内容生产串行节点 - Agent 7-8"""
    from ..agents.content_agents import run_content_serial
    return run_content_serial(state)


def voice_tts_node(state: CourseState) -> CourseState:
    """语音合成节点"""
    from ..agents.voice_agent import run_voice_tts
    return run_voice_tts(state)


def digital_human_node(state: CourseState) -> CourseState:
    """数字人视频合成节点"""
    from ..agents.digital_human_agent import run_digital_human
    return run_digital_human(state)


def quality_review_node(state: CourseState) -> CourseState:
    """质量审核节点 - Agent 9"""
    from ..agents.review_agent import run_quality_review
    return run_quality_review(state)


def packaging_node(state: CourseState) -> CourseState:
    """打包交付节点"""
    from ..agents.packager import run_packaging
    return run_packaging(state)


# ============================================================
# HITL 确认点 - 基于 LangGraph interrupt + Command
# ============================================================

# 定义每个 HITL 节点的路由目标（Command goto 必须是已注册节点名）
HITL_ROUTES = {
    "HITL-1": {
        "continue": NODE_IP_POSITIONING,       # 确认 → IP定位
        "regenerate": NODE_REQUIREMENT_ANALYSIS, # 重生成 → 需求解析
    },
    "HITL-2": {
        "continue": NODE_COURSE_ARCHITECTURE,
        "regenerate": NODE_IP_POSITIONING,
    },
    "HITL-3": {
        "continue": NODE_CONTENT_PARALLEL,
        "regenerate": NODE_COURSE_ARCHITECTURE,
    },
    "HITL-4": {
        "continue": NODE_VOICE_TTS,              # 内容确认 → 语音合成
        "regenerate": NODE_CONTENT_PARALLEL,
    },
    "HITL-5": {
        "continue": NODE_DIGITAL_HUMAN,          # 语音确认 → 数字人视频
        "regenerate": NODE_VOICE_TTS,
    },
    "HITL-6": {
        "continue": NODE_QUALITY_REVIEW,         # 数字人确认 → 质量审核
        "regenerate": NODE_DIGITAL_HUMAN,
    },
    "HITL-7": {
        "continue": NODE_PACKAGING,              # 审核确认 → 打包
        "regenerate": NODE_CONTENT_SERIAL,
    },
}


def _create_hitl_node(hitl_id: str):
    """工厂函数：为每个 HITL 创建统一的确认节点

    HITL 节点通过 Command(goto=实际节点名) 实现路由，
    不再依赖 conditional_edges，避免节点名不匹配问题。
    """
    routes = HITL_ROUTES[hitl_id]

    def hitl_node(state: CourseState) -> Command:
        # 如果用户已一键跳过全部 HITL，直接继续
        if state.get("skip_all_hitl"):
            state["hitl_status"][hitl_id] = "skipped"
            return Command(goto=routes["continue"])

        # HITL-5 特殊处理：无讲稿时自动跳过语音合成确认
        if hitl_id == "HITL-5":
            if state.get("tts_mode") == "none":
                state["hitl_status"][hitl_id] = "skipped"
                return Command(goto=routes["continue"])

        # HITL-6 特殊处理：数字人未启用时自动跳过
        if hitl_id == "HITL-6":
            dh_mode = state.get("digital_human_mode", "disabled")
            if dh_mode in ("disabled", "skipped_no_audio"):
                state["hitl_status"][hitl_id] = "skipped"
                return Command(goto=routes["continue"])

        # HITL-7 特殊处理：审核 ≥ 85 自动跳过
        if hitl_id == "HITL-7":
            score = state.get("review_score", 0)
            if score >= REVIEW_AUTO_SKIP_THRESHOLD:
                state["hitl_status"][hitl_id] = "skipped"
                return Command(goto=routes["continue"])

        # 调用 interrupt 暂停执行，等待用户响应
        hitl_def = HITL_DEFINITIONS[hitl_id]
        decision = interrupt({
            "hitl_id": hitl_id,
            "label": hitl_def["label"],
            "node": hitl_def["node"],
            "message": f"请确认 {hitl_def['label']} 的结果",
            "actions": ["confirm", "skip", "regenerate", "edit", "skip_all"],
        })

        # 处理用户决策
        action = decision.get("action", "confirm") if isinstance(decision, dict) else "confirm"

        if action == "skip_all":
            state["skip_all_hitl"] = True
            state["hitl_status"][hitl_id] = "skipped"
            return Command(goto=routes["continue"])
        elif action == "regenerate":
            state["hitl_status"][hitl_id] = "regenerating"
            return Command(goto=routes["regenerate"])
        elif action == "edit":
            edits = decision.get("edits", {})
            _apply_edits(state, hitl_def["node"], edits)
            state["hitl_status"][hitl_id] = "confirmed"
            return Command(goto=routes["continue"])
        else:  # confirm or skip
            state["hitl_status"][hitl_id] = "confirmed" if action == "confirm" else "skipped"
            return Command(goto=routes["continue"])

    return hitl_node


def _apply_edits(state: CourseState, node_name: str, edits: dict):
    """将用户编辑应用到状态中"""
    if node_name == NODE_REQUIREMENT_ANALYSIS and "user_profile" in edits:
        state["user_profile"] = edits["user_profile"]
    elif node_name == NODE_IP_POSITIONING and "ip_positioning" in edits:
        state["ip_positioning"] = edits["ip_positioning"]
    elif node_name == NODE_COURSE_ARCHITECTURE and "course_outline" in edits:
        state["course_outline"] = edits["course_outline"]


# ============================================================
# 条件路由函数（仅用于审核后的分支）
# ============================================================

def _route_after_review(state: CourseState) -> Literal["content_production_serial", "hitl_7", "packaging"]:
    """审核后路由：
    - <80分 且 <2轮修正 → 回到内容生产串行（修正循环）
    - ≥85分 → 直接打包
    - 80-84分 → 进入 HITL-7
    """
    score = state.get("review_score", 0)
    review_round = state.get("review_round", 0)

    if score < REVIEW_PASS_THRESHOLD and review_round <= MAX_REVIEW_ROUNDS:
        return NODE_CONTENT_SERIAL
    elif score >= REVIEW_AUTO_SKIP_THRESHOLD:
        return NODE_PACKAGING
    else:
        return "hitl_7"


MAX_REVIEW_ROUNDS = 2


# ============================================================
# 构建主图
# ============================================================

def build_course_graph(checkpointer: SqliteSaver = None) -> StateGraph:
    """构建课程生成主状态图

    Args:
        checkpointer: LangGraph Checkpointer 实例 (SQLite/Postgres/InMemory)

    Returns:
        编译后的 StateGraph
    """
    builder = StateGraph(CourseState)

    # --- 注册业务节点 ---
    builder.add_node(NODE_REQUIREMENT_ANALYSIS, requirement_analysis_node)
    builder.add_node(NODE_IP_POSITIONING, ip_positioning_node)
    builder.add_node(NODE_COURSE_ARCHITECTURE, course_architecture_node)
    builder.add_node(NODE_CONTENT_PARALLEL, content_parallel_node)
    builder.add_node(NODE_CONTENT_SERIAL, content_serial_node)
    builder.add_node(NODE_VOICE_TTS, voice_tts_node)
    builder.add_node(NODE_DIGITAL_HUMAN, digital_human_node)
    builder.add_node(NODE_QUALITY_REVIEW, quality_review_node)
    builder.add_node(NODE_PACKAGING, packaging_node)

    # --- 注册 HITL 确认节点 ---
    builder.add_node("hitl_1", _create_hitl_node("HITL-1"))
    builder.add_node("hitl_2", _create_hitl_node("HITL-2"))
    builder.add_node("hitl_3", _create_hitl_node("HITL-3"))
    builder.add_node("hitl_4", _create_hitl_node("HITL-4"))
    builder.add_node("hitl_5", _create_hitl_node("HITL-5"))
    builder.add_node("hitl_6", _create_hitl_node("HITL-6"))
    builder.add_node("hitl_7", _create_hitl_node("HITL-7"))

    # --- 编排边 ---

    # 入口 → 需求解析
    builder.add_edge(START, NODE_REQUIREMENT_ANALYSIS)

    # 需求解析 → HITL-1（HITL 节点内部通过 Command goto 路由）
    builder.add_edge(NODE_REQUIREMENT_ANALYSIS, "hitl_1")

    # IP 定位 → HITL-2
    builder.add_edge(NODE_IP_POSITIONING, "hitl_2")

    # 课程架构 → HITL-3
    builder.add_edge(NODE_COURSE_ARCHITECTURE, "hitl_3")

    # 内容生产并行 → 内容生产串行
    builder.add_edge(NODE_CONTENT_PARALLEL, NODE_CONTENT_SERIAL)

    # 内容生产串行 → HITL-4
    builder.add_edge(NODE_CONTENT_SERIAL, "hitl_4")

    # 语音合成 → HITL-5（语音确认）
    builder.add_edge(NODE_VOICE_TTS, "hitl_5")

    # 数字人视频 → HITL-6（数字人确认）
    builder.add_edge(NODE_DIGITAL_HUMAN, "hitl_6")

    # 质量审核 → 条件路由（修正/HITL-7/打包）
    builder.add_conditional_edges(
        NODE_QUALITY_REVIEW,
        _route_after_review,
        {
            NODE_CONTENT_SERIAL: NODE_CONTENT_SERIAL,
            NODE_PACKAGING: NODE_PACKAGING,
            "hitl_7": "hitl_7",
        }
    )

    # HITL-7 → 打包（HITL-7 内部 Command goto 到 packaging）
    builder.add_edge("hitl_7", NODE_PACKAGING)

    # 打包 → 结束
    builder.add_edge(NODE_PACKAGING, END)

    # --- 编译 ---
    return builder.compile(checkpointer=checkpointer)


def create_sqlite_checkpointer(db_path: str = None) -> SqliteSaver:
    """创建 SQLite Checkpointer

    Args:
        db_path: SQLite 数据库路径，默认使用配置中的路径

    Returns:
        SqliteSaver 实例
    """
    path = db_path or SQLITE_DB_PATH
    # 确保目录存在
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(path, check_same_thread=False)
    return SqliteSaver(conn)
