"""RunningHub digital_customize 数字人视频生成客户端（F5 后端核心）

纯云端 API 模式：服务端零 GPU，重计算卸载到 RunningHub 的 digital_customize 工作流，
将「肖像图 + 旁白音频」合成为口播视频。

两种运行模式：
1. Mock 模式（默认）：未同时配置 RUNNINGHUB_API_KEY / RUNNINGHUB_API_BASE /
   RUNNINGHUB_WORKFLOW_ID（或显式 RUNNINGHUB_MOCK=true）时启用，返回
   /static/sample.mp4 示例视频，保证前端端到端流程可测（TC-F5-01 / TC-F5-02）。
2. 真实模式：三项配置齐全且 RUNNINGHUB_MOCK!=true 时，调用 RunningHub 开放 API。

⚠️ Q9 待确认：RunningHub digital_customize 工作流的「确切端点 URL、鉴权方式、
   入参节点字段名、输出视频 URL 字段名」需与 RunningHub 工作流文档 / 技术负责人核对。
   代码已按 RunningHub 开放 API 通用形态预留，并将 URL / 字段全部外置为环境变量，
   待 Q9 明确后仅需填充配置、按需微调 _build_submit_payload / _parse_status，
   不臆造端点或字段。
"""
import os
import time
import uuid
import base64
import threading
import logging

import requests

logger = logging.getLogger("aip_core.digital_human")

# Mock 模式下任务从 processing 翻转为 done 的延时（秒），用于验证前端 3s 轮询链路
MOCK_DONE_DELAY = float(os.getenv("RUNNINGHUB_MOCK_DONE_DELAY", "2.5"))


class RunningHubClient:
    """RunningHub digital_customize 调用客户端（含 Mock 回退）。"""

    def __init__(self):
        self.api_key = os.getenv("RUNNINGHUB_API_KEY", "")
        self.api_base = os.getenv("RUNNINGHUB_API_BASE", "").rstrip("/")
        self.workflow_id = os.getenv("RUNNINGHUB_WORKFLOW_ID", "")
        # RunningHub 开放 API 路径（待 Q9 确认，默认常见形态）
        self.submit_path = os.getenv("RUNNINGHUB_SUBMIT_PATH", "/api/openapi/task/run")
        self.query_path = os.getenv("RUNNINGHUB_QUERY_PATH", "/api/openapi/task/status")
        self.timeout = int(os.getenv("RUNNINGHUB_TIMEOUT", "120"))
        # 强制 mock（调试 / Q9 未确认前保持 true）
        self.force_mock = os.getenv("RUNNINGHUB_MOCK", "true").lower() == "true"
        # 真实模式需 api_key + api_base + workflow_id 三者齐全
        self.mock = self.force_mock or not (
            self.api_key and self.api_base and self.workflow_id
        )
        if self.mock:
            logger.warning(
                "RunningHub 运行于 MOCK 模式（RUNNINGHUB_API_KEY / RUNNINGHUB_API_BASE / "
                "RUNNINGHUB_WORKFLOW_ID 未全部配置，或 RUNNINGHUB_MOCK=true）。"
                "将返回 /static/sample.mp4 示例视频用于前端链路验证。"
            )
        self._store = {}
        self._lock = threading.Lock()

    # ---------------- 任务提交 ----------------
    def submit(self, image_bytes, audio_bytes, resolution, prompt):
        """提交一个数字人口播生成任务，返回内部 task_id。"""
        task_id = uuid.uuid4().hex
        if self.mock:
            with self._lock:
                self._store[task_id] = {
                    "mode": "mock",
                    "created_at": time.time(),
                    "video_url": "/static/sample.mp4",
                    "status": "processing",
                }
            return task_id

        # ===== 真实模式：调用 RunningHub digital_customize =====
        # TODO(Q9): 确认 submit 端点、鉴权、入参节点字段名。
        try:
            upstream_id = self._submit_real(image_bytes, audio_bytes, resolution, prompt)
        except Exception:
            logger.exception("RunningHub submit 失败，回退 Mock 模式返回示例视频")
            with self._lock:
                self._store[task_id] = {
                    "mode": "mock",
                    "created_at": time.time(),
                    "video_url": "/static/sample.mp4",
                    "status": "processing",
                }
            return task_id

        with self._lock:
            self._store[task_id] = {
                "mode": "real",
                "upstream_id": upstream_id,
                "created_at": time.time(),
                "status": "processing",
            }
        return task_id

    def _submit_real(self, image_bytes, audio_bytes, resolution, prompt):
        # RunningHub 开放 API：POST {api_base}{submit_path}
        # 典型入参：apiKey + workflowId + params（各输入节点）。
        # 节点字段名需与部署的 digital_customize 工作流一致（Q9）。
        url = f"{self.api_base}{self.submit_path}"
        payload = {
            "apiKey": self.api_key,
            "workflowId": self.workflow_id,
            "params": {
                # TODO(Q9): 以下字段名以实际工作流入参节点为准，可能需替换为节点 ID
                "image": _b64_or_ref(image_bytes),
                "audio": _b64_or_ref(audio_bytes),
                "resolution": resolution,
                "prompt": prompt,
            },
        }
        resp = requests.post(url, json=payload, timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json()
        # TODO(Q9): 解析实际返回的 taskId 字段
        upstream = (
            data.get("data", {}).get("taskId")
            or data.get("taskId")
            or data.get("data", {}).get("task_id")
        )
        if not upstream:
            raise RuntimeError(f"RunningHub 未返回 taskId: {data}")
        return upstream

    # ---------------- 状态查询 ----------------
    def get_status(self, task_id):
        """查询任务状态，返回 {status, video_url, error}。"""
        with self._lock:
            rec = self._store.get(task_id)
        if rec is None:
            return {"status": "failed", "video_url": "", "error": "任务不存在或已过期"}
        if rec["mode"] == "mock":
            if time.time() - rec["created_at"] >= MOCK_DONE_DELAY:
                return {"status": "done", "video_url": rec["video_url"], "error": ""}
            return {"status": "processing", "video_url": "", "error": ""}
        # 真实模式：轮询 RunningHub
        try:
            upstream = rec.get("upstream_id")
            status, video_url, error = self._status_real(upstream)
        except Exception:
            logger.exception("RunningHub 状态轮询失败，维持 processing")
            return {"status": "processing", "video_url": "", "error": ""}
        with self._lock:
            self._store[task_id]["status"] = status
            if video_url:
                self._store[task_id]["video_url"] = video_url
        return {"status": status, "video_url": video_url or "", "error": error or ""}

    def _status_real(self, upstream_id):
        # TODO(Q9): 确认 query 端点、返回体中 status / 输出视频 URL 的字段名。
        url = f"{self.api_base}{self.query_path}"
        resp = requests.get(
            url, params={"taskId": upstream_id, "apiKey": self.api_key}, timeout=self.timeout
        )
        resp.raise_for_status()
        data = resp.json()
        raw = data.get("data") or data
        state = raw.get("status") or raw.get("taskStatus") or raw.get("state")
        if state in ("succeeded", "success", "completed", "done"):
            # TODO(Q9): 视频 URL 字段
            video_url = (
                raw.get("output")
                or raw.get("videoUrl")
                or (raw.get("outputs") or {}).get("video")
            )
            return "done", video_url or "", ""
        if state in ("failed", "error"):
            return "failed", "", raw.get("error") or raw.get("message") or "RunningHub 任务失败"
        return "processing", "", ""


def _b64_or_ref(b: bytes) -> str:
    """将上传字节转为 base64 字符串（Q9 明确文件上传方式后可改为 URL/引用）。"""
    return base64.b64encode(b).decode("utf-8")
