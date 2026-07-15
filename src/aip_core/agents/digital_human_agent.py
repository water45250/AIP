"""Digital Human Agent - 数字人视频合成节点

基于 Duix-Avatar 开源数字人引擎，将讲稿 + 克隆语音合成数字人讲课视频。

支持两种部署模式：
1. 本地部署 API（推荐，支持批量合成）
   - 视频合成: POST /easy/submit + GET /easy/query
   - 需 GPU 服务器（RTX 4070+, 32GB+ RAM）
2. 云端 API（实时交互场景，批量合成支持有限）
   - 参考: https://docs.duix.com/

环境变量：
- DUIX_API_BASE: Duix-Avatar API 地址（默认 http://127.0.0.1:8383）
- DUIX_API_KEY: 云端 API Key（使用云端时必填）
- DUIX_DEFAULT_VIDEO: 默认数字人视频模板路径
- DUIX_ENABLED: 是否启用数字人（默认 false，仅当部署了 Duix 服务时启用）
"""

import os
import uuid
import time
import json
from pathlib import Path
from typing import Optional

import requests

from ..graph.state import CourseState
from ..config import DATA_DIR


class DuixAvatarClient:
    """Duix-Avatar 数字人客户端

    封装视频合成 API 调用（本地部署模式）：
    - 提交合成任务: POST /easy/submit
    - 查询进度: GET /easy/query?code={taskCode}
    - 下载结果
    """

    def __init__(
        self,
        api_base: str = "http://127.0.0.1:8383",
        default_video: str = "",
        api_key: str = "",
        poll_interval: float = 3.0,
        max_wait_seconds: float = 600.0,
    ):
        """
        Args:
            api_base: Duix-Avatar API 地址
            default_video: 默认数字人视频模板路径
            api_key: 云端 API Key（本地部署可为空）
            poll_interval: 轮询间隔（秒）
            max_wait_seconds: 最大等待时间（秒）
        """
        self.api_base = api_base.rstrip("/")
        self.default_video = default_video
        self.api_key = api_key
        self.poll_interval = poll_interval
        self.max_wait_seconds = max_wait_seconds

    def _headers(self) -> dict:
        h = {"Content-Type": "application/json"}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        return h

    def submit_synthesis(
        self,
        audio_path: str,
        video_path: str = "",
        task_code: str = "",
    ) -> str:
        """提交视频合成任务

        Args:
            audio_path: 音频文件路径
            video_path: 数字人视频模板路径（为空则用 default_video）
            task_code: 任务唯一标识（为空则自动生成）

        Returns:
            task_code: 任务标识码，用于查询进度

        Raises:
            RuntimeError: API 调用失败
        """
        code = task_code or str(uuid.uuid4())
        video = video_path or self.default_video

        if not video:
            raise RuntimeError("未指定数字人视频模板，请设置 DUIX_DEFAULT_VIDEO 环境变量")

        payload = {
            "audio_url": audio_path,
            "video_url": video,
            "code": code,
            "chaofen": 0,
            "watermark_switch": 0,
            "pn": 1,
        }

        resp = requests.post(
            f"{self.api_base}/easy/submit",
            headers=self._headers(),
            json=payload,
            timeout=30,
        )

        if resp.status_code != 200:
            raise RuntimeError(f"Duix 视频合成提交失败 (HTTP {resp.status_code}): {resp.text[:500]}")

        data = resp.json()
        # 检查返回（本地 API 通常返回简单状态）
        if isinstance(data, dict) and data.get("error"):
            raise RuntimeError(f"Duix 视频合成提交失败: {data['error']}")

        return code

    def query_progress(self, task_code: str) -> dict:
        """查询合成进度

        Args:
            task_code: 任务标识码

        Returns:
            {"status": "processing"|"completed"|"failed", "video_url": str, "progress": float}
        """
        resp = requests.get(
            f"{self.api_base}/easy/query",
            params={"code": task_code},
            headers=self._headers(),
            timeout=15,
        )

        if resp.status_code != 200:
            return {"status": "failed", "error": f"HTTP {resp.status_code}", "video_url": "", "progress": 0}

        data = resp.json()

        # 解析不同响应格式
        if isinstance(data, dict):
            status = data.get("status", data.get("state", ""))
            video_url = data.get("video_url", data.get("result_url", ""))
            progress = data.get("progress", data.get("percent", 0))

            # 标准化状态
            if status in ("completed", "done", "success", "finish"):
                return {"status": "completed", "video_url": video_url, "progress": 100}
            elif status in ("failed", "error"):
                return {"status": "failed", "error": data.get("message", data.get("error", "")), "video_url": "", "progress": 0}
            else:
                return {"status": "processing", "video_url": "", "progress": progress or 50}
        else:
            return {"status": "processing", "video_url": "", "progress": 0}

    def wait_for_completion(self, task_code: str) -> dict:
        """轮询等待合成完成

        Args:
            task_code: 任务标识码

        Returns:
            {"status": "completed"|"failed", "video_url": str, "elapsed_seconds": float}
        """
        start_time = time.time()

        while True:
            elapsed = time.time() - start_time
            if elapsed > self.max_wait_seconds:
                return {
                    "status": "timeout",
                    "video_url": "",
                    "elapsed_seconds": elapsed,
                    "error": f"合成超时（>{self.max_wait_seconds}s）",
                }

            result = self.query_progress(task_code)

            if result["status"] in ("completed", "failed"):
                result["elapsed_seconds"] = elapsed
                return result

            time.sleep(self.poll_interval)

    def download_video(self, video_url: str, output_path: str) -> str:
        """下载合成后的视频文件

        Args:
            video_url: 视频 URL 或本地路径
            output_path: 输出路径

        Returns:
            输出文件路径
        """
        # 本地路径直接复制
        if not video_url.startswith(("http://", "https://")):
            import shutil
            if Path(video_url).exists():
                Path(output_path).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(video_url, output_path)
                return output_path
            else:
                raise FileNotFoundError(f"视频文件不存在: {video_url}")

        # HTTP 下载
        resp = requests.get(video_url, timeout=300, stream=True)
        resp.raise_for_status()

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)

        return output_path


# ============================================================
# LangGraph 节点函数
# ============================================================

def run_digital_human(state: CourseState) -> CourseState:
    """Digital Human 节点 - LangGraph 节点函数

    为每个课时的讲稿 + 配音生成数字人讲课视频。

    前置条件：
    - scripts（讲稿文本）
    - audio_files（TTS 合成的 MP3 音频）
    - Duix-Avatar 服务已部署

    无音频或未启用时自动跳过。
    """
    start_time = time.time()
    session_id = state.get("session_id", "unknown")

    # 检查是否启用
    duix_enabled = os.getenv("DUIX_ENABLED", "false").lower() == "true"
    if not duix_enabled:
        state["digital_human_mode"] = "disabled"
        state["digital_human_videos"] = {}
        state["digital_human_progress"] = {"total": 0, "completed": 0}
        state["current_node"] = "digital_human"
        return state

    audio_files = state.get("audio_files", {})
    scripts = state.get("scripts", {})

    # 无音频 → 跳过
    if not audio_files:
        state["digital_human_mode"] = "skipped_no_audio"
        state["digital_human_videos"] = {}
        state["digital_human_progress"] = {"total": 0, "completed": 0}
        state["current_node"] = "digital_human"
        return state

    # 创建客户端
    api_base = os.getenv("DUIX_API_BASE", "http://127.0.0.1:8383")
    api_key = os.getenv("DUIX_API_KEY", "")
    default_video = os.getenv("DUIX_DEFAULT_VIDEO", "")

    client = DuixAvatarClient(
        api_base=api_base,
        default_video=default_video,
        api_key=api_key,
    )

    # 输出目录
    output_dir = Path(DATA_DIR) / "digital_human" / session_id
    output_dir.mkdir(parents=True, exist_ok=True)

    total = len(audio_files)
    state["digital_human_mode"] = "duix_avatar"
    state["digital_human_progress"] = {"total": total, "completed": 0, "current_lesson": ""}

    video_files = {}
    failed_lessons = []

    for i, (lesson_id, mp3_path) in enumerate(sorted(audio_files.items())):
        state["digital_human_progress"]["current_lesson"] = lesson_id

        if not Path(mp3_path).exists():
            failed_lessons.append({"lesson_id": lesson_id, "error": f"音频文件不存在: {mp3_path}"})
            continue

        try:
            # Step 1: 提交合成任务
            task_code = client.submit_synthesis(
                audio_path=mp3_path,
                video_path="",  # 使用 default_video
                task_code=f"{session_id}_{lesson_id}",
            )

            # Step 2: 轮询等待完成
            result = client.wait_for_completion(task_code)

            if result["status"] == "completed" and result.get("video_url"):
                # Step 3: 下载视频
                video_path = output_dir / f"{lesson_id}.mp4"
                client.download_video(result["video_url"], str(video_path))
                video_files[lesson_id] = str(video_path)
                state["digital_human_progress"]["completed"] = i + 1
            else:
                error_msg = result.get("error", "未知错误")
                failed_lessons.append({"lesson_id": lesson_id, "error": error_msg})

        except Exception as e:
            failed_lessons.append({"lesson_id": lesson_id, "error": str(e)})
            state["errors"] = state.get("errors", []) + [{
                "node": "digital_human",
                "lesson_id": lesson_id,
                "error": f"数字人视频合成失败: {str(e)}",
                "time": time.time(),
            }]

    # 更新状态
    state["digital_human_videos"] = video_files
    state["current_node"] = "digital_human"
    state["node_history"] = state.get("node_history", []) + [{
        "node": "digital_human",
        "start": start_time,
        "end": time.time(),
        "status": "ok" if not failed_lessons else "partial",
        "mode": state["digital_human_mode"],
        "video_count": len(video_files),
        "total_lessons": total,
        "failed_lessons": failed_lessons,
    }]

    return state
