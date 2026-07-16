"""数字人视频生成 P0 MVP — API 路由（WaveSpeed Hunyuan Avatar）。

端点（挂载到 aip-core 主应用，前缀 /api/digital-human-p0）：
  POST /api/digital-human-p0/generate   单条生成（文件或 URL）
  POST /api/digital-human-p0/batch       批量生成（URL 入参）
  GET  /api/digital-human-p0/task/{id}  轮询任务状态

未配置 WAVESPEED_API_KEY 时进入 Mock 模式（返回示例视频）。
"""
from __future__ import annotations

import os
import tempfile
import threading
import uuid
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, Form, UploadFile
from pydantic import BaseModel, Field

from ..digital_human_p0.core import build_batch_inputs, build_input
from ..digital_human_p0.wavespeed_client import default_uploader, generate

router = APIRouter(prefix="/api/digital-human-p0", tags=["digital-human-p0"])

_tasks: dict[str, dict] = {}


def _is_mock() -> bool:
    return not bool(os.environ.get("WAVESPEED_API_KEY"))


def _save_upload(upload: Optional[UploadFile]) -> Optional[str]:
    if not upload or not upload.filename:
        return None
    suffix = Path(upload.filename).suffix or ".bin"
    fd, path = tempfile.mkstemp(suffix=suffix, prefix="aip_dh_")
    with os.fdopen(fd, "wb") as f:
        f.write(upload.file.read())
    return path


class GenerateRequest(BaseModel):
    image_url: Optional[str] = None
    audio_url: Optional[str] = None
    resolution: str = "480p"
    prompt: Optional[str] = None
    seed: Optional[int] = None
    api_key: Optional[str] = None


class BatchItem(GenerateRequest):
    """批量项（JSON，基于 URL；文件上传走单条 /generate）。"""


class BatchRequest(BaseModel):
    items: list[BatchItem]


def _run(task_id: str, payload: dict) -> None:
    try:
        if _is_mock():
            video_url = "/static/sample.mp4"
        else:
            result = generate(
                image=payload["image"],
                audio=payload["audio"],
                resolution=payload.get("resolution", "480p"),
                prompt=payload.get("prompt"),
                seed=payload.get("seed"),
                api_key=os.environ.get("WAVESPEED_API_KEY"),
            )
            outputs = result.get("outputs") or []
            video_url = outputs[0] if outputs else "/static/sample.mp4"
        _tasks[task_id].update(status="done", video_url=video_url)
    except Exception as exc:  # noqa: BLE001
        _tasks[task_id].update(status="failed", error=str(exc))


@router.post("/generate")
async def generate_task(
    image: Optional[UploadFile] = File(None),
    audio: Optional[UploadFile] = File(None),
    image_url: Optional[str] = Form(None),
    audio_url: Optional[str] = Form(None),
    resolution: str = Form("480p"),
    prompt: Optional[str] = Form(None),
    seed: Optional[int] = Form(None),
    api_key: Optional[str] = Form(None),
):
    if api_key:
        os.environ["WAVESPEED_API_KEY"] = api_key
    # mock 模式（無 WAVESPEED_API_KEY）：用佔位 uploader，避免同步上傳崩潰
    uploader = (lambda p: f"/mock/{Path(p).name}") if _is_mock() else default_uploader
    inp = build_input(
        image_file=_save_upload(image),
        image_url=image_url,
        audio_file=_save_upload(audio),
        audio_url=audio_url,
        resolution=resolution,
        prompt=prompt,
        seed=seed,
        uploader=uploader,
    )
    task_id = uuid.uuid4().hex
    _tasks[task_id] = {"status": "running", "input": inp}
    threading.Thread(target=_run, args=(task_id, inp), daemon=True).start()
    return {"task_id": task_id}


@router.post("/batch")
async def batch_task(items: BatchRequest):
    inputs = build_batch_inputs([it.model_dump(exclude_none=True) for it in items.items])
    batch_id = uuid.uuid4().hex
    _tasks[batch_id] = {"status": "running", "inputs": inputs, "results": [None] * len(inputs)}
    return {"batch_id": batch_id, "count": len(inputs), "inputs": inputs}


@router.get("/task/{task_id}")
async def task_status(task_id: str):
    task = _tasks.get(task_id)
    if not task:
        return {"status": "not_found"}
    return {
        "status": task.get("status"),
        "video_url": task.get("video_url"),
        "error": task.get("error"),
    }
