"""数字人视频生成 P0 MVP — 核心逻辑（可单测）。

把一次生成请求（图片 / 音频的「文件」或「URL」）转换为
WaveSpeed `hunyuan-avatar` 所需的 input 字典。

批量真实分支规则（即「批量真实分支」测试的断言依据）：
- 文件（image_file / audio_file）→ 经 upload 换取 WaveSpeed URL 后进入 input；
- URL（image_url / audio_url）→ 直接进 input，**不经过 upload**；
- 因此批量任务里，upload 调用次数 == 提供「文件」的项数，URL 项不产生 upload。

upload 行为通过依赖注入（`uploader`）传入，便于测试用 mock 替换，
也避免核心逻辑在单测时真正触达 WaveSpeed。
"""
from __future__ import annotations

from typing import Callable, Optional

from .wavespeed_client import default_uploader

# uploader: 本地路径 -> WaveSpeed 托管 URL（如 http://up/a1.mp3）
Uploader = Callable[[str], str]


class MediaMissingError(ValueError):
    """既没给文件也没给 URL 时抛出。"""


def resolve_media(file: Optional[str], url: Optional[str], uploader: Uploader) -> str:
    """把「文件或 URL」解析成最终进入 input 的 URL。

    - 给了文件：upload 后返回托管 URL（如 http://up/a1.mp3）
    - 给了 URL：原样返回（如 http://u/a2.mp3），**不触发 upload**
    - 都没给：报错
    """
    if file:
        return uploader(file)
    if url:
        return url
    raise MediaMissingError("必须提供 file 或 url 之一")


def build_input(
    *,
    image_file: Optional[str] = None,
    image_url: Optional[str] = None,
    audio_file: Optional[str] = None,
    audio_url: Optional[str] = None,
    resolution: str = "480p",
    prompt: Optional[str] = None,
    seed: Optional[int] = None,
    uploader: Optional[Uploader] = None,
) -> dict:
    """构建单次 hunyuan-avatar 生成的 input 字典。"""
    uploader = uploader or default_uploader
    return {
        "image": resolve_media(image_file, image_url, uploader),
        "audio": resolve_media(audio_file, audio_url, uploader),
        "resolution": resolution,
        **({"prompt": prompt} if prompt else {}),
        **({"seed": seed} if seed is not None else {}),
    }


def build_batch_inputs(tasks: list[dict], uploader: Optional[Uploader] = None) -> list[dict]:
    """批量构建 input 列表。每个 task 是 build_input 的关键字参数集合。"""
    return [build_input(uploader=uploader, **t) for t in tasks]
