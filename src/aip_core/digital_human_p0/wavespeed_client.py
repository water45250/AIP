"""WaveSpeed 客户端封装（真实调用）。

测试时通过依赖注入把 `uploader` 替换为 mock，因此本模块的真实依赖
（wavespeed）只在真正发起上传时才被导入，单测无需安装 wavespeed。
"""
from __future__ import annotations

from typing import Optional

_API_KEY_ENV = "WAVESPEED_API_KEY"


def _make_client(api_key: Optional[str] = None):
    from wavespeed import Client  # 真实依赖，仅在使用时导入

    import os

    key = api_key or os.environ.get(_API_KEY_ENV)
    if not key:
        raise RuntimeError("未配置 WAVESPEED_API_KEY，无法真实上传")
    return Client(api_key=key)


def default_uploader(local_path: str, api_key: Optional[str] = None) -> str:
    """把本地文件上传到 WaveSpeed，返回托管 URL。

    对应 README：文件先落本地临时文件，真实调用时由
    wavespeed.Client.upload() 上传到 WaveSpeed 换取 URL。
    """
    return _make_client(api_key).upload(local_path)


def generate(
    image: str,
    audio: str,
    resolution: str = "480p",
    prompt: Optional[str] = None,
    seed: Optional[int] = None,
    api_key: Optional[str] = None,
) -> dict:
    """提交一次 hunyuan-avatar 生成并轮询结果。"""
    client = _make_client(api_key)
    inp = {"image": image, "audio": audio, "resolution": resolution}
    if prompt:
        inp["prompt"] = prompt
    if seed is not None:
        inp["seed"] = seed
    return client.run("wavespeed-ai/hunyuan-avatar", inp)
