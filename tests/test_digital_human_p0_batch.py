"""批量真实分支（batch real-branch）断言测试 —— 已修正版本。

业务背景：批量生成数字人视频时，每个任务可以是
  - 上传的文件（image_file / audio_file）→ 需经 upload 换取 URL
  - 直接给的 URL（image_url / audio_url）→ 跳过 upload，直进 input

本测试验证：
  1) task1 的音频是文件 → 走 upload → input.audio == http://up/a1.mp3
  2) task2 的音频是 URL  → 不调 upload → input.audio == http://u/a2.mp3（原始 URL）
  3) upload 仅被调用 1 次（只为 a1 文件），call_args_list 不含 a2

────────────────────────────────────────────────────────────────────
【修复点：基于参数的 upload + 修正断言集合】
────────────────────────────────────────────────────────────────────
旧实现用的是「按调用顺序消耗」的 side_effect 列表：

    upload.side_effect = ["http://up/a1.mp3", "http://up/a2.mp3"]

它有两个问题：
  (a) 偶发不够 —— 该列表按调用顺序被耗尽，一旦批量里某个本不该 upload
      的任务（如 task2 的 URL 音频）被错误地计入，列表会提前 StopIteration；
  (b) 断言写错 —— 旧断言错误地认为 task2 也走了 upload，写成
        assert inputs[1]["audio"] == "http://up/a2.mp3"      # 错
        assert upload.call_args_list == [call(a1), call(a2)]  # 错
      但 task2 的音频是 URL，直接进 input，calls 里根本不该有 a2。

修复做法：
  * 把 upload 改成「基于参数的」——side_effect 是一个以入参为键的函数，
    无论调用次数 / 顺序如何都稳定返回对应 URL，永不会耗尽；
  * 断言集合如实反映「URL 音频不触发 upload」的真实分支。
"""
from pathlib import Path
from unittest import mock

import pytest

from src.aip_core.digital_human_p0.core import MediaMissingError, build_batch_inputs


def _param_upload(local_path: str) -> str:
    """基于参数的 upload：用文件名映射托管 URL，不依赖调用顺序/次数。"""
    return f"http://up/{Path(local_path).name}"


def test_batch_real_branch_url_audio_skips_upload():
    """task1 音频为文件走 upload；task2 音频为 URL 直进 input、不调 upload。"""
    tasks = [
        {"image_url": "http://u/i1.jpg", "audio_file": "/tmp/a1.mp3", "resolution": "480p"},
        {"image_url": "http://u/i2.jpg", "audio_url": "http://u/a2.mp3", "resolution": "720p"},
    ]

    with mock.patch(
        "src.aip_core.digital_human_p0.core.default_uploader", side_effect=_param_upload
    ) as up:
        inputs = build_batch_inputs(tasks)

    # —— 断言集合（修正后）——
    # task1 音频经 upload：http://u/a1.mp3 -> http://up/a1.mp3
    assert inputs[0]["audio"] == "http://up/a1.mp3"
    # task2 音频是 URL：原样保留，绝不应出现 http://up/a2.mp3
    assert inputs[1]["audio"] == "http://u/a2.mp3"
    # upload 只为 a1 调用一次，call_args_list 不含 a2
    assert up.call_count == 1
    assert up.call_args_list == [mock.call("/tmp/a1.mp3")]


def test_batch_real_branch_all_url_no_upload():
    """全部走 URL 时不产生任何 upload 调用。"""
    tasks = [
        {"image_url": "http://u/i1.jpg", "audio_url": "http://u/a1.mp3"},
        {"image_url": "http://u/i2.jpg", "audio_url": "http://u/a2.mp3"},
    ]

    with mock.patch(
        "src.aip_core.digital_human_p0.core.default_uploader", side_effect=_param_upload
    ) as up:
        inputs = build_batch_inputs(tasks)

    assert up.call_count == 0
    assert inputs[0]["audio"] == "http://u/a1.mp3"
    assert inputs[1]["audio"] == "http://u/a2.mp3"


def test_batch_real_branch_all_file_upload_count_matches():
    """全部走文件时，每个任务各 2 次 upload（image + audio）。"""
    tasks = [
        {"image_file": "/tmp/i1.jpg", "audio_file": "/tmp/a1.mp3"},
        {"image_file": "/tmp/i2.jpg", "audio_file": "/tmp/a2.mp3"},
    ]

    with mock.patch(
        "src.aip_core.digital_human_p0.core.default_uploader", side_effect=_param_upload
    ) as up:
        inputs = build_batch_inputs(tasks)

    assert up.call_count == 4
    assert inputs[1]["audio"] == "http://up/a2.mp3"
    assert inputs[1]["image"] == "http://up/i2.jpg"


def test_batch_real_branch_missing_media_raises():
    """既没给文件也没给 URL 时应报错。"""
    with mock.patch(
        "src.aip_core.digital_human_p0.core.default_uploader", side_effect=_param_upload
    ):
        with pytest.raises(MediaMissingError):
            build_batch_inputs([{"resolution": "480p"}])
