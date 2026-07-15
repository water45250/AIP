"""Voice Agent - 语音合成节点

在内容生产完成后，将课程讲稿批量合成为 MP3 音频文件。

引擎选择策略：
1. 硅基流动 CosyVoice2-0.5B（有 SILICONFLOW_API_KEY 时优先，支持语音克隆）
2. Edge TTS（零 API Key fallback，微软免费神经语音）

语音克隆流程（CosyVoice2）：
- 有 SILICONFLOW_CLONE_AUDIO 环境变量 → 上传音频 → 创建克隆音色(uri) → 合成
- 无克隆音频 → 使用 CosyVoice2 系统预置音色或 Edge TTS 预置音色
"""

import os
import time
from pathlib import Path

from ..graph.state import CourseState
from ..tools.tts_factory import create_tts_engine, BaseTTSEngine
from ..config import DATA_DIR, TTS_EDGE_VOICE, TTS_MAX_CHARS_PER_REQUEST


def run_voice_tts(state: CourseState) -> CourseState:
    """Voice TTS 节点 - LangGraph 节点函数

    将 scripts 中的每课时讲稿合成为 MP3 音频。
    单课时失败不阻断其他课时。
    """
    start_time = time.time()
    session_id = state.get("session_id", "unknown")
    scripts = state.get("scripts", {})

    # 无讲稿 → 跳过
    if not scripts:
        state["tts_mode"] = "none"
        state["audio_files"] = {}
        state["voice_progress"] = {"total": 0, "completed": 0}
        state["current_node"] = "voice_tts"
        return state

    # 创建输出目录
    output_dir = Path(DATA_DIR) / "audio" / session_id
    output_dir.mkdir(parents=True, exist_ok=True)

    # 获取配置
    siliconflow_api_key = os.getenv("SILICONFLOW_API_KEY", "")
    clone_audio_path = os.getenv("SILICONFLOW_CLONE_AUDIO", "")
    clone_text = os.getenv("SILICONFLOW_CLONE_TEXT", "")
    voice = os.getenv("SILICONFLOW_VOICE", "FunAudioLLM/CosyVoice2-0.5B:alex")
    tts_speed = float(os.getenv("TTS_SPEED", "1.0"))
    tts_gain = float(os.getenv("TTS_GAIN", "0.0"))

    # 创建引擎（优先硅基流动 CosyVoice2）
    engine = create_tts_engine(
        output_dir=output_dir,
        siliconflow_api_key=siliconflow_api_key,
        clone_audio_path=clone_audio_path or None,
        clone_text=clone_text,
        voice=voice,
        edge_voice=TTS_EDGE_VOICE,
        max_chars=TTS_MAX_CHARS_PER_REQUEST,
        speed=tts_speed,
        gain=tts_gain,
    )

    tts_mode = engine.get_mode_name()

    # 硅基流动语音克隆（如果配置了参考音频）
    clone_voice_uri = None
    if tts_mode == "siliconflow_cosyvoice2" and clone_audio_path:
        try:
            clone_voice_uri = engine.clone_voice()
            tts_mode = "siliconflow_cloned"
        except Exception as e:
            state["errors"] = state.get("errors", []) + [{
                "node": "voice_tts",
                "error": f"硅基流动语音克隆失败，降级使用预置音色: {str(e)}",
                "time": time.time(),
            }]
            tts_mode = "siliconflow_preset"

    state["tts_mode"] = tts_mode
    state["voice_progress"] = {
        "total": len(scripts),
        "completed": 0,
        "current_lesson": "",
        "clone_voice_uri": clone_voice_uri,
    }

    audio_files = {}
    failed_lessons = []

    # 逐课时合成（按 lesson_id 排序）
    for i, (lesson_id, script_text) in enumerate(sorted(scripts.items())):
        state["voice_progress"]["current_lesson"] = lesson_id
        try:
            mp3_path = engine.synthesize(script_text, lesson_id)
            audio_files[lesson_id] = mp3_path
            state["voice_progress"]["completed"] = i + 1
        except Exception as e:
            failed_lessons.append({"lesson_id": lesson_id, "error": str(e)})
            state["errors"] = state.get("errors", []) + [{
                "node": "voice_tts",
                "lesson_id": lesson_id,
                "error": f"TTS 合成失败: {str(e)}",
                "time": time.time(),
            }]

    # 更新状态
    state["audio_files"] = audio_files
    state["current_node"] = "voice_tts"
    state["node_history"] = state.get("node_history", []) + [{
        "node": "voice_tts",
        "start": start_time,
        "end": time.time(),
        "status": "ok" if not failed_lessons else "partial",
        "tts_mode": state["tts_mode"],
        "audio_count": len(audio_files),
        "total_lessons": len(scripts),
        "failed_lessons": failed_lessons,
    }]

    return state
