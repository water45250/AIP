"""TTS Factory - 语音合成引擎

支持双引擎：
1. SiliconFlow CosyVoice2-0.5B — 阿里通义语音克隆 + 高表现力合成（需 API Key）
2. Edge TTS — 微软免费神经语音（零 API Key，fallback）

CosyVoice2 语音克隆流程：
  上传参考音频(base64) → 创建克隆音色(uri) → 使用 uri 合成所有课时

使用方式:
    from .tts_factory import create_tts_engine, get_available_voices

    engine = create_tts_engine(output_dir)
    mp3_path = engine.synthesize("讲稿文本...", "M1L1")
"""

import os
import io
import json
import base64
import time
import hashlib
import subprocess
import tempfile
from pathlib import Path
from typing import Optional
from abc import ABC, abstractmethod

import requests


class AudioTooLongError(RuntimeError):
    """硅基流动 CosyVoice2 单次合成输出音频超过 30s 限制（code 20015）"""


# CosyVoice2 单次合成输出音频上限 30s，按中文 ~4.5 字/秒估算，
# 留足安全余量（语速可能 <1.0 使音频变长），单请求字符上限取保守值。
COSYVOICE_MAX_CHARS = 90


# ============================================================
# 抽象基类
# ============================================================

class BaseTTSEngine(ABC):
    """TTS 引擎抽象基类"""

    def __init__(self, output_dir: Path, max_chars: int = 4000):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_chars = max_chars

    @abstractmethod
    def synthesize(self, text: str, lesson_id: str) -> str:
        """合成语音，返回 MP3 文件路径"""
        ...

    @abstractmethod
    def get_mode_name(self) -> str:
        """返回引擎标识名称"""
        ...

    def _clean_markdown(self, text: str) -> str:
        """将 Markdown 讲稿转为纯文本"""
        import re
        text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)
        text = re.sub(r'\*{1,3}([^*]+)\*{1,3}', r'\1', text)
        text = re.sub(r'_{1,3}([^_]+)_{1,3}', r'\1', text)
        text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
        text = re.sub(r'!\[([^\]]*)\]\([^)]+\)', '', text)
        text = re.sub(r'`([^`]+)`', r'\1', text)
        text = re.sub(r'```[\s\S]*?```', '', text)
        text = re.sub(r'^[\s]*[-*+]\s+', '', text, flags=re.MULTILINE)
        text = re.sub(r'^[\s]*\d+\.\s+', '', text, flags=re.MULTILINE)
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r'^[-*_]{3,}\s*$', '', text, flags=re.MULTILINE)
        return text.strip()

    def _split_text(self, text: str, max_chars: Optional[int] = None) -> list[str]:
        """将长文本按段落/句子边界分段，每段不超过 max_chars"""
        max_chars = max_chars or self.max_chars
        paragraphs = text.split("\n\n")
        chunks = []
        current = ""

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            if len(current) + len(para) + 2 <= max_chars:
                current = (current + "\n\n" + para) if current else para
            else:
                if current:
                    chunks.append(current)
                if len(para) > max_chars:
                    for i in range(0, len(para), max_chars):
                        chunks.append(para[i:i + max_chars])
                    current = ""
                else:
                    current = para

        if current:
            chunks.append(current)

        return chunks if chunks else [text[:max_chars]]

    def _merge_mp3_files(self, file_paths: list[str], output_path: str):
        """简单二进制拼接 MP3 文件（容错：跳过缺失/空文件）"""
        existing = [
            p for p in file_paths
            if p and os.path.exists(p) and os.path.getsize(p) > 0
        ]
        if not existing:
            raise RuntimeError(
                "语音合成失败：未能生成任何音频片段。"
                "音色可能已失效（克隆音色请重新上传参考音频克隆），或文本为空。"
            )
        with open(output_path, "wb") as out:
            for fp in existing:
                with open(fp, "rb") as f:
                    out.write(f.read())


# ============================================================
# SiliconFlow CosyVoice2-0.5B 引擎
# ============================================================

class SiliconFlowTTS(BaseTTSEngine):
    """硅基流动 CosyVoice2-0.5B — 语音克隆 + 高表现力 TTS

    基于阿里通义 CosyVoice2 模型，通过硅基流动 API 代理访问。

    功能：
    - 上传参考音频 → 创建克隆音色 (voice_uri)
    - 使用克隆音色或系统预置音色合成所有课时讲稿
    - 支持情感控制、多语言（中/英/日/韩/粤语/方言）
    - 支持流式和非流式合成

    环境变量：
    - SILICONFLOW_API_KEY: 硅基流动 API Key（必填）
    - SILICONFLOW_CLONE_AUDIO: 克隆参考音频路径（可选）
    - SILICONFLOW_CLONE_TEXT: 参考音频对应文本（可选）
    """

    # API 端点
    BASE_URL = "https://api.siliconflow.cn/v1"

    # CosyVoice2 系统预置音色（8 种）
    PRESET_VOICES = {
        "FunAudioLLM/CosyVoice2-0.5B:alex":     {"name": "沉稳男声", "gender": "男"},
        "FunAudioLLM/CosyVoice2-0.5B:benjamin": {"name": "低沉男声", "gender": "男"},
        "FunAudioLLM/CosyVoice2-0.5B:charles":  {"name": "磁性男声", "gender": "男"},
        "FunAudioLLM/CosyVoice2-0.5B:david":    {"name": "欢快男声", "gender": "男"},
        "FunAudioLLM/CosyVoice2-0.5B:anna":     {"name": "沉稳女声", "gender": "女"},
        "FunAudioLLM/CosyVoice2-0.5B:bella":    {"name": "激情女声", "gender": "女"},
        "FunAudioLLM/CosyVoice2-0.5B:claire":   {"name": "温柔女声", "gender": "女"},
        "FunAudioLLM/CosyVoice2-0.5B:diana":    {"name": "欢快女声", "gender": "女"},
    }

    def __init__(
        self,
        output_dir: Path,
        api_key: str,
        voice: str = "FunAudioLLM/CosyVoice2-0.5B:alex",
        clone_audio_path: Optional[str] = None,
        clone_text: str = "",
        max_chars: int = 4000,
        speed: float = 1.0,
        gain: float = 0.0,
    ):
        """
        Args:
            output_dir: 音频输出目录
            api_key: 硅基流动 API Key
            voice: 语音标识（系统预置音色或克隆后的 uri）
            clone_audio_path: 参考音频文件路径（触发语音克隆）
            clone_text: 参考音频对应的文本（提升克隆效果）
            max_chars: 单次合成最大字符数
            speed: 语速 (0.25-4.0)
            gain: 音量增益 dB (-10 到 10)
        """
        super().__init__(output_dir, max_chars)
        self.api_key = api_key
        self.voice = voice
        self.clone_audio_path = clone_audio_path
        self.clone_text = clone_text
        self.speed = speed
        self.gain = gain
        self._cloned_voice_uri: Optional[str] = None
        self.model = "FunAudioLLM/CosyVoice2-0.5B"

    def get_mode_name(self) -> str:
        return "siliconflow_cosyvoice2"

    def _headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    # ---- 语音克隆 ----

    def clone_voice(self) -> str:
        """执行语音克隆，返回 voice_uri

        流程：读取参考音频 → base64 编码 → 上传创建音色 → 返回 uri

        CosyVoice2 的克隆是通过上传参考音频创建「用户预置音色」实现的。
        """
        if self._cloned_voice_uri:
            return self._cloned_voice_uri

        if not self.clone_audio_path:
            # 没有参考音频，使用系统预置音色
            return self.voice

        audio_path = Path(self.clone_audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"克隆参考音频不存在: {self.clone_audio_path}")

        # Step 1: 读取音频文件并 base64 编码
        with open(audio_path, "rb") as f:
            audio_bytes = f.read()

        # 检测 MIME 类型
        suffix = audio_path.suffix.lower()
        mime_map = {".mp3": "audio/mpeg", ".wav": "audio/wav", ".opus": "audio/opus", ".pcm": "audio/pcm"}
        mime_type = mime_map.get(suffix, "audio/mpeg")

        audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
        audio_data_uri = f"data:{mime_type};base64,{audio_b64}"

        # Step 2: 调用硅基流动上传接口创建音色
        custom_name = f"opc_{hashlib.md5(audio_path.name.encode()).hexdigest()[:8]}"

        payload = {
            "model": self.model,
            "customName": custom_name,
            "audio": audio_data_uri,
            "text": self.clone_text or "",
        }

        resp = requests.post(
            f"{self.BASE_URL}/uploads/audio/voice",
            headers=self._headers(),
            json=payload,
            timeout=60,
        )

        if resp.status_code != 200:
            error_text = resp.text[:500]
            raise RuntimeError(f"硅基流动音色上传失败 (HTTP {resp.status_code}): {error_text}")

        data = resp.json()
        voice_uri = data.get("uri", "")

        if not voice_uri:
            raise RuntimeError(f"硅基流动音色上传返回空 uri: {json.dumps(data, ensure_ascii=False)[:500]}")

        self._cloned_voice_uri = voice_uri
        return voice_uri

    # ---- TTS 合成 ----

    def synthesize(self, text: str, lesson_id: str) -> str:
        """同步合成语音（使用 requests，兼容已有事件循环环境）

        文本按 CosyVoice2 单次输出 30s 上限进行安全分片；若某片仍触发
        20015（audio longer than 30s），_synthesize_chunk_safe 会自动二分重试。
        """
        clean_text = self._clean_markdown(text)
        if not clean_text.strip():
            clean_text = "（本课时无讲稿内容）"

        mp3_path = self.output_dir / f"{lesson_id}.mp3"

        # 按 CosyVoice2 安全上限分片
        chunks = self._split_text(clean_text, max_chars=COSYVOICE_MAX_CHARS)
        if len(chunks) == 1:
            self._synthesize_chunk_safe(chunks[0], str(mp3_path))
            return str(mp3_path)

        # 多片分段合成 + 合并
        chunk_paths = []
        for i, chunk in enumerate(chunks):
            chunk_path = self.output_dir / f"{lesson_id}_chunk_{i}.mp3"
            self._synthesize_chunk_safe(chunk, str(chunk_path))
            chunk_paths.append(str(chunk_path))

        self._merge_mp3_files(chunk_paths, str(mp3_path))

        for cp in chunk_paths:
            try:
                Path(cp).unlink()
            except Exception:
                pass

        return str(mp3_path)

    def _split_two(self, text: str) -> tuple[str, str]:
        """在接近中点处按句子边界把文本切成两段，避免截断词语"""
        if len(text) <= 2:
            return text, ""
        mid = len(text) // 2
        cut = -1
        for i in range(mid, -1, -1):
            if text[i] in "。！？!?；;\n":
                cut = i + 1
                break
        if cut <= 0 or cut >= len(text):
            return text[:mid], text[mid:]
        return text[:cut], text[cut:]

    def _synthesize_chunk_safe(self, text: str, output_path: str, depth: int = 0):
        """合成单段文本；若触发 30s 上限（code 20015）则二分拆分后重试并合并。

        关键修复：任何分片的真实合成失败（音色无效/过期、鉴权失败、网络错误等）
        都必须**原样抛出明确错误**，绝不能吞掉后让父级合并读到缺失文件而
        崩溃为隐晦的 FileNotFoundError。仅在确认是「30s 超限」时才做二分重试。
        """
        text = text.strip()
        if not text:
            text = "。"

        try:
            self._synthesize_chunk(text, output_path)
        except (AudioTooLongError, RuntimeError) as e:
            msg = str(e)
            is_too_long = ("20015" in msg) or ("audio longer than 30s" in msg.lower())
            if not is_too_long:
                raise  # 非 30s 超限错误（音色无效/鉴权/网络等）原样抛出
            if depth >= 6 or len(text) <= 4:
                # 已达最大递归深度仍超长：说明音色可能已失效或文本异常，给出明确提示
                raise RuntimeError(
                    f"语音合成失败：分句后仍超出单句时长上限，或当前音色不可用。"
                    f"（{msg[:160]}）若为克隆音色请重新上传参考音频克隆；"
                    f"预置音色可换一个试试。"
                )
            p1, p2 = self._split_two(text)
            if not p2:
                raise RuntimeError(
                    f"语音合成失败：文本无法继续分割（{msg[:160]}）。"
                    f"若为克隆音色请重新克隆；预置音色可换一个试试。"
                )
            with tempfile.TemporaryDirectory() as tmp:
                f1 = os.path.join(tmp, "a.mp3")
                f2 = os.path.join(tmp, "b.mp3")
                self._synthesize_chunk_safe(p1, f1, depth + 1)
                self._synthesize_chunk_safe(p2, f2, depth + 1)
                self._merge_mp3_files([f1, f2], output_path)

    def _synthesize_chunk(self, text: str, output_path: str):
        """调用硅基流动 CosyVoice2 API 合成单段文本

        使用 OpenAI 兼容的 audio/speech 接口，返回二进制音频数据。
        若响应为 20015（audio longer than 30s）则抛出 AudioTooLongError，
        由上层 _synthesize_chunk_safe 进行二分重试。

        检测策略（鲁棒）：除解析顶层 JSON code==20015 外，还会扫描原始响应文本中的
        "20015" / "audio longer than 30s"。克隆音色等场景下硅基流动可能把错误放在
        非顶层结构或直接以纯文本返回，仅靠 JSON code 会漏判导致错误冒泡。
        """
        effective_voice = self._cloned_voice_uri or self.voice

        payload = {
            "model": self.model,
            "input": text,
            "voice": effective_voice,
            "response_format": "mp3",
            "sample_rate": 32000,
            "speed": self.speed,
            "gain": self.gain,
            "stream": False,
        }

        resp = requests.post(
            f"{self.BASE_URL}/audio/speech",
            headers=self._headers(),
            json=payload,
            timeout=120,
        )

        if resp.status_code != 200:
            body = resp.text or ""
            # 鲁棒检测 30s 超限
            is_too_long = False
            try:
                ej = resp.json()
                if ej.get("code") == 20015:
                    is_too_long = True
            except Exception:
                pass
            if (not is_too_long) and ("20015" in body or "audio longer than 30s" in body.lower()):
                is_too_long = True
            if is_too_long:
                raise AudioTooLongError(
                    f"硅基流动 TTS 合成失败 (HTTP {resp.status_code}): {body[:300]}"
                )
            # 克隆音色参考文本缺失/失效：硅基流动返回 50507 Unknown error，
            # 该残缺克隆有时还会把参考音频原样回吐当成合成结果（内容/时长都不对）。
            if "50507" in body:
                raise RuntimeError(
                    "克隆音色合成失败：该音色可能缺少参考文本或已失效。"
                    "请重新上传参考音频，并填写『参考音频对应文本』（念出的实际文字）后再次克隆，再试听。"
                )
            raise RuntimeError(f"硅基流动 TTS 合成失败 (HTTP {resp.status_code}): {body[:500]}")

        # 硅基流动 audio/speech 返回二进制音频数据
        content_type = resp.headers.get("Content-Type", "")
        if "application/json" in content_type:
            # 某些错误以 JSON 返回
            data = resp.json()
            raise RuntimeError(f"硅基流动 TTS 返回异常: {json.dumps(data, ensure_ascii=False)[:500]}")

        with open(output_path, "wb") as f:
            f.write(resp.content)

    # ---- 辅助方法 ----

    def get_available_preset_voices(self) -> list[dict]:
        """获取系统预置音色列表"""
        return [
            {
                "id": vid,
                "name": info["name"],
                "gender": info["gender"],
                "type": "siliconflow_preset",
            }
            for vid, info in self.PRESET_VOICES.items()
        ]

    def list_cloned_voices(self) -> list[dict]:
        """获取已克隆的音色列表"""
        resp = requests.get(
            f"{self.BASE_URL}/audio/voice/list",
            headers=self._headers(),
            timeout=15,
        )
        if resp.status_code != 200:
            return []
        data = resp.json()
        if isinstance(data, list):
            return data
        return data.get("voices", data.get("data", []))

    def delete_cloned_voice(self, voice_uri: str) -> bool:
        """删除克隆音色"""
        resp = requests.post(
            f"{self.BASE_URL}/audio/voice/deletions",
            headers=self._headers(),
            json={"uri": voice_uri},
            timeout=15,
        )
        return resp.status_code == 200


# ============================================================
# Edge TTS 引擎 (fallback)
# ============================================================

class EdgeTTS(BaseTTSEngine):
    """Edge TTS 引擎 — 微软免费神经语音

    支持的语音:
    - zh-CN-XiaoxiaoNeural  (女声，温暖)   [默认]
    - zh-CN-YunxiNeural     (男声，专业)
    - zh-CN-YunjianNeural   (男声，播报风格)
    - zh-CN-XiaoyiNeural    (女声，活泼)
    """

    AVAILABLE_VOICES = [
        "zh-CN-XiaoxiaoNeural",
        "zh-CN-YunxiNeural",
        "zh-CN-YunjianNeural",
        "zh-CN-XiaoyiNeural",
    ]

    def __init__(
        self,
        output_dir: Path,
        voice: str = "zh-CN-XiaoxiaoNeural",
        max_chars: int = 4000,
    ):
        super().__init__(output_dir, max_chars)
        self.voice = voice

    def get_mode_name(self) -> str:
        return "edge_tts"

    def synthesize(self, text: str, lesson_id: str) -> str:
        """同步合成语音（使用 edge-tts CLI 子进程）"""
        clean_text = self._clean_markdown(text)
        if not clean_text.strip():
            clean_text = "（本课时无讲稿内容）"

        mp3_path = self.output_dir / f"{lesson_id}.mp3"

        if len(clean_text) <= self.max_chars:
            self._synthesize_via_cli(clean_text, str(mp3_path))
            return str(mp3_path)

        chunks = self._split_text(clean_text)
        chunk_paths = []
        for i, chunk in enumerate(chunks):
            chunk_path = self.output_dir / f"{lesson_id}_chunk_{i}.mp3"
            self._synthesize_via_cli(chunk, str(chunk_path))
            chunk_paths.append(str(chunk_path))

        self._merge_mp3_files(chunk_paths, str(mp3_path))

        for cp in chunk_paths:
            try:
                Path(cp).unlink()
            except Exception:
                pass

        return str(mp3_path)

    def _synthesize_via_cli(self, text: str, output_path: str):
        """通过 edge-tts CLI 子进程合成语音"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
            f.write(text)
            tmp_path = f.name

        try:
            result = subprocess.run(
                ["edge-tts", "--voice", self.voice, "-f", tmp_path, "--write-media", output_path],
                capture_output=True, text=True, timeout=120,
            )
            if result.returncode != 0:
                raise RuntimeError(f"edge-tts failed: {result.stderr.strip()}")
        finally:
            try:
                Path(tmp_path).unlink()
            except Exception:
                pass


# ============================================================
# 工厂函数
# ============================================================

def create_tts_engine(
    output_dir: Path,
    siliconflow_api_key: str = "",
    clone_audio_path: Optional[str] = None,
    clone_text: str = "",
    voice: str = "FunAudioLLM/CosyVoice2-0.5B:alex",
    edge_voice: str = "zh-CN-XiaoxiaoNeural",
    max_chars: int = 4000,
    speed: float = 1.0,
    gain: float = 0.0,
) -> BaseTTSEngine:
    """创建 TTS 引擎实例

    优先使用硅基流动 CosyVoice2（有 API Key 时），fallback 到 Edge TTS。

    Args:
        output_dir: 音频输出目录
        siliconflow_api_key: 硅基流动 API Key（为空时使用 Edge TTS）
        clone_audio_path: 克隆参考音频路径（触发语音克隆）
        clone_text: 参考音频对应文本
        voice: CosyVoice2 系统预置音色（如 FunAudioLLM/CosyVoice2-0.5B:alex）
        edge_voice: Edge TTS 语音名称（fallback 时使用）
        max_chars: 单次合成最大字符数
        speed: 语速 (0.25-4.0)
        gain: 音量增益 dB (-10 到 10)

    Returns:
        BaseTTSEngine 实例
    """
    if siliconflow_api_key:
        return SiliconFlowTTS(
            output_dir=output_dir,
            api_key=siliconflow_api_key,
            voice=voice,
            clone_audio_path=clone_audio_path,
            clone_text=clone_text,
            max_chars=max_chars,
            speed=speed,
            gain=gain,
        )
    else:
        return EdgeTTS(
            output_dir=output_dir,
            voice=edge_voice,
            max_chars=max_chars,
        )


# ---- 语音选择辅助 ----

def get_available_voices(api_key: str = "") -> list[dict]:
    """获取可用的中文语音列表

    Args:
        api_key: 硅基流动 API Key（有则返回 CosyVoice2 + Edge 语音）

    Returns:
        [{id, name, gender, type, ...}]
    """
    if api_key:
        # 硅基流动可用时，CosyVoice2 预置音色优先
        voices = [
            {"id": "FunAudioLLM/CosyVoice2-0.5B:alex",     "name": "沉稳男声 (CosyVoice2)", "gender": "男", "type": "siliconflow_preset", "default": True},
            {"id": "FunAudioLLM/CosyVoice2-0.5B:benjamin", "name": "低沉男声 (CosyVoice2)", "gender": "男", "type": "siliconflow_preset", "default": False},
            {"id": "FunAudioLLM/CosyVoice2-0.5B:charles",  "name": "磁性男声 (CosyVoice2)", "gender": "男", "type": "siliconflow_preset", "default": False},
            {"id": "FunAudioLLM/CosyVoice2-0.5B:david",    "name": "欢快男声 (CosyVoice2)", "gender": "男", "type": "siliconflow_preset", "default": False},
            {"id": "FunAudioLLM/CosyVoice2-0.5B:anna",     "name": "沉稳女声 (CosyVoice2)", "gender": "女", "type": "siliconflow_preset", "default": False},
            {"id": "FunAudioLLM/CosyVoice2-0.5B:bella",    "name": "激情女声 (CosyVoice2)", "gender": "女", "type": "siliconflow_preset", "default": False},
            {"id": "FunAudioLLM/CosyVoice2-0.5B:claire",   "name": "温柔女声 (CosyVoice2)", "gender": "女", "type": "siliconflow_preset", "default": False},
            {"id": "FunAudioLLM/CosyVoice2-0.5B:diana",    "name": "欢快女声 (CosyVoice2)", "gender": "女", "type": "siliconflow_preset", "default": False},
        ]
        # 追加 Edge TTS 作为备选
        voices += [
            {"id": "zh-CN-XiaoxiaoNeural", "name": "晓晓 (Edge)", "gender": "女", "style": "温暖", "type": "edge_tts", "default": False},
            {"id": "zh-CN-YunxiNeural",    "name": "云希 (Edge)", "gender": "男", "style": "专业", "type": "edge_tts", "default": False},
        ]
    else:
        voices = [
            {"id": "zh-CN-XiaoxiaoNeural", "name": "晓晓", "gender": "女", "style": "温暖", "type": "edge_tts", "default": True},
            {"id": "zh-CN-YunxiNeural",    "name": "云希", "gender": "男", "style": "专业", "type": "edge_tts", "default": False},
            {"id": "zh-CN-YunjianNeural",  "name": "云健", "gender": "男", "style": "播报", "type": "edge_tts", "default": False},
            {"id": "zh-CN-XiaoyiNeural",   "name": "晓伊", "gender": "女", "style": "活泼", "type": "edge_tts", "default": False},
        ]

    return voices
