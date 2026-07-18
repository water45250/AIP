"""文经客 AIP OPC - 全局配置模块"""

import os
from pathlib import Path
from dotenv import load_dotenv

# 加载 .env 文件
load_dotenv()

# === 项目路径 ===
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CHECKPOINT_DIR = DATA_DIR / "checkpoints"

# === LLM 配置 ===
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# Claude 4 为默认模型，GPT-5 为 fallback
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "anthropic")  # "anthropic" | "openai"
LLM_MODEL = os.getenv("LLM_MODEL", "claude-sonnet-4-20250514")

# === 搜索引擎 ===
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")
BRAVE_API_KEY = os.getenv("BRAVE_API_KEY", "")

# === MCP Server 地址 ===
MCP_SERVERS = {
    "knowledge-rag": os.getenv("MCP_KNOWLEDGE_RAG_URL", "http://localhost:8001"),
    "web-search": os.getenv("MCP_WEB_SEARCH_URL", "http://localhost:8002"),
    "doc-generator": os.getenv("MCP_DOC_GENERATOR_URL", "http://localhost:8003"),
    "image-generator": os.getenv("MCP_IMAGE_GENERATOR_URL", "http://localhost:8004"),
    "data-analyzer": os.getenv("MCP_DATA_ANALYZER_URL", "http://localhost:8005"),
}

# === SQLite ===
SQLITE_DB_PATH = os.getenv("SQLITE_DB_PATH", str(DATA_DIR / "aip_sessions.db"))

# === API ===
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8080"))

# === 内容安全 ===
CONTENT_SAFETY_API_URL = os.getenv("CONTENT_SAFETY_API_URL", "")
CONTENT_SAFETY_API_KEY = os.getenv("CONTENT_SAFETY_API_KEY", "")
CONTENT_SAFETY_ENABLED = os.getenv("CONTENT_SAFETY_ENABLED", "true").lower() == "true"

# === Voice TTS 配置 ===
TTS_EDGE_VOICE = os.getenv("TTS_EDGE_VOICE", "zh-CN-XiaoxiaoNeural")
TTS_MAX_CHARS_PER_REQUEST = int(os.getenv("TTS_MAX_CHARS_PER_REQUEST", "4000"))

# === 硅基流动 CosyVoice2 TTS 配置 ===
SILICONFLOW_API_KEY = os.getenv("SILICONFLOW_API_KEY", "")
SILICONFLOW_VOICE = os.getenv("SILICONFLOW_VOICE", "FunAudioLLM/CosyVoice2-0.5B:alex")
SILICONFLOW_CLONE_AUDIO = os.getenv("SILICONFLOW_CLONE_AUDIO", "")
SILICONFLOW_CLONE_TEXT = os.getenv("SILICONFLOW_CLONE_TEXT", "")
TTS_SPEED = float(os.getenv("TTS_SPEED", "1.0"))
TTS_GAIN = float(os.getenv("TTS_GAIN", "0.0"))

# === Duix-Avatar 数字人配置 ===
DUIX_API_BASE = os.getenv("DUIX_API_BASE", "http://127.0.0.1:8383")
DUIX_API_KEY = os.getenv("DUIX_API_KEY", "")
DUIX_DEFAULT_VIDEO = os.getenv("DUIX_DEFAULT_VIDEO", "")
DUIX_ENABLED = os.getenv("DUIX_ENABLED", "false").lower() == "true"

# === Agent 限制 ===
MAX_FOLLOWUP_ROUNDS = 5            # 需求追問最多輪數（5輪後強制進入確認）
REQUIREMENT_MIN_COMPLETENESS = 70  # 需求完整度閾值（5/7字段≈70%，信息更足才确认）
REVIEW_PASS_THRESHOLD = 80         # 审核通过分数线
REVIEW_AUTO_SKIP_THRESHOLD = 85    # 审核 ≥85 自动跳过 HITL-7（审核报告确认）
MAX_REVIEW_ROUNDS = 2             # 修正最多轮数
MAX_CONCURRENT_SESSIONS = 3       # 单用户最大并发 session

# === 超时配置 ===
AGENT_TIMEOUT_SECONDS = 120       # 单 Agent 执行超时
MCP_TIMEOUT_SECONDS = 30          # MCP 调用超时
CHECKPOINT_RETENTION_DAYS = 7     # Checkpoint 保留天数
