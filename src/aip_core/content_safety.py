"""内容安全审核模块

提供多层次内容安全检测能力：
1. 本地敏感词过滤（零延迟，离线可用）
2. 外部审核 API 接入（高精度，按需启用）
3. 内容安全评分与阻断策略

使用方式:
    from .content_safety import ContentSafety

    safety = ContentSafety()
    result = safety.check_text("用户输入或AI生成内容")
    if not result.passed:
        raise HTTPException(status_code=422, detail=result.message)
"""

import os
import re
import time
import hashlib
from typing import Optional, Literal
from dataclasses import dataclass, field

import httpx

from .config import MCP_TIMEOUT_SECONDS


# ============================================================
# 安全审核结果
# ============================================================

@dataclass
class SafetyResult:
    """内容安全审核结果"""
    passed: bool                          # 是否通过
    risk_level: Literal["none", "low", "medium", "high", "blocked"] = "none"
    message: str = ""                     # 阻断原因说明
    matched_keywords: list[str] = field(default_factory=list)
    external_score: Optional[float] = None  # 外部 API 评分 (0-1, 越高越危险)
    checked_via: str = "none"             # 审核方式：local / external / none

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "risk_level": self.risk_level,
            "message": self.message,
            "checked_via": self.checked_via,
        }


# ============================================================
# 内容安全审核器
# ============================================================

class ContentSafety:
    """内容安全审核器

    两层防护：
    1. 本地敏感词库 → 毫秒级拦截，离线可用
    2. 外部审核 API → 高精度语义理解，按需启用
    """

    # ---- 敏感词分类 ----

    # 政治敏感词（最小集合，仅作示例，生产环境需从专业词库加载）
    POLITICAL_KEYWORDS = [
        "颠覆国家政权", "分裂国家", "煽动民族仇恨",
        "邪教组织", "恐怖主义", "极端主义",
    ]

    # 色情/低俗敏感词
    ADULT_KEYWORDS = [
        "色情", "淫秽", "裸体", "性交",
        "成人影片", "一夜情", "约炮",
    ]

    # 暴力/赌博/毒品敏感词
    ILLEGAL_KEYWORDS = [
        "赌博", "赌场", "毒品", "吸毒",
        "枪支买卖", "杀人", "自杀",
    ]

    # 欺诈/广告敏感词
    FRAUD_KEYWORDS = [
        "快速致富", "日赚", "网络兼职",
        "传销", "直销", "刷单",
    ]

    # 人身攻击/歧视敏感词
    HATE_KEYWORDS = [
        "种族歧视", "地域歧视", "性别歧视",
        "人身攻击", "辱骂",
    ]

    # ---- 编译后正则 ----

    _POLITICAL_RE = None
    _ADULT_RE = None
    _ILLEGAL_RE = None
    _FRAUD_RE = None
    _HATE_RE = None

    def __init__(self, enable_external: bool = None):
        """
        Args:
            enable_external: 是否启用外部审核 API。
                             默认从环境变量 CONTENT_SAFETY_API_URL 判断。
        """
        self._external_url = os.getenv("CONTENT_SAFETY_API_URL", "")
        self._external_key = os.getenv("CONTENT_SAFETY_API_KEY", "")
        self._enable_external = (
            enable_external if enable_external is not None
            else bool(self._external_url)
        )
        self._compile_patterns()

    def _compile_patterns(self):
        """编译敏感词正则（线程安全，懒加载）"""
        if ContentSafety._POLITICAL_RE is None:
            ContentSafety._POLITICAL_RE = self._build_re(self.POLITICAL_KEYWORDS)
            ContentSafety._ADULT_RE = self._build_re(self.ADULT_KEYWORDS)
            ContentSafety._ILLEGAL_RE = self._build_re(self.ILLEGAL_KEYWORDS)
            ContentSafety._FRAUD_RE = self._build_re(self.FRAUD_KEYWORDS)
            ContentSafety._HATE_RE = self._build_re(self.HATE_KEYWORDS)

    @staticmethod
    def _build_re(keywords: list[str]) -> re.Pattern:
        return re.compile("|".join(re.escape(k) for k in keywords), re.IGNORECASE)

    # ---- 核心 API ----

    def check_text(self, text: str, context: str = "general") -> SafetyResult:
        """对文本内容进行安全审核

        Args:
            text: 待审核文本
            context: 上下文类型 — "user_input" / "ai_output" / "course_content" / "general"

        Returns:
            SafetyResult: 审核结果
        """
        if not text or not text.strip():
            return SafetyResult(passed=True, risk_level="none", checked_via="none")

        # 第一层：本地敏感词过滤
        local_result = self._local_check(text)
        if local_result.risk_level in ("high", "blocked"):
            return local_result

        # 第二层：外部 API 审核（如果启用）
        if self._enable_external:
            external_result = self._external_check(text, context)
            if external_result and external_result.risk_level in ("high", "blocked"):
                return external_result

        # 通过
        return SafetyResult(passed=True, risk_level="none", checked_via="local")

    def check_batch(self, items: list[dict]) -> list[SafetyResult]:
        """批量审核多条内容

        Args:
            items: [{"text": "...", "context": "ai_output"}, ...]

        Returns:
            [SafetyResult, ...] 与输入一一对应
        """
        return [self.check_text(item.get("text", ""), item.get("context", "general")) for item in items]

    def sanitize(self, text: str) -> str:
        """对文本进行脱敏处理（用 *** 替换敏感词）

        适用于需要保留内容但去除敏感信息的场景。
        """
        for pattern, replacement in [
            (self._POLITICAL_RE, "***"),
            (self._ADULT_RE, "***"),
            (self._ILLEGAL_RE, "***"),
            (self._FRAUD_RE, "***"),
            (self._HATE_RE, "***"),
        ]:
            if pattern:
                text = pattern.sub(replacement, text)
        return text

    # ---- 内部方法 ----

    def _local_check(self, text: str) -> SafetyResult:
        """本地敏感词检查"""
        checks = [
            ("political", self._POLITICAL_RE, "blocked", "内容包含政治敏感信息"),
            ("adult", self._ADULT_RE, "high", "内容包含不当成人信息"),
            ("illegal", self._ILLEGAL_RE, "blocked", "内容包含违法信息"),
            ("fraud", self._FRAUD_RE, "high", "内容包含欺诈或广告信息"),
            ("hate", self._HATE_RE, "high", "内容包含歧视或攻击性信息"),
        ]

        for category, pattern, risk, msg in checks:
            if pattern and pattern.search(text):
                return SafetyResult(
                    passed=False,
                    risk_level=risk,
                    message=msg,
                    checked_via="local",
                )

        return SafetyResult(passed=True, risk_level="none", checked_via="local")

    def _external_check(self, text: str, context: str) -> Optional[SafetyResult]:
        """调用外部审核 API

        使用 HTTP POST 同步调用，超时时间与 MCP 一致。
        外部 API 预期返回格式：
        {
            "safe": true/false,
            "risk_level": "none/low/medium/high/blocked",
            "score": 0.0-1.0,
            "categories": ["..."]
        }
        """
        if not self._external_url:
            return None

        try:
            headers = {"Content-Type": "application/json"}
            if self._external_key:
                headers["Authorization"] = f"Bearer {self._external_key}"

            payload = {
                "text": text[:8000],  # 截断长文本
                "context": context,
                "timestamp": time.time(),
            }

            # 使用同步 httpx（FastAPI 异步路由内可用）
            with httpx.Client(timeout=MCP_TIMEOUT_SECONDS) as client:
                resp = client.post(self._external_url, json=payload, headers=headers)
                if resp.status_code != 200:
                    # 外部 API 不可用时降级为仅本地检查
                    return None

                data = resp.json()
                safe = data.get("safe", True)
                risk = data.get("risk_level", "none")
                score = data.get("score", 0.0)

                if not safe or risk in ("high", "blocked"):
                    return SafetyResult(
                        passed=False,
                        risk_level=risk,
                        message=data.get("reason", "内容未通过外部安全审核"),
                        external_score=score,
                        checked_via="external",
                    )

                return SafetyResult(
                    passed=True,
                    risk_level=risk,
                    external_score=score,
                    checked_via="external",
                )

        except Exception:
            # 外部服务不可用时，不阻断业务流程
            return None


# ============================================================
# 全局单例
# ============================================================

_safety_instance: Optional[ContentSafety] = None


def get_safety() -> ContentSafety:
    """获取全局 ContentSafety 实例（懒加载）"""
    global _safety_instance
    if _safety_instance is None:
        _safety_instance = ContentSafety()
    return _safety_instance


# ============================================================
# 快速审核装饰器 / 辅助函数
# ============================================================

def check_user_input(text: str) -> SafetyResult:
    """快速审核用户输入（context=user_input）"""
    return get_safety().check_text(text, context="user_input")


def check_ai_output(text: str) -> SafetyResult:
    """快速审核 AI 生成内容（context=ai_output）"""
    return get_safety().check_text(text, context="ai_output")


def check_course_content(text: str) -> SafetyResult:
    """快速审核课程内容（context=course_content）"""
    return get_safety().check_text(text, context="course_content")
