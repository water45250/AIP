"""LLM 工厂 - 统一 LLM 创建

支持 Claude 4（默认）和 GPT-5（备选）。
用于 CrewAI Agent 和 LangChain 链。
"""

import os
from crewai import LLM as CrewLLM


def create_llm(model: str = None, temperature: float = 0.7) -> CrewLLM:
    """创建 CrewAI 兼容的 LLM 实例

    Args:
        model: 模型名称，默认从环境变量读取
        temperature: 温度参数

    Returns:
        CrewAI LLM 实例
    """
    model = model or os.getenv("LLM_MODEL", "claude-sonnet-4-20250514")
    provider = os.getenv("LLM_PROVIDER", "anthropic")

    if provider == "anthropic":
        api_key = os.getenv("ANTHROPIC_API_KEY", "")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY 未设置")
        return CrewLLM(
            model=f"anthropic/{model}",
            api_key=api_key,
            temperature=temperature,
        )
    elif provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY", "")
        if not api_key:
            raise ValueError("OPENAI_API_KEY 未设置")
        return CrewLLM(
            model=f"openai/{model}",
            api_key=api_key,
            temperature=temperature,
        )
    else:
        raise ValueError(f"不支持的 LLM provider: {provider}")


# 默认 LLM 实例（懒加载）
_default_llm = None


def get_default_llm() -> CrewLLM:
    """获取默认 LLM 实例（单例）"""
    global _default_llm
    if _default_llm is None:
        _default_llm = create_llm()
    return _default_llm
