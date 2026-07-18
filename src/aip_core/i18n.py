"""繁體中文輸出支援。

- TRADITIONAL_DIRECTIVE：注入到各 Agent 系統提示詞（backstory/goal），要求 LLM 以繁體中文輸出。
- create_agent：包裝 crewai.Agent，自動為 backstory/goal 附加繁體中文指令。
- to_traditional：遞迴將回傳前端的 JSON 中所有字串轉為繁體中文（OpenCC s2t），
  作為兜底——確保即使 LLM 偶發輸出簡體，或存在寫死的簡體文案（如追問問題、降級模板），
  前端看到的都是繁體中文。
"""
import json
import shutil
import subprocess

from crewai import Agent as _CrewAgent

TRADITIONAL_DIRECTIVE = (
    "\n\n【語言要求】你所有輸出（包含標題、正文、範例、建議、追問，以及結構化 JSON 中"
    "的全部中文字串）都必須使用繁體中文（Traditional Chinese），嚴禁使用簡體中文。"
)

_OPENCC_BIN = shutil.which("opencc")


def create_agent(*args, **kwargs):
    """建立 CrewAI Agent，並在系統提示詞中強制繁體中文輸出。"""
    backstory = kwargs.get("backstory") or ""
    if isinstance(backstory, str) and TRADITIONAL_DIRECTIVE not in backstory:
        kwargs["backstory"] = backstory + TRADITIONAL_DIRECTIVE
    goal = kwargs.get("goal") or ""
    if isinstance(goal, str) and TRADITIONAL_DIRECTIVE not in goal:
        kwargs["goal"] = goal + TRADITIONAL_DIRECTIVE
    return _CrewAgent(*args, **kwargs)


def to_traditional(obj):
    """遞迴將可序列化物件中的字串轉為繁體中文；失敗時原樣回傳。"""
    if not _OPENCC_BIN:
        return obj
    try:
        text = json.dumps(obj, ensure_ascii=False, default=str)
        proc = subprocess.run(
            [_OPENCC_BIN, "-c", "s2t"],
            input=text,
            capture_output=True,
            text=True,
            timeout=15,
        )
        if proc.returncode == 0 and proc.stdout:
            return json.loads(proc.stdout)
        return obj
    except Exception:
        return obj
