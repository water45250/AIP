"""IP 定位 Agent - 个人IP战略顾问 (M2 实现)

基于 CrewAI 框架，调用知识库 RAG + 联网搜索，
生成包含定位宣言、差异化标签、信任飞轮、内容矩阵的 IP 定位报告。
"""

import json
import time
from typing import Optional

from crewai import Agent, Task, Crew, Process

from ..graph.state import CourseState, IPPositioning
from ..tools import get_default_llm
from ._crew_runner import run_crew
from .course_architect_agent import _is_exam_oriented


# ============================================================
# CrewAI Agent 定义
# ============================================================

def _create_ip_agent() -> Agent:
    """创建 IP 定位 Agent"""
    return Agent(
        role="个人IP战略顾问",
        goal="基于用户背景和行业数据，提炼差异化定位，设计人设锚点和信任飞轮，输出专业IP定位报告",
        backstory=(
            "你是一位资深个人品牌战略顾问，曾帮助数百位知识付费创作者找到自己的差异化定位。\n"
            "你擅长：\n"
            "1. 通过精准提问挖掘用户独特优势\n"
            "2. 调用行业数据支撑定位建议\n"
            "3. 设计可落地的内容矩阵和信任建立路径\n"
            "你的方法论核心是'定位三角'：我是谁 × 帮谁 × 解决什么问题。"
        ),
        llm=get_default_llm(),
        verbose=False,
        allow_delegation=False,
        max_iter=5,
    )


def _create_ip_task(agent: Agent, user_profile: dict) -> Task:
    """创建 IP 定位任务"""
    identity = user_profile.get("identity", "知识创作者")
    expertise = user_profile.get("expertise", "专业领域")
    experience = user_profile.get("experience", "")
    audience = user_profile.get("target_audience", "目标学员")
    topic = user_profile.get("course_topic", expertise)
    style = user_profile.get("style_preference", "实战干货")
    exam_oriented = _is_exam_oriented(user_profile)

    if exam_oriented:
        exam_note = (
            "【课程定位】政府/企业强制的ESG上岗考试培训。\n"
            "定位宣言中的『结果』应为『通过ESG上岗考试、合规持证上岗』，不要出现『变现/加薪/副业/个人品牌』等商业化表述。\n"
            "信任飞轮改为『学习动员→考点精讲→模拟演练→考前陪伴』；内容矩阵围绕备考支持（考点卡片、答疑、模拟卷）。"
        )
    else:
        exam_note = (
            "【课程定位】知识付费/技能变现类课程。\n"
            "定位宣言的『结果』可为『实现能力跃迁/变现』，可使用个人品牌与信任飞轮方法论。"
        )

    task_description = f"""
请基于以下用户画像，生成一份完整的 IP 定位报告。

## 用户画像
- 身份：{identity}
- 核心专长：{expertise}
- 经验背景：{experience}
- 目标受众：{audience}
- 课程主题：{topic}
- 风格偏好：{style}

{exam_note}

## 任务步骤
1. 基于你的个人品牌方法论专业知识，为用户设计差异化定位
2. 直接综合用户优势与目标市场，输出：
   - 一句定位宣言（格式：帮 [谁] 通过 [什么方法] 实现 [什么结果]）
   - 3-5个差异化标签
   - 4步信任飞轮（考试类请按备考支持设计）
   - 内容矩阵建议

## 输出格式（严格 JSON）
```json
{{
  "positioning_statement": "一句话定位宣言",
  "differentiation_tags": ["标签1", "标签2", "标签3"],
  "trust_flywheel": [
    {{"step": 1, "action": "具体行动", "goal": "达成目标"}},
    {{"step": 2, "action": "具体行动", "goal": "达成目标"}},
    {{"step": 3, "action": "具体行动", "goal": "达成目标"}},
    {{"step": 4, "action": "具体行动", "goal": "达成目标"}}
  ],
  "content_matrix": {{
    "platforms": ["平台1", "平台2"],
    "content_types": ["内容类型1", "内容类型2"],
    "frequency": "发布频率建议"
  }},
  "analysis_notes": "简短分析说明（2-3句话）"
}}
```
"""
    return Task(
        description=task_description,
        expected_output="符合 JSON 格式的 IP 定位报告",
        agent=agent,
    )


# ============================================================
# 降级方案（无 LLM API 时使用）
# ============================================================

def _generate_fallback_positioning(profile: dict) -> IPPositioning:
    """当 LLM 不可用时，基于规则生成 IP 定位（降级方案）"""
    topic = profile.get("course_topic", "专业领域")
    expertise = profile.get("expertise", topic)
    audience = profile.get("target_audience", "目标学员")
    identity = profile.get("identity", "知识创作者")

    return {
        "positioning_statement": f"帮{audience}通过{expertise}的实战方法论实现能力跃迁",
        "differentiation_tags": [
            f"{expertise}实战专家",
            f"{audience}专属教练",
            "可复制的方法论",
            "结果导向",
        ],
        "trust_flywheel": [
            {"step": 1, "action": "持续输出免费干货内容（公众号/小红书/视频号）", "goal": "建立专业认知"},
            {"step": 2, "action": "推出低价体验课或免费公开课", "goal": "建立初步信任"},
            {"step": 3, "action": "正价训练营/课程转化 + 社群服务", "goal": "深度服务与收入"},
            {"step": 4, "action": "学员成果展示 + 口碑裂变 + 复购/升单", "goal": "持续增长"},
        ],
        "content_matrix": {
            "platforms": ["公众号", "小红书", "视频号"],
            "content_types": ["干货文章/图文", "案例拆解", "短视频/直播"],
            "frequency": "每周 2-3 篇图文 + 1 场直播/短视频",
        },
        "analysis_notes": f"作为{identity}，你在{expertise}领域的积累是核心优势。建议聚焦{audience}群体，用实战案例建立信任。",
    }


# ============================================================
# 节点入口
# ============================================================

def run_ip_positioning(state: CourseState) -> CourseState:
    """IP 定位 Agent 主逻辑 - LangGraph 节点函数"""
    profile = state.get("user_profile", {})
    if not profile:
        state["errors"] = state.get("errors", []) + [{
            "node": "ip_positioning",
            "error": "缺少用户画像，使用默认值",
            "time": time.time(),
        }]
        profile = {"course_topic": "专业技能", "identity": "知识博主"}

    # 尝试调用 CrewAI Agent
    positioning = None
    try:
        agent = _create_ip_agent()
        task = _create_ip_task(agent, profile)
        crew = Crew(
            agents=[agent],
            tasks=[task],
            process=Process.sequential,
            verbose=False,
        )
        result = run_crew(crew)

        # 从结果中提取 JSON
        raw_output = str(result.raw) if hasattr(result, 'raw') else str(result)
        positioning = _extract_json(raw_output)

    except Exception as e:
        state["errors"] = state.get("errors", []) + [{
            "node": "ip_positioning",
            "error": f"LLM 调用失败: {str(e)}，使用降级方案",
            "time": time.time(),
        }]

    # 降级方案
    if not positioning:
        positioning = _generate_fallback_positioning(profile)

    # 更新状态
    state["ip_positioning"] = positioning
    state["current_node"] = "ip_positioning"
    state["hitl_status"]["HITL-2"] = "pending"
    state["node_history"] = state.get("node_history", []) + [{
        "node": "ip_positioning",
        "start": time.time(),
        "end": time.time(),
        "status": "ok",
    }]

    return state


def _extract_json(raw_output: str) -> Optional[dict]:
    """从 CrewAI 输出中提取 JSON"""
    # 尝试直接解析
    try:
        return json.loads(raw_output)
    except json.JSONDecodeError:
        pass

    # 尝试从 markdown 代码块中提取
    import re
    json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', raw_output)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass

    # 尝试找到 JSON 对象
    brace_match = re.search(r'\{[\s\S]*\}', raw_output)
    if brace_match:
        try:
            return json.loads(brace_match.group(0))
        except json.JSONDecodeError:
            pass

    return None
