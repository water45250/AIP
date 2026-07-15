"""课程架构 Agent - 课程产品架构师 (M2 实现)

基于 CrewAI 框架，按"认知→方法→实战→变现"四阶递进模型设计课程大纲。
输出符合 CourseOutline Schema 的结构化数据。
"""

import json
import time
import re
from typing import Optional

from crewai import Agent, Task, Crew, Process

from ..graph.state import CourseState, CourseOutline
from ..tools import get_default_llm


# ============================================================
# CrewAI Agent 定义
# ============================================================

def _create_architect_agent() -> Agent:
    """创建课程架构 Agent"""
    return Agent(
        role="课程产品架构师",
        goal="设计结构清晰、递进合理的课程大纲，确保每个模块和课时都有明确的学习目标和交付价值",
        backstory=(
            "你是一位资深课程产品架构师，曾为上百位知识付费创作者设计过课程体系。\n"
            "你的核心设计理念是'四阶递进模型'：\n"
            "  认知重塑 → 方法体系 → 实战演练 → 变现闭环\n"
            "你擅长：\n"
            "1. 将模糊的知识领域拆解为清晰的学习路径\n"
            "2. 设计课程钩子，提升完课率和转化率\n"
            "3. 平衡知识深度和学员接受度\n"
            "你的原则：每节课必须有可量化的学习目标和可操作的课后任务。"
        ),
        llm=get_default_llm(),
        verbose=False,
        allow_delegation=False,
        max_iter=3,
    )


def _create_architect_task(agent: Agent, profile: dict, ip_positioning: dict) -> Task:
    """创建课程架构任务"""
    topic = profile.get("course_topic", "专业技能")
    expertise = profile.get("expertise", topic)
    audience = profile.get("target_audience", "目标学员")
    identity = profile.get("identity", "知识创作者")
    experience = profile.get("experience", "")
    style = profile.get("style_preference", "实战干货")
    positioning_statement = ip_positioning.get("positioning_statement", "")

    task_description = f"""
请基于以下信息，设计一份完整的课程大纲。

## 课程背景
- 课程主题：{topic}
- 讲师身份：{identity}（{experience}）
- 核心专长：{expertise}
- 目标学员：{audience}
- 风格偏好：{style}
- IP定位：{positioning_statement}

## 设计约束
1. 必须遵循"认知→方法→实战→变现"四阶递进模型
2. 总模块数：4-6个（至少包含认知/方法/实战/变现各一个）
3. 每模块课时数：2-4课时
4. 每课时必须包含：学习目标、3-5个核心要点、可操作的课后任务、预计时长（20-45分钟）
5. 在模块1第1课时设计"开场钩子"，在最后模块最后课时设计"转化钩子"
6. 课程标题要吸引人，不能太学术化

## 输出格式（严格 JSON）
```json
{{
  "course_title": "课程主标题",
  "total_modules": 4,
  "total_lessons": 12,
  "total_duration_minutes": 420,
  "modules": [
    {{
      "id": "M1",
      "title": "模块标题",
      "description": "模块简介（1-2句）",
      "phase": "认知",
      "lessons": [
        {{
          "id": "M1L1",
          "title": "课时标题",
          "learning_objective": "学完这节课你能...",
          "key_points": ["要点1", "要点2", "要点3"],
          "homework": "可操作的课后任务",
          "duration_minutes": 30,
          "hook_type": "opening"
        }}
      ]
    }}
  ],
  "hooks": [
    {{"type": "opening", "location": "M1L1", "content": "开场钩子文案"}},
    {{"type": "conversion", "location": "M4L3", "content": "转化钩子文案"}}
  ]
}}
```

注意：phase 必须从 ["认知", "方法", "实战", "变现"] 中选择。
hook_type 可选值：null（无钩子）、"opening"（开场钩子）、"transition"（过渡钩子）、"conversion"（转化钩子）
只返回 JSON，不要其他文字。
"""
    return Task(
        description=task_description,
        expected_output="符合 JSON 格式的课程大纲",
        agent=agent,
    )


# ============================================================
# 降级方案（无 LLM API 时使用）
# ============================================================

def _generate_fallback_outline(profile: dict, ip: dict) -> CourseOutline:
    """当 LLM 不可用时，基于模板生成课程大纲（降级方案）"""
    topic = profile.get("course_topic", "专业技能")
    audience = profile.get("target_audience", "目标学员")

    modules = [
        {
            "id": "M1", "title": f"认知重塑：重新理解{topic}",
            "description": "打破认知误区，建立正确的学习框架",
            "phase": "认知",
            "lessons": [
                {"id": "M1L1", "title": f"为什么你做{topic}总是不见效？",
                 "learning_objective": "识别3个常见认知误区，建立正确学习路径",
                 "key_points": ["3个常见误区诊断", "正确的方法论框架", "你的个人学习路线图"],
                 "homework": "列出你在该领域踩过的3个坑，分析根本原因",
                 "duration_minutes": 30, "hook_type": "opening"},
                {"id": "M1L2", "title": f"{topic}的底层逻辑与核心原理",
                 "learning_objective": "掌握核心原理，建立系统认知框架",
                 "key_points": ["核心原理拆解", "关键概念定义", "框架全景图"],
                 "homework": "用一句话说清楚该领域的核心原理",
                 "duration_minutes": 35, "hook_type": None},
                {"id": "M1L3", "title": f"设定你的{topic}学习目标与路径",
                 "learning_objective": "学会用SMART原则设定可执行的学习目标",
                 "key_points": ["SMART目标设定法", "目标拆解为周计划", "常见目标设定错误"],
                 "homework": "写下你的3个月学习目标和第一周执行计划",
                 "duration_minutes": 25, "hook_type": None},
            ],
        },
        {
            "id": "M2", "title": f"方法体系：{topic}的系统化打法",
            "description": "从零散技巧到系统方法论",
            "phase": "方法",
            "lessons": [
                {"id": "M2L1", "title": f"{topic}的5步核心方法论",
                 "learning_objective": "掌握从0到1的系统化操作流程",
                 "key_points": ["5步法详解", "每步的输入与输出", "常见卡点与破解方法"],
                 "homework": "按照5步法完成一次完整练习并记录过程",
                 "duration_minutes": 40, "hook_type": None},
                {"id": "M2L2", "title": "必备工具与效率模板",
                 "learning_objective": "掌握3个核心工具，提升3倍效率",
                 "key_points": ["工具1：XXX的使用方法", "工具2：XXX模板", "效率提升的5个技巧"],
                 "homework": "用模板完成一次实际操作并截图提交",
                 "duration_minutes": 35, "hook_type": None},
                {"id": "M2L3", "title": "避开90%新手都会犯的错",
                 "learning_objective": "识别并规避10个高频错误",
                 "key_points": ["10个高频错误清单", "每个错误的根因分析", "预防策略"],
                 "homework": "对照清单自查，记录你犯过的3个错误及改进方案",
                 "duration_minutes": 35, "hook_type": None},
            ],
        },
        {
            "id": "M3", "title": "实战演练：真实案例全流程拆解",
            "description": "用真实案例贯穿方法论，动手做出成果",
            "phase": "实战",
            "lessons": [
                {"id": "M3L1", "title": "案例拆解：从0到1的全流程还原",
                 "learning_objective": "通过真实成功案例理解完整操作流程",
                 "key_points": ["案例背景与初始条件", "关键操作步骤详解", "决策点与取舍逻辑"],
                 "homework": "拆解一个你欣赏的案例，写300字分析",
                 "duration_minutes": 40, "hook_type": None},
                {"id": "M3L2", "title": "动手实战：你的第一个完整项目",
                 "learning_objective": "独立完成一个从规划到执行的完整项目",
                 "key_points": ["项目任务书解读", "分步执行指南", "自检清单"],
                 "homework": "提交你的项目成果（至少包含3个关键产出物）",
                 "duration_minutes": 45, "hook_type": None},
                {"id": "M3L3", "title": "项目点评与优化迭代",
                 "learning_objective": "学会从反馈中识别改进点并迭代优化",
                 "key_points": ["常见问题总结", "优秀案例对比", "你的个人优化方向"],
                 "homework": "根据反馈清单优化你的项目，提交v2.0版本",
                 "duration_minutes": 35, "hook_type": None},
            ],
        },
        {
            "id": "M4", "title": "变现闭环：把能力变成可持续收入",
            "description": "将所学技能转化为商业价值",
            "phase": "变现",
            "lessons": [
                {"id": "M4L1", "title": f"{topic}的3种变现模式拆解",
                 "learning_objective": "了解主流变现路径，选择最适合你的模式",
                 "key_points": ["模式1：接单/服务变现", "模式2：知识付费变现", "模式3：咨询/顾问变现"],
                 "homework": "分析自己的优势，选择一种变现模式并制定3步计划",
                 "duration_minutes": 35, "hook_type": None},
                {"id": "M4L2", "title": "个人品牌打造与精准获客",
                 "learning_objective": "学会持续吸引精准客户的策略",
                 "key_points": ["个人品牌定位与包装", "内容获客的3个渠道", "从流量到付费的转化漏斗"],
                 "homework": "发布你的第一篇专业内容并记录数据",
                 "duration_minutes": 40, "hook_type": None},
                {"id": "M4L3", "title": "结课总结：你的90天行动计划",
                 "learning_objective": "制定可持续的成长和变现计划",
                 "key_points": ["课程核心回顾", "你的90天行动路线图", "持续进阶资源推荐"],
                 "homework": "制定并公开承诺你的90天行动计划",
                 "duration_minutes": 30, "hook_type": "conversion"},
            ],
        },
    ]

    total_lessons = sum(len(m["lessons"]) for m in modules)
    total_duration = sum(l["duration_minutes"] for m in modules for l in m["lessons"])

    return {
        "course_title": f"《{topic}实战训练营》：从入门到变现的系统方法论",
        "total_modules": len(modules),
        "total_lessons": total_lessons,
        "total_duration_minutes": total_duration,
        "modules": modules,
        "hooks": [
            {"type": "opening", "location": "M1L1",
             "content": f"你是否花了大量时间学习{topic}，却始终看不到成果？问题不在你，在于方法——今天开始，我们用一套被验证过的系统方法论，重新定义你的学习路径。"},
            {"type": "conversion", "location": "M4L3",
             "content": "恭喜你完成了这段学习旅程！但真正的成长才刚刚开始。加入我们的进阶社群，获得1v1辅导、资源对接和持续迭代——你的下一个里程碑，我们一起达成。"},
        ],
    }


# ============================================================
# 节点入口
# ============================================================

def run_course_architecture(state: CourseState) -> CourseState:
    """课程架构 Agent 主逻辑 - LangGraph 节点函数"""
    profile = state.get("user_profile", {})
    ip = state.get("ip_positioning", {})

    if not profile:
        state["errors"] = state.get("errors", []) + [{
            "node": "course_architecture",
            "error": "缺少用户画像，使用默认值",
            "time": time.time(),
        }]
        profile = {"course_topic": "专业技能", "identity": "知识博主"}

    if not ip:
        ip = {"positioning_statement": "帮目标学员通过系统方法论实现能力跃迁"}

    # 尝试调用 CrewAI Agent
    outline = None
    try:
        agent = _create_architect_agent()
        task = _create_architect_task(agent, profile, ip)
        crew = Crew(
            agents=[agent],
            tasks=[task],
            process=Process.sequential,
            verbose=False,
        )
        result = crew.kickoff()
        raw_output = str(result.raw) if hasattr(result, 'raw') else str(result)
        outline = _extract_json(raw_output)
    except Exception as e:
        state["errors"] = state.get("errors", []) + [{
            "node": "course_architecture",
            "error": f"LLM 调用失败: {str(e)}，使用降级方案",
            "time": time.time(),
        }]

    # 降级方案
    if not outline:
        outline = _generate_fallback_outline(profile, ip)

    # 验证大纲结构
    outline = _validate_and_fix_outline(outline)

    # 更新状态
    state["course_outline"] = outline
    state["current_node"] = "course_architecture"
    state["hitl_status"]["HITL-3"] = "pending"
    state["node_history"] = state.get("node_history", []) + [{
        "node": "course_architecture",
        "start": time.time(),
        "end": time.time(),
        "status": "ok",
    }]

    return state


def _extract_json(raw_output: str) -> Optional[dict]:
    """从 CrewAI 输出中提取 JSON"""
    try:
        return json.loads(raw_output)
    except json.JSONDecodeError:
        pass

    json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', raw_output)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass

    brace_match = re.search(r'\{[\s\S]*\}', raw_output)
    if brace_match:
        try:
            return json.loads(brace_match.group(0))
        except json.JSONDecodeError:
            pass

    return None


def _validate_and_fix_outline(outline: dict) -> dict:
    """验证并修正大纲结构"""
    modules = outline.get("modules", [])

    # 确保模块数在合理范围
    if len(modules) < 3:
        # 太少，不合法，用降级方案
        return _generate_fallback_outline(
            {"course_topic": "专业技能"}, {"positioning_statement": ""}
        )
    if len(modules) > 8:
        modules = modules[:8]

    # 修复每个模块
    valid_phases = {"认知", "方法", "实战", "变现"}
    for i, mod in enumerate(modules):
        if "id" not in mod:
            mod["id"] = f"M{i+1}"
        if mod.get("phase") not in valid_phases:
            phase_map = {0: "认知", 1: "方法", 2: "实战", 3: "变现"}
            mod["phase"] = phase_map.get(i, "方法")

        # 修复课时
        lessons = mod.get("lessons", [])
        for j, les in enumerate(lessons):
            if "id" not in les:
                les["id"] = f"{mod['id']}L{j+1}"
            if "duration_minutes" not in les or not isinstance(les.get("duration_minutes"), int):
                les["duration_minutes"] = 30
            if "hook_type" not in les:
                les["hook_type"] = None

    # 重新计算统计数据
    total_lessons = sum(len(m.get("lessons", [])) for m in modules)
    total_duration = sum(
        l.get("duration_minutes", 30)
        for m in modules
        for l in m.get("lessons", [])
    )

    outline["total_modules"] = len(modules)
    outline["total_lessons"] = total_lessons
    outline["total_duration_minutes"] = total_duration

    # 确保有 hooks
    if "hooks" not in outline or not outline["hooks"]:
        outline["hooks"] = [
            {"type": "opening", "location": "M1L1",
             "content": "开场钩子：用痛点共鸣抓住学员注意力"},
            {"type": "conversion", "location": f"M{len(modules)}L{len(modules[-1].get('lessons', [{}]))}",
             "content": "转化钩子：自然引出进阶服务"},
        ]

    return outline
