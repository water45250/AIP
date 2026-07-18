"""内容生产 Agent 集群 - 执行层 (M3/M4 实现)

M3: 并行执行组
  - 脚本写手 (Agent 4): 逐课时 Markdown 讲稿
  - 课件设计师 (Agent 5): 结构化幻灯片数据
  - 案例挖掘师 (Agent 6): 行业案例匹配

M4: 串行执行组
  - 营销文案师 (Agent 7): 多版本营销物料
  - 定价策略师 (Agent 8): 定价方案建议

并行策略: asyncio.gather 同时执行三个 Agent，
任一失败不影响其他两个，全部完成后统一汇总。
"""

import asyncio
import json
import re
import time
from typing import Optional

from crewai import Agent, Task, Crew, Process

from ..graph.state import CourseState
from ..tools import get_default_llm
from ._crew_runner import run_crew


# ============================================================
# Agent 4: 脚本写手
# ============================================================

SCRIPT_WRITER_BACKSTORY = """你是一位资深课程讲稿撰写专家，擅长将课程大纲转化为生动、有料的逐字讲稿。
你的写作风格：
1. 开场用痛点/故事/反常识观点抓住注意力
2. 核心内容层层递进，用案例穿插降低认知负荷
3. 每15分钟设置一个互动点（提问/小练习）
4. 结尾有明确的小结和行动指引
你的讲稿模板：
## 开场白（2-3分钟）
## 核心内容（N分钟）
## 互动设计（2分钟）
## 小结与过渡（1-2分钟）
"""


def _create_script_writer() -> Agent:
    return Agent(
        role="课程脚本写手",
        goal="为每课时撰写高质量逐字讲稿，确保内容生动、结构清晰、学员易懂",
        backstory=SCRIPT_WRITER_BACKSTORY,
        llm=get_default_llm(),
        verbose=False,
        allow_delegation=False,
        max_iter=3,
    )


def _write_scripts_sync(outline: dict, profile: dict, ip: dict) -> dict:
    """同步版本：为每个课时生成讲稿（单个 Crew 处理整个大纲）"""
    topic = profile.get("course_topic", "专业技能")
    style = profile.get("style_preference", "实战干货")
    positioning = ip.get("positioning_statement", "")

    # 构建大纲摘要
    outline_summary = _format_outline_for_agent(outline)

    task = Task(
        description=f"""请基于以下课程大纲，为每一课时撰写完整的逐字讲稿。

## 课程信息
- 主题：{topic}
- 风格：{style}
- 定位：{positioning}

## 课程大纲
{outline_summary}

## 输出格式（严格 JSON）
```json
{{
  "M1L1": "## 开场白\\n...\\n\\n## 核心内容\\n...\\n\\n## 互动设计\\n...\\n\\n## 小结\\n...",
  "M1L2": "...",
  ...
}}
```
每个课时的讲稿必须包含：开场白、核心内容（含案例/举例）、互动设计、小结与过渡。
讲稿总字数每课时控制在 800-2000 字之间。
只返回 JSON，不要其他文字。""",
        expected_output="符合 JSON 格式的讲稿字典",
        agent=_create_script_writer(),
    )

    try:
        crew = Crew(agents=[task.agent], tasks=[task], process=Process.sequential, verbose=False)
        result = run_crew(crew)
        raw = str(result.raw) if hasattr(result, 'raw') else str(result)
        return _extract_json(raw) or {}
    except Exception:
        return {}


def _generate_fallback_scripts(outline: dict, profile: dict) -> dict:
    """降级方案：基于模板生成讲稿"""
    scripts = {}
    topic = profile.get("course_topic", "专业技能")

    for mod in outline.get("modules", []):
        module_title = mod.get("title", "")
        for les in mod.get("lessons", []):
            lid = les.get("id", "")
            title = les.get("title", "")
            key_points = les.get("key_points", [])
            homework = les.get("homework", "")
            hook_type = les.get("hook_type")

            # 开场白
            if hook_type == "opening":
                opening = f"""## 开场白（3分钟）

大家好，欢迎来到《{topic}实战训练营》！

在开始之前，我想问你一个问题：你是否花了大量时间学习{topic}，却始终感觉看不到成果？学了很多方法，但一到实际操作就无从下手？

如果你有这样的感受，恭喜你——今天开始，这套课程将彻底改变你的学习方式。

我是这门课的讲师，过去这些年，我帮助了数百位学员在{topic}领域实现了从0到1的突破。今天第一节课，我们先从最核心的问题开始：{title}

"""
            else:
                opening = f"""## 开场白（2分钟）

大家好，欢迎回来！上节课我们聊了{topic}的一个重要话题，今天继续深入。

本节课的主题是：**{title}**。这可能是整个课程中最实用的一节课之一，建议大家准备好纸笔，边听边记。

"""

            # 核心内容
            core_parts = []
            for i, point in enumerate(key_points):
                core_parts.append(f"""### 要点{i+1}：{point}

关于{point}，我想分享一个关键洞察——

（此处展开讲解{point}的核心原理、操作步骤和注意事项。建议结合具体案例说明，让学员能够立即理解并应用。）

举个例子：假设你正在实际操作中遇到这个问题，你应该怎么做？第一步...第二步...第三步...

""")

            # 互动设计
            engagement = f"""## 互动设计（2分钟）

思考题：回想一下你在{topic}领域的具体经历——

{homework if homework else "你觉得今天讲的哪个要点对你最有启发？为什么？"}

欢迎在学习群里分享你的思考，我会挑选精彩回答在下节课点评！

"""

            # 小结
            summary = f"""## 小结与过渡（1分钟）

本节课我们学习了{title}的核心要点。记住一句话：{'、'.join(key_points[:2]) if key_points else '知行合一，学以致用'}。

下节课我们将继续深入，带来更精彩的内容。记得完成课后任务，我们下节课见！

"""

            scripts[lid] = opening + "## 核心内容\n\n" + "\n".join(core_parts) + engagement + summary

    return scripts


# ============================================================
# Agent 5: 课件设计师
# ============================================================

SLIDE_DESIGNER_BACKSTORY = """你是一位专业的课件设计师，擅长将讲稿内容转化为结构清晰的幻灯片。
你的设计原则：
1. 每页幻灯片不超过5个要点
2. 要点用短语而非完整句子（简洁有力）
3. 每页包含讲师备注（帮助讲师理解该页的讲解重点）
4. 配图建议要具体（如"此处适合放一个对比表格"）
"""


def _create_slide_designer() -> Agent:
    return Agent(
        role="课件设计师",
        goal="将课程讲稿转化为结构清晰、视觉友好的幻灯片",
        backstory=SLIDE_DESIGNER_BACKSTORY,
        llm=get_default_llm(),
        verbose=False,
        allow_delegation=False,
        max_iter=3,
    )


def _design_slides_sync(outline: dict, scripts: dict) -> dict:
    """同步版本：为每课时设计幻灯片"""
    if not scripts:
        return {}

    # 取第一个课时做样例，批量设计所有幻灯片
    sample_lid = list(scripts.keys())[0]
    sample_script = scripts[sample_lid][:500]  # 取前500字做样例

    # 构建大纲摘要
    outline_summary = _format_outline_for_agent(outline)

    task = Task(
        description=f"""请基于课程大纲和讲稿样例，为每一课时设计幻灯片。

## 课程大纲
{outline_summary}

## 讲稿样例（{sample_lid}）
{sample_script}

## 输出格式（严格 JSON）
```json
{{
  "M1L1": {{
    "lesson_id": "M1L1",
    "total_slides": 5,
    "slides": [
      {{"page": 1, "title": "封面标题", "bullets": [], "image_suggestion": null, "speaker_notes": "开场白要点"}},
      {{"page": 2, "title": "要点页标题", "bullets": ["要点1", "要点2"], "image_suggestion": "对比表格", "speaker_notes": "展开讲解核心概念"}},
      ...
    ]
  }},
  ...
}}
```
每课时设计 4-8 页幻灯片，包含封面页、核心内容页、互动页、小结页。
每页 bullets 不超过 5 条。只返回 JSON。""",
        expected_output="符合 JSON 格式的幻灯片数据",
        agent=_create_slide_designer(),
    )

    try:
        crew = Crew(agents=[task.agent], tasks=[task], process=Process.sequential, verbose=False)
        result = run_crew(crew)
        raw = str(result.raw) if hasattr(result, 'raw') else str(result)
        return _extract_json(raw) or {}
    except Exception:
        return {}


def _generate_fallback_slides(outline: dict) -> dict:
    """降级方案：基于大纲模板生成幻灯片"""
    slides = {}

    for mod in outline.get("modules", []):
        for les in mod.get("lessons", []):
            lid = les.get("id", "")
            title = les.get("title", "")
            key_points = les.get("key_points", [])
            homework = les.get("homework", "")
            learning_obj = les.get("learning_objective", "")

            slide_pages = [
                {"page": 1, "title": title, "bullets": [f"学习目标：{learning_obj}"],
                 "image_suggestion": None, "speaker_notes": f"简短介绍本节课主题：{title}。预告本节课将解决的核心问题。"},
            ]

            for i, point in enumerate(key_points):
                slide_pages.append({
                    "page": i + 2,
                    "title": f"要点{i+1}",
                    "bullets": [point, f"展开说明{point}的具体方法", "案例/数据支撑"],
                    "image_suggestion": "流程图" if i == 0 else "案例截图" if i == 1 else None,
                    "speaker_notes": f"重点讲解{point}，建议用时3-5分钟。可以结合个人经历或学员案例展开。",
                })

            slide_pages.append({
                "page": len(key_points) + 2,
                "title": "互动与练习",
                "bullets": [homework, "小组讨论：分享你的想法"],
                "image_suggestion": None,
                "speaker_notes": "引导学员参与互动，预留2-3分钟讨论时间。",
            })

            slide_pages.append({
                "page": len(key_points) + 3,
                "title": "本节课小结",
                "bullets": key_points[:3] + ["课后任务：" + homework],
                "image_suggestion": None,
                "speaker_notes": "快速回顾本节课核心要点，强调课后任务的重要性，预告下节课内容。",
            })

            slides[lid] = {
                "lesson_id": lid,
                "total_slides": len(slide_pages),
                "slides": slide_pages,
            }

    return slides


# ============================================================
# Agent 6: 案例挖掘师
# ============================================================

CASE_MINER_BACKSTORY = """你是一位专注ESG（环境、社会、治理）领域的行业研究专家，熟悉真实的ESG法规、事件与企业实践。
你的案例标准：
1. 真实可信：必须基于真实发生的事件/政策/企业实践，包含具体的公司或机构名称与年份，禁止虚构
2. 可考据：尽量给出信息来源（如监管文件、企业年报、权威媒体）
3. 相关性强：案例必须与课程主题高度相关
4. 结构化呈现：背景→挑战→方案→结果→启示
5. 数据支撑：尽可能提供真实的具体数字
注意：不要依赖外部搜索工具，直接基于你自身的专业知识储备输出真实案例。
"""


def _create_case_miner() -> Agent:
    return Agent(
        role="ESG行业案例研究专家",
        goal="为ESG课程挖掘3-5个真实、可考据的高质量行业案例，支撑教学内容",
        backstory=CASE_MINER_BACKSTORY,
        llm=get_default_llm(),
        verbose=False,
        allow_delegation=False,
        max_iter=3,
    )


def _mine_cases_sync(topic: str, tags: list) -> list:
    """同步版本：挖掘行业案例"""
    tags_str = ", ".join(tags) if tags else topic

    task = Task(
        description=f"""请为以下课程主题挖掘3-5个真实、可考据的高质量行业案例。

## 课程主题
{topic}

## 相关标签
{tags_str}

## 任务
直接基于你的专业知识储备，列出3-5个真实发生的ESG案例（例如：真实的环保处罚/诉讼事件、碳关税/碳交易政策、企业ESG报告实践、绿色债券/可持续融资、供应链尽职调查、社会责任争议等）。每个案例必须具体、可查证，包含真实的公司/机构名称与年份，禁止编造。

## 输出格式（严格 JSON 数组）
```json
[
  {{
    "title": "案例标题（含公司/机构与年份）",
    "background": "案例背景（2-3句，含真实事实）",
    "challenge": "面临的核心挑战",
    "solution": "解决方案（3-5步，具体措施）",
    "results": "具体结果（尽可能有真实数据/数字）",
    "source": "信息来源（如监管文件/企业年报/权威媒体）",
    "relevance": "与本课程ESG知识体系的关联说明"
  }}
]
```
只返回 JSON 数组，不要其他文字。""",
        expected_output="符合 JSON 格式的案例列表",
        agent=_create_case_miner(),
    )

    try:
        crew = Crew(agents=[task.agent], tasks=[task], process=Process.sequential, verbose=False)
        result = run_crew(crew)
        raw = str(result.raw) if hasattr(result, 'raw') else str(result)
        return _extract_json(raw) or []
    except Exception:
        return []


def _generate_fallback_cases(topic: str, tags: list) -> list:
    """降级方案：基于模板生成案例"""
    return [
        {
            "title": f"{topic}领域典型成功案例",
            "background": f"在{topic}领域，许多从业者通过系统化方法论实现了显著突破。",
            "challenge": "缺乏系统方法论，学习效率低下，难以形成可复制的成功路径。",
            "solution": "1. 建立系统化知识框架\n2. 制定分阶段学习计划\n3. 实战中迭代优化\n4. 寻求专业指导和反馈",
            "results": "通过系统化学习，多数人在3-6个月内实现能力跃迁，效率提升50%以上。",
            "source": "行业公开数据",
            "relevance": f"本案例展示了{topic}领域从零散学习到系统掌握的关键路径，与课程的方法论体系高度契合。",
        },
        {
            "title": f"从入门到变现：{topic}实践者的成长路径",
            "background": f"一位{topic}的初学者，从完全不懂到能够独立接单/变现的完整历程。",
            "challenge": "知识碎片化，不知道学什么、怎么学、学完怎么用。",
            "solution": "1. 选择正确的学习路径和资源\n2. 以项目驱动学习，边学边做\n3. 积累作品/案例建立信任\n4. 逐步从低价到高价定位",
            "results": "6个月内从0基础到月收入突破万元（效果因人而异）。",
            "source": "行业访谈与公开分享",
            "relevance": "本案例完整呈现了'认知→方法→实战→变现'的四阶递进过程，是课程核心理念的最佳佐证。",
        },
        {
            "title": f"数据驱动的{topic}效率提升实践",
            "background": "通过数据分析和流程优化，大幅提升{topic}领域的工作效率。",
            "challenge": "重复性工作多，缺乏标准化流程，产出不稳定。",
            "solution": "1. 建立标准化操作流程(SOP)\n2. 引入工具和模板提升效率\n3. 数据跟踪和持续优化\n4. 团队协作和经验沉淀",
            "results": "标准化后效率提升3倍，产出质量更加稳定可控。",
            "source": "行业最佳实践总结",
            "relevance": "本案例展示了系统方法论如何落地为标准化流程，是课程中'方法体系'模块的实践验证。",
        },
    ]


# ============================================================
# 并行编排
# ============================================================

def run_content_parallel(state: CourseState) -> CourseState:
    """内容生产并行阶段入口 - LangGraph 节点函数

    使用 asyncio.gather 并行执行三个 Agent：
    - 脚本写手
    - 课件设计师
    - 案例挖掘师
    """
    outline = state.get("course_outline", {})
    profile = state.get("user_profile", {})
    ip = state.get("ip_positioning") or {}
    topic = profile.get("course_topic", "专业技能")
    tags = ip.get("differentiation_tags", [])

    start_time = time.time()

    # 使用 asyncio 并行执行
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # 已有事件循环在运行，使用 nest_asyncio 或同步降级
            scripts, slides, cases = _run_parallel_sync(outline, profile, ip, topic, tags)
        else:
            scripts, slides, cases = loop.run_until_complete(
                _run_parallel_async(outline, profile, ip, topic, tags)
            )
    except RuntimeError:
        # 无事件循环，创建新的
        scripts, slides, cases = asyncio.run(
            _run_parallel_async(outline, profile, ip, topic, tags)
        )

    # 降级处理：如果 LLM 返回空，使用模板
    if not scripts:
        scripts = _generate_fallback_scripts(outline, profile)
    if not slides:
        slides = _generate_fallback_slides(outline)
    if not cases:
        cases = _generate_fallback_cases(topic, tags)

    # 更新状态
    state["scripts"] = scripts
    state["slides"] = slides
    state["cases"] = cases
    state["current_node"] = "content_production_parallel"
    state["node_history"] = state.get("node_history", []) + [{
        "node": "content_production_parallel",
        "start": start_time,
        "end": time.time(),
        "status": "ok",
        "scripts_count": len(scripts),
        "slides_count": len(slides),
        "cases_count": len(cases),
    }]

    return state


async def _run_parallel_async(outline: dict, profile: dict, ip: dict, topic: str, tags: list):
    """异步并行执行三个 Agent"""
    results = await asyncio.gather(
        asyncio.to_thread(_write_scripts_sync, outline, profile, ip),
        asyncio.to_thread(_design_slides_sync, outline, {}),
        asyncio.to_thread(_mine_cases_sync, topic, tags),
        return_exceptions=True,
    )

    scripts, slides, cases = {}, {}, []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            continue  # 某个 Agent 失败，其他继续
        if i == 0:
            scripts = result or {}
        elif i == 1:
            slides = result or {}
        elif i == 2:
            cases = result or []

    return scripts, slides, cases


def _run_parallel_sync(outline: dict, profile: dict, ip: dict, topic: str, tags: list):
    """同步降级：顺序执行（当事件循环不可用时）"""
    try:
        scripts = _write_scripts_sync(outline, profile, ip) or _generate_fallback_scripts(outline, profile)
    except Exception:
        scripts = _generate_fallback_scripts(outline, profile)
    try:
        slides = _design_slides_sync(outline, scripts) or _generate_fallback_slides(outline)
    except Exception:
        slides = _generate_fallback_slides(outline)
    try:
        cases = _mine_cases_sync(topic, tags) or _generate_fallback_cases(topic, tags)
    except Exception:
        cases = _generate_fallback_cases(topic, tags)
    return scripts, slides, cases


# ============================================================
# M4: 串行执行组 - 营销文案师 (Agent 7)
# ============================================================

MARKETING_AGENT_BACKSTORY = """你是一位资深课程营销文案专家，擅长将课程卖点转化为高转化率文案。
你的写作原则：
1. 痛点驱动：先戳痛点，再给方案
2. 信任背书：用数据、案例、身份建立信任
3. 行动号召：每段文案必须有明确的 CTA
4. 多版本适配：同一卖点，不同场景用不同角度和语气
5. 合规底线：不承诺具体收益，不使用绝对化用语"""


def _create_marketing_agent() -> Agent:
    return Agent(
        role="课程营销文案师",
        goal="生成多版本、高转化的营销物料，覆盖销售页、朋友圈、社群和短视频场景",
        backstory=MARKETING_AGENT_BACKSTORY,
        llm=get_default_llm(),
        verbose=False,
        allow_delegation=False,
        max_iter=3,
    )


def _generate_marketing_sync(profile: dict, ip: dict, outline: dict, cases: list) -> dict:
    """同步版本：生成营销物料"""
    topic = profile.get("course_topic", "专业技能")
    audience = profile.get("target_audience", "目标学员")
    statement = ip.get("positioning_statement", "")
    tags = ip.get("differentiation_tags", [])
    course_title = outline.get("course_title", f"{topic}实战训练营")
    total_modules = outline.get("total_modules", 4)
    total_lessons = outline.get("total_lessons", 12)
    case_names = [c.get("title", "") for c in cases[:2]]

    task = Task(
        description=f"""请为以下课程生成多版本营销物料。

## 课程信息
- 课程标题：{course_title}
- 课程主题：{topic}
- 目标学员：{audience}
- IP定位：{statement}
- 差异化标签：{', '.join(tags)}
- 课程规模：{total_modules}模块/{total_lessons}课时
- 相关案例：{', '.join(case_names) if case_names else '行业典型案例'}

## 输出格式（严格 JSON）
```json
{{
  "sales_page": "课程销售页完整文案（包含痛点引入、课程亮点、大纲预览、信任背书、CTA）",
  "moments": [
    "朋友圈文案1：痛点切入型",
    "朋友圈文案2：成果展示型",
    "朋友圈文案3：故事驱动型"
  ],
  "community_script": {{
    "welcome": "入群欢迎话术",
    "teaser": "价值预告话术",
    "flash_sale": "限时优惠话术",
    "close": "成交逼单话术"
  }},
  "video_outlines": [
    {{"duration": "60秒", "content": "短视频脚本概要"}},
    {{"duration": "3分钟", "content": "短视频脚本概要"}}
  ]
}}
```
营销文案必须与IP定位保持一致。禁止承诺具体收益数字。只返回 JSON。""",
        expected_output="符合 JSON 格式的营销物料",
        agent=_create_marketing_agent(),
    )

    try:
        crew = Crew(agents=[task.agent], tasks=[task], process=Process.sequential, verbose=False)
        result = run_crew(crew)
        raw = str(result.raw) if hasattr(result, 'raw') else str(result)
        return _extract_json(raw) or {}
    except Exception:
        return {}


def _generate_fallback_marketing(profile: dict, ip: dict, outline: dict, cases: list) -> dict:
    """降级方案：基于模板生成营销文案"""
    topic = profile.get("course_topic", "专业技能")
    audience = profile.get("target_audience", "目标学员")
    statement = ip.get("positioning_statement", "")
    course_title = outline.get("course_title", f"{topic}实战训练营")
    total_modules = outline.get("total_modules", 4)
    total_lessons = outline.get("total_lessons", 12)

    return {
        "sales_page": (
            f"# {course_title}\n\n"
            f"## 你是否也面临这些问题？\n"
            f"- 学了很长时间{topic}却始终看不到成果\n"
            f"- 缺乏系统化的方法论，东学一点西学一点\n"
            f"- 不知道如何将所学转化为实际收入\n\n"
            f"## 这门课能给你什么\n"
            f"{statement}\n\n"
            f"## 课程亮点\n"
            f"- {total_modules}大模块，{total_lessons}课时系统学习\n"
            f"- 从认知→方法→实战→变现的完整闭环\n"
            f"- 真实案例拆解 + 动手实战项目\n"
            f"- 专属学习社群 + 讲师答疑\n\n"
            f"## 适合谁学\n"
            f"{audience}\n\n"
            f"## 学员反馈\n"
            f'"这是我学过最系统的{topic}课程，终于不再碎片化学习了！"\n\n'
            f"## 现在加入\n"
            f"🔥 限时早鸟价优惠中，立即报名开启你的蜕变之旅 →\n\n"
            f"*效果因人而异，本课程不承诺具体收益。"
        ),
        "moments": [
            f"做了这么久{topic}，你真的掌握系统方法论了吗？大多数人缺的不是努力，而是一套可复制的方法。#个人成长 #{topic}",
            f"学员3个月实现突破的秘密：不是天赋，而是正确的方法+持续的行动。你想成为下一个吗？#干货分享 #{topic}",
            f"从0到1的{topic}实战之路，我踩过的坑和方法都在这里了。扫码了解详情 👇 #真实经历 #{topic}",
        ],
        "community_script": {
            "welcome": f"🎉 欢迎加入《{course_title}》学习社群！接下来的21天，我们将一起完成从认知到变现的蜕变。请先看群公告，然后做自我介绍：我是谁+我想通过这门课解决什么问题？",
            "teaser": f"📣 明晚8点，我将分享{topic}领域一个让效率翻倍的秘密武器。这个技巧我在实战中反复验证过，记得定好闹钟！",
            "flash_sale": f"🔥 限时福利：今晚12点前报名《{course_title}》立减300元！仅剩最后5个名额，手慢无 →",
            "close": f"⏰ 最后3小时！《{course_title}》早鸟价即将截止。错过这波只能等下一期涨价了，立即锁定名额 →",
        },
        "video_outlines": [
            {"duration": "60秒",
             "content": f"开头：做{topic}的你是否也有这些困惑？（3个痛点快速闪过）→ 转折：其实你缺的不是努力 → 解决方案：一套系统方法论 → CTA：点击主页了解详情"},
            {"duration": "3分钟",
             "content": f"开头：我的故事/学员案例引入 → 问题分析：为什么大多数人做不好{topic} → 方法拆解：3个核心步骤 → 成果展示 → CTA：课程介绍+限时优惠"},
        ],
    }


# ============================================================
# M4: 定价策略师 (Agent 8)
# ============================================================

PRICING_AGENT_BACKSTORY = """你是一位课程定价策略专家，擅长基于市场数据和课程规模制定科学的定价方案。
你的定价原则：
1. 基于课程实际价值（课时数、深度、服务配套）
2. 参考知识付费领域同类ESG/职业技能课程的真实市场价格区间（如训练营常见 ¥299-¥3999）
3. 设计阶梯定价，让不同预算的学员都有选择
4. 早鸟价要有真实的紧迫感
5. 定价依据必须清晰可解释，基于课程规模给出合理判断"""


def _create_pricing_agent() -> Agent:
    return Agent(
        role="课程定价策略师",
        goal="基于课程规模和市场数据，制定科学的定价方案和促销策略",
        backstory=PRICING_AGENT_BACKSTORY,
        llm=get_default_llm(),
        verbose=False,
        allow_delegation=False,
        max_iter=3,
    )


def _generate_pricing_sync(outline: dict, topic: str) -> dict:
    """同步版本：生成定价方案"""
    total_lessons = outline.get("total_lessons", 12)
    total_duration = outline.get("total_duration_minutes", 480)

    task = Task(
        description=f"""请为以下课程制定定价方案。

## 课程信息
- 主题：{topic}
- 课时数：{total_lessons}课时
- 总时长：{total_duration}分钟
- 课程形式：训练营（录播+直播+社群）

## 任务
1. 基于课程规模（课时数、时长、服务配套）和市场同类知识付费课程的真实价格区间，制定阶梯定价方案
2. 价格需合理、有竞争力，并给出清晰可解释的定价依据

## 输出格式（严格 JSON）
```json
{{
  "standard_price": 1999,
  "early_bird_price": 1299,
  "tiered_pricing": [
    {{"tier": "基础版", "price": 999, "includes": ["内容1", "内容2"]}},
    {{"tier": "进阶版", "price": 1999, "includes": ["内容1", "内容2", "内容3"]}},
    {{"tier": "全套版", "price": 3999, "includes": ["内容1", "内容2", "内容3", "内容4"]}}
  ],
  "rationale": "定价依据说明（2-3句话）"
}}
```
价格单位：人民币元。只返回 JSON。""",
        expected_output="符合 JSON 格式的定价方案",
        agent=_create_pricing_agent(),
    )

    try:
        crew = Crew(agents=[task.agent], tasks=[task], process=Process.sequential, verbose=False)
        result = run_crew(crew)
        raw = str(result.raw) if hasattr(result, 'raw') else str(result)
        return _extract_json(raw) or {}
    except Exception:
        return {}


def _generate_fallback_pricing(outline: dict) -> dict:
    """降级方案：基于课程规模估算定价"""
    total_lessons = outline.get("total_lessons", 12)
    total_duration = outline.get("total_duration_minutes", 480)

    # 基于课时数估算
    if total_lessons <= 8:
        base_price = 999
    elif total_lessons <= 15:
        base_price = 1999
    else:
        base_price = 2999

    return {
        "standard_price": base_price,
        "early_bird_price": int(base_price * 0.65),
        "tiered_pricing": [
            {"tier": "基础版", "price": int(base_price * 0.5),
             "includes": ["录播课程", "课件下载", "学习社群"]},
            {"tier": "进阶版", "price": base_price,
             "includes": ["全部录播课程", "每周直播答疑", "作业点评", "专属社群"]},
            {"tier": "全套版", "price": base_price * 2,
             "includes": ["全部内容", "1v1辅导(3次)", "资源对接", "结业证书"]},
        ],
        "rationale": f"基于{total_lessons}课时/{total_duration}分钟课程规模，参考同领域知识付费课程定价区间（¥500-¥5000），建议标准定价¥{base_price}。早鸟价¥{int(base_price*0.65)}有助于前期快速积累学员口碑。",
    }


# ============================================================
# M4: 串行编排入口
# ============================================================

def _call_with_timeout(func, args=(), kwargs={}, timeout_seconds=15):
    """v12: timeout wrapper to prevent CrewAI hang (content_production_serial)"""
    import threading as _th
    result = [None]; exc = [None]
    def target():
        try: result[0] = func(*args, **kwargs)
        except Exception as e: exc[0] = e
    t = _th.Thread(target=target, daemon=True); t.start()
    t.join(timeout=timeout_seconds)
    if t.is_alive(): return None
    if exc[0] is not None: return None
    return result[0]


def run_content_serial(state: CourseState) -> CourseState:
    """内容生产串行阶段 - LangGraph 节点函数

    串行执行两个 Agent：
    1. 营销文案师 → marketing_copy
    2. 定价策略师 → pricing_plan
    """
    profile = state.get("user_profile", {})
    ip = state.get("ip_positioning") or {}
    outline = state.get("course_outline", {})
    cases = state.get("cases", [])

    topic = profile.get("course_topic", "专业技能")
    start_time = time.time()

    # Step 1: 营销文案
    # v12: 用 threading 超时包裹 CrewAI 调用，防止 deepseek 兼容性问题（OpenAI function name
    # cannot be empty）导致 crew.kickoff() 挂死不返回而卡死整个 content_production_serial 节点
    marketing = _call_with_timeout(
        _generate_marketing_sync, args=(profile, ip, outline, cases), timeout_seconds=15
    )
    if not marketing or not isinstance(marketing, dict) or not marketing.get("sales_page"):
        marketing = _generate_fallback_marketing(profile, ip, outline, cases)

    # Step 2: 定价方案
    pricing = _call_with_timeout(
        _generate_pricing_sync, args=(outline, topic), timeout_seconds=15
    )
    if not pricing or not isinstance(pricing, dict) or not pricing.get("standard_price"):
        pricing = _generate_fallback_pricing(outline)

    # 更新状态
    state["marketing_copy"] = marketing
    state["pricing_plan"] = pricing
    state["current_node"] = "content_production_serial"
    # v13: 只在尚未确认/跳过时才设为 pending，防止 _advance_to_next 重入时覆盖 confirmed → 死循环
    if state.get("hitl_status", {}).get("HITL-4") not in ("confirmed", "skipped"):
        state["hitl_status"]["HITL-4"] = "pending"
    state["node_history"] = state.get("node_history", []) + [{
        "node": "content_production_serial",
        "start": start_time,
        "end": time.time(),
        "status": "ok",
    }]

    return state


# ============================================================
# 工具函数
# ============================================================

def _format_outline_for_agent(outline: dict) -> str:
    """将大纲格式化为 Agent 可读的文本"""
    if not outline or not isinstance(outline, dict):
        return "（课程大纲尚未生成，请先完成课程架构设计）"
    lines = [f"## {outline.get('course_title', '课程大纲')}", ""]
    for mod in outline.get("modules", []):
        lines.append(f"### {mod.get('id', '')} {mod.get('title', '')}（{mod.get('phase', '')}）")
        lines.append(f"描述：{mod.get('description', '')}")
        for les in mod.get("lessons", []):
            lines.append(f"  - {les.get('id', '')} {les.get('title', '')}")
            lines.append(f"    目标：{les.get('learning_objective', '')}")
            lines.append(f"    要点：{', '.join(les.get('key_points', []))}")
            lines.append(f"    作业：{les.get('homework', '')}")
            lines.append(f"    时长：{les.get('duration_minutes', 30)}分钟")
        lines.append("")
    return "\n".join(lines)


def _extract_json(raw_output: str):
    """从 CrewAI 输出中提取 JSON"""
    try:
        return json.loads(raw_output)
    except (json.JSONDecodeError, TypeError):
        pass
    json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', raw_output)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except (json.JSONDecodeError, TypeError):
            pass
    brace_match = re.search(r'[\[{][\s\S]*[}\]]', raw_output)
    if brace_match:
        try:
            return json.loads(brace_match.group(0))
        except (json.JSONDecodeError, TypeError):
            pass
    return None
