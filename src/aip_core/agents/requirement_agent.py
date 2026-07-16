"""需求解析 Agent - 课程需求分析师

M1 核心 Agent，负责：
1. 解析用户自由文本输入
2. 提取结构化用户画像
3. 判断完整度，触发追问或进入下一节点
"""

import json
import re
from typing import Optional

from ..graph.state import CourseState, UserProfile
from ..config import REQUIREMENT_MIN_COMPLETENESS, MAX_FOLLOWUP_ROUNDS

# ============================================================
# 结构化提取 Prompt
# ============================================================

EXTRACT_PROFILE_SYSTEM = """你是一位经验丰富的课程需求分析师。你的任务是从用户的自由文本中提取结构化信息。

请从以下用户输入中提取尽可能多的字段，以 JSON 格式返回：

{
    "identity": "用户身份，可选值：知识博主/独立讲师/咨询顾问/企业培训师/其他",
    "expertise": "核心专长和知识领域",
    "experience": "从业或实践年限及背景",
    "target_audience": "希望服务的学员画像",
    "course_topic": "期望的课程方向和主题",
    "delivery_format": "交付形式，可选值：录播视频/直播课/训练营/图文专栏/混合",
    "style_preference": "语言风格和人设倾向",
    "completeness": 0-100 的整数，表示信息完整度百分比
}

规则：
- 如果用户未提及某字段，该字段设为 null
- 不要编造信息
- completeness 基于已提取的非 null 字段数计算（共 7 个字段，每个约 14 分）
- 只返回 JSON，不要有其他文字"""

FOLLOWUP_QUESTIONS = {
    "identity": "在开始之前，想确认一下：你的身份更接近哪一类？\nA. 知识博主\nB. 独立讲师\nC. 咨询顾问\nD. 企业培训师\nE. 其他",
    "expertise": "你的核心专长是什么领域？比如：小红书运营、Python编程、个人品牌打造...",
    "experience": "在这个领域你有多少年的经验？有什么特别的成果或背书吗？",
    "target_audience": "你的课程主要面向什么样的学员？\nA. 零基础新手\nB. 有一定基础但没做出成果的\nC. 想提升变现效率的成熟玩家",
    "course_topic": "你想做什么方向的课程？越具体越好，比如'小红书从0到1万粉的实操方法'",
    "delivery_format": "你期望的课程交付形式是什么？\nA. 录播视频课程（学员自学）\nB. 直播授课+社群答疑\nC. 训练营模式（带作业和辅导）\nD. 图文专栏\nE. 混合形式",
    "style_preference": "你希望课程呈现什么样的风格？\nA. 专业严谨\nB. 轻松幽默\nC. 实战干货\nD. 故事驱动",
}

# 追问优先级：先问最影响后续流程的字段
FOLLOWUP_PRIORITY = [
    "course_topic",
    "expertise",
    "identity",
    "target_audience",
    "experience",
    "delivery_format",
    "style_preference",
]


def _get_missing_fields(profile: UserProfile) -> list[str]:
    """获取缺失的关键字段，按优先级排序"""
    missing = []
    for field in FOLLOWUP_PRIORITY:
        if not profile.get(field):
            missing.append(field)
    return missing


def _calculate_completeness(profile: UserProfile) -> int:
    """计算需求完整度 0-100"""
    fields = FOLLOWUP_PRIORITY
    filled = sum(1 for f in fields if profile.get(f))
    return int(filled / len(fields) * 100)


def _generate_followup(missing_fields: list[str], round_num: int) -> str:
    """生成追问消息

    每轮挑选不同的缺失字段追问，避免重复同一问题；
    并把后续还会问的字段先列出来，减少来回轮次。
    """
    if not missing_fields:
        return "我已经了解了你的需求，正在为你分析课程定位..."

    # 轮换缺失字段，保证每轮问的是不同问题
    idx = round_num % len(missing_fields)
    field = missing_fields[idx]
    question = FOLLOWUP_QUESTIONS.get(field, f"能再聊聊你的{field}吗？")

    # 把本轮之外、后续还会问的字段先提示出来
    _field_names = {
        "course_topic": "课程方向",
        "expertise": "核心专长",
        "identity": "你的身份",
        "target_audience": "目标学员",
        "experience": "从业经验",
        "delivery_format": "交付形式",
        "style_preference": "课程风格",
    }
    others = [f for f in missing_fields if f != field][:2]
    if others:
        names = "、".join(_field_names.get(f, f) for f in others)
        question = question.rstrip() + f"（后续还想了解：{names}）"

    if round_num == 0:
        return question
    return f"好的，还有一个问题：{question}"


def _parse_user_input(user_message: str, existing_profile: Optional[UserProfile] = None) -> UserProfile:
    """
    解析用户输入，提取结构化画像。

    实际生产环境调用 Claude 4 API，这里用规则 + 正则做轻量实现，
    后续替换为 LLM 调用即可。
    """
    profile = dict(existing_profile) if existing_profile else {}

    text = user_message.strip()

    # 1. 尝试检测身份
    identity_keywords = {
        "知识博主": ["博主", "自媒体", "内容创作者", "KOL", "up主"],
        "独立讲师": ["讲师", "培训师", "老师", "教练", "导师"],
        "咨询顾问": ["顾问", "咨询", "专家"],
        "企业培训师": ["企业培训", "内训", "企业"],
    }
    if "identity" not in profile or not profile["identity"]:
        for identity, keywords in identity_keywords.items():
            if any(kw in text for kw in keywords):
                profile["identity"] = identity
                break

    # 2. 尝试检测交付形式
    format_keywords = {
        "训练营": ["训练营", "陪跑", "带练", "辅导", "作业"],
        "直播课": ["直播", "实时", "在线授课"],
        "录播视频": ["录播", "视频课", "录课", "录制"],
        "图文专栏": ["图文", "专栏", "文章"],
        "混合": ["混合"],
    }
    if "delivery_format" not in profile or not profile["delivery_format"]:
        for fmt, keywords in format_keywords.items():
            if any(kw in text for kw in keywords):
                profile["delivery_format"] = fmt
                break

    # 3. 尝试检测风格偏好
    style_keywords = {
        "实战干货": ["干货", "实操", "落地", "实战", "可操作"],
        "专业严谨": ["专业", "严谨", "系统", "学术"],
        "轻松幽默": ["轻松", "幽默", "有趣", "好玩"],
        "故事驱动": ["故事", "案例", "经历"],
    }
    if "style_preference" not in profile or not profile["style_preference"]:
        for style, keywords in style_keywords.items():
            if any(kw in text for kw in keywords):
                profile["style_preference"] = style
                break

    # 4. 提取课程主题 - 宽松匹配自然表达（"ESG的系列课程" / "X培训" 等）
    #    关键：用「认识/了解/做...」等动词就近锚定课程词，避免 (.+?) 从句首吞到句尾。
    if "course_topic" not in profile or not profile["course_topic"]:
        topic_patterns = [
            r'关于(.+?)的课程',
            r'(?:认识|了解|做|讲|讲讲|想做|打算|希望|需要)(.+?)的(?:系列)?(?:课程|课|培训|训练营)',
            r'做[一个门]*[关于]*(.+?)[的之]*(课程|课|培训|训练营)',
            r'(.+?)运营',
            r'(.+?)教程',
        ]
        _lead_verbs = ("认识", "了解", "想做", "开始", "打算", "希望", "需要", "做", "讲", "讲讲")
        for pattern in topic_patterns:
            match = re.search(pattern, text)
            if match:
                topic = match.group(1).strip()
                for _v in _lead_verbs:
                    if topic.startswith(_v):
                        topic = topic[len(_v):].strip()
                if topic:
                    profile["course_topic"] = topic
                    break

    # 5. 提取经验
    if "experience" not in profile or not profile["experience"]:
        exp_patterns = [
            r'(\d+)\s*年',
            r'做了?\s*(\d+)\s*年',
        ]
        for pattern in exp_patterns:
            match = re.search(pattern, text)
            if match:
                profile["experience"] = f"{match.group(1)}年"
                break

    # 6. 提取受众 - 支持「针对/面向/服务 + 地区/人群」等自然表达
    #    关键：把「地区 + 人口词（如务工人员）」整体保留，而非只抓到地区。
    if "target_audience" not in profile or not profile["target_audience"]:
        audience_patterns = [
            r'(?:针对|面向|服务[于]?)(.+?(?:务工人员|学员|用户|人群|客户|读者|受众))',
            r'(.+?(?:务工人员|学员|用户|人群|客户|读者|受众))',
            r'(新手|零基础|小白|入门)',
            r'(进阶|提升|有一定基础)',
            r'(变现|赚钱|商业|盈利)',
        ]
        for pattern in audience_patterns:
            match = re.search(pattern, text)
            if match:
                _g = match.group(1).strip() if (match.groups() and match.group(1)) else match.group(0).strip()
                profile["target_audience"] = _g
                break

    # 7. 提取专长
    if "expertise" not in profile or not profile["expertise"]:
        # 如果提到了课程主题，专长通常就是那个领域
        if profile.get("course_topic"):
            profile["expertise"] = profile["course_topic"]

    return profile


# ============================================================
# 节点入口
# ============================================================

def run_requirement_analysis(state: CourseState) -> CourseState:
    """需求解析 Agent 主逻辑

    在 LangGraph 中作为节点函数调用。
    """
    # 获取当前对话消息
    messages = state.get("messages", [])
    last_user_message = ""
    for msg in reversed(messages):
        if isinstance(msg, dict) and msg.get("role") == "user":
            last_user_message = msg.get("content", "")
            break

    # 获取已有画像（如果是追问后再次进入）
    existing_profile = state.get("user_profile")

    # 解析用户输入
    profile = _parse_user_input(last_user_message, existing_profile)

    # 计算完整度
    completeness = _calculate_completeness(profile)
    followup_rounds = state.get("followup_rounds", 0)

    # 更新状态
    state["user_profile"] = profile
    state["requirement_completeness"] = completeness
    state["followup_rounds"] = followup_rounds
    state["current_node"] = "requirement_analysis"

    # 记录执行历史
    import time
    state["node_history"] = [{
        "node": "requirement_analysis",
        "start": time.time(),
        "end": time.time(),
        "status": "ok",
        "completeness": completeness,
    }]

    # 判断是否需要追问
    missing = _get_missing_fields(profile)
    if completeness < REQUIREMENT_MIN_COMPLETENESS and followup_rounds < MAX_FOLLOWUP_ROUNDS:
        followup_msg = _generate_followup(missing, followup_rounds)
        state["messages"] = state.get("messages", []) + [{
            "role": "assistant",
            "content": followup_msg,
            "type": "followup",
            "followup_round": followup_rounds + 1,
            "missing_fields": missing,
        }]
        state["followup_rounds"] = followup_rounds + 1
        state["hitl_status"]["HITL-1"] = "asking"
    else:
        # 完整度足够或达到追问上限
        state["hitl_status"]["HITL-1"] = "pending"
        # 生成确认摘要
        summary = _generate_profile_summary(profile, completeness)
        state["messages"] = state.get("messages", []) + [{
            "role": "assistant",
            "content": summary,
            "type": "hitl_preview",
        }]

    return state


def _generate_profile_summary(profile: UserProfile, completeness: int) -> str:
    """生成用户画像确认摘要"""
    identity_map = {
        "知识博主": "🎙️ 知识博主",
        "独立讲师": "👨‍🏫 独立讲师",
        "咨询顾问": "💼 咨询顾问",
        "企业培训师": "🏢 企业培训师",
    }
    format_map = {
        "训练营": "🔥 训练营（带作业辅导）",
        "直播课": "📡 直播授课+社群答疑",
        "录播视频": "🎬 录播视频课程",
        "图文专栏": "📝 图文专栏",
        "混合": "🔀 混合形式",
    }

    parts = ["📋 **课程需求分析报告**\n"]

    if profile.get("identity"):
        parts.append(f"**身份**：{identity_map.get(profile['identity'], profile['identity'])}")
    if profile.get("expertise"):
        parts.append(f"**核心专长**：{profile['expertise']}")
    if profile.get("experience"):
        parts.append(f"**经验背景**：{profile['experience']}")
    if profile.get("course_topic"):
        parts.append(f"**课程主题**：{profile['course_topic']}")
    if profile.get("target_audience"):
        parts.append(f"**目标学员**：{profile['target_audience']}")
    if profile.get("delivery_format"):
        parts.append(f"**交付形式**：{format_map.get(profile['delivery_format'], profile['delivery_format'])}")
    if profile.get("style_preference"):
        parts.append(f"**风格偏好**：{profile['style_preference']}")

    parts.append(f"\n📊 信息完整度：{completeness}%")

    if completeness >= REQUIREMENT_MIN_COMPLETENESS:
        parts.append("\n✅ 信息充足，可以开始课程生成。请确认以上信息，或选择修改/跳过。")
    else:
        parts.append(f"\n⚠️ 信息完整度偏低（{completeness}%），已达到追问上限，将基于现有信息继续。")

    return "\n".join(parts)
