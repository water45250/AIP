"""课程生成全局状态定义

LangGraph 的 CourseState 是整个编排系统的核心数据结构。
所有 7 个 Agent 节点通过此 State 交换数据。
"""

from typing import TypedDict, Literal, Annotated, Optional
import operator


# ============================================================
# 子数据结构
# ============================================================

class UserProfile(TypedDict, total=False):
    """用户画像 - 需求解析 Agent 产出"""
    identity: str           # 用户身份：知识博主/独立讲师/咨询顾问/企业培训师/其他
    expertise: str          # 核心专长领域
    experience: str         # 经验年限/背景
    target_audience: str    # 目标学员画像
    course_topic: str       # 期望课程主题
    delivery_format: str    # 交付形式：录播视频/直播课/训练营/图文专栏/混合
    style_preference: str   # 语言风格/人设倾向


class IPPositioning(TypedDict, total=False):
    """IP 定位报告 - IP 定位 Agent 产出"""
    positioning_statement: str          # 一句话定位宣言
    differentiation_tags: list[str]     # 差异化标签 3-5 个
    trust_flywheel: list[dict]          # 信任飞轮 [{step, action, goal}]
    content_matrix: dict                # 内容矩阵 {platforms, content_types, frequency}


class Lesson(TypedDict, total=False):
    """课时结构"""
    id: str                      # "M1L2"
    title: str
    learning_objective: str
    key_points: list[str]        # 3-5 个
    homework: str
    duration_minutes: int
    hook_type: Optional[str]     # "opening" | "transition" | "conversion" | None


class Module(TypedDict, total=False):
    """课程模块"""
    id: str                      # "M1"
    title: str
    description: str
    phase: str                   # "认知" | "方法" | "实战" | "变现"
    lessons: list[Lesson]


class CourseOutline(TypedDict, total=False):
    """课程大纲 - 课程架构 Agent 产出"""
    course_title: str
    total_modules: int
    total_lessons: int
    total_duration_minutes: int
    modules: list[Module]
    hooks: list[dict]            # [{type, location, content}]


class ReviewDetail(TypedDict, total=False):
    """审核明细"""
    total_score: int
    dimensions: dict             # {dim_name: {score, max, issues}}
    pass_: bool                  # ≥80
    auto_skip_hitl: bool         # ≥85


# ============================================================
# 全局状态
# ============================================================

class CourseState(TypedDict, total=False):
    """课程生成全局状态 - 贯穿整个 LangGraph 主图"""

    # === 会话标识 ===
    session_id: str
    user_id: str
    current_node: str                    # 当前所在节点名

    # === 需求解析 (Agent 1 产出) ===
    user_profile: Optional[UserProfile]
    requirement_completeness: int        # 0-100
    followup_rounds: int                 # 已追问轮数

    # === IP 定位 (Agent 2 产出) ===
    ip_positioning: Optional[IPPositioning]

    # === 课程架构 (Agent 3 产出) ===
    course_outline: Optional[CourseOutline]

    # === 内容生产 (Agent 4-6 并行产出) ===
    scripts: Optional[dict]              # {lesson_id: markdown_text}
    slides: Optional[dict]               # {lesson_id: slide_data}
    cases: Optional[list]                # [{title, background, challenge, ...}]

    # === 内容生产 (Agent 7-8 串行产出) ===
    marketing_copy: Optional[dict]       # {物料类型: 内容}
    pricing_plan: Optional[dict]         # {standard_price, early_bird, tiered, ...}

    # === 审核 (Agent 9 产出) ===
    review_score: Optional[int]          # 总分 0-100
    review_detail: Optional[ReviewDetail]
    review_round: int                    # 修正轮次 (0-2)

    # === 语音合成 (Voice Agent 产出) ===
    audio_files: Optional[dict]          # {lesson_id: mp3_file_path}
    voice_progress: Optional[dict]       # {total, completed, current_lesson}
    tts_mode: Optional[str]              # "minimax_hd" | "minimax_cloned" | "minimax_preset" | "edge_tts" | "none"

    # === 数字人视频 (Digital Human Agent 产出) ===
    digital_human_videos: Optional[dict]     # {lesson_id: mp4_file_path}
    digital_human_progress: Optional[dict]   # {total, completed, current_lesson}
    digital_human_mode: Optional[str]        # "duix_avatar" | "disabled" | "skipped_no_audio"

    # === 流程控制 ===
    hitl_status: dict                    # {HITL_ID: "pending"|"confirmed"|"skipped"|"regenerating"}
    skip_all_hitl: bool                  # 一键跳过全部
    errors: Annotated[list, operator.add]  # 错误日志
    node_history: Annotated[list, operator.add]  # 执行历史

    # === 对话消息 ===
    messages: Annotated[list, operator.add]  # 对话历史


# ============================================================
# 节点名称常量
# ============================================================

NODE_REQUIREMENT_ANALYSIS = "requirement_analysis"
NODE_IP_POSITIONING = "ip_positioning"
NODE_COURSE_ARCHITECTURE = "course_architecture"
NODE_CONTENT_PARALLEL = "content_production_parallel"
NODE_CONTENT_SERIAL = "content_production_serial"
NODE_VOICE_TTS = "voice_tts"
NODE_DIGITAL_HUMAN = "digital_human"
NODE_QUALITY_REVIEW = "quality_review"
NODE_PACKAGING = "packaging"

ALL_NODES = [
    NODE_REQUIREMENT_ANALYSIS,
    NODE_IP_POSITIONING,
    NODE_COURSE_ARCHITECTURE,
    NODE_CONTENT_PARALLEL,
    NODE_CONTENT_SERIAL,
    NODE_QUALITY_REVIEW,
    NODE_PACKAGING,
]

# HITL 确认点定义
HITL_DEFINITIONS = {
    "HITL-1": {"node": NODE_REQUIREMENT_ANALYSIS, "label": "需求解析确认", "order": 1},
    "HITL-2": {"node": NODE_IP_POSITIONING, "label": "IP 定位确认", "order": 2},
    "HITL-3": {"node": NODE_COURSE_ARCHITECTURE, "label": "课程大纲确认", "order": 3},
    "HITL-4": {"node": NODE_CONTENT_SERIAL, "label": "内容预览确认", "order": 4},
    "HITL-7": {"node": NODE_QUALITY_REVIEW, "label": "审核报告确认", "order": 7},
}
