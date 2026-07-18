"""需求解析 Agent - 课程需求分析师

M1 核心 Agent，负责：
1. 解析用户自由文本输入
2. 提取结构化用户画像
3. 判断完整度，触发追问或进入下一节点
"""

import json
import re
import os
from typing import Optional

from ..graph.state import CourseState, UserProfile
from ..config import REQUIREMENT_MIN_COMPLETENESS, MAX_FOLLOWUP_ROUNDS


# ============================================================
# LLM 结构化提取（DeepSeek）—— 需求解析主路径
# ============================================================
# 需求分析阶段必须用 LLM 真正理解用户自然语言的复杂输入（如
# "针对东南亚务工人员开始讲识ESG的系列课程"），脆弱正则无法可靠处理。
# 正则 _parse_user_input 仅作为 LLM 不可用时的兜底。
try:
    from openai import OpenAI
    _OPENAI_OK = True
except Exception:
    OpenAI = None
    _OPENAI_OK = False

_LLM_PROFILE_FIELDS = [
    "identity", "expertise", "experience", "target_audience",
    "course_topic", "delivery_format", "style_preference",
]

_EXTRACT_SYSTEM = """你是一位经验严谨的课程需求分析师。请从用户的自由文本中提取结构化信息，合并到「已有画像」中，输出 JSON。

字段（未提及或无法判断则设 null）：
- identity: 用户身份，枚举之一或自填——["知识博主","独立讲师","咨询顾问","企业培训师","其他"]
- expertise: 核心专长/知识领域（如 "ESG"、"小红书运营"、"Python编程"）
- experience: 从业/实践年限与背景（如 "5年ESG实战"，无则 null）
- target_audience: 希望服务的学员人群（如 "东南亚务工人员"、"零基础宝妈"），必须是「人/群体」，不要包含课程描述
- course_topic: 简洁纯净的课程主题短语（如 "ESG入门系列课程"、"小红书涨粉实操课"）。
  规则：① 必须是纯净主题，剔除 "讲/认识/了解/开始/做" 等动词；
        ② 若用户说 "讲识X的系列课程" 或 "认识X"，主题是 "X系列课程"；
        ③ 不要以 "系列课程/课程" 开头。
- delivery_format: 枚举之一或自填——["录播视频","直播课","训练营","图文专栏","混合"]，无则 null
- style_preference: 枚举之一或自填——["实战干货","专业严谨","轻松幽默","故事驱动"]，无则 null
- completeness: 0-100 整数，基于已提取的非 null 字段数估计（共 7 个字段，每约 14 分）

只输出 JSON，不要其他文字。示例：{"identity":"咨询顾问","expertise":"ESG","target_audience":"东南亚务工人员","course_topic":"ESG入门系列课程","experience":null,"delivery_format":null,"style_preference":null,"completeness":57}"""


def _call_deepseek(messages: list, temperature: float = 0.3, max_tokens: int = 700) -> Optional[str]:
    """调用 DeepSeek 返回文本；失败/超时返回 None（由调用方回退正则）。"""
    if not _OPENAI_OK:
        return None
    try:
        client = OpenAI(
            api_key=os.getenv("DEEPSEEK_API_KEY", ""),
            base_url="https://api.deepseek.com",
        )
        resp = client.chat.completions.create(
            model=os.getenv("LLM_MODEL", "deepseek-chat"),
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=25,
        )
        return resp.choices[0].message.content or ""
    except Exception:
        return None


def _clean_llm_topic(t: Optional[str]) -> Optional[str]:
    """清洗 LLM 输出的课程主题：剔除动词冗余、规整为纯净主题。"""
    if not t:
        return None
    t = t.strip("的 之 ，,。；;、（）() ")
    # 剔除开头动词/冗余词
    for _v in ("讲识", "讲", "认识", "了解", "开始", "做", "讲讲", "打算",
               "希望", "需要", "准备", "搞", "弄", "关于", "系列课程", "课程"):
        if t.startswith(_v):
            t = t[len(_v):].strip("的 之 ，,。；;、（）() ")
    if not t:
        return None
    # "识ESG"/"讲ESG" 等残留规整
    t = t.replace("识ESG", "ESG").replace("讲ESG", "ESG")
    t = t.strip("的 之 ，,。；;、（）() ")
    if not t or len(t) < 2:
        return None
    # 过短（如 "ESG"）补成 "系列课程" 兜底
    if len(t) <= 4 and "课程" not in t and "课" not in t:
        t = t + "系列课程"
    if t in ("一个", "个", "这门", "那个", "这个", "什么", "相关", "方面", "方向", "领域", "赛道", "内容", "账号"):
        return None
    return t


def _llm_extract_profile(text: str, existing: Optional[dict]) -> Optional[dict]:
    """用 DeepSeek 提取并合并结构化画像；失败返回 None。"""
    if not _OPENAI_OK:
        return None
    existing = existing or {}
    user_msg = (
        f"已有画像：{json.dumps(existing, ensure_ascii=False)}\n"
        f"用户新输入：{text}\n\n"
        "请基于「已有画像」合并更新（已有字段优先保留，除非新输入明确补充/修正），"
        "输出合并后的完整 JSON。"
    )
    content = _call_deepseek([
        {"role": "system", "content": _EXTRACT_SYSTEM},
        {"role": "user", "content": user_msg},
    ])
    if not content:
        return None
    # 容错解析：截取首个 {...} 片段
    try:
        s = content.strip()
        if not s.startswith("{"):
            _m = re.search(r"\{.*\}", s, re.DOTALL)
            if not _m:
                return None
            s = _m.group(0)
        data = json.loads(s)
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    profile = {}
    for f in _LLM_PROFILE_FIELDS:
        v = data.get(f)
        if v is None:
            continue
        v = str(v).strip()
        if not v:
            continue
        if f == "course_topic":
            v = _clean_llm_topic(v)
            if not v:
                continue
        profile[f] = v
    # 若 LLM 未给 course_topic 但给了 expertise，沿用（兜底）
    if "course_topic" not in profile and profile.get("expertise"):
        profile["course_topic"] = _clean_llm_topic(profile["expertise"]) or profile["expertise"]
    return profile or None

# ============================================================
# 结构化提取 Prompt
# ============================================================

EXTRACT_PROFILE_SYSTEM = """你是一位经验严谨的课程需求分析师。你的任务是从用户的自由文本中提取结构化信息。

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
- 只返回 JSON，不要有其他文字

重要规则：
- expertise 和 course_topic 必须区分
- 时间/级别限定词不属于 expertise 也不属于 course_topic 核心词"""

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


def _clean_topic(raw: str) -> Optional[str]:
    """清洗正则提取出的课程主题：去首尾虚词、前置动词，剔除过短/无意义碎片。

    避免「帮我做个课程」被提成「个」、「做副业变现的内容」被提成无意义片段。
    """
    if not raw:
        return None
    _strip = "的之了，,。；;、（）() "
    topic = raw.strip(_strip)
    for _v in ("认识", "了解", "想做", "开始", "打算", "希望", "需要", "准备",
               "做", "讲", "讲讲", "开", "搞", "弄", "关于",
               "始讲", "始做", "始讲讲", "想要做"):  # 正则拆字殘留（如“开始”被拆成“开”+“始”）
        if topic.startswith(_v):
            topic = topic[len(_v):].strip(_strip)
    # 再剝一輪純數字/字母前綴（如“ESG”前的零碎）
    while topic and not topic[0].isalnum() and topic[0] not in "一二三四五六七八九十":
        topic = topic[1:].strip(_strip)
    topic = topic.strip(_strip)
    if not topic or len(topic) < 2:
        return None
    if topic in ("一个", "个", "这门", "那个", "这个", "什么", "相关",
                 "方面", "方向", "领域", "赛道", "内容", "账号"):
        return None
    return topic


def _trad_to_simp(text: str) -> str:
    """繁體→簡體轉換（覆蓋課程工坊常見繁體字，避免逐個正則加變體）。

    不依賴 opencc，用靜態映射表保證零外部依賴、確定性執行。
    """
    _MAP = {
        # 課程相關
        "課程":"课程","課":"课","講識":"讲","講":"讲","識":"识","系列":"系列",
        "培訓":"培训","訓練":"训练","營營":"营营","初級":"初级","高級":"高级",
        "內容":"内容","帶領":"带领","關於":"关于","針對":"针对","面向":"面向",
        "開始":"开始","準備":"准备","想要":"想要","打算":"打算","希望":"希望",
        "需要":"需要","做":"做","講講":"讲讲",
        # 身份相關
        "顧問":"顾问","講師":"讲师","教練":"教练","導師":"导师","博主":"博主",
        "創業":"创业","職場":"职场","職人":"职人","專業":"专业","專長":"专长",
        "經驗":"经验","背景":"背景","身份":"身份",
        # 受眾相關
        "學員":"学员","新手":"新手","小白":"小白","入門":"入门","進階":"进階",
        "媽媽":"妈妈","寶媽":"宝妈","上班族":"上班族","學生":"学生","企業":"企业",
        # 格式
        "錄播":"录播","直播":"直播","圖文":"图文","專欄":"专栏","影片":"视频",
        "線上":"线上","實戰":"实战","實操":"实操","落地":"落地",
        # 風格
        "輕鬆":"轻松","幽默":"幽默","有趣":"有趣","嚴謹":"严谨","系統":"系统",
        # 通用
        "並":"并","與":"与","個":"个","們":"们","為":"为","這":"这",
        "來":"来","時":"时","間":"间","題":"题","項":"项","點":"点",
        "長":"长","專":"专","車":"车","門":"门","頭":"头","見":"见",
        "會":"会","選":"选","標":"标","記":"记","設":"设","認":"认",
        "證":"证","識":"识","達":"达","過":"过","還":"还","進":"进",
        "運":"运","動":"动","區":"区","醫":"医","兩":"两","國":"国",
        "號":"号","質":"质","響":"响","應":"应","義":"义","機":"机",
        "樹":"树","種":"种","樣":"样","氣":"气","壓":"压","廣":"广",
        "條":"条","極":"极","構":"构","務":"务","級":"级","範":"范",
        "網":"网","終":"终","細":"细","組":"组","維":"维","統":"统",
        "業":"业","義":"义","萬":"万","幾":"几","確":"确","雜":"杂",
        "類":"类","離":"离","護":"护","現":"现","對":"对","導":"导",
    }
    # 按长度降序排列（避免短 key 先误替换长 key 的子串）
    for trad, simp in sorted(_MAP.items(), key=lambda x: -len(x[0])):
        text = text.replace(trad, simp)
    return text


def _parse_user_input(user_message: str, existing_profile: Optional[UserProfile] = None) -> UserProfile:
    """
    解析用户输入，提取结构化画像。

    实际生产环境调用 Claude 4 API，这里用规则 + 正则做轻量实现，
    后续替换为 LLM 调用即可。
    """
    profile = dict(existing_profile) if existing_profile else {}

    text = user_message.strip()
    text = _trad_to_simp(text)  # 繁體→簡體：後續所有正則統一生效，不遺漏任何繁體變體

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

    # 4. 提取课程主题 - 宽松匹配自然表达
    #    覆盖：「关于X的课程」「做X的课/培训」「X运营」「X教程」
    #         「X方向/赛道/领域的内容/账号」「做X内容」
    #    关键：用动词就近锚定课程词，避免 (.+?) 从句首吞到句尾；并用 _clean_topic 兜底。
    if "course_topic" not in profile or not profile["course_topic"]:
        topic_patterns = [
            r'关于(.+?)(?:的)?(?:系列)?课程',
            r'(?:认识|了解|做|讲|讲讲|想做|打算|希望|需要|准备|开|搞|弄)(.+?)(?:方向|赛道|领域|方面)?的?(?:系列)?(?:课程|课|培训|训练营|内容|账号|IP|ip)',
            r'做[一个门]*[关于]*(.+?)[的之]*(课程|课|培训|训练营)',
            r'(.+?)运营',
            r'(.+?)教程',
        ]
        _lead_verbs = ("认识", "了解", "想做", "开始", "打算", "希望", "需要",
                       "准备", "做", "讲", "讲讲", "开", "搞", "弄")
        for pattern in topic_patterns:
            match = re.search(pattern, text)
            if match:
                topic = _clean_topic(match.group(1))
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

    # 6. 提取受众 - 支持「针对/面向/服务 + 人群词」「新手/宝妈/职场人」等自然表达
    #    注意：变现/赚钱/商业/盈利 是「目的」而非「受众」，严禁误提；
    #          要求捕获片段长度 >= 2，避免把单字碎片当受众。
    if "target_audience" not in profile or not profile["target_audience"]:
        audience_patterns = [
            r'(?:针对|面向|服务[于]?)(.+?)(?:，|,|。|；|、|$)',
            r'(新手|零基础|小白|入门)',
            r'(进阶|提升|有一定基础)',
            r'(职场人|宝妈|上班族|学生|自由职业|创业者|职场新人|银发族)',
            r'(学员|用户|人群|客户|读者|受众)',
        ]
        for pattern in audience_patterns:
            match = re.search(pattern, text)
            if match:
                _g = match.group(1).strip() if (match.groups() and match.group(1)) else match.group(0).strip()
                if _g and len(_g) >= 2:
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

    # 解析用户输入：优先 LLM(DeepSeek) 提取，失败回退正则
    profile = _llm_extract_profile(last_user_message, existing_profile)
    if not profile:
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
    MIN_ASK_ROUNDS = 2
    need_ask = (completeness < REQUIREMENT_MIN_COMPLETENESS or followup_rounds < MIN_ASK_ROUNDS) and followup_rounds < MAX_FOLLOWUP_ROUNDS

    # 清洗 target_audience 冗余课程修饰词
    if profile.get("target_audience"):
        aud = str(profile["target_audience"]).strip()
        for jk in ("关于","關於","的系列課程","的系列课程","的課程","的课程","做","系列課程","培训","課程"):
            if aud.endswith(jk) and len(aud)>len(jk)+2: aud = aud[:-len(jk)].rstrip("的，, ")
        if len(aud)>=2: profile["target_audience"] = aud

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

    # 诚实文案：只有真正达标才说「信息充足」；信息不足时像真顾问一样
    # 明确指出缺口，把「补充」与「确认继续」的选择权交还给用户，绝不谎称。
    if completeness >= REQUIREMENT_MIN_COMPLETENESS:
        _sm = [v for k,v in {"identity":"身份","expertise":"核心专长","experience":"经验","course_topic":"课程主题","target_audience":"目标学员","delivery_format":"交付形式","style_preference":"风格偏好"}.items() if not profile.get(k)]
        if _sm: parts.append(f"\n⚠️ 仍有 {_sm} 未填（将用通用方案），建议补充后重新分析以获得更好结果。")
        parts.append("\n✅ 可以开始课程生成。请确认以上信息，或选择修改/补充后重新分析。")
    else:
        _miss_names = {
            "identity": "你的身份",
            "expertise": "核心专长",
            "experience": "从业经验",
            "course_topic": "课程主题",
            "target_audience": "目标学员",
            "delivery_format": "交付形式",
            "style_preference": "课程风格",
        }
        _miss = [v for k, v in _miss_names.items() if not profile.get(k)]
        if _miss:
            parts.append("\n⚠️ 目前信息还不足以设计出针对性强的课程，仍建议补充：" + "、".join(_miss) + "。")
            parts.append("你可以继续补充，我再重新确认；或直接确认用现有信息开始（缺口模块会用通用方案兜底）。")
        else:
            parts.append("\n✅ 信息充足，可以开始课程生成。请确认以上信息，或选择修改/跳过。")

    return "\n".join(parts)
