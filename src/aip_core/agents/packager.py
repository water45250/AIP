"""打包交付模块 (M5 实现)

将全部课程交付物打包为 ZIP 文件供下载。
目录结构：
  课程包_{标题}.zip
  ├── 01_课程大纲.md
  ├── 02_IP定位报告.md
  ├── 03_讲稿/
  ├── 04_课件数据.json
  ├── 05_案例库.md
  ├── 06_营销物料/
  ├── 07_定价方案.md
  ├── 08_审核报告.md
  ├── 09_音频/
  └── 10_数字人视频/
"""

import io
import json
import time
import zipfile
from pathlib import Path
from typing import Optional

from ..graph.state import CourseState

# 打包输出目录（必须与容器内可写路径一致；/workspace/ 在容器内不存在且无权限创建）
PACKAGE_DIR = Path("/app/data/packages")


def run_packaging(state: CourseState) -> CourseState:
    """打包交付节点 - LangGraph 节点函数"""
    start_time = time.time()
    pkg_ok = False

    try:
        zip_data = _build_package(state)
        # 保存到本地（供 API 下载）
        PACKAGE_DIR.mkdir(parents=True, exist_ok=True)
        session_id = state.get("session_id", "unknown")
        filepath = PACKAGE_DIR / f"course_package_{session_id}.zip"
        filepath.write_bytes(zip_data)

        state["_package_path"] = str(filepath)
        state["_package_size"] = len(zip_data)
        pkg_ok = True
    except Exception as e:
        state["errors"] = state.get("errors", []) + [{
            "node": "packaging",
            "error": f"打包失败: {str(e)}",
            "time": time.time(),
        }]

    state["current_node"] = "packaging"
    state["node_history"] = state.get("node_history", []) + [{
        "node": "packaging",
        "start": start_time,
        "end": time.time(),
        "status": "ok" if pkg_ok else "error",
    }]

    return state


def _build_package(state: CourseState) -> bytes:
    """构建课程 ZIP 包"""
    profile = state.get("user_profile", {})
    ip = state.get("ip_positioning", {})
    outline = state.get("course_outline", {})
    scripts = state.get("scripts", {})
    slides = state.get("slides", {})
    cases = state.get("cases", [])
    marketing = state.get("marketing_copy", {})
    pricing = state.get("pricing_plan", {})
    review = state.get("review_detail", {})

    course_title = outline.get("course_title", "课程")
    # 清理文件名
    safe_title = _safe_filename(course_title)

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        # 01. 课程大纲
        zf.writestr(f"{safe_title}/01_课程大纲.md", _render_outline_md(outline))

        # 02. IP 定位报告
        zf.writestr(f"{safe_title}/02_IP定位报告.md", _render_ip_report_md(profile, ip))

        # 03. 讲稿
        for lid, script in sorted(scripts.items()):
            zf.writestr(f"{safe_title}/03_讲稿/{lid}.md", script)

        # 04. 课件数据
        zf.writestr(f"{safe_title}/04_课件数据.json", json.dumps(slides, ensure_ascii=False, indent=2))

        # 05. 案例库
        zf.writestr(f"{safe_title}/05_案例库.md", _render_cases_md(cases))

        # 06. 营销物料
        zf.writestr(f"{safe_title}/06_营销物料/销售页文案.md", marketing.get("sales_page", ""))
        moments = marketing.get("moments", [])
        zf.writestr(f"{safe_title}/06_营销物料/朋友圈文案.md",
                     "\n\n---\n\n".join(f"**版本{i+1}**\n{m}" for i, m in enumerate(moments)))
        community = marketing.get("community_script", {})
        zf.writestr(f"{safe_title}/06_营销物料/社群话术.md", _render_community_md(community))
        video = marketing.get("video_outlines", [])
        zf.writestr(f"{safe_title}/06_营销物料/短视频概要.md",
                     "\n\n".join(f"## {v.get('duration', '')}\n{v.get('content', '')}" for v in video))

        # 07. 定价方案
        zf.writestr(f"{safe_title}/07_定价方案.md", _render_pricing_md(pricing))

        # 08. 审核报告
        zf.writestr(f"{safe_title}/08_审核报告.md", _render_review_md(review))

        # 09. 音频文件
        audio_files = state.get("audio_files", {})
        if audio_files:
            for lid, mp3_path in sorted(audio_files.items()):
                if Path(mp3_path).exists():
                    zf.write(mp3_path, f"{safe_title}/09_音频/{lid}.mp3")

        # 10. 数字人视频
        video_files = state.get("digital_human_videos", {})
        if video_files:
            for lid, mp4_path in sorted(video_files.items()):
                if Path(mp4_path).exists():
                    zf.write(mp4_path, f"{safe_title}/10_数字人视频/{lid}.mp4")

    return buf.getvalue()


# ============================================================
# 渲染函数
# ============================================================

def _render_outline_md(outline: dict) -> str:
    """渲染课程大纲为 Markdown"""
    lines = [f"# {outline.get('course_title', '课程大纲')}", ""]
    lines.append(f"> 共 {outline.get('total_modules', 0)} 模块 / {outline.get('total_lessons', 0)} 课时")
    lines.append(f"> 总时长约 {outline.get('total_duration_minutes', 0)} 分钟")
    lines.append("")

    for mod in outline.get("modules", []):
        lines.append(f"## {mod.get('id', '')} {mod.get('title', '')}")
        lines.append(f"**阶段**：{mod.get('phase', '')} | **描述**：{mod.get('description', '')}")
        lines.append("")
        for les in mod.get("lessons", []):
            lines.append(f"### {les.get('id', '')} {les.get('title', '')}")
            lines.append(f"- **学习目标**：{les.get('learning_objective', '')}")
            lines.append(f"- **核心要点**：{'、'.join(les.get('key_points', []))}")
            lines.append(f"- **课后任务**：{les.get('homework', '')}")
            lines.append(f"- **预计时长**：{les.get('duration_minutes', 30)} 分钟")
            if les.get("hook_type"):
                lines.append(f"- **钩子类型**：{les['hook_type']}")
            lines.append("")
        lines.append("")

    # 钩子汇总
    hooks = outline.get("hooks", [])
    if hooks:
        lines.append("## 课程钩子设计")
        for h in hooks:
            lines.append(f"- **{h.get('type', '')}** ({h.get('location', '')})：{h.get('content', '')}")

    return "\n".join(lines)


def _render_ip_report_md(profile: dict, ip: dict) -> str:
    """渲染 IP 定位报告为 Markdown"""
    lines = ["# IP 定位报告", ""]

    lines.append("## 用户画像")
    lines.append(f"- 身份：{profile.get('identity', '')}")
    lines.append(f"- 专长：{profile.get('expertise', '')}")
    lines.append(f"- 经验：{profile.get('experience', '')}")
    lines.append(f"- 受众：{profile.get('target_audience', '')}")
    lines.append(f"- 主题：{profile.get('course_topic', '')}")
    lines.append(f"- 风格：{profile.get('style_preference', '')}")
    lines.append("")

    lines.append("## 定位宣言")
    lines.append(f"> {ip.get('positioning_statement', '')}")
    lines.append("")

    lines.append("## 差异化标签")
    for tag in ip.get("differentiation_tags", []):
        lines.append(f"- {tag}")
    lines.append("")

    lines.append("## 信任飞轮")
    for step in ip.get("trust_flywheel", []):
        lines.append(f"{step.get('step', '')}. **{step.get('action', '')}** → {step.get('goal', '')}")
    lines.append("")

    cm = ip.get("content_matrix", {})
    if cm:
        lines.append("## 内容矩阵建议")
        lines.append(f"- 平台：{'、'.join(cm.get('platforms', []))}")
        lines.append(f"- 内容类型：{'、'.join(cm.get('content_types', []))}")
        lines.append(f"- 发布频率：{cm.get('frequency', '')}")

    if ip.get("analysis_notes"):
        lines.append(f"\n## 分析说明\n{ip['analysis_notes']}")

    return "\n".join(lines)


def _render_cases_md(cases: list) -> str:
    """渲染案例库为 Markdown"""
    lines = ["# 课程案例库", ""]
    for i, case in enumerate(cases):
        lines.append(f"## 案例 {i+1}：{case.get('title', '')}")
        lines.append(f"**背景**：{case.get('background', '')}")
        lines.append(f"**挑战**：{case.get('challenge', '')}")
        lines.append(f"**解决方案**：\n{case.get('solution', '')}")
        lines.append(f"**成果**：{case.get('results', '')}")
        lines.append(f"**来源**：{case.get('source', '')}")
        lines.append(f"**课程关联**：{case.get('relevance', '')}")
        lines.append("")
    return "\n".join(lines)


def _render_community_md(community: dict) -> str:
    """渲染社群话术为 Markdown"""
    lines = ["# 社群运营话术", ""]
    labels = {
        "welcome": "入群欢迎",
        "teaser": "价值预告",
        "flash_sale": "限时优惠",
        "close": "成交逼单",
    }
    for key, label in labels.items():
        if key in community:
            lines.append(f"## {label}")
            lines.append(community[key])
            lines.append("")
    return "\n".join(lines)


def _render_pricing_md(pricing: dict) -> str:
    """渲染定价方案为 Markdown"""
    lines = ["# 课程定价方案", ""]
    lines.append(f"**标准定价**：¥{pricing.get('standard_price', '--')}")
    lines.append(f"**早鸟价**：¥{pricing.get('early_bird_price', '--')}")
    lines.append("")

    lines.append("## 阶梯定价")
    for tier in pricing.get("tiered_pricing", []):
        lines.append(f"### {tier.get('tier', '')} — ¥{tier.get('price', '--')}")
        for item in tier.get("includes", []):
            lines.append(f"- {item}")
        lines.append("")

    lines.append(f"## 定价依据\n{pricing.get('rationale', '')}")
    return "\n".join(lines)


def _render_review_md(review: dict) -> str:
    """渲染审核报告为 Markdown"""
    lines = ["# 课程质量审核报告", ""]
    lines.append(f"**总分**：{review.get('total_score', '--')} / 100")
    lines.append(f"**审核结论**：{'✅ 通过' if review.get('pass_') else '⚠️ 需修正'}")
    lines.append("")

    lines.append("## 各维度评分")
    dimensions = review.get("dimensions", {})
    dim_labels = {
        "ip_consistency": "IP 一致性",
        "content_depth": "内容深度",
        "structure_logic": "结构逻辑",
        "marketing_compliance": "营销合规",
        "user_experience": "用户体验",
    }
    for key, label in dim_labels.items():
        dim = dimensions.get(key, {})
        score = dim.get("score", 0)
        max_s = dim.get("max", 0)
        issues = dim.get("issues", [])
        bar = "█" * int(score / max_s * 20) if max_s > 0 else ""
        lines.append(f"### {label}：{score}/{max_s} {bar}")
        for issue in issues:
            lines.append(f"- ⚠️ {issue}")
        lines.append("")

    if review.get("summary"):
        lines.append(f"## 总体评价\n{review['summary']}")

    suggestions = review.get("improvement_suggestions", [])
    if suggestions:
        lines.append("\n## 改进建议")
        for s in suggestions:
            lines.append(f"- {s}")

    return "\n".join(lines)


def _safe_filename(name: str) -> str:
    """将课程标题转换为安全的文件名"""
    import re
    # 移除或替换非法字符
    safe = re.sub(r'[<>:"/\\|?*]', '', name)
    safe = safe.replace(' ', '_')
    # 限制长度
    if len(safe) > 50:
        safe = safe[:50]
    return safe or "课程包"
