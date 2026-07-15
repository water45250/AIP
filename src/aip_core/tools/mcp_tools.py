"""MCP 工具封装

将 MCP Server 封装为 CrewAI 可用的 Tool。
M2 阶段实现 knowledge-rag 和 web-search。
"""

import json
import time
from typing import Optional

import httpx
from crewai.tools import BaseTool

from ..config import MCP_SERVERS, MCP_TIMEOUT_SECONDS


class MCPToolBase(BaseTool):
    """MCP 工具基类 - 通过 HTTP 调用 MCP Server"""

    mcp_server: str = ""
    tool_name: str = ""
    timeout: int = MCP_TIMEOUT_SECONDS

    def _call_mcp(self, arguments: dict) -> dict:
        """调用 MCP Server"""
        server_url = MCP_SERVERS.get(self.mcp_server, "")
        if not server_url:
            return {"status": "degraded", "error": f"MCP Server '{self.mcp_server}' 未配置", "data": None}

        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(
                    f"{server_url}/tools/{self.tool_name}",
                    json={"arguments": arguments},
                )
                response.raise_for_status()
                return response.json()
        except httpx.TimeoutException:
            return {"status": "degraded", "error": f"MCP Server '{self.mcp_server}' 超时", "data": None}
        except Exception as e:
            return {"status": "degraded", "error": str(e), "data": None}


class KnowledgeSearchTool(MCPToolBase):
    """知识库 RAG 搜索工具

    调用 knowledge-rag MCP Server，检索行业案例和定位方法论。
    """

    name: str = "知识库搜索"
    description: str = "搜索行业知识库，获取同领域成功IP案例和定位方法论。输入搜索查询字符串。"
    mcp_server: str = "knowledge-rag"
    tool_name: str = "search_knowledge"

    def _run(self, query: str, top_k: int = 5) -> str:
        result = self._call_mcp({"query": query, "top_k": top_k})

        if result.get("status") == "degraded":
            return json.dumps({
                "status": "degraded",
                "message": "知识库暂时不可用，使用通用知识",
                "query": query,
            }, ensure_ascii=False)

        return json.dumps(result.get("data", {}), ensure_ascii=False)


class WebSearchTool(MCPToolBase):
    """联网搜索工具

    调用 web-search MCP Server，获取竞品信息和行业动态。
    """

    name: str = "联网搜索"
    description: str = "搜索互联网获取竞品课程信息、行业动态和市场数据。输入搜索查询字符串。"
    mcp_server: str = "web-search"
    tool_name: str = "web_search"

    def _run(self, query: str, num_results: int = 5) -> str:
        result = self._call_mcp({"query": query, "num_results": num_results})

        if result.get("status") == "degraded":
            return json.dumps({
                "status": "degraded",
                "message": "联网搜索暂时不可用",
                "query": query,
            }, ensure_ascii=False)

        return json.dumps(result.get("data", {}), ensure_ascii=False)


class LocalKnowledgeSearchTool(BaseTool):
    """本地知识库搜索工具（降级方案）

    当 MCP knowledge-rag Server 不可用时使用。
    基于预置的定位方法论文档进行搜索。
    """

    name: str = "本地知识库搜索"
    description: str = "搜索本地知识库获取IP定位方法论文档和案例。输入搜索查询字符串。"

    # 预置定位方法论知识库
    _knowledge_base = {
        "ip定位": """
IP定位方法论（四步法）：
1. 定位三角：我是谁（身份） × 帮谁（受众） × 解决什么问题（痛点）
2. 差异化公式：在 [领域] 中，我专注帮 [特定人群] 通过 [独特方法] 实现 [具体结果]
3. 人设锚点：选1-2个核心标签反复强化，如"实战派""结果导向"
4. 信任飞轮：免费内容→低价体验→正价服务→口碑裂变

成功案例：
- 小红书运营教练：定位"帮新手博主3个月涨粉1万"，差异化是"真实数据+可复制方法"
- Python培训师：定位"帮转行人员6个月拿到offer"，差异化是"企业真题+1v1简历辅导"
- 个人品牌顾问：定位"帮专业人士打造知识IP年入百万"，差异化是"定位-内容-变现全链路"
""",
        "课程设计": """
课程设计四阶递进模型：
1. 认知重塑（占20%）：打破认知误区，建立正确框架
2. 方法体系（占40%）：系统化方法论，可操作工具
3. 实战演练（占30%）：真实案例拆解+动手练习
4. 变现闭环（占10%）：商业转化路径

钩子设计原则：
- 开场钩子（第1课）：用反常识观点或痛点共鸣抓住注意力
- 过渡钩子（每模块结尾）：预告下一模块的高价值内容
- 转化钩子（最后一课）：自然引出进阶服务或课程
""",
        "营销合规": """
知识付费营销合规要点：
- 禁止承诺具体收益数字（如"保证月入10万"）
- 禁止使用绝对化用语（如"最好的""唯一的"）
- 必须标注"效果因人而异"
- 案例展示需获得当事人授权
- 价格优惠需明确原价依据
""",
    }

    def _run(self, query: str) -> str:
        # 简单的关键词匹配搜索
        results = []
        query_lower = query.lower()

        for key, content in self._knowledge_base.items():
            if any(kw in query_lower for kw in key.split()):
                results.append(f"## {key}\n{content.strip()}")
            elif any(kw in query_lower for kw in content[:50].lower().split()):
                results.append(f"## {key}\n{content.strip()}")

        if not results:
            # 返回最相关的通用知识
            results.append(self._knowledge_base.get("ip定位", ""))

        return "\n\n---\n\n".join(results)
