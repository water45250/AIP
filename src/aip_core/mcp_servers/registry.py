"""MCP Server 注册管理中心

统一管理 5 个 MCP Server 的注册、健康检查和降级策略。
提供统一的工具调用接口供所有 Agent 使用。
"""

import time
from typing import Optional, Any
from dataclasses import dataclass, field

import httpx

from ..config import MCP_SERVERS, MCP_TIMEOUT_SECONDS


@dataclass
class MCPServerStatus:
    """MCP Server 状态"""
    name: str
    url: str
    status: str = "unknown"       # "active" | "degraded" | "unavailable"
    last_check: float = 0.0
    error_count: int = 0
    max_errors: int = 3


class MCPRegistry:
    """MCP Server 注册中心（单例）"""

    _instance: Optional["MCPRegistry"] = None

    def __init__(self):
        self._servers: dict[str, MCPServerStatus] = {}
        self._init_servers()

    def _init_servers(self):
        for name, url in MCP_SERVERS.items():
            self._servers[name] = MCPServerStatus(name=name, url=url)

    @classmethod
    def get_instance(cls) -> "MCPRegistry":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def get_server_url(self, name: str) -> Optional[str]:
        """获取 Server URL"""
        server = self._servers.get(name)
        return server.url if server else None

    async def health_check(self, name: str) -> bool:
        """健康检查单个 Server"""
        server = self._servers.get(name)
        if not server:
            return False

        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{server.url}/health")
                if resp.status_code == 200:
                    server.status = "active"
                    server.error_count = 0
                    server.last_check = time.time()
                    return True
        except Exception:
            pass

        server.error_count += 1
        if server.error_count >= server.max_errors:
            server.status = "unavailable"
        else:
            server.status = "degraded"
        server.last_check = time.time()
        return False

    async def health_check_all(self) -> dict[str, bool]:
        """健康检查所有 Server"""
        results = {}
        for name in self._servers:
            results[name] = await self.health_check(name)
        return results

    def get_all_status(self) -> dict:
        """获取所有 Server 状态"""
        return {
            name: {
                "status": s.status,
                "url": s.url,
                "error_count": s.error_count,
                "last_check": s.last_check,
            }
            for name, s in self._servers.items()
        }

    def is_available(self, name: str) -> bool:
        """检查 Server 是否可用"""
        server = self._servers.get(name)
        return server is not None and server.status != "unavailable"

    def call_tool(self, server_name: str, tool_name: str, arguments: dict) -> dict:
        """同步调用 MCP Tool（供 Agent 使用）"""
        server = self._servers.get(server_name)
        if not server or server.status == "unavailable":
            return {
                "status": "degraded",
                "error": f"MCP Server '{server_name}' 不可用",
                "data": None,
            }

        try:
            with httpx.Client(timeout=MCP_TIMEOUT_SECONDS) as client:
                resp = client.post(
                    f"{server.url}/tools/{tool_name}",
                    json={"arguments": arguments},
                )
                resp.raise_for_status()
                server.error_count = 0  # 重置错误计数
                return {"status": "ok", "data": resp.json()}
        except Exception as e:
            server.error_count += 1
            if server.error_count >= server.max_errors:
                server.status = "unavailable"
            return {
                "status": "degraded",
                "error": str(e),
                "data": None,
            }


# 全局单例
mcp_registry = MCPRegistry.get_instance()
