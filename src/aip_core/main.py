"""文经客 AIP OPC - 服务入口

启动命令:
    python -m uvicorn src.aip_core.main:app --host 0.0.0.0 --port 8080 --reload
"""

from .api import app

__all__ = ["app"]
