"""数字人视频生成（F5）子模块

纯云端 API 模式：服务端零 GPU，重计算卸载到 RunningHub 的 digital_customize 工作流，
将「肖像图 + 旁白音频」合成为口播视频。
"""

from .runninghub_client import RunningHubClient

__all__ = ["RunningHubClient"]
