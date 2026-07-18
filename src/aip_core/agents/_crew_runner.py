"""线程池执行 CrewAI kickoff，规避 async 事件循环冲突。

问题背景
--------
aip-core 的 API 路由是 async（FastAPI）。在「运行中的事件循环」线程里同步调用
``crew.kickoff()`` 会触发 CrewAI 报错：

    Agent execution was invoked synchronously from within a running event loop.
    Use `agent.kickoff_async()` / `crew.kickoff_async()` (or ...)

降级方案随后产不出内容，导致 ``review_detail`` 等字段为 ``None``/``{}``，进而打包
阶段 ``None.get()`` 崩溃、ZIP 永不生成（即 v20 修复前「课程包尚未生成」的深层根因）。

修复方式
--------
``run_crew()`` 检测当前是否处于 running loop 线程：
- 若否（纯同步上下文），直接 ``crew.kickoff()``；
- 若是，将执行提交到「无 running loop 的独立线程」执行（``loop.run_in_executor``），
  CrewAI 在该线程内不再检测到 running loop，可正常同步完成。
这与本仓 ``content_agents.py`` 已验证可行的 ``asyncio.to_thread`` 模式等价。
"""

import asyncio


def run_crew(crew):
    """同步执行 ``crew.kickoff()``，规避事件循环冲突。返回 CrewOutput。"""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        # 当前线程无 running loop：直接同步执行
        return crew.kickoff()
    # 当前位于 running loop 线程：提交到独立线程执行（规避 CrewAI 的 loop 检测）
    return loop.run_in_executor(None, crew.kickoff).result()
