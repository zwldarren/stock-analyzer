"""全局 aiohttp ClientSession 管理"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import aiohttp

_session: aiohttp.ClientSession | None = None


def get_aiohttp_session() -> aiohttp.ClientSession:
    """获取全局 aiohttp session（必须在 aiohttp_session context 内调用）"""
    if _session is None:
        raise RuntimeError("aiohttp session not initialized. Use aiohttp_session_manager().")
    return _session


@asynccontextmanager
async def aiohttp_session_manager() -> AsyncIterator[aiohttp.ClientSession]:
    """应用级 aiohttp session 生命周期管理"""
    global _session

    _session = aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=30),
        connector=aiohttp.TCPConnector(limit=20, limit_per_host=5),
    )

    try:
        yield _session
    finally:
        await _session.close()
        _session = None
