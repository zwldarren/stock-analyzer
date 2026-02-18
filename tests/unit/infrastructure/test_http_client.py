import pytest

from stock_analyzer.infrastructure.http_client import (
    aiohttp_session_manager,
    get_aiohttp_session,
)


@pytest.mark.asyncio
async def test_session_manager_creates_session():
    """测试 session manager 创建有效 session"""
    async with aiohttp_session_manager() as session:
        assert session is not None
        assert not session.closed


@pytest.mark.asyncio
async def test_session_manager_closes_session():
    """测试 session manager 退出时关闭 session"""
    async with aiohttp_session_manager() as session:
        pass

    assert session.closed


@pytest.mark.asyncio
async def test_get_session_within_context():
    """测试在 context 内可以获取 session"""
    async with aiohttp_session_manager():
        session = get_aiohttp_session()
        assert session is not None
        assert not session.closed


@pytest.mark.asyncio
async def test_get_session_outside_context_raises():
    """测试在 context 外获取 session 抛出异常"""
    with pytest.raises(RuntimeError, match="aiohttp session not initialized"):
        get_aiohttp_session()


@pytest.mark.asyncio
async def test_session_is_singleton():
    """测试多次调用返回同一 session"""
    async with aiohttp_session_manager():
        session1 = get_aiohttp_session()
        session2 = get_aiohttp_session()
        assert session1 is session2
