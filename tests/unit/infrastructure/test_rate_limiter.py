import asyncio
import time

import pytest

from stock_analyzer.infrastructure.rate_limiter import AsyncRateLimiter


@pytest.mark.asyncio
async def test_rate_limiter_allows_burst():
    """测试突发请求允许立即通过"""
    limiter = AsyncRateLimiter(rate=1.0, burst=3)

    start = time.monotonic()
    for _ in range(3):
        await limiter.acquire()
    elapsed = time.monotonic() - start

    assert elapsed < 0.1


@pytest.mark.asyncio
async def test_rate_limiter_enforces_rate():
    """测试限流器强制执行速率限制"""
    limiter = AsyncRateLimiter(rate=5.0, burst=1)

    await limiter.acquire()

    start = time.monotonic()
    for _ in range(3):
        await limiter.acquire()
    elapsed = time.monotonic() - start

    assert elapsed >= 0.5
    assert elapsed < 1.0


@pytest.mark.asyncio
async def test_rate_limiter_concurrent_access():
    """测试并发访问时锁正常工作"""
    limiter = AsyncRateLimiter(rate=10.0, burst=1)

    async def make_request():
        await limiter.acquire()
        return time.monotonic()

    start = time.monotonic()
    _ = await asyncio.gather(*[make_request() for _ in range(5)])
    elapsed = time.monotonic() - start

    assert elapsed >= 0.3
