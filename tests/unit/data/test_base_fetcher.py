import time

import pytest

from ashare_analyzer.data.base import BaseFetcher
from ashare_analyzer.infrastructure.rate_limiter import AsyncRateLimiter


class MockFetcher(BaseFetcher):
    """测试用 Mock Fetcher"""

    name = "MockFetcher"
    priority = 1

    async def _fetch_raw_data(self, stock_code: str, start_date: str, end_date: str):
        return None

    def _normalize_data(self, df, stock_code: str):
        return df


@pytest.mark.asyncio
async def test_base_fetcher_with_rate_limiter():
    """测试 BaseFetcher 使用限流器"""
    limiter = AsyncRateLimiter(rate=100.0, burst=10)
    fetcher = MockFetcher(rate_limiter=limiter)

    start = time.monotonic()
    await fetcher._enforce_rate_limit()
    elapsed = time.monotonic() - start

    assert elapsed < 0.1


@pytest.mark.asyncio
async def test_base_fetcher_without_rate_limiter():
    """测试 BaseFetcher 无限流器时正常工作"""
    fetcher = MockFetcher(rate_limiter=None)

    await fetcher._enforce_rate_limit()
