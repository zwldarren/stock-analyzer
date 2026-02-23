# tests/unit/search/test_filter.py

import asyncio

import pytest

from ashare_analyzer.models import SearchResult


@pytest.fixture
def sample_results() -> list[SearchResult]:
    """Sample search results for testing."""
    return [
        SearchResult(
            title="贵州茅台股价创新高",
            snippet="贵州茅台股价今日创新高，分析师看好后市...",
            url="https://example.com/1",
            source="finance.example.com",
            published_date="2026-02-22",
        ),
        SearchResult(
            title="茅台酒价格走势分析",
            snippet="茅台酒零售价格走势，与股价相关性分析...",
            url="https://example.com/2",
            source="news.example.com",
            published_date="2026-02-20",
        ),
    ]


def test_filter_disabled_returns_original(sample_results):
    """Test that disabled filter returns original results."""
    from ashare_analyzer.config import NewsFilterConfig
    from ashare_analyzer.search.filter import NewsFilter

    config = NewsFilterConfig(news_filter_enabled=False)
    news_filter = NewsFilter(config=config)

    result = news_filter.filter(
        results=sample_results,
        stock_code="600519",
        stock_name="贵州茅台",
    )

    assert result == sample_results
    assert len(result) == 2


def test_filter_empty_input():
    """Test that empty input returns empty list."""
    from ashare_analyzer.config import NewsFilterConfig
    from ashare_analyzer.search.filter import NewsFilter

    config = NewsFilterConfig(news_filter_enabled=True)
    news_filter = NewsFilter(config=config)

    result = news_filter.filter(
        results=[],
        stock_code="600519",
        stock_name="贵州茅台",
    )

    assert result == []


def test_filter_min_results_threshold(sample_results):
    """Test that filter returns original if filtered count below threshold."""
    from ashare_analyzer.config import NewsFilterConfig
    from ashare_analyzer.search.filter import NewsFilter

    config = NewsFilterConfig(
        news_filter_enabled=True,
        news_filter_min_results=10,  # Higher than sample size
    )
    news_filter = NewsFilter(config=config)

    result = news_filter.filter(
        results=sample_results,
        stock_code="600519",
        stock_name="贵州茅台",
    )

    # Should return original because filtered count < min_results
    assert result == sample_results


def test_filter_llm_failure_fallback(sample_results):
    """Test that filter returns original on LLM failure."""
    from ashare_analyzer.config import NewsFilterConfig
    from ashare_analyzer.search.filter import NewsFilter

    config = NewsFilterConfig(news_filter_enabled=True)
    news_filter = NewsFilter(config=config)  # No LLM client

    result = news_filter.filter(
        results=sample_results,
        stock_code="600519",
        stock_name="贵州茅台",
    )

    # Should return original because no LLM client
    assert result == sample_results


class TestRunAsync:
    """Tests for _run_async helper function."""

    def test_run_async_in_sync_context(self):
        """Test _run_async works in sync context."""
        from ashare_analyzer.search.filter import _run_async

        async def async_func():
            await asyncio.sleep(0.01)
            return "success"

        result = _run_async(async_func())
        assert result == "success"

    @pytest.mark.asyncio
    async def test_run_async_in_async_context(self):
        """Test _run_async works when called from async context."""
        from ashare_analyzer.search.filter import _run_async

        async def async_func():
            await asyncio.sleep(0.01)
            return "async_success"

        # This is tricky: when already in async context, _run_async
        # uses ThreadPoolExecutor to run the coroutine
        result = _run_async(async_func())
        assert result == "async_success"
