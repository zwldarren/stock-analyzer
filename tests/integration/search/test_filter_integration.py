# tests/integration/search/test_filter_integration.py
"""Integration tests for NewsFilter with mock LLM client."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from ashare_analyzer.config import NewsFilterConfig
from ashare_analyzer.models import SearchResult
from ashare_analyzer.search.filter import NewsFilter


@pytest.fixture
def mock_llm_client():
    """Create mock LLM client that returns filter results."""
    client = MagicMock()

    # Mock generate_with_tool to return filter results
    async def mock_generate_with_tool(*args, **kwargs):
        return {
            "results": [
                {"index": 0, "is_relevant": True, "freshness": "fresh"},
                {"index": 1, "is_relevant": False, "freshness": "acceptable"},
                {"index": 2, "is_relevant": True, "freshness": "stale"},
            ]
        }

    client.generate_with_tool = AsyncMock(side_effect=mock_generate_with_tool)
    client.is_available = MagicMock(return_value=True)
    return client


@pytest.fixture
def sample_news_results() -> list[SearchResult]:
    """Sample news results for integration testing."""
    return [
        SearchResult(
            title="贵州茅台发布年报，净利润增长15%",
            snippet="贵州茅台今日发布年报，净利润同比增长15%，超出市场预期...",
            url="https://finance.example.com/1",
            source="finance.example.com",
            published_date="2026-02-22",
        ),
        SearchResult(
            title="茅台镇旅游攻略",
            snippet="茅台镇是著名的白酒产地，这里介绍旅游攻略...",
            url="https://travel.example.com/2",
            source="travel.example.com",
            published_date="2026-02-21",
        ),
        SearchResult(
            title="白酒行业分析报告",
            snippet="白酒行业2025年分析报告，市场整体走势...",
            url="https://report.example.com/3",
            source="report.example.com",
            published_date="2026-02-15",  # Stale
        ),
    ]


def test_filter_integration_with_mock_llm(mock_llm_client, sample_news_results):
    """Test news filter with mock LLM client."""
    # Use model_construct to bypass validation_alias that reads from env
    # This ensures our test values are used instead of defaults
    config = NewsFilterConfig.model_construct(
        news_filter_enabled=True,
        news_filter_min_results=1,
        news_filter_model=None,
    )

    news_filter = NewsFilter(config=config, llm_client=mock_llm_client)

    filtered = news_filter.filter(
        results=sample_news_results,
        stock_code="600519",
        stock_name="贵州茅台",
    )

    # Should filter out index 1 (not relevant) and index 2 (stale)
    # Only index 0 should remain
    assert len(filtered) == 1
    assert filtered[0].title == "贵州茅台发布年报，净利润增长15%"
