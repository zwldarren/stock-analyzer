# tests/unit/search/test_akshare_provider.py

"""Tests for AkshareNewsProvider."""

from unittest.mock import patch

import pandas as pd
import pytest

from ashare_analyzer.search.impl import AkshareNewsProvider


@pytest.fixture
def akshare_provider():
    """Create an AkshareNewsProvider instance."""
    return AkshareNewsProvider()


def test_provider_is_always_available(akshare_provider):
    """Test that AkshareNewsProvider is always available."""
    assert akshare_provider.is_available is True


def test_provider_skip_filter_flag(akshare_provider):
    """Test that AkshareNewsProvider has skip_filter flag set."""
    assert akshare_provider.skip_filter is True


def test_provider_name(akshare_provider):
    """Test provider name."""
    assert akshare_provider.name == "AkshareNews"


def test_search_success(akshare_provider):
    """Test successful search with mock data."""
    # Create mock DataFrame
    mock_df = pd.DataFrame(
        {
            "关键词": ["600519"] * 3,
            "新闻标题": ["标题1", "标题2", "标题3"],
            "新闻内容": ["内容1", "内容2", "内容3"],
            "发布时间": ["2026-02-22 10:00:00", "2026-02-21 15:30:00", "2026-02-20 09:00:00"],
            "文章来源": ["东方财富", "新浪财经", "同花顺"],
            "新闻链接": [
                "http://example.com/1",
                "http://example.com/2",
                "http://example.com/3",
            ],
        }
    )

    with patch("akshare.stock_news_em", return_value=mock_df):
        response = akshare_provider.search("600519", max_results=10)

    assert response.success is True
    assert response.provider == "AkshareNews"
    assert len(response.results) == 3
    assert response.results[0].title == "标题1"
    assert response.results[0].source == "东方财富"
    assert response.results[0].published_date == "2026-02-22"


def test_search_with_max_results_limit(akshare_provider):
    """Test that max_results limits the number of results."""
    mock_df = pd.DataFrame(
        {
            "关键词": ["600519"] * 10,
            "新闻标题": [f"标题{i}" for i in range(10)],
            "新闻内容": [f"内容{i}" for i in range(10)],
            "发布时间": [f"2026-02-22 10:00:0{i}" for i in range(10)],
            "文章来源": ["东方财富"] * 10,
            "新闻链接": [f"http://example.com/{i}" for i in range(10)],
        }
    )

    with patch("akshare.stock_news_em", return_value=mock_df):
        response = akshare_provider.search("600519", max_results=5)

    assert response.success is True
    assert len(response.results) == 5


def test_search_empty_results(akshare_provider):
    """Test search with empty DataFrame."""
    mock_df = pd.DataFrame()

    with patch("akshare.stock_news_em", return_value=mock_df):
        response = akshare_provider.search("600519", max_results=10)

    assert response.success is True
    assert len(response.results) == 0
    assert "No news found" in response.error_message


def test_search_none_results(akshare_provider):
    """Test search with None returned from akshare."""
    with patch("akshare.stock_news_em", return_value=None):
        response = akshare_provider.search("600519", max_results=10)

    assert response.success is True
    assert len(response.results) == 0
    assert "No news found" in response.error_message


def test_search_handles_exception(akshare_provider):
    """Test that exceptions are handled gracefully."""
    with patch("akshare.stock_news_em", side_effect=Exception("Network error")):
        response = akshare_provider.search("600519", max_results=10)

    assert response.success is False
    assert len(response.results) == 0
    assert "获取新闻失败" in response.error_message


def test_search_handles_missing_columns(akshare_provider):
    """Test that missing columns are handled gracefully."""
    # DataFrame with some missing columns
    mock_df = pd.DataFrame(
        {
            "关键词": ["600519"],
            "新闻标题": ["标题"],
            # Missing: 新闻内容, 发布时间, 文章来源, 新闻链接
        }
    )

    with patch("akshare.stock_news_em", return_value=mock_df):
        response = akshare_provider.search("600519", max_results=10)

    assert response.success is True
    assert len(response.results) == 1
    assert response.results[0].title == "标题"
    assert response.results[0].snippet == ""
    assert response.results[0].url == ""
    assert response.results[0].source == "东方财富"  # Default source


def test_search_truncates_long_snippet(akshare_provider):
    """Test that long snippets are truncated."""
    long_content = "x" * 1000
    mock_df = pd.DataFrame(
        {
            "关键词": ["600519"],
            "新闻标题": ["标题"],
            "新闻内容": [long_content],
            "发布时间": ["2026-02-22 10:00:00"],
            "文章来源": ["东方财富"],
            "新闻链接": ["http://example.com/1"],
        }
    )

    with patch("akshare.stock_news_em", return_value=mock_df):
        response = akshare_provider.search("600519", max_results=10)

    assert response.success is True
    assert len(response.results[0].snippet) == 500
