# tests/unit/search/test_service.py

"""Tests for SearchService with NewsFilter integration."""

from unittest.mock import MagicMock, patch

import pytest

from ashare_analyzer.models import SearchResponse, SearchResult


@pytest.fixture
def mock_provider():
    """Create a mock search provider."""
    provider = MagicMock()
    provider.is_available = True
    provider.name = "mock"
    provider.search.return_value = SearchResponse(
        query="test",
        results=[
            SearchResult(
                title="Test News",
                snippet="Test snippet",
                url="https://example.com",
                source="example.com",
                published_date="2026-02-22",
            )
        ],
        provider="mock",
        success=True,
    )
    return provider


@pytest.fixture
def mock_akshare_provider():
    """Create a mock Akshare provider."""
    provider = MagicMock()
    provider.is_available = True
    provider.name = "AkshareNews"
    provider.search.return_value = SearchResponse(
        query="600519",
        results=[
            SearchResult(
                title="Akshare News",
                snippet="Akshare snippet",
                url="https://example.com/akshare",
                source="东方财富",
                published_date="2026-02-22",
            )
        ],
        provider="AkshareNews",
        success=True,
    )
    return provider


@pytest.fixture
def mock_config():
    """Create a mock config with news filter disabled."""
    config = MagicMock()
    config.news_filter.news_filter_enabled = False
    return config


def test_is_ashare_detection():
    """Test A-share stock code detection."""
    from ashare_analyzer.search.service import SearchService

    assert SearchService._is_ashare("600519") is True  # Shanghai main board
    assert SearchService._is_ashare("000001") is True  # Shenzhen main board
    assert SearchService._is_ashare("300750") is True  # ChiNext
    assert SearchService._is_ashare("688981") is True  # STAR Market
    assert SearchService._is_ashare("AAPL") is False  # US stock
    assert SearchService._is_ashare("00700") is False  # HK stock (5-digit, not A-share pattern)
    assert SearchService._is_ashare("hk00700") is False  # HK stock


def test_should_skip_filter():
    """Test that akshare provider should skip filter."""
    from ashare_analyzer.search.service import SearchService

    assert SearchService._should_skip_filter("AkshareNews") is True
    assert SearchService._should_skip_filter("akshare") is True
    assert SearchService._should_skip_filter("Akshare") is True
    assert SearchService._should_skip_filter("Tavily") is False
    assert SearchService._should_skip_filter("Brave") is False


def test_search_stock_news_uses_akshare_for_ashare(mock_akshare_provider, mock_config):
    """Test that A-shares use Akshare provider directly."""
    from ashare_analyzer.search.service import SearchService

    service = SearchService()
    service._providers = [mock_akshare_provider]
    service._news_filter = None

    with patch("ashare_analyzer.search.service.get_config", return_value=mock_config):
        response = service.search_stock_news(
            stock_code="600519",
            stock_name="贵州茅台",
            use_cache=False,
        )

    assert response.success
    assert response.provider == "AkshareNews"
    assert len(response.results) == 1
    assert response.results[0].title == "Akshare News"
    # Akshare provider should be called with stock code directly
    mock_akshare_provider.search.assert_called_once()


def test_search_stock_news_uses_other_provider_for_foreign_stocks(mock_provider, mock_config):
    """Test that foreign stocks use other providers, not Akshare."""
    from ashare_analyzer.search.service import SearchService

    mock_akshare = MagicMock()
    mock_akshare.is_available = True
    mock_akshare.name = "AkshareNews"
    # Akshare should fail for foreign stocks (not found)
    mock_akshare.search.return_value = SearchResponse(
        query="AAPL",
        results=[],
        provider="AkshareNews",
        success=False,
        error_message="No news found",
    )

    service = SearchService()
    service._providers = [mock_akshare, mock_provider]
    service._news_filter = None

    with patch("ashare_analyzer.search.service.get_config", return_value=mock_config):
        response = service.search_stock_news(
            stock_code="AAPL",
            stock_name="Apple",
            use_cache=False,
        )

    assert response.success
    # For foreign stocks, akshare might be tried first but then falls back to other providers
    assert response.provider == "mock"


def test_search_stock_news_skips_filter_for_akshare(mock_akshare_provider):
    """Test that filter is skipped for Akshare provider."""
    from ashare_analyzer.search.service import SearchService

    # Create mock filter
    mock_filter = MagicMock()
    mock_filter.filter.return_value = [
        SearchResult(
            title="Filtered News",
            snippet="Filtered snippet",
            url="https://filtered.example.com",
            source="filtered.example.com",
            published_date="2026-02-22",
        )
    ]

    service = SearchService()
    service._providers = [mock_akshare_provider]
    service._news_filter = mock_filter

    response = service.search_stock_news(
        stock_code="600519",
        stock_name="贵州茅台",
        use_cache=False,
    )

    assert response.success
    # Should NOT be filtered since Akshare returns stock-specific news
    assert response.results[0].title == "Akshare News"
    mock_filter.filter.assert_not_called()


def test_search_stock_news_with_filter_disabled(mock_provider, mock_config):
    """Test that search_stock_news calls filter when enabled."""
    from ashare_analyzer.search.service import SearchService

    # Create service with filter disabled
    service = SearchService()
    service._providers = [mock_provider]

    with patch("ashare_analyzer.search.service.get_config", return_value=mock_config):
        response = service.search_stock_news(
            stock_code="600519",
            stock_name="贵州茅台",
            use_cache=False,
        )

    assert response.success
    assert len(response.results) == 1
    assert response.results[0].title == "Test News"


def test_search_stock_news_with_filter_enabled(mock_provider):
    """Test that filter is applied when enabled."""
    from ashare_analyzer.search.service import SearchService

    # Create mock config with filter enabled
    mock_config = MagicMock()
    mock_config.news_filter.news_filter_enabled = True
    mock_config.news_filter.news_filter_min_results = 1

    # Create mock filter
    mock_filter = MagicMock()
    mock_filter.filter.return_value = [
        SearchResult(
            title="Filtered News",
            snippet="Filtered snippet",
            url="https://filtered.example.com",
            source="filtered.example.com",
            published_date="2026-02-22",
        )
    ]

    service = SearchService()
    service._providers = [mock_provider]
    service._news_filter = mock_filter

    response = service.search_stock_news(
        stock_code="600519",
        stock_name="贵州茅台",
        use_cache=False,
    )

    assert response.success
    assert len(response.results) == 1
    assert response.results[0].title == "Filtered News"
    # Verify filter was called with correct arguments
    mock_filter.filter.assert_called_once_with(
        results=mock_provider.search.return_value.results,
        stock_code="600519",
        stock_name="贵州茅台",
    )


def test_search_stock_news_filter_failure_returns_original_results(mock_provider):
    """Test that filter failure returns original results."""
    from ashare_analyzer.search.service import SearchService

    # Create mock config with filter enabled
    mock_config = MagicMock()
    mock_config.news_filter.news_filter_enabled = True

    # Create mock filter that raises exception
    mock_filter = MagicMock()
    mock_filter.filter.side_effect = Exception("Filter error")

    service = SearchService()
    service._providers = [mock_provider]
    service._news_filter = mock_filter

    response = service.search_stock_news(
        stock_code="600519",
        stock_name="贵州茅台",
        use_cache=False,
    )

    # Should still return original results despite filter failure
    assert response.success
    assert len(response.results) == 1
    assert response.results[0].title == "Test News"


def test_search_stock_news_no_filter_instance(mock_provider, mock_config):
    """Test that search works when filter is None."""
    from ashare_analyzer.search.service import SearchService

    service = SearchService()
    service._providers = [mock_provider]
    service._news_filter = None

    with patch("ashare_analyzer.search.service.get_config", return_value=mock_config):
        response = service.search_stock_news(
            stock_code="600519",
            stock_name="贵州茅台",
            use_cache=False,
        )

    assert response.success
    assert len(response.results) == 1


def test_init_news_filter_creates_filter_when_enabled():
    """Test that _init_news_filter creates filter when enabled."""
    from ashare_analyzer.search.service import SearchService

    mock_config = MagicMock()
    mock_config.news_filter.news_filter_enabled = True

    mock_llm_client = MagicMock()

    with patch("ashare_analyzer.search.service.get_config", return_value=mock_config):
        service = SearchService()
        service._init_news_filter(mock_llm_client)

    assert service._news_filter is not None


def test_init_news_filter_skips_when_disabled():
    """Test that _init_news_filter skips when disabled."""
    from ashare_analyzer.search.service import SearchService

    mock_config = MagicMock()
    mock_config.news_filter.news_filter_enabled = False

    with patch("ashare_analyzer.search.service.get_config", return_value=mock_config):
        service = SearchService()
        service._init_news_filter()

    assert service._news_filter is None


def test_search_stock_news_with_focus_keywords_uses_search(mock_provider, mock_config):
    """Test that focus_keywords bypasses Akshare and uses regular search."""
    from ashare_analyzer.search.service import SearchService

    mock_akshare = MagicMock()
    mock_akshare.is_available = True
    mock_akshare.name = "AkshareNews"
    # Akshare fails for keyword search
    mock_akshare.search.return_value = SearchResponse(
        query="业绩 利好",
        results=[],
        provider="AkshareNews",
        success=False,
        error_message="Not a stock code",
    )

    service = SearchService()
    service._providers = [mock_akshare, mock_provider]
    service._news_filter = None

    with patch("ashare_analyzer.search.service.get_config", return_value=mock_config):
        response = service.search_stock_news(
            stock_code="600519",
            stock_name="贵州茅台",
            focus_keywords=["业绩", "利好"],
            use_cache=False,
        )

    assert response.success
    # The response comes from the regular search provider (mock)
    assert response.provider == "mock"
