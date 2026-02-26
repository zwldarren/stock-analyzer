"""
Dependency injection container - Centralized service management.

This module provides a unified dependency injection container using simple
factory functions with caching. All services are created and managed here
to avoid scattered global state.
"""

from __future__ import annotations

from functools import cache
from typing import TYPE_CHECKING

from ashare_analyzer.config import get_config
from ashare_analyzer.infrastructure.rate_limiter import AsyncRateLimiter
from ashare_analyzer.search.base import register_builtin_providers
from ashare_analyzer.storage import DatabaseManager

register_builtin_providers()

if TYPE_CHECKING:
    from ashare_analyzer.ai.analyzer import AIAnalyzer
    from ashare_analyzer.data.manager import DataManager
    from ashare_analyzer.search.service import SearchService


_akshare_limiter = AsyncRateLimiter(rate=2.0, burst=3)
_tushare_limiter = AsyncRateLimiter(rate=0.5, burst=1)
_efinance_limiter = AsyncRateLimiter(rate=5.0, burst=10)
_baostock_limiter = AsyncRateLimiter(rate=1.0, burst=2)
_yfinance_limiter = AsyncRateLimiter(rate=1.0, burst=2)


@cache
def get_db() -> DatabaseManager:
    """
    Get database manager instance (singleton).

    Returns:
        DatabaseManager singleton instance
    """
    return DatabaseManager.get_instance()


@cache
def get_data_manager() -> DataManager:
    """
    Get data manager instance (singleton).

    Data manager provides access to stock data from multiple sources
    with caching and fallback mechanisms.

    Returns:
        DataManager singleton instance
    """
    from ashare_analyzer.data import DataManager

    config = get_config()
    rate_limiters = {
        "akshare": _akshare_limiter,
        "tushare": _tushare_limiter,
        "efinance": _efinance_limiter,
        "baostock": _baostock_limiter,
        "yfinance": _yfinance_limiter,
    }
    return DataManager(config=config, storage=get_db(), rate_limiters=rate_limiters)


@cache
def get_ai_analyzer() -> AIAnalyzer:
    """
    Get AI analyzer instance (singleton).

    AI analyzer performs stock analysis using LLM models.

    Returns:
        AIAnalyzer singleton instance
    """
    from ashare_analyzer.ai.analyzer import AIAnalyzer

    return AIAnalyzer()


@cache
def get_search_service() -> SearchService:
    """
    Get search service instance (singleton).

    Search service provides web search capabilities for news
    and market intelligence with database caching.

    Returns:
        SearchService singleton instance
    """
    from ashare_analyzer.ai.clients import get_filter_llm_client
    from ashare_analyzer.search import SearchService

    config = get_config()
    service = SearchService(
        bocha_keys=config.search.bocha_api_keys,
        tavily_keys=config.search.tavily_api_keys,
        brave_keys=config.search.brave_api_keys,
        serpapi_keys=config.search.serpapi_keys,
        searxng_base_url=config.search.searxng_base_url,
        searxng_username=config.search.searxng_username,
        searxng_password=config.search.searxng_password,
        llm_client=get_filter_llm_client(),
    )
    service.set_db(get_db())
    return service
