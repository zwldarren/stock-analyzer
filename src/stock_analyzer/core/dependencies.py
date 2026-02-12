"""
Dependency injection container - Centralized service management.

This module provides a unified dependency injection container using simple
factory functions with caching. All services are created and managed here
to avoid scattered global state.
"""

from __future__ import annotations

from functools import cache
from typing import TYPE_CHECKING, Any

from stock_analyzer.config import get_config
from stock_analyzer.infrastructure.persistence.database import DatabaseManager

if TYPE_CHECKING:
    from stock_analyzer.ai.analyzer import AIAnalyzer
    from stock_analyzer.domain.services.data_service import DataService
    from stock_analyzer.infrastructure.notification.service import NotificationService
    from stock_analyzer.infrastructure.search.service import SearchService


@cache
def get_db() -> DatabaseManager:
    """
    Get database manager instance (singleton).

    Returns:
        DatabaseManager singleton instance
    """
    return DatabaseManager.get_instance()


@cache
def get_data_service() -> DataService:
    """
    Get data service instance (singleton).

    Data service provides access to stock data from multiple sources
    with caching and fallback mechanisms.

    Returns:
        DataService singleton instance
    """
    from stock_analyzer.domain.services.data_service import DataService
    from stock_analyzer.infrastructure.data_sources.base import (
        DataFetcherManager,
    )

    config = get_config()
    fetcher_manager = DataFetcherManager()

    return DataService(
        stock_repo=get_db(),
        fetcher_manager=fetcher_manager,
        config=config,
    )


@cache
def get_ai_analyzer() -> AIAnalyzer:
    """
    Get AI analyzer instance (singleton).

    AI analyzer performs stock analysis using LLM models.

    Returns:
        AIAnalyzer singleton instance
    """
    from stock_analyzer.ai.analyzer import AIAnalyzer

    return AIAnalyzer()


@cache
def get_search_service() -> SearchService:
    """
    Get search service instance (singleton).

    Search service provides web search capabilities for news
    and market intelligence.

    Returns:
        SearchService singleton instance
    """
    from stock_analyzer.infrastructure.search.service import SearchService

    config = get_config()
    return SearchService(
        bocha_keys=config.search.bocha_api_keys,
        tavily_keys=config.search.tavily_api_keys,
        brave_keys=config.search.brave_api_keys,
        serpapi_keys=config.search.serpapi_keys,
        searxng_base_url=config.search.searxng_base_url,
        searxng_username=config.search.searxng_username,
        searxng_password=config.search.searxng_password,
    )


def get_notification_service(context: Any | None = None) -> NotificationService:
    """
    Get notification service instance (factory).

    Creates a new instance each time to support different contexts.
    Notification service sends alerts through multiple channels.

    Args:
        context: Optional message context for the notification

    Returns:
        NotificationService instance
    """
    from stock_analyzer.infrastructure.notification.service import NotificationService

    return NotificationService(context=context)
