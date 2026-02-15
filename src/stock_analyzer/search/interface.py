"""
Search Service Interface

Defines the abstract interface for search service,
following the Dependency Inversion Principle.
"""

from abc import ABC, abstractmethod
from typing import Any

from stock_analyzer.models import SearchResponse


class ISearchService(ABC):
    """
    Search Service Interface

    Responsible for searching stock-related news, intelligence, and market information.
    Supports multiple search providers (Bocha, Tavily, SerpAPI, etc.).
    """

    @abstractmethod
    def search_comprehensive_intel(
        self, stock_code: str, stock_name: str, max_searches: int = 5, use_cache: bool = True
    ) -> dict[str, Any] | None:
        """
        Comprehensive intelligence search.

        Searches stock-related information from multiple dimensions:
        - Company news and announcements
        - Industry dynamics
        - Market sentiment

        Args:
            stock_code: Stock code
            stock_name: Stock name
            max_searches: Maximum number of searches
            use_cache: Whether to use cache

        Returns:
            Dict with search results or None if failed
        """

    @abstractmethod
    def search(self, query: str, max_results: int = 10) -> SearchResponse:
        """
        Simple search.

        Args:
            query: Search query
            max_results: Maximum number of results

        Returns:
            SearchResponse with results
        """

    @abstractmethod
    def is_available(self) -> bool:
        """
        Check if search service is available.

        Returns:
            bool: Whether at least one search provider is available
        """
