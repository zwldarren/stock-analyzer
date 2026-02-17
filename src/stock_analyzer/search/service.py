"""
Search service.

Provides a unified search service interface, managing multiple search engines.
"""

import logging
import re
from datetime import datetime
from typing import TYPE_CHECKING

from stock_analyzer.models import SearchResponse, SearchResult
from stock_analyzer.search.base import (
    ApiKeyProviderConfig,
    ProviderRegistry,
    SearxngProviderConfig,
)

if TYPE_CHECKING:
    from stock_analyzer.storage.database import DatabaseManager

logger = logging.getLogger(__name__)


class SearchService:
    """
    Search service.

    Features:
    1. Manages multiple search engines with priority
    2. Automatic failover to next provider
    3. Database caching for news results
    4. Automatic English keywords for HK/US stocks
    """

    # Cache settings
    NEWS_CACHE_DAYS = 1  # Consider news cached if fetched within 1 day

    def __init__(
        self,
        bocha_keys: list[str] | None = None,
        tavily_keys: list[str] | None = None,
        brave_keys: list[str] | None = None,
        serpapi_keys: list[str] | None = None,
        searxng_base_url: str = "",
        searxng_username: str | None = None,
        searxng_password: str | None = None,
        db: "DatabaseManager | None" = None,
    ):
        """
        Initialize the search service.

        Args:
            bocha_keys: List of Bocha search API keys.
            tavily_keys: List of Tavily API keys.
            brave_keys: List of Brave Search API keys.
            serpapi_keys: List of SerpAPI keys.
            searxng_base_url: SearXNG base URL.
            searxng_username: SearXNG Basic Auth username.
            searxng_password: SearXNG Basic Auth password.
            db: DatabaseManager instance for caching news.
        """
        self._providers = []
        self._db = db

        # Use registry to create providers
        # Order: Tavily -> Brave -> SerpAPI -> Bocha -> SearXNG

        # 1. Tavily
        if tavily_keys:
            provider = ProviderRegistry.create_provider("tavily", ApiKeyProviderConfig(api_keys=tavily_keys))
            if provider:
                self._providers.append(provider)
                logger.debug(f"已配置 tavily 搜索，共 {len(tavily_keys)} 个 API Key")

        # 2. Brave Search
        if brave_keys:
            provider = ProviderRegistry.create_provider("brave", ApiKeyProviderConfig(api_keys=brave_keys))
            if provider:
                self._providers.append(provider)
                logger.debug(f"已配置 brave 搜索，共 {len(brave_keys)} 个 API Key")

        # 3. SerpAPI
        if serpapi_keys:
            provider = ProviderRegistry.create_provider("serpapi", ApiKeyProviderConfig(api_keys=serpapi_keys))
            if provider:
                self._providers.append(provider)
                logger.debug(f"已配置 serpapi 搜索，共 {len(serpapi_keys)} 个 API Key")

        # 4. Bocha
        if bocha_keys:
            provider = ProviderRegistry.create_provider("bocha", ApiKeyProviderConfig(api_keys=bocha_keys))
            if provider:
                self._providers.append(provider)
                logger.debug(f"已配置 bocha 搜索，共 {len(bocha_keys)} 个 API Key")

        # 5. SearXNG (lowest priority due to noisy results)
        if searxng_base_url:
            provider = ProviderRegistry.create_provider(
                "searxng",
                SearxngProviderConfig(
                    base_url=searxng_base_url,
                    username=searxng_username,
                    password=searxng_password,
                ),
            )
            if provider:
                self._providers.append(provider)
                logger.debug("已配置 searxng 搜索")

    def set_db(self, db: "DatabaseManager") -> None:
        """Set database manager for caching."""
        self._db = db

    def _get_cached_news(self, stock_code: str, dimension: str = "latest_news") -> list[SearchResult] | None:
        """
        Get cached news from database.

        Args:
            stock_code: Stock code
            dimension: Search dimension (e.g., 'latest_news', 'risk_check')

        Returns:
            List of cached SearchResult if available and fresh, None otherwise
        """
        if not self._db:
            return None

        try:
            # Get recent news from database
            news_list = self._db.get_recent_news(
                code=stock_code,
                days=self.NEWS_CACHE_DAYS,
                limit=30,
            )

            if not news_list:
                return None

            # Filter by dimension if specified
            if dimension:
                news_list = [n for n in news_list if n.dimension == dimension][:10]

            if not news_list:
                return None

            # Convert to SearchResult
            results = []
            for news in news_list:
                results.append(
                    SearchResult(
                        title=str(news.title) if news.title else "",
                        snippet=str(news.snippet) if news.snippet else "",
                        url=str(news.url) if news.url else "",
                        source=str(news.source) if news.source else "",
                        published_date=news.published_date.isoformat() if news.published_date else None,
                    )
                )

            logger.debug(f"[缓存命中] {stock_code} 获取到 {len(results)} 条缓存新闻")
            return results

        except Exception as e:
            logger.warning(f"[缓存读取失败] {stock_code}: {e}")
            return None

    def _save_news_to_cache(
        self,
        stock_code: str,
        stock_name: str,
        dimension: str,
        query: str,
        response: SearchResponse,
    ) -> int:
        """
        Save search results to database cache.

        Args:
            stock_code: Stock code
            stock_name: Stock name
            dimension: Search dimension
            query: Search query
            response: Search response to save

        Returns:
            Number of results saved
        """
        if not self._db or not response.success or not response.results:
            return 0

        try:
            saved = self._db.save_news_intel(
                code=stock_code,
                name=stock_name,
                dimension=dimension,
                query=query,
                response=response,
            )
            if saved > 0:
                logger.debug(f"[缓存保存] {stock_code} 保存了 {saved} 条新闻")
            return saved
        except Exception as e:
            logger.warning(f"[缓存保存失败] {stock_code}: {e}")
            return 0

    @staticmethod
    def _is_foreign_stock(stock_code: str) -> bool:
        """Check if the stock is HK or US stock."""
        code = stock_code.strip()
        # US stocks: 1-5 uppercase letters, may contain dot (e.g., BRK.B)
        if re.match(r"^[A-Za-z]{1,5}(\.[A-Za-z])?$", code):
            return True
        # HK stocks: starts with 'hk' prefix or 5-digit number
        lower = code.lower()
        if lower.startswith("hk"):
            return True
        return bool(code.isdigit() and len(code) == 5)

    @property
    def is_available(self) -> bool:
        """Check if any search engine is available."""
        return any(p.is_available for p in self._providers)

    def search(self, query: str, max_results: int = 10, days: int = 7) -> SearchResponse:
        """
        Simple search with automatic failover.

        Args:
            query: Search query
            max_results: Maximum number of results
            days: Time range in days

        Returns:
            SearchResponse from the first successful provider
        """
        if not self._providers:
            return SearchResponse(
                query=query,
                results=[],
                provider="None",
                success=False,
                error_message="No search providers configured",
            )

        for provider in self._providers:
            if not provider.is_available:
                continue

            response = provider.search(query, max_results, days=days)
            if response.success:
                return response

        return SearchResponse(
            query=query,
            results=[],
            provider="None",
            success=False,
            error_message="All search providers failed",
        )

    def search_stock_news(
        self,
        stock_code: str,
        stock_name: str,
        max_results: int = 10,
        focus_keywords: list[str] | None = None,
        use_cache: bool = True,
    ) -> SearchResponse:
        """
        Search for stock-related news.

        Uses the first successful search provider (no aggregation).

        Args:
            stock_code: Stock code.
            stock_name: Stock name.
            max_results: Maximum number of results to return.
            focus_keywords: List of keywords to focus on.
            use_cache: Whether to use cached results if available.

        Returns:
            SearchResponse object.
        """
        # Try to get from cache first
        if use_cache:
            cached = self._get_cached_news(stock_code, dimension="latest_news")
            if cached:
                return SearchResponse(
                    query=f"{stock_name} {stock_code} (cached)",
                    results=cached[:max_results],
                    provider="cache",
                    success=True,
                )

        # Smart time range determination
        today_weekday = datetime.now().weekday()
        if today_weekday == 0:  # Monday
            search_days = 3
        elif today_weekday >= 5:  # Saturday(5), Sunday(6)
            search_days = 2
        else:  # Tuesday(1) - Friday(4)
            search_days = 1

        # Build search query (select language based on stock type)
        is_foreign = self._is_foreign_stock(stock_code)
        if focus_keywords:
            query = " ".join(focus_keywords)
        elif is_foreign:
            # Use English keywords for HK/US stocks
            query = f"{stock_name} {stock_code} stock latest news"
        else:
            # Use Chinese keywords for A-shares
            query = f"{stock_name} {stock_code} 股票 最新消息"

        logger.debug(f"搜索股票新闻: {stock_name}({stock_code}), query='{query}', 时间范围: 近{search_days}天")

        # Use simple search (first successful provider)
        response = self.search(query, max_results, days=search_days)

        if response.success:
            # Save to cache
            self._save_news_to_cache(
                stock_code=stock_code,
                stock_name=stock_name,
                dimension="latest_news",
                query=query,
                response=response,
            )
            logger.debug(f"使用 {response.provider} 搜索成功，获取 {len(response.results)} 条结果")
        else:
            logger.warning(f"所有搜索引擎都失败: {response.error_message}")

        return response
