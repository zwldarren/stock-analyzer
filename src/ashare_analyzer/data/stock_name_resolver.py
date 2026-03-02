"""Stock name resolution service with TTL caching.

Provides unified stock name lookup logic, integrating multiple data sources:
1. Analysis context (priority)
2. Real-time quote data
3. Dynamic data sources
"""

import logging
from typing import TYPE_CHECKING, Any

from cachetools import TTLCache

if TYPE_CHECKING:
    from ashare_analyzer.data.manager import DataManager

logger = logging.getLogger(__name__)

STOCK_NAME_CACHE_TTL = 86400

# Fallback sources for stock name resolution (in priority order)
FALLBACK_REALTIME_SOURCES = ["tencent", "akshare_em", "akshare_sina"]


class StockNameResolver:
    """Stock name resolver with TTL caching.

    Unified stock name lookup logic supporting multiple data sources and caching.
    """

    def __init__(self, data_manager: "DataManager | None" = None):
        """Initialize resolver.

        Args:
            data_manager: Data manager (optional, for dynamic queries)
        """
        self._data_manager = data_manager
        self._cache: TTLCache[str, str] = TTLCache(maxsize=5000, ttl=STOCK_NAME_CACHE_TTL)

    @classmethod
    def from_context(
        cls,
        stock_code: str,
        context: dict[str, Any] | None = None,
    ) -> str:
        """Quickly resolve stock name from context (class method, no instantiation needed).

        Lookup priority:
        1. stock_name field in context
        2. realtime.name field in context
        3. Default name (股票{code})

        Args:
            stock_code: Stock code
            context: Analysis context (optional)

        Returns:
            Chinese stock name
        """
        # 1. Get from context directly
        if context:
            # Priority: stock_name field
            name = context.get("stock_name")
            if name and not name.startswith("股票") and not name.startswith("Stock"):
                return name

            # Second: realtime data
            realtime = context.get("realtime")
            if isinstance(realtime, dict):
                name = realtime.get("name")
                if name and not name.startswith("股票"):
                    return name

        # 2. Return default name
        return f"股票{stock_code}"

    async def resolve(
        self,
        stock_code: str,
        context: dict[str, Any] | None = None,
        use_cache: bool = True,
    ) -> str:
        """Resolve stock name (full version, async).

        Lookup priority:
        1. Local cache
        2. Context data
        3. Dynamic data sources
        4. Default name

        Args:
            stock_code: Stock code
            context: Analysis context (optional)
            use_cache: Whether to use cache

        Returns:
            Chinese stock name
        """
        # 1. Check local cache
        if use_cache and stock_code in self._cache:
            return self._cache[stock_code]

        # 2. Try to get from context
        name = self.from_context(stock_code, context)
        if name and not name.startswith("股票") and not name.startswith("Stock"):
            self._cache[stock_code] = name
            return name

        # 3. Try data sources if available
        if self._data_manager:
            name = await self._fetch_from_data_source(stock_code)
            if name and not name.startswith("股票"):
                self._cache[stock_code] = name
                return name

        # 4. Return default
        default_name = f"股票{stock_code}"
        self._cache[stock_code] = default_name
        return default_name

    async def _fetch_from_data_source(self, stock_code: str) -> str | None:
        """Fetch stock name from data source (async).

        Tries multiple data sources in sequence:
        1. DataManager's get_stock_name (uses cached realtime quote)
        2. Direct realtime quote from fallback sources (tencent, akshare_em, etc.)
        """
        if self._data_manager is None:
            return None

        # 1. Try DataManager's get_stock_name first
        try:
            name = await self._data_manager.get_stock_name(stock_code)
            if name and not name.startswith("股票"):
                logger.debug(f"[{stock_code}] 从DataManager获取股票名称成功: {name}")
                return str(name)
        except Exception as e:
            logger.debug(f"从DataManager获取股票名称失败: {e}")

        # 2. Try fallback realtime sources directly
        if hasattr(self._data_manager, "_try_realtime_by_source"):
            for source in FALLBACK_REALTIME_SOURCES:
                try:
                    quote = await self._data_manager._try_realtime_by_source(stock_code, source)
                    if quote and hasattr(quote, "name") and quote.name:
                        name = quote.name
                        if name and not name.startswith("股票"):
                            logger.debug(f"[{stock_code}] 从{source}获取股票名称成功: {name}")
                            return str(name)
                except Exception as e:
                    logger.debug(f"[{stock_code}] 从{source}获取股票名称失败: {e}")
                    continue

        return None
