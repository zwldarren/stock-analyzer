"""Stock name resolution service with TTL caching.

Provides unified stock name lookup logic, integrating multiple data sources:
1. Analysis context (priority)
2. Real-time quote data
3. Dynamic data sources
"""

import logging
from typing import TYPE_CHECKING, Any

from cachetools import TTLCache

from stock_analyzer.domain.exceptions import handle_errors

if TYPE_CHECKING:
    from stock_analyzer.infrastructure.data_sources.base import DataFetcherManager

logger = logging.getLogger(__name__)

STOCK_NAME_CACHE_TTL = 86400


class StockNameResolver:
    """Stock name resolver with TTL caching.

    Unified stock name lookup logic supporting multiple data sources and caching.
    """

    def __init__(self, data_manager: "DataFetcherManager | None" = None):
        """Initialize resolver.

        Args:
            data_manager: Data fetcher manager (optional, for dynamic queries)
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
            if name and not name.startswith("股票"):
                return name

            # Second: realtime data
            realtime = context.get("realtime")
            if isinstance(realtime, dict):
                name = realtime.get("name")
                if name:
                    return name

        # 2. Return default name
        return f"股票{stock_code}"

    def resolve(
        self,
        stock_code: str,
        context: dict[str, Any] | None = None,
        use_cache: bool = True,
    ) -> str:
        """Resolve stock name (full version).

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

        # 2. Get from context
        name = self._resolve_from_context(stock_code, context)
        if name and not name.startswith("股票"):
            if use_cache:
                self._cache[stock_code] = name
            return name

        # 3. Get from dynamic data source
        if self._data_manager:
            name = self._resolve_from_data_source(stock_code)
            if name:
                if use_cache:
                    self._cache[stock_code] = name
                return name

        # 4. Return default name
        default_name = f"股票{stock_code}"
        logger.debug(f"无法解析股票名称，使用默认值: {default_name}")
        return default_name

    def _resolve_from_context(
        self,
        stock_code: str,
        context: dict[str, Any] | None,
    ) -> str | None:
        """从上下文解析名称"""
        if not context:
            return None

        # 优先从 stock_name 字段获取
        name = context.get("stock_name")
        if name and not name.startswith("股票"):
            return name

        # 其次从 realtime 数据获取
        realtime = context.get("realtime")
        if isinstance(realtime, dict):
            name = realtime.get("name")
            if name:
                return name

        return None

    @handle_errors(
        "从数据源获取股票名称失败",
        default_return=None,
        log_level="debug",
    )
    def _resolve_from_data_source(self, stock_code: str) -> str | None:
        """从动态数据源解析名称"""
        if not self._data_manager:
            return None

        name = self._data_manager.get_stock_name(stock_code)
        if name:
            logger.debug(f"从数据源获取股票名称: {stock_code} -> {name}")
        return name

    def batch_resolve(
        self,
        stock_codes: list[str],
        use_cache: bool = True,
    ) -> dict[str, str]:
        """Batch resolve stock names.

        Args:
            stock_codes: List of stock codes
            use_cache: Whether to use cache

        Returns:
            Dictionary of {stock_code: stock_name}
        """
        result = {}
        missing_codes = []

        # 1. Get from cache
        if use_cache:
            for code in stock_codes:
                if code in self._cache:
                    result[code] = self._cache[code]
                else:
                    missing_codes.append(code)
        else:
            missing_codes = stock_codes

        # 2. Batch fetch from data source
        if missing_codes and self._data_manager:
            try:
                batch_result = self._data_manager.batch_get_stock_names(missing_codes)
                for code, name in batch_result.items():
                    if name:
                        result[code] = name
                        self._cache[code] = name

                # Record unfound codes
                for code in missing_codes:
                    if code not in result:
                        result[code] = f"股票{code}"
            except Exception as e:
                logger.warning(f"批量获取股票名称失败: {e}")
                for code in missing_codes:
                    result[code] = f"股票{code}"
        elif missing_codes:
            for code in missing_codes:
                result[code] = f"股票{code}"

        return result

    def clear_cache(self) -> None:
        """Clear local cache."""
        self._cache.clear()
        logger.debug("股票名称本地缓存已清空")

    def register(self, code: str, name: str) -> None:
        """Register stock name to local cache.

        Args:
            code: Stock code
            name: Stock name
        """
        self._cache[code] = name
        logger.debug(f"注册股票名称: {code} -> {name}")


# 便捷函数


def get_stock_name(
    stock_code: str,
    context: dict[str, Any] | None = None,
    data_manager: "DataFetcherManager | None" = None,
) -> str:
    """获取股票名称（便捷函数）

    Args:
        stock_code: 股票代码
        context: 分析上下文（可选）
        data_manager: 数据获取管理器（可选）

    Returns:
        股票中文名称
    """
    resolver = StockNameResolver(data_manager)
    return resolver.resolve(stock_code, context)


def get_stock_name_from_context(stock_code: str, context: dict[str, Any] | None = None) -> str:
    """仅从上下文获取股票名称（快速版，无需实例化）

    Args:
        stock_code: 股票代码
        context: 分析上下文（可选）

    Returns:
        股票中文名称
    """
    return StockNameResolver.from_context(stock_code, context)
