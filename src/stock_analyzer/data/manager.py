"""
Data manager: unified data access with source management and caching.
"""

import logging
from datetime import date
from typing import TYPE_CHECKING, Any

import pandas as pd

from stock_analyzer.data.cache import TTLCache
from stock_analyzer.data.fetchers.akshare import AkshareFetcher
from stock_analyzer.data.fetchers.baostock import BaostockFetcher
from stock_analyzer.data.fetchers.efinance import EfinanceFetcher
from stock_analyzer.data.fetchers.pytdx import PytdxFetcher
from stock_analyzer.data.fetchers.tushare import TushareFetcher
from stock_analyzer.data.fetchers.yfinance import YfinanceFetcher
from stock_analyzer.models import ChipDistribution, UnifiedRealtimeQuote
from stock_analyzer.utils.stock_code import is_us_code

if TYPE_CHECKING:
    from stock_analyzer.config import Config
    from stock_analyzer.storage import DatabaseManager

logger = logging.getLogger(__name__)


class DataManager:
    """
    Unified data manager with source management and caching.

    Provides a single entry point for all data operations with:
    - Automatic source failover
    - TTL caching
    - Database persistence (optional)
    """

    def __init__(
        self,
        config: "Config | None" = None,
        storage: "DatabaseManager | None" = None,
    ) -> None:
        """
        Initialize data manager.

        Args:
            config: Application configuration
            storage: Optional database manager for persistence
        """
        self._config = config
        self._storage = storage
        self._cache = TTLCache(default_ttl=600, max_size=1000)
        self._fetchers: list[Any] = []
        self._init_fetchers()

    def _init_fetchers(self) -> None:
        """Initialize data fetchers by priority."""
        # Fetchers are already ordered by priority in their class definitions
        fetcher_classes = [
            EfinanceFetcher,  # priority 1
            AkshareFetcher,  # priority 2
            TushareFetcher,  # priority 3
            PytdxFetcher,  # priority 4
            BaostockFetcher,  # priority 5
            YfinanceFetcher,  # priority 6
        ]

        for fetcher_class in fetcher_classes:
            try:
                fetcher = fetcher_class()
                self._fetchers.append(fetcher)
            except Exception as e:
                logger.debug(f"Failed to initialize {fetcher_class.__name__}: {e}")

        # Sort by priority (defined in each fetcher class)
        self._fetchers.sort(key=lambda f: getattr(f, "priority", 99))

        if not self._fetchers:
            logger.warning("No data sources available!")

    def get_daily_data(
        self,
        stock_code: str,
        days: int = 30,
        target_date: date | None = None,
        use_cache: bool = True,
    ) -> tuple[pd.DataFrame | None, str]:
        """
        Fetch daily stock data with caching strategy.

        Strategy:
        1. If use_cache=True, try to get from local DB first
        2. If local data is insufficient or expired, fetch from external API
        3. Save new data to local DB

        Args:
            stock_code: Stock code
            days: Number of trading days
            target_date: Target date (default today)
            use_cache: Whether to use cache

        Returns:
            Tuple of (DataFrame, source_name) or (None, "")
        """
        if target_date is None:
            target_date = date.today()

        # 1. Try local database
        if use_cache and self._storage is not None:
            local_data = self._storage.get_daily_data(stock_code, days=days)
            if local_data is not None and not local_data.empty:
                latest_date = pd.to_datetime(local_data["date"].iloc[-1]).date()
                if latest_date >= target_date:
                    logger.debug(f"[DataManager] Got {stock_code} from database, {len(local_data)} rows")
                    return local_data, "database"

        # 2. Fetch from external sources
        df, source = self._fetch_with_fallback(
            lambda f: f.get_daily_data(stock_code, days=days),
            stock_code=stock_code,
        )

        if df is not None and not df.empty:
            # 3. Save to database
            if self._storage is not None:
                self._storage.save_daily_data(df, stock_code, data_source=source)
            logger.debug(f"[DataManager] Got {stock_code} from {source}, {len(df)} rows")
            return df, source

        return None, ""

    def get_realtime_quote(self, stock_code: str) -> UnifiedRealtimeQuote | None:
        """
        Get realtime stock quote with caching.

        Args:
            stock_code: Stock code

        Returns:
            UnifiedRealtimeQuote or None
        """
        cache_key = f"realtime:{stock_code}"

        # Check cache
        if self._cache.is_valid(cache_key):
            logger.debug(f"[DataManager] Cache hit for realtime {stock_code}")
            return self._cache.get(cache_key)

        # US stock handling
        if is_us_code(stock_code):
            for fetcher in self._fetchers:
                if fetcher.name == "YfinanceFetcher":
                    quote = self._try_fetch(lambda f: f.get_realtime_quote(stock_code), fetcher)
                    if quote:
                        return quote
            return None

        # Get from configured priority
        source_priority = self._get_realtime_source_priority()
        for source in source_priority:
            quote = self._try_realtime_by_source(stock_code, source)
            if quote is not None and hasattr(quote, "has_basic_data") and quote.has_basic_data():
                self._cache.set(cache_key, quote, ttl=600)
                logger.debug(f"[DataManager] Got realtime {stock_code} from {source}")
                return quote

        logger.warning(f"[DataManager] Failed to get realtime quote for {stock_code}")
        return None

    def get_chip_distribution(self, stock_code: str) -> ChipDistribution | None:
        """
        Get chip distribution data.

        Args:
            stock_code: Stock code

        Returns:
            ChipDistribution or None
        """
        chip_sources = [
            ("AkshareFetcher", "akshare_chip"),
            ("TushareFetcher", "tushare_chip"),
            ("EfinanceFetcher", "efinance_chip"),
        ]

        for fetcher_name, _source_key in chip_sources:
            for fetcher in self._fetchers:
                if fetcher.name == fetcher_name:
                    chip = self._try_fetch(lambda f: f.get_chip_distribution(stock_code), fetcher)
                    if chip is not None:
                        return chip

        return None

    def get_stock_name(self, stock_code: str) -> str | None:
        """
        Get stock Chinese name.

        Args:
            stock_code: Stock code

        Returns:
            Stock name or None
        """
        cache_key = f"stock_name:{stock_code}"

        # Check cache
        cached = self._cache.get(cache_key)
        if cached:
            return cached

        # Try realtime quote first
        quote = self.get_realtime_quote(stock_code)
        if quote and hasattr(quote, "name") and quote.name:
            self._cache.set(cache_key, quote.name)
            return quote.name

        # Try fetchers
        for fetcher in self._fetchers:
            if hasattr(fetcher, "get_stock_name"):
                name = self._try_fetch(lambda f: f.get_stock_name(stock_code), fetcher)
                if name:
                    self._cache.set(cache_key, name)
                    return name

        return None

    def batch_get_stock_names(self, stock_codes: list[str]) -> dict[str, str]:
        """
        Get stock names for multiple codes.

        Args:
            stock_codes: List of stock codes

        Returns:
            Dict of {code: name}
        """
        result = {}
        missing_codes = []

        # Check cache first
        for code in stock_codes:
            cache_key = f"stock_name:{code}"
            cached = self._cache.get(cache_key)
            if cached:
                result[code] = cached
            else:
                missing_codes.append(code)

        if not missing_codes:
            return result

        # Batch fetch from data sources
        for fetcher in self._fetchers:
            if hasattr(fetcher, "get_stock_list") and missing_codes:
                stock_list = self._try_fetch(lambda f: f.get_stock_list(), fetcher)
                if stock_list is not None and not stock_list.empty:
                    for _, row in stock_list.iterrows():
                        code = row.get("code")
                        name = row.get("name")
                        if code and name and code in missing_codes:
                            result[code] = name
                            self._cache.set(f"stock_name:{code}", name)
                            missing_codes.remove(code)

                    if not missing_codes:
                        break

        # Fetch remaining individually
        for code in list(missing_codes):
            name = self.get_stock_name(code)
            if name:
                result[code] = name

        return result

    def get_main_indices(self) -> list[dict[str, Any]]:
        """Get main stock indices data."""
        for fetcher in self._fetchers:
            result = self._try_fetch(lambda f: f.get_main_indices(), fetcher)
            if result:
                return result
        return []

    def get_market_stats(self) -> dict[str, Any]:
        """Get market statistics (up/down counts, volume, etc.)."""
        for fetcher in self._fetchers:
            result = self._try_fetch(lambda f: f.get_market_stats(), fetcher)
            if result:
                return result
        return {}

    def get_sector_rankings(self, n: int = 5) -> tuple[list[dict], list[dict]]:
        """Get sector rankings (top gainers, top losers)."""
        for fetcher in self._fetchers:
            result = self._try_fetch(lambda f: f.get_sector_rankings(n), fetcher)
            if result:
                return result
        return ([], [])

    def prefetch_realtime_quotes(self, stock_codes: list[str]) -> int:
        """
        Prefetch realtime quotes for multiple stocks.

        Args:
            stock_codes: List of stock codes

        Returns:
            Number of prefetched stocks
        """
        if len(stock_codes) < 5:
            return 0

        # Try to trigger bulk fetch with first stock
        first_code = stock_codes[0]
        quote = self.get_realtime_quote(first_code)

        if quote:
            logger.debug("[DataManager] Prefetch complete, cache filled")
            return len(stock_codes)

        return 0

    def invalidate_cache(self, pattern: str | None = None) -> None:
        """
        Invalidate cache entries.

        Args:
            pattern: Glob pattern to match keys (None = clear all)
        """
        self._cache.invalidate(pattern)

    def _fetch_with_fallback(self, fetch_fn: Any, stock_code: str | None = None) -> tuple[Any, str]:
        """
        Try fetchers in order until one succeeds.

        Args:
            fetch_fn: Function that takes a fetcher and returns data
            stock_code: Stock code for logging

        Returns:
            Tuple of (data, source_name)

        Raises:
            DataFetchError: If all fetchers fail
        """
        errors = []

        for fetcher in self._fetchers:
            try:
                result = fetch_fn(fetcher)
                if result is not None:
                    if isinstance(result, pd.DataFrame) and result.empty:
                        continue
                    return result, fetcher.name
            except Exception as e:
                errors.append(f"{fetcher.name}: {e}")
                logger.debug(f"Fetcher {fetcher.name} failed: {e}")
                continue

        if errors:
            error_msg = f"All fetchers failed for {stock_code or 'request'}: {'; '.join(errors)}"
            logger.warning(error_msg)

        return None, ""

    def _try_fetch(self, fetch_fn: Any, fetcher: Any) -> Any:
        """Try to fetch from a specific fetcher, returning None on error."""
        try:
            return fetch_fn(fetcher)
        except Exception as e:
            logger.debug(f"{fetcher.name} fetch failed: {e}")
            return None

    def _get_realtime_source_priority(self) -> list[str]:
        """Get realtime source priority from config."""
        if self._config is not None:
            priority = getattr(self._config, "realtime_source_priority", "efinance,akshare_em")
            return [s.strip().lower() for s in priority.split(",")]
        return ["efinance", "akshare_em", "akshare_sina", "tencent"]

    def _try_realtime_by_source(self, stock_code: str, source: str) -> Any:
        """Try to get realtime quote from specific source."""
        source_map = {
            "efinance": ("EfinanceFetcher", {}),
            "akshare_em": ("AkshareFetcher", {"source": "em"}),
            "akshare_sina": ("AkshareFetcher", {"source": "sina"}),
            "tencent": ("AkshareFetcher", {"source": "tencent"}),
            "akshare_qq": ("AkshareFetcher", {"source": "tencent"}),
            "tushare": ("TushareFetcher", {}),
        }

        if source not in source_map:
            return None

        fetcher_name, extra_kwargs = source_map[source]
        for fetcher in self._fetchers:
            if fetcher.name == fetcher_name:
                return self._try_fetch(
                    lambda f: f.get_realtime_quote(stock_code, **extra_kwargs),
                    fetcher,
                )
        return None

    @property
    def available_fetchers(self) -> list[str]:
        """Return available fetcher names."""
        return [f.name for f in self._fetchers]
