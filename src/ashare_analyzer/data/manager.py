"""
Data manager: unified data access with source management and caching.
"""

import logging
from datetime import date
from typing import TYPE_CHECKING, Any

import pandas as pd

from ashare_analyzer.data.cache import TTLCache
from ashare_analyzer.data.fetchers.akshare import AkshareFetcher
from ashare_analyzer.data.fetchers.baostock import BaostockFetcher
from ashare_analyzer.data.fetchers.efinance import EfinanceFetcher
from ashare_analyzer.data.fetchers.tushare import TushareFetcher
from ashare_analyzer.data.fetchers.yfinance import YfinanceFetcher
from ashare_analyzer.infrastructure.rate_limiter import AsyncRateLimiter
from ashare_analyzer.models import ChipDistribution, UnifiedRealtimeQuote
from ashare_analyzer.utils.stock_code import is_us_code

if TYPE_CHECKING:
    from ashare_analyzer.config import Config
    from ashare_analyzer.storage import DatabaseManager

logger = logging.getLogger(__name__)


class DataManager:
    """
    Unified data manager with source management and caching (async).

    Provides a single entry point for all data operations with:
    - Automatic source failover
    - TTL caching
    - Database persistence (optional)
    """

    def __init__(
        self,
        config: "Config | None" = None,
        storage: "DatabaseManager | None" = None,
        rate_limiters: dict[str, AsyncRateLimiter] | None = None,
    ) -> None:
        self._config = config
        self._storage = storage
        self._cache = TTLCache(default_ttl=600, max_size=1000)
        self._fetchers: list[Any] = []
        self._rate_limiters = rate_limiters or {}
        self._init_fetchers()

    def _init_fetchers(self) -> None:
        fetcher_configs = [
            (EfinanceFetcher, "efinance"),
            (AkshareFetcher, "akshare"),
            (TushareFetcher, "tushare"),
            (BaostockFetcher, "baostock"),
            (YfinanceFetcher, "yfinance"),
        ]

        for fetcher_class, name in fetcher_configs:
            try:
                rate_limiter = self._rate_limiters.get(name)
                fetcher = fetcher_class(rate_limiter=rate_limiter)
                self._fetchers.append(fetcher)
            except Exception as e:
                logger.debug(f"Failed to initialize {fetcher_class.__name__}: {e}")

        self._fetchers.sort(key=lambda f: getattr(f, "priority", 99))

        if not self._fetchers:
            logger.warning("No data sources available!")

    async def get_daily_data(
        self,
        stock_code: str,
        days: int = 30,
        target_date: date | None = None,
        use_cache: bool = True,
    ) -> tuple[pd.DataFrame | None, str]:
        if target_date is None:
            target_date = date.today()

        if use_cache and self._storage is not None:
            local_data = self._storage.get_daily_data(stock_code, days=days)
            if local_data is not None and not local_data.empty:
                latest_date = pd.to_datetime(local_data["date"].iloc[-1]).date()
                if latest_date >= target_date:
                    logger.debug(f"[DataManager] Got {stock_code} from database, {len(local_data)} rows")
                    return local_data, "database"

        df, source = await self._fetch_with_fallback(
            lambda f: f.get_daily_data(stock_code, days=days),
            stock_code=stock_code,
        )

        if df is not None and not df.empty:
            if self._storage is not None:
                self._storage.save_daily_data(df, stock_code, data_source=source)
            logger.debug(f"[DataManager] Got {stock_code} from {source}, {len(df)} rows")
            return df, source

        return None, ""

    async def get_realtime_quote(self, stock_code: str) -> UnifiedRealtimeQuote | None:
        cache_key = f"realtime:{stock_code}"

        if self._cache.is_valid(cache_key):
            logger.debug(f"[DataManager] Cache hit for realtime {stock_code}")
            return self._cache.get(cache_key)

        if is_us_code(stock_code):
            for fetcher in self._fetchers:
                if fetcher.name == "YfinanceFetcher":
                    quote = await self._try_fetch(lambda f: f.get_realtime_quote(stock_code), fetcher)
                    if quote:
                        return quote
            return None

        source_priority = self._get_realtime_source_priority()
        for source in source_priority:
            quote = await self._try_realtime_by_source(stock_code, source)
            if quote is not None and hasattr(quote, "has_basic_data") and quote.has_basic_data():
                self._cache.set(cache_key, quote, ttl=600)
                logger.debug(f"[DataManager] Got realtime {stock_code} from {source}")
                return quote

        logger.warning(f"[DataManager] Failed to get realtime quote for {stock_code}")
        return None

    async def get_chip_distribution(self, stock_code: str) -> ChipDistribution | None:
        """
        Get chip distribution data with database caching.

        Args:
            stock_code: Stock code

        Returns:
            ChipDistribution or None
        """
        from datetime import date

        def _chip_from_dict(data: dict, source: str | None) -> ChipDistribution:
            """Create ChipDistribution from dict."""
            return ChipDistribution(
                code=data.get("code", stock_code),
                date=str(data.get("date", "")),
                profit_ratio=float(data.get("profit_ratio", 0.0) or 0.0),
                avg_cost=float(data.get("avg_cost", 0.0) or 0.0),
                cost_90_low=float(data.get("cost_90_low", 0.0) or 0.0),
                cost_90_high=float(data.get("cost_90_high", 0.0) or 0.0),
                concentration_90=float(data.get("concentration_90", 0.0) or 0.0),
                cost_70_low=float(data.get("cost_70_low", 0.0) or 0.0),
                cost_70_high=float(data.get("cost_70_high", 0.0) or 0.0),
                concentration_70=float(data.get("concentration_70", 0.0) or 0.0),
                source=source if source else "database",
            )

        # 1. Try to get from database cache first
        if self._storage is not None:
            cached_data = self._storage.get_chip_data(stock_code)
            # Check if data is from today (fresh enough)
            if cached_data is not None and cached_data.date == date.today():
                logger.debug(f"[DataManager] Got {stock_code} chip data from database cache")
                return _chip_from_dict(cached_data.to_dict(), str(cached_data.data_source or "database"))

        # 2. Fetch from API
        chip_sources = [
            ("AkshareFetcher", "akshare_chip"),
            ("TushareFetcher", "tushare_chip"),
            ("EfinanceFetcher", "efinance_chip"),
        ]

        for fetcher_name, source_key in chip_sources:
            for fetcher in self._fetchers:
                if fetcher.name == fetcher_name:
                    chip = await self._try_fetch(lambda f: f.get_chip_distribution(stock_code), fetcher)
                    if chip is not None:
                        # 3. Save to database cache
                        if self._storage is not None:
                            self._storage.save_chip_data(
                                stock_code,
                                chip.to_dict(),
                                data_source=source_key,
                            )
                        return chip

        # 4. If API failed, try to return cached data even if stale
        if self._storage is not None:
            cached_data = self._storage.get_chip_data(stock_code)
            if cached_data is not None:
                logger.debug(f"[DataManager] Using stale cached chip data for {stock_code} (API failed)")
                return _chip_from_dict(cached_data.to_dict(), str(cached_data.data_source or "database_stale"))

        return None

    async def get_stock_name(self, stock_code: str) -> str | None:
        cache_key = f"stock_name:{stock_code}"

        cached = self._cache.get(cache_key)
        if cached:
            return cached

        quote = await self.get_realtime_quote(stock_code)
        if quote and hasattr(quote, "name") and quote.name:
            self._cache.set(cache_key, quote.name)
            return quote.name

        for fetcher in self._fetchers:
            if hasattr(fetcher, "get_stock_name"):
                name = await self._try_fetch(lambda f: f.get_stock_name(stock_code), fetcher)
                if name:
                    self._cache.set(cache_key, name)
                    return name

        return None

    async def batch_get_stock_names(self, stock_codes: list[str]) -> dict[str, str]:
        result = {}
        missing_codes = []

        for code in stock_codes:
            cache_key = f"stock_name:{code}"
            cached = self._cache.get(cache_key)
            if cached:
                result[code] = cached
            else:
                missing_codes.append(code)

        if not missing_codes:
            return result

        for fetcher in self._fetchers:
            if hasattr(fetcher, "get_stock_list") and missing_codes:
                stock_list = await self._try_fetch(lambda f: f.get_stock_list(), fetcher)
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

        for code in list(missing_codes):
            name = await self.get_stock_name(code)
            if name:
                result[code] = name

        return result

    async def get_main_indices(self) -> list[dict[str, Any]]:
        for fetcher in self._fetchers:
            result = await self._try_fetch(lambda f: f.get_main_indices(), fetcher)
            if result:
                return result
        return []

    async def get_market_stats(self) -> dict[str, Any]:
        for fetcher in self._fetchers:
            result = await self._try_fetch(lambda f: f.get_market_stats(), fetcher)
            if result:
                return result
        return {}

    async def get_sector_rankings(self, n: int = 5) -> tuple[list[dict], list[dict]]:
        for fetcher in self._fetchers:
            result = await self._try_fetch(lambda f: f.get_sector_rankings(n), fetcher)
            if result:
                return result
        return ([], [])

    async def prefetch_realtime_quotes(self, stock_codes: list[str]) -> int:
        if len(stock_codes) < 5:
            return 0

        first_code = stock_codes[0]
        quote = await self.get_realtime_quote(first_code)

        if quote:
            logger.debug("[DataManager] Prefetch complete, cache filled")
            return len(stock_codes)

        return 0

    def invalidate_cache(self, pattern: str | None = None) -> None:
        self._cache.invalidate(pattern)

    async def _fetch_with_fallback(self, fetch_fn: Any, stock_code: str | None = None) -> tuple[Any, str]:
        errors = []

        for fetcher in self._fetchers:
            try:
                result = await fetch_fn(fetcher)
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

    async def _try_fetch(self, fetch_fn: Any, fetcher: Any) -> Any:
        try:
            return await fetch_fn(fetcher)
        except Exception as e:
            logger.debug(f"{fetcher.name} fetch failed: {e}")
            return None

    def _get_realtime_source_priority(self) -> list[str]:
        if self._config is not None:
            priority = getattr(self._config, "realtime_source_priority", "efinance,akshare_em")
            return [s.strip().lower() for s in priority.split(",")]
        return ["efinance", "akshare_em", "akshare_sina", "tencent"]

    async def _try_realtime_by_source(self, stock_code: str, source: str) -> Any:
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
                return await self._try_fetch(
                    lambda f: f.get_realtime_quote(stock_code, **extra_kwargs),
                    fetcher,
                )
        return None

    @property
    def available_fetchers(self) -> list[str]:
        return [f.name for f in self._fetchers]
