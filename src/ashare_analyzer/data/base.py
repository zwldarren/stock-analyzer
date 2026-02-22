"""
Data fetcher base classes and utilities.

Design Pattern: Strategy Pattern
- BaseFetcher: Abstract base class defining unified interface
- Standard columns and anti-scraping utilities shared by all fetchers
"""

import logging
import random
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Any

import pandas as pd

from ashare_analyzer.exceptions import DataFetchError
from ashare_analyzer.infrastructure.rate_limiter import AsyncRateLimiter

logger = logging.getLogger(__name__)

STANDARD_COLUMNS = ["date", "open", "high", "low", "close", "volume", "amount", "pct_chg"]

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 "
    "(KHTML, like Gecko) Version/17.2 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
]


class BaseFetcher(ABC):
    """
    Abstract base class for data fetchers (async).

    Responsibilities:
    1. Define unified data fetching interface
    2. Provide data standardization methods
    3. Provide rate limiting support
    """

    name: str = "BaseFetcher"
    priority: int = 99

    def __init__(self, rate_limiter: AsyncRateLimiter | None = None):
        """
        Initialize BaseFetcher.

        Args:
            rate_limiter: Optional rate limiter for API calls
        """
        self._rate_limiter = rate_limiter

    def _set_random_user_agent(self) -> None:
        """Set random User-Agent (key anti-scraping strategy)."""
        try:
            random_ua = random.choice(USER_AGENTS)
            logger.debug(f"Setting User-Agent: {random_ua[:50]}...")
        except Exception as e:
            logger.debug(f"Failed to set User-Agent: {e}")

    async def _enforce_rate_limit(self) -> None:
        """Enforce rate limiting (async)."""
        if self._rate_limiter:
            await self._rate_limiter.acquire()

    @abstractmethod
    async def _fetch_raw_data(self, stock_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch raw data from data source (subclass must implement).

        Args:
            stock_code: Stock code, e.g. '600519', '000001'
            start_date: Start date, format 'YYYY-MM-DD'
            end_date: End date, format 'YYYY-MM-DD'

        Returns:
            Raw data DataFrame (column names vary by data source)
        """

    @abstractmethod
    def _normalize_data(self, df: pd.DataFrame, stock_code: str) -> pd.DataFrame:
        """
        Standardize column names (subclass must implement).

        Convert different data source column names to standard:
        ['date', 'open', 'high', 'low', 'close', 'volume', 'amount', 'pct_chg']
        """

    async def get_daily_data(
        self,
        stock_code: str,
        start_date: str | None = None,
        end_date: str | None = None,
        days: int = 30,
    ) -> pd.DataFrame:
        """
        Get daily data (unified entry point, async).

        Args:
            stock_code: Stock code
            start_date: Start date (optional)
            end_date: End date (optional, default today)
            days: Days to fetch (used when start_date not specified)

        Returns:
            Standardized DataFrame
        """
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")

        if start_date is None:
            start_dt = datetime.strptime(end_date, "%Y-%m-%d") - timedelta(days=days * 2)
            start_date = start_dt.strftime("%Y-%m-%d")

        logger.debug(f"[{self.name}] Fetching {stock_code} data: {start_date} ~ {end_date}")

        try:
            await self._enforce_rate_limit()

            raw_df = await self._fetch_raw_data(stock_code, start_date, end_date)

            if raw_df is None or raw_df.empty:
                raise DataFetchError(f"[{self.name}] No data for {stock_code}")

            df = self._normalize_data(raw_df, stock_code)

            logger.debug(f"[{self.name}] {stock_code} fetch success, {len(df)} rows")
            return df

        except Exception as e:
            logger.error(f"[{self.name}] Failed to fetch {stock_code}: {str(e)}")
            raise DataFetchError(f"[{self.name}] {stock_code}: {str(e)}") from e

    async def get_main_indices(self) -> list[dict[str, Any]] | None:
        """Get main index realtime quotes. Returns: List of dicts."""
        return None

    async def get_market_stats(self) -> dict[str, Any] | None:
        """Get market up/down statistics. Returns: Dict with up/down counts."""
        return None

    async def get_sector_rankings(self, n: int = 5) -> tuple[list[dict], list[dict]] | None:
        """Get sector up/down rankings. Returns: Tuple of (top gainers, top losers)."""
        return None

    async def get_realtime_quote(self, stock_code: str, **kwargs):
        """Get realtime quote data. Returns: UnifiedRealtimeQuote or None."""
        return None

    async def get_chip_distribution(self, stock_code: str):
        """Get chip distribution data. Returns: ChipDistribution or None."""
        return None

    async def get_stock_name(self, stock_code: str) -> str | None:
        """Get stock Chinese name. Returns: Name string or None."""
        return None

    async def get_stock_list(self) -> pd.DataFrame | None:
        """Get stock list. Returns: DataFrame or None."""
        return None
