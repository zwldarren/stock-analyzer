"""DataService domain data service with TTL caching."""

import fnmatch
import logging
import threading
from datetime import date
from typing import TYPE_CHECKING, Any

import pandas as pd
from cachetools import TTLCache

from stock_analyzer.domain.exceptions import DataFetchError
from stock_analyzer.domain.types import (
    ChipDistribution,
    UnifiedRealtimeQuote,
)
from stock_analyzer.infrastructure.data_sources.base import DataFetcherManager
from stock_analyzer.infrastructure.persistence.database import DatabaseManager

if TYPE_CHECKING:
    from stock_analyzer.config import Config

logger = logging.getLogger(__name__)


class DataService:
    """DataService provide unified data access for stock analysis, with caching and multiple data sources."""

    def __init__(
        self,
        stock_repo: DatabaseManager | None = None,
        fetcher_manager: DataFetcherManager | None = None,
        config: "Config | None" = None,
    ):
        """Init DataService with optional repository and fetcher manager"""
        self._stock_repo = stock_repo
        self._fetcher_manager = fetcher_manager
        self._config = config

        self._cache: TTLCache[str, Any] = TTLCache(maxsize=1000, ttl=600)
        self._cache_expiry: dict[str, float] = {}
        self._cache_lock = threading.RLock()

        logger.info("DataService initialized")

    def get_daily_data(
        self,
        stock_code: str,
        days: int = 30,
        target_date: date | None = None,
        use_cache: bool = True,
    ) -> tuple[pd.DataFrame | None, str]:
        """
        Fetch daily stock data with caching strategy

        Strategy:
        1. If use_cache=True, try to get from local DB first
        2. If local data is insufficient or expired, fetch from external API
        3. Save new data to local DB
        """
        if target_date is None:
            target_date = date.today()

        # 1. 尝试从本地数据库获取
        if use_cache and self._stock_repo is not None:
            local_data = self._stock_repo.get_daily_data(stock_code, days=days)
            if local_data is not None and not local_data.empty:
                # 检查是否包含目标日期数据
                latest_date = pd.to_datetime(local_data["date"].iloc[-1]).date()
                if latest_date >= target_date:
                    logger.info(f"[DataService] 从本地数据库获取 {stock_code} 数据，共 {len(local_data)} 条")
                    return local_data, "database"

        # 2. 从外部数据源获取
        if self._fetcher_manager is not None:
            try:
                df, source = self._fetcher_manager.get_daily_data(stock_code, days=days)

                if df is not None and not df.empty:
                    # 3. 保存到本地数据库
                    if self._stock_repo is not None:
                        self._stock_repo.save_daily_data(df, stock_code, data_source=source)
                    logger.info(f"[DataService] 从 {source} 获取 {stock_code} 数据并缓存，共 {len(df)} 条")
                    return df, source

            except DataFetchError as e:
                logger.error(f"[DataService] 获取 {stock_code} 日线数据失败: {e}")
        else:
            logger.warning(f"[DataService] 数据获取器未配置，无法获取 {stock_code} 的外部数据")

        return None, ""

    def get_realtime_quote(self, stock_code: str) -> UnifiedRealtimeQuote | None:
        """Fetch real-time stock quote with caching strategy"""
        if self._fetcher_manager is None:
            logger.warning(f"[DataService] 数据获取器未配置，无法获取 {stock_code} 的实时行情")
            return None

        cache_key = f"realtime:{stock_code}"

        # 检查缓存
        if self._is_cache_valid(cache_key):
            logger.debug(f"[DataService] 实时行情缓存命中: {stock_code}")
            return self._cache.get(cache_key)

        # 从数据源获取
        quote = self._fetcher_manager.get_realtime_quote(stock_code)

        if quote is not None and self._config is not None:
            # Update cache with configured TTL
            ttl = self._config.realtime_quote.realtime_cache_ttl
            self._set_cache(cache_key, quote, ttl)

        return quote

    def get_chip_distribution(self, stock_code: str) -> ChipDistribution | None:
        """Fetch chip distribution data for a stock, no caching implemented"""
        if self._fetcher_manager is None:
            logger.warning(f"[DataService] 数据获取器未配置，无法获取 {stock_code} 的筹码分布")
            return None
        return self._fetcher_manager.get_chip_distribution(stock_code)

    def get_stock_name(self, stock_code: str) -> str | None:
        """Get Chinese stock name by code, with caching strategy"""
        cache_key = f"stock_name:{stock_code}"

        # 1. 检查内存缓存（线程安全）
        with self._cache_lock:
            if cache_key in self._cache:
                return self._cache[cache_key]

        # 2. 尝试从实时行情获取
        quote = self.get_realtime_quote(stock_code)
        if quote and hasattr(quote, "name") and quote.name:
            with self._cache_lock:
                self._cache[cache_key] = quote.name
            return quote.name

        # 3. 从数据源获取
        if self._fetcher_manager is None:
            logger.warning(f"[DataService] 数据获取器未配置，无法获取 {stock_code} 的股票名称")
            return None

        name = self._fetcher_manager.get_stock_name(stock_code)
        if name:
            with self._cache_lock:
                self._cache[cache_key] = name

        return name

    def batch_get_stock_names(self, stock_codes: list[str]) -> dict[str, str]:
        """Get stock names for a list of stock codes, with caching strategy"""
        result = {}
        missing_codes = []

        # 1. 先检查内存缓存（线程安全）
        with self._cache_lock:
            for code in stock_codes:
                cache_key = f"stock_name:{code}"
                if cache_key in self._cache:
                    result[code] = self._cache[cache_key]
                else:
                    missing_codes.append(code)

        if not missing_codes:
            return result

        # 2. 批量获取剩余的股票名称
        if self._fetcher_manager is None:
            logger.warning("[DataService] 数据获取器未配置，无法批量获取股票名称")
            return result

        names = self._fetcher_manager.batch_get_stock_names(missing_codes)
        result.update(names)

        # 3. 更新缓存（线程安全）
        with self._cache_lock:
            for code, name in names.items():
                self._cache[f"stock_name:{code}"] = name

        return result

    def get_analysis_context(self, stock_code: str, target_date: date | None = None) -> dict[str, Any] | None:
        """Get analysis context for a stock, including today's and yesterday's data, and technical indicators"""
        if target_date is None:
            target_date = date.today()

        # 从数据库获取最近2天数据
        if self._stock_repo is None:
            logger.warning(f"[DataService] 仓储未配置，无法获取 {stock_code} 的数据")
            return None

        recent_data = self._stock_repo.get_latest_data(stock_code, days=2)

        if not recent_data:
            logger.warning(f"[DataService] 未找到 {stock_code} 的数据")
            return None

        today_data = recent_data[0]
        yesterday_data = recent_data[1] if len(recent_data) > 1 else None

        context = {
            "code": stock_code,
            "date": today_data.date.isoformat(),
            "today": today_data.to_dict(),
        }

        if yesterday_data:
            context["yesterday"] = yesterday_data.to_dict()

            # 计算相比昨日的变化
            if yesterday_data.volume and yesterday_data.volume > 0:
                context["volume_change_ratio"] = round(today_data.volume / yesterday_data.volume, 2)

            if yesterday_data.close and yesterday_data.close > 0:
                context["price_change_ratio"] = round(
                    (today_data.close - yesterday_data.close) / yesterday_data.close * 100, 2
                )

            # 均线形态判断
            close = today_data.close
            ma5 = today_data.ma5
            ma10 = today_data.ma10
            ma20 = today_data.ma20
            if close > ma5 > ma10 > ma20 > 0:
                context["ma_status"] = "多头排列 📈"
            elif close < ma5 < ma10 < ma20 and ma20 > 0:
                context["ma_status"] = "空头排列 📉"
            elif close > ma5 and ma5 > ma10:
                context["ma_status"] = "短期向好 🔼"
            elif close < ma5 and ma5 < ma10:
                context["ma_status"] = "短期走弱 🔽"
            else:
                context["ma_status"] = "震荡整理 ↔️"

        return context

    def get_main_indices(self) -> list[dict[str, Any]]:
        """Get main stock indices data"""
        if self._fetcher_manager is None:
            logger.warning("[DataService] 数据获取器未配置，无法获取主要指数")
            return []
        return self._fetcher_manager.get_main_indices()

    def get_market_stats(self) -> dict[str, Any]:
        """Get overall market statistics, such as涨跌家数, 成交额, etc."""
        if self._fetcher_manager is None:
            logger.warning("[DataService] 数据获取器未配置，无法获取市场统计")
            return {}
        return self._fetcher_manager.get_market_stats()

    def get_sector_rankings(self, n: int = 5) -> tuple[list[dict], list[dict]]:
        """Get sector rankings"""
        if self._fetcher_manager is None:
            logger.warning("[DataService] 数据获取器未配置，无法获取板块排名")
            return [], []
        return self._fetcher_manager.get_sector_rankings(n)

    def get_stock_industry(self, stock_code: str) -> dict[str, Any] | None:
        """
        获取股票所属行业信息

        使用efinance的get_base_info获取股票基本信息，包括所处行业

        Args:
            stock_code: 股票代码

        Returns:
            包含行业信息的字典，获取失败返回None
        """
        cache_key = f"industry:{stock_code}"

        # 检查缓存
        if self._is_cache_valid(cache_key):
            logger.debug(f"[DataService] 行业信息缓存命中: {stock_code}")
            return self._cache.get(cache_key)

        # 从数据源获取
        if self._fetcher_manager is None:
            logger.warning(f"[DataService] 数据获取器未配置，无法获取 {stock_code} 的行业信息")
            return None

        try:
            # 尝试从efinance获取基本信息
            base_info = self._fetcher_manager.get_base_info(stock_code)
            if base_info:
                industry_info = {
                    "stock_code": stock_code,
                    "industry": base_info.get("所处行业"),
                    "sector_code": base_info.get("板块编号"),
                }
                # 缓存行业信息（1小时TTL）
                self._set_cache(cache_key, industry_info, 3600)
                logger.info(f"[DataService] 获取 {stock_code} 行业信息: {industry_info.get('industry')}")
                return industry_info
        except DataFetchError as e:
            logger.warning(f"[DataService] 获取 {stock_code} 行业信息失败: {e}")

        return None

    def get_industry_valuation(self, industry_name: str) -> dict[str, Any] | None:
        """
        获取行业平均估值数据（PE/PB）

        使用akshare获取申万一级行业的平均估值数据

        Args:
            industry_name: 行业名称（如"光学光电子"、"银行"等）

        Returns:
            包含行业平均PE/PB的字典，获取失败返回None
        """
        if not industry_name:
            return None

        cache_key = f"industry_valuation:{industry_name}"

        # 检查缓存
        if self._is_cache_valid(cache_key):
            logger.debug(f"[DataService] 行业估值缓存命中: {industry_name}")
            return self._cache.get(cache_key)

        try:
            import akshare as ak

            # 获取申万一级行业信息
            df = ak.sw_index_first_info()
            if df is None or df.empty:
                logger.warning("[DataService] 无法获取申万行业数据")
                return None

            # 查找匹配的行业
            # 申万行业数据列：行业代码, 行业名称, 成份个数, 静态市盈率, TTM市盈率, 市净率, 静态股息率
            industry_row = None
            for _, row in df.iterrows():
                sw_name = str(row.get("行业名称", ""))
                # 模糊匹配行业名称
                if industry_name in sw_name or sw_name in industry_name:
                    industry_row = row
                    break

            if industry_row is None:
                logger.warning(f"[DataService] 未找到行业 '{industry_name}' 的估值数据")
                return None

            # 提取估值数据
            valuation_data = {
                "industry_name": industry_row.get("行业名称"),
                "industry_code": industry_row.get("行业代码"),
                "component_count": int(industry_row.get("成份个数", 0))
                if pd.notna(industry_row.get("成份个数"))
                else 0,
                "avg_pe_static": float(industry_row.get("静态市盈率", 0))
                if pd.notna(industry_row.get("静态市盈率"))
                else 0,
                "avg_pe_ttm": float(industry_row.get("TTM(滚动)市盈率", 0))
                if pd.notna(industry_row.get("TTM(滚动)市盈率"))
                else 0,
                "avg_pb": float(industry_row.get("市净率", 0)) if pd.notna(industry_row.get("市净率")) else 0,
                "dividend_yield": float(industry_row.get("静态股息率", 0))
                if pd.notna(industry_row.get("静态股息率"))
                else 0,
            }

            # 缓存行业估值数据（30分钟TTL，因为估值数据会随市场变化）
            self._set_cache(cache_key, valuation_data, 1800)
            logger.info(
                f"[DataService] 获取行业 '{industry_name}' 估值数据: "
                f"PE_TTM={valuation_data.get('avg_pe_ttm')}, PB={valuation_data.get('avg_pb')}"
            )
            return valuation_data

        except DataFetchError as e:
            logger.warning(f"[DataService] 获取行业 '{industry_name}' 估值数据失败: {e}")
            return None

    def invalidate_cache(self, pattern: str | None = None) -> None:
        """Invalidate cache entries matching the pattern. If pattern is None, clear all cache."""
        with self._cache_lock:
            if pattern is None:
                self._cache.clear()
                self._cache_expiry.clear()
                logger.info("[DataService] All cache cleared")
            else:
                keys_to_remove = [k for k in self._cache if fnmatch.fnmatch(k, pattern)]
                for key in keys_to_remove:
                    self._remove_cache_entry(key)
                logger.info(f"[DataService] Cache cleared: {pattern} ({len(keys_to_remove)} entries)")

    def _is_cache_valid(self, key: str) -> bool:
        """Check if cache entry is valid (exists and not expired)."""
        with self._cache_lock:
            if key not in self._cache:
                return False
            # Check custom expiry if set
            if key in self._cache_expiry:
                import time

                if time.time() > self._cache_expiry[key]:
                    # Expired, remove from cache
                    self._remove_cache_entry(key)
                    return False
            return True

    def _set_cache(self, key: str, value: Any, ttl_seconds: int) -> None:
        """Set cache entry with TTL (time-to-live) in seconds."""
        import time

        with self._cache_lock:
            self._cache[key] = value
            # Store custom expiry time for this entry
            self._cache_expiry[key] = time.time() + ttl_seconds

    def _remove_cache_entry(self, key: str) -> None:
        """Remove a single cache entry and its expiry record."""
        self._cache.pop(key, None)
        self._cache_expiry.pop(key, None)
