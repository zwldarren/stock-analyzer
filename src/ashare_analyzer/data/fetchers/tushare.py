"""
===================================
TushareFetcher - 备用数据源 1 (Priority 2)
===================================

数据来源：Tushare Pro API（挖地兔）
特点：需要 Token、有请求配额限制
优点：数据质量高、接口稳定

流控策略：
1. 实现"每分钟调用计数器"
2. 超过免费配额（80次/分）时，强制休眠到下一分钟
3. 使用 tenacity 实现指数退避重试
"""

import asyncio
import logging
import time
from datetime import datetime
from typing import TYPE_CHECKING, Any, cast

import pandas as pd
from cachetools import LRUCache
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

if TYPE_CHECKING:
    from tushare.pro.client import DataApi

from ashare_analyzer.config import get_config
from ashare_analyzer.data.base import STANDARD_COLUMNS, BaseFetcher
from ashare_analyzer.exceptions import DataFetchError, RateLimitError
from ashare_analyzer.infrastructure import AsyncRateLimiter
from ashare_analyzer.models import UnifiedRealtimeQuote
from ashare_analyzer.utils.stock_code import convert_to_provider_format, is_us_code

logger = logging.getLogger(__name__)


class TushareFetcher(BaseFetcher):
    """
    Tushare Pro 数据源实现 (async)

    优先级：2
    数据来源：Tushare Pro API

    关键策略：
    - 每分钟调用计数器，防止超出配额
    - 超过 80 次/分钟时强制等待
    - 失败后指数退避重试

    配额说明（Tushare 免费用户）：
    - 每分钟最多 80 次请求
    - 每天最多 500 次请求
    """

    name = "TushareFetcher"

    @property
    def priority(self) -> int:
        config = get_config()

        if not hasattr(self, "_api") or self._api is None:
            return config.datasource.tushare_priority

        if config.datasource.tushare_token and self._api is not None:
            return -1

        return config.datasource.tushare_priority

    def __init__(self, rate_limiter: AsyncRateLimiter | None = None, rate_limit_per_minute: int = 80):
        super().__init__(rate_limiter=rate_limiter)
        self.rate_limit_per_minute = rate_limit_per_minute
        self._call_count = 0
        self._minute_start: float | None = None
        self._api: DataApi | None = None
        self._stock_name_cache: LRUCache[str, str] = LRUCache(maxsize=1000)
        self._init_api()

    def _init_api(self) -> None:
        config = get_config()

        if not config.datasource.tushare_token:
            logger.debug("Tushare Token 未配置，此数据源不可用")
            return

        try:
            import tushare as ts

            ts.set_token(config.datasource.tushare_token)
            self._api = ts.pro_api()
            logger.debug("Tushare API 初始化成功")

        except Exception as e:
            logger.error(f"Tushare API 初始化失败: {e}")
            self._api = None

    def is_available(self) -> bool:
        return self._api is not None

    async def _check_rate_limit(self) -> None:
        current_time = time.time()

        if self._minute_start is None:
            self._minute_start = current_time
            self._call_count = 0
        elif current_time - self._minute_start >= 60:
            self._minute_start = current_time
            self._call_count = 0
            logger.debug("速率限制计数器已重置")

        if self._call_count >= self.rate_limit_per_minute:
            elapsed = current_time - self._minute_start
            sleep_time = max(0, 60 - elapsed) + 1

            logger.warning(
                f"Tushare 达到速率限制 ({self._call_count}/{self.rate_limit_per_minute} 次/分钟)，"
                f"等待 {sleep_time:.1f} 秒..."
            )

            await asyncio.sleep(sleep_time)

            self._minute_start = time.time()
            self._call_count = 0

        self._call_count += 1
        logger.debug(f"Tushare 当前分钟调用次数: {self._call_count}/{self.rate_limit_per_minute}")

    def _convert_stock_code(self, stock_code: str) -> str:
        code = stock_code.strip()

        if "." in code:
            return code.upper()

        return convert_to_provider_format(stock_code, "tushare")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type((ConnectionError, TimeoutError)),
        before_sleep=before_sleep_log(cast(Any, logger), logging.WARNING),
    )
    async def _fetch_raw_data(self, stock_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        if self._api is None:
            raise DataFetchError("Tushare API 未初始化，请检查 Token 配置")

        if is_us_code(stock_code):
            raise DataFetchError(f"TushareFetcher 不支持美股 {stock_code}，请使用 AkshareFetcher 或 YfinanceFetcher")

        await self._check_rate_limit()

        ts_code = self._convert_stock_code(stock_code)
        ts_start = start_date.replace("-", "")
        ts_end = end_date.replace("-", "")

        logger.debug(f"调用 Tushare daily({ts_code}, {ts_start}, {ts_end})")

        api = self._api
        if api is None:
            raise DataFetchError("Tushare API 未初始化")

        try:

            def _call_api():
                return api.daily(ts_code=ts_code, start_date=ts_start, end_date=ts_end)

            df = await asyncio.to_thread(_call_api)
            return df

        except Exception as e:
            error_msg = str(e).lower()

            if any(keyword in error_msg for keyword in ["quota", "配额", "limit", "权限"]):
                logger.warning(f"Tushare 配额可能超限: {e}")
                raise RateLimitError(f"Tushare 配额超限: {e}") from e

            raise DataFetchError(f"Tushare 获取数据失败: {e}") from e

    def _normalize_data(self, df: pd.DataFrame, stock_code: str) -> pd.DataFrame:
        df = df.copy()

        column_mapping = {
            "trade_date": "date",
            "vol": "volume",
        }

        df = df.rename(columns=column_mapping)

        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")

        if "volume" in df.columns:
            df["volume"] = df["volume"] * 100

        if "amount" in df.columns:
            df["amount"] = df["amount"] * 1000

        df["code"] = stock_code

        keep_cols = ["code"] + STANDARD_COLUMNS
        existing_cols = [col for col in keep_cols if col in df.columns]
        df = df[existing_cols]

        return df

    async def get_stock_name(self, stock_code: str) -> str | None:
        if self._api is None:
            logger.warning("Tushare API 未初始化，无法获取股票名称")
            return None

        if stock_code in self._stock_name_cache:
            return self._stock_name_cache[stock_code]

        return None

    async def get_stock_list(self) -> pd.DataFrame | None:
        if self._api is None:
            logger.warning("Tushare API 未初始化，无法获取股票列表")
            return None

        try:
            await self._check_rate_limit()

            api = self._api

            def _call_api():
                return api.stock_basic(exchange="", list_status="L", fields="ts_code,name,industry,area,market")

            df = await asyncio.to_thread(_call_api)

            if df is not None and not df.empty:
                df["code"] = df["ts_code"].apply(lambda x: x.split(".")[0])

                for _, row in df.iterrows():
                    self._stock_name_cache[row["code"]] = row["name"]

                logger.info(f"Tushare 获取股票列表成功: {len(df)} 条")
                return df[["code", "name", "industry", "area", "market"]]

        except Exception as e:
            logger.warning(f"Tushare 获取股票列表失败: {e}")

        return None

    async def get_realtime_quote(self, stock_code: str, **kwargs) -> UnifiedRealtimeQuote | None:
        if self._api is None:
            return None

        from ashare_analyzer.models import RealtimeSource

        from .realtime_types import safe_float, safe_int

        await self._check_rate_limit()

        try:
            ts_code = self._convert_stock_code(stock_code)
            api = self._api

            def _call_api():
                return api.quotation(ts_code=ts_code)

            df = await asyncio.to_thread(_call_api)

            if df is not None and not df.empty:
                row = df.iloc[0]
                logger.debug(f"Tushare Pro 实时行情获取成功: {stock_code}")

                return UnifiedRealtimeQuote(
                    code=stock_code,
                    name=str(row.get("name", "")),
                    source=RealtimeSource.TUSHARE,
                    price=safe_float(row.get("price")),
                    change_pct=safe_float(row.get("pct_chg")),
                    change_amount=safe_float(row.get("change")),
                    volume=safe_int(row.get("vol")),
                    amount=safe_float(row.get("amount")),
                    high=safe_float(row.get("high")),
                    low=safe_float(row.get("low")),
                    open_price=safe_float(row.get("open")),
                    pre_close=safe_float(row.get("pre_close")),
                    turnover_rate=safe_float(row.get("turnover_ratio")),
                    pe_ratio=safe_float(row.get("pe")),
                    pb_ratio=safe_float(row.get("pb")),
                    total_mv=safe_float(row.get("total_mv")),
                )
        except Exception as e:
            logger.debug(f"Tushare Pro 实时行情不可用 (可能是积分不足): {e}")

        try:
            import tushare as ts

            code_6 = stock_code.split(".")[0] if "." in stock_code else stock_code

            if code_6 == "000001":
                symbol = "sh000001"
            elif code_6 == "399001":
                symbol = "sz399001"
            elif code_6 == "399006":
                symbol = "sz399006"
            elif code_6 == "000300":
                symbol = "sh000300"
            else:
                symbol = code_6

            def _call_api():
                return ts.get_realtime_quotes(symbol)

            df = await asyncio.to_thread(_call_api)

            if df is None or df.empty:
                return None

            row = df.iloc[0]

            price = safe_float(row["price"])
            pre_close = safe_float(row["pre_close"])
            change_pct = 0.0
            change_amount = 0.0

            if price and pre_close and pre_close > 0:
                change_amount = price - pre_close
                change_pct = (change_amount / pre_close) * 100

            from ashare_analyzer.models import RealtimeSource

            return UnifiedRealtimeQuote(
                code=stock_code,
                name=str(row["name"]),
                source=RealtimeSource.TUSHARE,
                price=price,
                change_pct=round(change_pct, 2),
                change_amount=round(change_amount, 2),
                volume=(safe_int(row["volume"], default=0) or 0) // 100,
                amount=safe_float(row["amount"]),
                high=safe_float(row["high"]),
                low=safe_float(row["low"]),
                open_price=safe_float(row["open"]),
                pre_close=pre_close,
            )

        except Exception as e:
            logger.warning(f"Tushare (旧版) 获取实时行情失败 {stock_code}: {e}")
            return None

    async def get_main_indices(self) -> list[dict] | None:
        if self._api is None:
            return None

        from .realtime_types import safe_float

        indices_map = {
            "000001.SH": "上证指数",
            "399001.SZ": "深证成指",
            "399006.SZ": "创业板指",
            "000688.SH": "科创50",
            "000016.SH": "上证50",
            "000300.SH": "沪深300",
        }

        try:
            await self._check_rate_limit()

            end_date = datetime.now().strftime("%Y%m%d")
            start_date = (datetime.now() - pd.Timedelta(days=5)).strftime("%Y%m%d")

            results = []

            api = self._api
            for ts_code, name in indices_map.items():
                try:
                    code = ts_code

                    def _call_api(c=code):
                        return api.index_daily(ts_code=c, start_date=start_date, end_date=end_date)

                    df = await asyncio.to_thread(_call_api)
                    if df is not None and not df.empty:
                        row = df.iloc[0]

                        current = safe_float(row["close"])
                        prev_close = safe_float(row["pre_close"])

                        results.append(
                            {
                                "code": ts_code.split(".")[0],
                                "name": name,
                                "current": current,
                                "change": safe_float(row["change"]),
                                "change_pct": safe_float(row["pct_chg"]),
                                "open": safe_float(row["open"]),
                                "high": safe_float(row["high"]),
                                "low": safe_float(row["low"]),
                                "prev_close": prev_close,
                                "volume": safe_float(row["vol"]),
                                "amount": (safe_float(row["amount"], default=0.0) or 0.0) * 1000,
                                "amplitude": 0.0,
                            }
                        )
                except Exception as e:
                    logger.debug(f"Tushare 获取指数 {name} 失败: {e}")
                    continue

            if results:
                return results
            else:
                logger.warning("[Tushare] 未获取到指数行情数据")

        except Exception as e:
            logger.error(f"[Tushare] 获取指数行情失败: {e}")

        return None

    async def get_market_stats(self) -> dict | None:
        if self._api is None:
            return None

        api = self._api
        if api is None:
            return None

        try:
            await self._check_rate_limit()

            start_date = (datetime.now() - pd.Timedelta(days=20)).strftime("%Y%m%d")

            def _call_trade_cal():
                return api.trade_cal(
                    exchange="",
                    start_date=start_date,
                    end_date=datetime.now().strftime("%Y%m%d"),
                    is_open="1",
                )

            trade_cal = await asyncio.to_thread(_call_trade_cal)

            if trade_cal is None or trade_cal.empty:
                return None

            trade_cal = trade_cal.sort_values("cal_date")
            last_date = trade_cal.iloc[-1]["cal_date"]
            logger.debug(f"[Tushare] Calendar suggests last trading date: {last_date}")

            def _call_daily():
                return api.daily(trade_date=last_date)

            df = await asyncio.to_thread(_call_daily)

            current_len = len(df) if df is not None else 0
            logger.debug(f"[Tushare] Initial fetch for {last_date} returned {current_len} records")

            if df is None or len(df) < 100:
                if len(trade_cal) > 1:
                    prev_date = trade_cal.iloc[-2]["cal_date"]
                    logger.warning(
                        f"Data for {last_date} is incomplete (count={current_len}), falling back to {prev_date}"
                    )
                    last_date = prev_date

                    def _call_daily_prev():
                        return api.daily(trade_date=last_date)

                    df = await asyncio.to_thread(_call_daily_prev)
                else:
                    logger.warning(f"[Tushare] {last_date} 数据不足且无可用历史交易日")

            logger.debug(f"Calculating stats using data from date: {last_date}")

            if df is not None and not df.empty:
                logger.debug(f"[Tushare] 使用交易日 {last_date} 进行市场统计分析")
                up_count = len(df[df["pct_chg"] > 0])
                down_count = len(df[df["pct_chg"] < 0])
                flat_count = len(df[df["pct_chg"] == 0])

                limit_up = len(df[df["pct_chg"] >= 9.9])
                limit_down = len(df[df["pct_chg"] <= -9.9])

                total_amount = df["amount"].sum() * 1000 / 1e8

                return {
                    "up_count": up_count,
                    "down_count": down_count,
                    "flat_count": flat_count,
                    "limit_up_count": limit_up,
                    "limit_down_count": limit_down,
                    "total_amount": total_amount,
                }
            else:
                logger.warning("[Tushare] 获取市场统计数据为空")

        except Exception as e:
            logger.error(f"[Tushare] 获取市场统计失败: {e}")

        return None

    async def get_sector_rankings(self, n: int = 5) -> tuple[list, list] | None:
        return None
