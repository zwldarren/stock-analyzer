"""
===================================
PytdxFetcher - 通达信数据源 (Priority 2)
===================================

数据来源：通达信行情服务器（pytdx 库）
特点：免费、无需 Token、直连行情服务器
优点：实时数据、稳定、无配额限制

关键策略：
1. 多服务器自动切换
2. 连接超时自动重连
3. 失败后指数退避重试
"""

import asyncio
import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any, cast

import pandas as pd
from cachetools import LRUCache
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from stock_analyzer.data.base import STANDARD_COLUMNS, BaseFetcher, DataFetchError
from stock_analyzer.infrastructure import AsyncRateLimiter
from stock_analyzer.utils.stock_code import is_us_code

logger = logging.getLogger(__name__)


class PytdxFetcher(BaseFetcher):
    """
    通达信数据源实现 (async)

    优先级：2（与 Tushare 同级）
    数据来源：通达信行情服务器

    关键策略：
    - 自动选择最优服务器
    - 连接失败自动切换服务器
    - 失败后指数退避重试

    Pytdx 特点：
    - 免费、无需注册
    - 直连行情服务器
    - 支持实时行情和历史数据
    - 支持股票名称查询
    """

    name = "PytdxFetcher"

    @property
    def priority(self) -> int:
        from stock_analyzer.config import get_config

        return get_config().datasource.pytdx_priority

    DEFAULT_HOSTS = [
        ("119.147.212.81", 7709),
        ("112.74.214.43", 7727),
        ("221.231.141.60", 7709),
        ("101.227.73.20", 7709),
        ("101.227.77.254", 7709),
        ("14.215.128.18", 7709),
        ("59.173.18.140", 7709),
        ("180.153.39.51", 7709),
    ]

    def __init__(self, hosts: list[tuple[str, int]] | None = None, rate_limiter: AsyncRateLimiter | None = None):
        super().__init__(rate_limiter=rate_limiter)
        self._hosts = hosts or self.DEFAULT_HOSTS
        self._current_host_idx = 0
        self._stock_list_cache: dict[str, str] | None = None
        self._stock_name_cache: LRUCache[str, str] = LRUCache(maxsize=1000)

    def _get_pytdx(self):
        try:
            from pytdx.hq import TdxHq_API

            return TdxHq_API
        except ImportError:
            logger.warning("pytdx 未安装，请运行: pip install pytdx")
            return None

    @asynccontextmanager
    async def _pytdx_session(self) -> AsyncGenerator:
        TdxHq_API = self._get_pytdx()
        if TdxHq_API is None:
            raise DataFetchError("pytdx 库未安装")

        api = TdxHq_API()
        connected = False

        try:
            for i in range(len(self._hosts)):
                host_idx = (self._current_host_idx + i) % len(self._hosts)
                host, port = self._hosts[host_idx]

                try:

                    def _connect(api=api, host=host, port=port):
                        return api.connect(host, port, time_out=5)

                    if await asyncio.to_thread(_connect):
                        connected = True
                        self._current_host_idx = host_idx
                        logger.debug(f"Pytdx 连接成功: {host}:{port}")
                        break
                except Exception as e:
                    logger.debug(f"Pytdx 连接 {host}:{port} 失败: {e}")
                    continue

            if not connected:
                raise DataFetchError("Pytdx 无法连接任何服务器")

            yield api

        finally:
            try:

                def _disconnect(api=api):
                    return api.disconnect()

                await asyncio.to_thread(_disconnect)
                logger.debug("Pytdx 连接已断开")
            except Exception as e:
                logger.warning(f"Pytdx 断开连接时出错: {e}")

    def _get_market_code(self, stock_code: str) -> tuple[int, str]:
        code = stock_code.strip()

        code = code.replace(".SH", "").replace(".SZ", "")
        code = code.replace(".sh", "").replace(".sz", "")
        code = code.replace("sh", "").replace("sz", "")

        if code.startswith(("60", "68")):
            return 1, code
        else:
            return 0, code

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type((ConnectionError, TimeoutError)),
        before_sleep=before_sleep_log(cast(Any, logger), logging.WARNING),
    )
    async def _fetch_raw_data(self, stock_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        if is_us_code(stock_code):
            raise DataFetchError(f"PytdxFetcher 不支持美股 {stock_code}，请使用 AkshareFetcher 或 YfinanceFetcher")

        market, code = self._get_market_code(stock_code)

        from datetime import datetime as dt

        start_dt = dt.strptime(start_date, "%Y-%m-%d")
        end_dt = dt.strptime(end_date, "%Y-%m-%d")
        days = (end_dt - start_dt).days
        count = min(max(days * 5 // 7 + 10, 30), 800)

        logger.debug(f"调用 Pytdx get_security_bars(market={market}, code={code}, count={count})")

        async with self._pytdx_session() as api:
            try:
                await self._enforce_rate_limit()

                def _call_api(api=api, market=market, code=code, count=count):
                    return api.get_security_bars(
                        category=9,
                        market=market,
                        code=code,
                        start=0,
                        count=count,
                    )

                data = await asyncio.to_thread(_call_api)

                if data is None or len(data) == 0:
                    raise DataFetchError(f"Pytdx 未查询到 {stock_code} 的数据")

                df = api.to_df(data)

                df["datetime"] = pd.to_datetime(df["datetime"])
                df = df[(df["datetime"] >= start_date) & (df["datetime"] <= end_date)]

                return df

            except Exception as e:
                if isinstance(e, DataFetchError):
                    raise
                raise DataFetchError(f"Pytdx 获取数据失败: {e}") from e

    def _normalize_data(self, df: pd.DataFrame, stock_code: str) -> pd.DataFrame:
        df = df.copy()

        column_mapping = {
            "datetime": "date",
            "vol": "volume",
        }

        df = df.rename(columns=column_mapping)

        if "pct_chg" not in df.columns and "close" in df.columns:
            df["pct_chg"] = df["close"].pct_change() * 100
            df["pct_chg"] = df["pct_chg"].fillna(0).round(2)

        df["code"] = stock_code

        keep_cols = ["code"] + STANDARD_COLUMNS
        existing_cols = [col for col in keep_cols if col in df.columns]
        df = df[existing_cols]

        return df

    async def get_stock_name(self, stock_code: str) -> str | None:
        if stock_code in self._stock_name_cache:
            return self._stock_name_cache[stock_code]

        try:
            market, code = self._get_market_code(stock_code)

            async with self._pytdx_session() as api:
                if self._stock_list_cache is None:

                    def _get_stock_list(api=api):
                        sz_stocks = api.get_security_list(0, 0)
                        sh_stocks = api.get_security_list(1, 0)
                        return sz_stocks, sh_stocks

                    sz_stocks, sh_stocks = await asyncio.to_thread(_get_stock_list)

                    self._stock_list_cache = {}
                    for stock in (sz_stocks or []) + (sh_stocks or []):
                        self._stock_list_cache[stock["code"]] = stock["name"]

                name = self._stock_list_cache.get(code)
                if name:
                    self._stock_name_cache[stock_code] = name
                    return name

                def _get_finance_info(api=api, market=market, code=code):
                    return api.get_finance_info(market, code)

                finance_info = await asyncio.to_thread(_get_finance_info)

                if finance_info and "name" in finance_info:
                    name = finance_info["name"]
                    self._stock_name_cache[stock_code] = name
                    return name

        except Exception as e:
            logger.warning(f"Pytdx 获取股票名称失败 {stock_code}: {e}")

        return None

    async def get_realtime_quote(self, stock_code: str, **kwargs) -> dict | None:
        try:
            market, code = self._get_market_code(stock_code)

            async with self._pytdx_session() as api:
                await self._enforce_rate_limit()

                def _get_quotes(api=api, market=market, code=code):
                    return api.get_security_quotes([(market, code)])

                data = await asyncio.to_thread(_get_quotes)

                if data and len(data) > 0:
                    quote = data[0]
                    return {
                        "code": stock_code,
                        "name": quote.get("name", ""),
                        "price": quote.get("price", 0),
                        "open": quote.get("open", 0),
                        "high": quote.get("high", 0),
                        "low": quote.get("low", 0),
                        "pre_close": quote.get("last_close", 0),
                        "volume": quote.get("vol", 0),
                        "amount": quote.get("amount", 0),
                        "bid_prices": [quote.get(f"bid{i}", 0) for i in range(1, 6)],
                        "ask_prices": [quote.get(f"ask{i}", 0) for i in range(1, 6)],
                    }
        except Exception as e:
            logger.warning(f"Pytdx 获取实时行情失败 {stock_code}: {e}")

        return None
