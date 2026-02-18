"""
===================================
BaostockFetcher - 备用数据源 2 (Priority 3)
===================================

数据来源：证券宝（Baostock）
特点：免费、无需 Token、需要登录管理
优点：稳定、无配额限制

关键策略：
1. 管理 bs.login() 和 bs.logout() 生命周期
2. 使用上下文管理器防止连接泄露
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
from stock_analyzer.utils.stock_code import convert_to_provider_format, is_us_code

logger = logging.getLogger(__name__)


class BaostockFetcher(BaseFetcher):
    """
    Baostock 数据源实现 (async)

    优先级：3
    数据来源：证券宝 Baostock API

    关键策略：
    - 使用异步上下文管理器管理连接生命周期
    - 每次请求都重新登录/登出，防止连接泄露
    - 失败后指数退避重试

    Baostock 特点：
    - 免费、无需注册
    - 需要显式登录/登出
    - 数据更新略有延迟（T+1）
    """

    name = "BaostockFetcher"

    @property
    def priority(self) -> int:
        from stock_analyzer.config import get_config

        return get_config().datasource.baostock_priority

    def __init__(self, rate_limiter: AsyncRateLimiter | None = None):
        super().__init__(rate_limiter=rate_limiter)
        self._bs_module = None
        self._stock_name_cache: LRUCache[str, str] = LRUCache(maxsize=1000)

    def _get_baostock(self):
        if self._bs_module is None:
            import baostock as bs

            self._bs_module = bs
        return self._bs_module

    @asynccontextmanager
    async def _baostock_session(self) -> AsyncGenerator:
        bs = self._get_baostock()
        login_result = None

        try:
            login_result = await asyncio.to_thread(bs.login)

            if login_result.error_code != "0":
                raise DataFetchError(f"Baostock 登录失败: {login_result.error_msg}")

            logger.debug("Baostock 登录成功")

            yield bs

        finally:
            try:
                logout_result = await asyncio.to_thread(bs.logout)
                if logout_result.error_code == "0":
                    logger.debug("Baostock 登出成功")
                else:
                    logger.warning(f"Baostock 登出异常: {logout_result.error_msg}")
            except Exception as e:
                logger.warning(f"Baostock 登出时发生错误: {e}")

    def _convert_stock_code(self, stock_code: str) -> str:
        code = stock_code.strip()

        if code.startswith(("sh.", "sz.")):
            return code.lower()

        return convert_to_provider_format(stock_code, "baostock")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type((ConnectionError, TimeoutError)),
        before_sleep=before_sleep_log(cast(Any, logger), logging.WARNING),
    )
    async def _fetch_raw_data(self, stock_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        if is_us_code(stock_code):
            raise DataFetchError(f"BaostockFetcher 不支持美股 {stock_code}，请使用 AkshareFetcher 或 YfinanceFetcher")

        bs_code = self._convert_stock_code(stock_code)

        logger.debug(f"调用 Baostock query_history_k_data_plus({bs_code}, {start_date}, {end_date})")

        async with self._baostock_session() as bs:
            try:
                await self._enforce_rate_limit()

                def _call_api():
                    return bs.query_history_k_data_plus(
                        code=bs_code,
                        fields="date,open,high,low,close,volume,amount,pctChg",
                        start_date=start_date,
                        end_date=end_date,
                        frequency="d",
                        adjustflag="2",
                    )

                rs = await asyncio.to_thread(_call_api)

                if rs.error_code != "0":
                    raise DataFetchError(f"Baostock 查询失败: {rs.error_msg}")

                data_list = []
                while rs.next():
                    data_list.append(rs.get_row_data())

                if not data_list:
                    raise DataFetchError(f"Baostock 未查询到 {stock_code} 的数据")

                df = pd.DataFrame(data_list, columns=rs.fields)

                return df

            except Exception as e:
                if isinstance(e, DataFetchError):
                    raise
                raise DataFetchError(f"Baostock 获取数据失败: {e}") from e

    def _normalize_data(self, df: pd.DataFrame, stock_code: str) -> pd.DataFrame:
        df = df.copy()

        column_mapping = {
            "pctChg": "pct_chg",
        }

        df = df.rename(columns=column_mapping)

        numeric_cols = ["open", "high", "low", "close", "volume", "amount", "pct_chg"]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        df["code"] = stock_code

        keep_cols = ["code"] + STANDARD_COLUMNS
        existing_cols = [col for col in keep_cols if col in df.columns]
        df = df[existing_cols]

        return df

    async def get_stock_name(self, stock_code: str) -> str | None:
        if stock_code in self._stock_name_cache:
            return self._stock_name_cache[stock_code]

        try:
            bs_code = self._convert_stock_code(stock_code)

            async with self._baostock_session() as bs:
                await self._enforce_rate_limit()

                def _call_api():
                    return bs.query_stock_basic(code=bs_code)

                rs = await asyncio.to_thread(_call_api)

                if rs.error_code == "0":
                    data_list = []
                    while rs.next():
                        data_list.append(rs.get_row_data())

                    if data_list:
                        fields = rs.fields
                        name_idx = fields.index("code_name") if "code_name" in fields else None
                        if name_idx is not None and len(data_list[0]) > name_idx:
                            name = data_list[0][name_idx]
                            self._stock_name_cache[stock_code] = name
                            logger.debug(f"Baostock 获取股票名称成功: {stock_code} -> {name}")
                            return name

        except Exception as e:
            logger.warning(f"Baostock 获取股票名称失败 {stock_code}: {e}")

        return None

    async def get_stock_list(self) -> pd.DataFrame | None:
        try:
            async with self._baostock_session() as bs:
                await self._enforce_rate_limit()

                def _call_api():
                    return bs.query_stock_basic()

                rs = await asyncio.to_thread(_call_api)

                if rs.error_code == "0":
                    data_list = []
                    while rs.next():
                        data_list.append(rs.get_row_data())

                    if data_list:
                        df = pd.DataFrame(data_list, columns=rs.fields)

                        df["code"] = df["code"].apply(lambda x: x.split(".")[1] if "." in x else x)
                        df = df.rename(columns={"code_name": "name"})

                        for _, row in df.iterrows():
                            self._stock_name_cache[row["code"]] = row["name"]

                        logger.info(f"Baostock 获取股票列表成功: {len(df)} 条")
                        return df[["code", "name"]]

        except Exception as e:
            logger.warning(f"Baostock 获取股票列表失败: {e}")

        return None
