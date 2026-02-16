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

import logging
from collections.abc import Generator
from contextlib import contextmanager

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
from stock_analyzer.utils.stock_code import convert_to_provider_format, is_us_code

logger = logging.getLogger(__name__)


class BaostockFetcher(BaseFetcher):
    """
    Baostock 数据源实现

    优先级：3
    数据来源：证券宝 Baostock API

    关键策略：
    - 使用上下文管理器管理连接生命周期
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

    def __init__(self):
        """Initialize BaostockFetcher."""
        self._bs_module = None
        # Stock name cache with LRU eviction policy
        self._stock_name_cache: LRUCache[str, str] = LRUCache(maxsize=1000)

    def _get_baostock(self):
        """
        延迟加载 baostock 模块

        只在首次使用时导入，避免未安装时报错
        """
        if self._bs_module is None:
            import baostock as bs

            self._bs_module = bs
        return self._bs_module

    @contextmanager
    def _baostock_session(self) -> Generator:
        """
        Baostock 连接上下文管理器

        确保：
        1. 进入上下文时自动登录
        2. 退出上下文时自动登出
        3. 异常时也能正确登出

        使用示例：
            with self._baostock_session():
                # 在这里执行数据查询
        """
        bs = self._get_baostock()
        login_result = None

        try:
            # 登录 Baostock
            login_result = bs.login()

            if login_result.error_code != "0":
                raise DataFetchError(f"Baostock 登录失败: {login_result.error_msg}")

            logger.debug("Baostock 登录成功")

            yield bs

        finally:
            # 确保登出，防止连接泄露
            try:
                logout_result = bs.logout()
                if logout_result.error_code == "0":
                    logger.debug("Baostock 登出成功")
                else:
                    logger.warning(f"Baostock 登出异常: {logout_result.error_msg}")
            except Exception as e:
                logger.warning(f"Baostock 登出时发生错误: {e}")

    def _convert_stock_code(self, stock_code: str) -> str:
        """
        Convert stock code to Baostock format.

        Baostock required format:
        - Shanghai: sh.600519
        - Shenzhen: sz.000001

        Args:
            stock_code: Original code (e.g., '600519', '000001')

        Returns:
            Baostock formatted code (e.g., 'sh.600519', 'sz.000001')
        """
        code = stock_code.strip()

        # Already has prefix
        if code.startswith(("sh.", "sz.")):
            return code.lower()

        # Use unified conversion function
        return convert_to_provider_format(stock_code, "baostock")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type((ConnectionError, TimeoutError)),
        before_sleep=before_sleep_log(logger, logging.WARNING),  # type: ignore[arg-type]
    )
    def _fetch_raw_data(self, stock_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        从 Baostock 获取原始数据

        使用 query_history_k_data_plus() 获取日线数据

        流程：
        1. 检查是否为美股（不支持）
        2. 使用上下文管理器管理连接
        3. 转换股票代码格式
        4. 调用 API 查询数据
        5. 将结果转换为 DataFrame
        """
        # 美股不支持，抛出异常让 DataFetcherManager 切换到其他数据源
        if is_us_code(stock_code):
            raise DataFetchError(f"BaostockFetcher 不支持美股 {stock_code}，请使用 AkshareFetcher 或 YfinanceFetcher")

        # 转换代码格式
        bs_code = self._convert_stock_code(stock_code)

        logger.debug(f"调用 Baostock query_history_k_data_plus({bs_code}, {start_date}, {end_date})")

        with self._baostock_session() as bs:
            try:
                # 查询日线数据
                # adjustflag: 1-后复权，2-前复权，3-不复权
                rs = bs.query_history_k_data_plus(
                    code=bs_code,
                    fields="date,open,high,low,close,volume,amount,pctChg",
                    start_date=start_date,
                    end_date=end_date,
                    frequency="d",  # 日线
                    adjustflag="2",  # 前复权
                )

                if rs.error_code != "0":
                    raise DataFetchError(f"Baostock 查询失败: {rs.error_msg}")

                # 转换为 DataFrame
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
        """
        标准化 Baostock 数据

        Baostock 返回的列名：
        date, open, high, low, close, volume, amount, pctChg

        需要映射到标准列名：
        date, open, high, low, close, volume, amount, pct_chg
        """
        df = df.copy()

        # 列名映射（只需要处理 pctChg）
        column_mapping = {
            "pctChg": "pct_chg",
        }

        df = df.rename(columns=column_mapping)

        # 数值类型转换（Baostock 返回的都是字符串）
        numeric_cols = ["open", "high", "low", "close", "volume", "amount", "pct_chg"]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # 添加股票代码列
        df["code"] = stock_code

        # 只保留需要的列
        keep_cols = ["code"] + STANDARD_COLUMNS
        existing_cols = [col for col in keep_cols if col in df.columns]
        df = df[existing_cols]

        return df

    def get_stock_name(self, stock_code: str) -> str | None:
        """Get stock name using Baostock query_stock_basic API.

        Args:
            stock_code: Stock code

        Returns:
            Stock name, None if failed
        """
        # Check cache
        if stock_code in self._stock_name_cache:
            return self._stock_name_cache[stock_code]

        try:
            bs_code = self._convert_stock_code(stock_code)

            with self._baostock_session() as bs:
                # Query stock basic info
                rs = bs.query_stock_basic(code=bs_code)

                if rs.error_code == "0":
                    data_list = []
                    while rs.next():
                        data_list.append(rs.get_row_data())

                    if data_list:
                        # Baostock returns fields: code, code_name, ipoDate, outDate, type, status
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

    def get_stock_list(self) -> pd.DataFrame | None:
        """Get stock list using Baostock query_stock_basic API.

        Returns:
            DataFrame with code and name columns, None if failed
        """
        try:
            with self._baostock_session() as bs:
                # Query all stock basic info
                rs = bs.query_stock_basic()

                if rs.error_code == "0":
                    data_list = []
                    while rs.next():
                        data_list.append(rs.get_row_data())

                    if data_list:
                        df = pd.DataFrame(data_list, columns=rs.fields)

                        # Convert code format (remove sh. or sz. prefix)
                        df["code"] = df["code"].apply(lambda x: x.split(".")[1] if "." in x else x)
                        df = df.rename(columns={"code_name": "name"})

                        # Update cache
                        for _, row in df.iterrows():
                            self._stock_name_cache[row["code"]] = row["name"]

                        logger.info(f"Baostock 获取股票列表成功: {len(df)} 条")
                        return df[["code", "name"]]

        except Exception as e:
            logger.warning(f"Baostock 获取股票列表失败: {e}")

        return None
