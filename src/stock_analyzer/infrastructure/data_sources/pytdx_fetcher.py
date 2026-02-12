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

from stock_analyzer.utils.stock_code import is_us_code

from .base import STANDARD_COLUMNS, BaseFetcher, DataFetchError

logger = logging.getLogger(__name__)


class PytdxFetcher(BaseFetcher):
    """
    通达信数据源实现

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

    # 默认通达信行情服务器列表
    DEFAULT_HOSTS = [
        ("119.147.212.81", 7709),  # 深圳
        ("112.74.214.43", 7727),  # 深圳
        ("221.231.141.60", 7709),  # 上海
        ("101.227.73.20", 7709),  # 上海
        ("101.227.77.254", 7709),  # 上海
        ("14.215.128.18", 7709),  # 广州
        ("59.173.18.140", 7709),  # 武汉
        ("180.153.39.51", 7709),  # 杭州
    ]

    def __init__(self, hosts: list[tuple[str, int]] | None = None):
        """
        初始化 PytdxFetcher

        Args:
            hosts: 服务器列表 [(host, port), ...]，默认使用内置列表
        """
        self._hosts = hosts or self.DEFAULT_HOSTS
        self._api = None
        self._connected = False
        self._current_host_idx = 0
        self._stock_list_cache: dict[str, str] | None = None  # Stock list cache
        self._stock_name_cache: LRUCache[str, str] = LRUCache(maxsize=1000)  # Stock name cache with LRU eviction

    def _get_pytdx(self):
        """
        延迟加载 pytdx 模块

        只在首次使用时导入，避免未安装时报错
        """
        try:
            from pytdx.hq import TdxHq_API

            return TdxHq_API
        except ImportError:
            logger.warning("pytdx 未安装，请运行: pip install pytdx")
            return None

    @contextmanager
    def _pytdx_session(self) -> Generator:
        """
        Pytdx 连接上下文管理器

        确保：
        1. 进入上下文时自动连接
        2. 退出上下文时自动断开
        3. 异常时也能正确断开

        使用示例：
            with self._pytdx_session() as api:
                # 在这里执行数据查询
        """
        TdxHq_API = self._get_pytdx()
        if TdxHq_API is None:
            raise DataFetchError("pytdx 库未安装")

        api = TdxHq_API()
        connected = False

        try:
            # 尝试连接服务器（自动选择最优）
            for i in range(len(self._hosts)):
                host_idx = (self._current_host_idx + i) % len(self._hosts)
                host, port = self._hosts[host_idx]

                try:
                    if api.connect(host, port, time_out=5):
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
            # 确保断开连接
            try:
                api.disconnect()
                logger.debug("Pytdx 连接已断开")
            except Exception as e:
                logger.warning(f"Pytdx 断开连接时出错: {e}")

    def _get_market_code(self, stock_code: str) -> tuple[int, str]:
        """
        根据股票代码判断市场

        Pytdx 市场代码：
        - 0: 深圳
        - 1: 上海

        Args:
            stock_code: 股票代码

        Returns:
            (market, code) 元组
        """
        code = stock_code.strip()

        # 去除可能的前缀后缀
        code = code.replace(".SH", "").replace(".SZ", "")
        code = code.replace(".sh", "").replace(".sz", "")
        code = code.replace("sh", "").replace("sz", "")

        # 根据代码前缀判断市场
        # 上海：60xxxx, 68xxxx（科创板）
        # 深圳：00xxxx, 30xxxx（创业板）, 002xxx（中小板）
        if code.startswith(("60", "68")):
            return 1, code  # 上海
        else:
            return 0, code  # 深圳

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type((ConnectionError, TimeoutError)),
        before_sleep=before_sleep_log(logger, logging.WARNING),  # type: ignore[arg-type]
    )
    def _fetch_raw_data(self, stock_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        从通达信获取原始数据

        使用 get_security_bars() 获取日线数据

        流程：
        1. 检查是否为美股（不支持）
        2. 使用上下文管理器管理连接
        3. 判断市场代码
        4. 调用 API 获取 K 线数据
        """
        # 美股不支持，抛出异常让 DataFetcherManager 切换到其他数据源
        if is_us_code(stock_code):
            raise DataFetchError(f"PytdxFetcher 不支持美股 {stock_code}，请使用 AkshareFetcher 或 YfinanceFetcher")

        market, code = self._get_market_code(stock_code)

        # 计算需要获取的交易日数量（估算）
        from datetime import datetime as dt

        start_dt = dt.strptime(start_date, "%Y-%m-%d")
        end_dt = dt.strptime(end_date, "%Y-%m-%d")
        days = (end_dt - start_dt).days
        count = min(max(days * 5 // 7 + 10, 30), 800)  # 估算交易日，最大 800 条

        logger.debug(f"调用 Pytdx get_security_bars(market={market}, code={code}, count={count})")

        with self._pytdx_session() as api:
            try:
                # 获取日 K 线数据
                # category: 9-日线, 0-5分钟, 1-15分钟, 2-30分钟, 3-1小时
                data = api.get_security_bars(
                    category=9,  # 日线
                    market=market,
                    code=code,
                    start=0,  # 从最新开始
                    count=count,
                )

                if data is None or len(data) == 0:
                    raise DataFetchError(f"Pytdx 未查询到 {stock_code} 的数据")

                # 转换为 DataFrame
                df = api.to_df(data)

                # 过滤日期范围
                df["datetime"] = pd.to_datetime(df["datetime"])
                df = df[(df["datetime"] >= start_date) & (df["datetime"] <= end_date)]

                return df

            except Exception as e:
                if isinstance(e, DataFetchError):
                    raise
                raise DataFetchError(f"Pytdx 获取数据失败: {e}") from e

    def _normalize_data(self, df: pd.DataFrame, stock_code: str) -> pd.DataFrame:
        """
        标准化 Pytdx 数据

        Pytdx 返回的列名：
        datetime, open, high, low, close, vol, amount

        需要映射到标准列名：
        date, open, high, low, close, volume, amount, pct_chg
        """
        df = df.copy()

        # 列名映射
        column_mapping = {
            "datetime": "date",
            "vol": "volume",
        }

        df = df.rename(columns=column_mapping)

        # 计算涨跌幅（pytdx 不返回涨跌幅，需要自己计算）
        if "pct_chg" not in df.columns and "close" in df.columns:
            df["pct_chg"] = df["close"].pct_change() * 100
            df["pct_chg"] = df["pct_chg"].fillna(0).round(2)

        # 添加股票代码列
        df["code"] = stock_code

        # 只保留需要的列
        keep_cols = ["code"] + STANDARD_COLUMNS
        existing_cols = [col for col in keep_cols if col in df.columns]
        df = df[existing_cols]

        return df

    def get_stock_name(self, stock_code: str) -> str | None:
        """Get stock name.

        Args:
            stock_code: Stock code

        Returns:
            Stock name, None if failed
        """
        # Check cache first
        if stock_code in self._stock_name_cache:
            return self._stock_name_cache[stock_code]

        try:
            market, code = self._get_market_code(stock_code)

            with self._pytdx_session() as api:
                # Get stock list (cached)
                if self._stock_list_cache is None:
                    # Get Shenzhen and Shanghai stock lists
                    sz_stocks = api.get_security_list(0, 0)  # Shenzhen
                    sh_stocks = api.get_security_list(1, 0)  # Shanghai

                    self._stock_list_cache = {}
                    for stock in (sz_stocks or []) + (sh_stocks or []):
                        self._stock_list_cache[stock["code"]] = stock["name"]

                # Look up stock name
                name = self._stock_list_cache.get(code)
                if name:
                    self._stock_name_cache[stock_code] = name
                    return name

                # Try using get_finance_info
                finance_info = api.get_finance_info(market, code)
                if finance_info and "name" in finance_info:
                    name = finance_info["name"]
                    self._stock_name_cache[stock_code] = name
                    return name

        except Exception as e:
            logger.warning(f"Pytdx 获取股票名称失败 {stock_code}: {e}")

        return None

    def get_realtime_quote(self, stock_code: str, **kwargs) -> dict | None:
        """
        获取实时行情

        Args:
            stock_code: 股票代码

        Returns:
            实时行情数据字典，失败返回 None
        """
        try:
            market, code = self._get_market_code(stock_code)

            with self._pytdx_session() as api:
                data = api.get_security_quotes([(market, code)])

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
