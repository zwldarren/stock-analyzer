"""
数据源基类与管理器

设计模式：策略模式 (Strategy Pattern)
- BaseFetcher: 抽象基类，定义统一接口
- DataFetcherManager: 策略管理器，实现自动切换

职责边界：
- 只负责外部数据获取和故障切换
- 不包含缓存逻辑（由 DataService 处理）
- 不包含指标计算（由 IndicatorService 处理）
"""

import logging
import random
import time
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any

import pandas as pd

from stock_analyzer.domain.exceptions import DataFetchError
from stock_analyzer.utils.stock_code import is_us_code

logger = logging.getLogger(__name__)

# 标准列名定义
STANDARD_COLUMNS = ["date", "open", "high", "low", "close", "volume", "amount", "pct_chg"]

# User-Agent 池，用于随机轮换（共享）
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
    数据源抽象基类

    职责：
    1. 定义统一的数据获取接口
    2. 提供数据标准化方法
    3. 提供共享的反爬策略（User-Agent轮换、速率限制）

    子类实现：
    - _fetch_raw_data(): 从具体数据源获取原始数据
    - _normalize_data(): 将原始数据转换为标准格式
    """

    name: str = "BaseFetcher"
    priority: int = 99  # 优先级数字越小越优先

    def __init__(self, sleep_min: float = 1.5, sleep_max: float = 3.0):
        """
        初始化 BaseFetcher

        Args:
            sleep_min: 最小休眠时间（秒）
            sleep_max: 最大休眠时间（秒）
        """
        self.sleep_min = sleep_min
        self.sleep_max = sleep_max
        self._last_request_time: float | None = None

    def _set_random_user_agent(self) -> None:
        """
        设置随机 User-Agent

        这是关键的反爬策略之一
        """
        try:
            random_ua = random.choice(USER_AGENTS)
            logger.debug(f"设置 User-Agent: {random_ua[:50]}...")
        except Exception as e:
            logger.debug(f"设置 User-Agent 失败: {e}")

    def _enforce_rate_limit(self) -> None:
        """
        强制执行速率限制

        策略：
        1. 检查距离上次请求的时间间隔
        2. 如果间隔不足，补充休眠时间
        3. 然后再执行随机 jitter 休眠
        """
        if self._last_request_time is not None:
            elapsed = time.time() - self._last_request_time
            min_interval = self.sleep_min
            if elapsed < min_interval:
                additional_sleep = min_interval - elapsed
                logger.debug(f"补充休眠 {additional_sleep:.2f} 秒")
                time.sleep(additional_sleep)

        # 执行随机 jitter 休眠
        sleep_time = random.uniform(self.sleep_min, self.sleep_max)
        logger.debug(f"随机休眠 {sleep_time:.2f} 秒...")
        time.sleep(sleep_time)
        self._last_request_time = time.time()

    @abstractmethod
    def _fetch_raw_data(self, stock_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        从数据源获取原始数据（子类必须实现）

        Args:
            stock_code: 股票代码，如 '600519', '000001'
            start_date: 开始日期，格式 'YYYY-MM-DD'
            end_date: 结束日期，格式 'YYYY-MM-DD'

        Returns:
            原始数据 DataFrame（列名因数据源而异）
        """

    @abstractmethod
    def _normalize_data(self, df: pd.DataFrame, stock_code: str) -> pd.DataFrame:
        """
        标准化数据列名（子类必须实现）

        将不同数据源的列名统一为：
        ['date', 'open', 'high', 'low', 'close', 'volume', 'amount', 'pct_chg']
        """

    def get_daily_data(
        self,
        stock_code: str,
        start_date: str | None = None,
        end_date: str | None = None,
        days: int = 30,
    ) -> pd.DataFrame:
        """
        获取日线数据（统一入口）

        流程：
        1. 计算日期范围
        2. 调用子类获取原始数据
        3. 标准化列名

        Args:
            stock_code: 股票代码
            start_date: 开始日期（可选）
            end_date: 结束日期（可选，默认今天）
            days: 获取天数（当 start_date 未指定时使用）

        Returns:
            标准化的 DataFrame（不包含技术指标，由 IndicatorService 计算）
        """
        # 计算日期范围
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")

        if start_date is None:
            from datetime import timedelta

            start_dt = datetime.strptime(end_date, "%Y-%m-%d") - timedelta(days=days * 2)
            start_date = start_dt.strftime("%Y-%m-%d")

        logger.info(f"[{self.name}] 获取 {stock_code} 数据: {start_date} ~ {end_date}")

        try:
            # Step 1: 获取原始数据
            raw_df = self._fetch_raw_data(stock_code, start_date, end_date)

            if raw_df is None or raw_df.empty:
                raise DataFetchError(f"[{self.name}] 未获取到 {stock_code} 的数据")

            # Step 2: 标准化列名
            df = self._normalize_data(raw_df, stock_code)

            logger.info(f"[{self.name}] {stock_code} 获取成功，共 {len(df)} 条数据")
            return df

        except Exception as e:
            logger.error(f"[{self.name}] 获取 {stock_code} 失败: {str(e)}")
            raise DataFetchError(f"[{self.name}] {stock_code}: {str(e)}") from e

    def get_main_indices(self) -> list[dict[str, Any]] | None:
        """
        获取主要指数实时行情

        Returns:
            指数列表，每个元素为字典
        """
        return None

    def get_market_stats(self) -> dict[str, Any] | None:
        """
        获取市场涨跌统计

        Returns:
            包含涨跌家数统计
        """
        return None

    def get_sector_rankings(self, n: int = 5) -> tuple[list[dict], list[dict]] | None:
        """
        获取板块涨跌榜

        Args:
            n: 返回前n个

        Returns:
            Tuple: (领涨板块列表, 领跌板块列表)
        """
        return None

    def get_realtime_quote(self, stock_code: str, **kwargs):
        """
        获取实时行情数据

        Args:
            stock_code: 股票代码
            **kwargs: 额外参数（如 source）

        Returns:
            UnifiedRealtimeQuote 对象，不支持返回 None
        """
        return None

    def get_chip_distribution(self, stock_code: str):
        """
        获取筹码分布数据

        Args:
            stock_code: 股票代码

        Returns:
            ChipDistribution 对象，不支持返回 None
        """
        return None

    def get_stock_name(self, stock_code: str) -> str | None:
        """
        获取股票中文名称

        Args:
            stock_code: 股票代码

        Returns:
            股票中文名称，不支持返回 None
        """
        return None

    def get_stock_list(self) -> pd.DataFrame | None:
        """
        获取股票列表

        Returns:
            股票列表 DataFrame，不支持返回 None
        """
        return None


class DataFetcherManager:
    """
    数据源策略管理器

    职责：
    1. 管理多个数据源（按优先级排序）
    2. 自动故障切换（Failover）
    3. 提供统一的数据获取接口

    注意：
    - 不包含缓存逻辑（由 DataService 处理）
    - 不包含指标计算（由 IndicatorService 处理）
    - 实现 IDataFetcher 接口，供领域层使用
    """

    def __init__(self, fetchers: list[BaseFetcher] | None = None):
        """
        初始化管理器

        Args:
            fetchers: 数据源列表（可选，默认按优先级自动创建）
        """
        self._fetchers: list[BaseFetcher] = []

        if fetchers:
            self._fetchers = sorted(fetchers, key=lambda f: f.priority)
        else:
            self._init_default_fetchers()

    def _init_default_fetchers(self) -> None:
        """初始化默认数据源列表"""
        from .akshare_fetcher import AkshareFetcher
        from .baostock_fetcher import BaostockFetcher
        from .efinance_fetcher import EfinanceFetcher
        from .pytdx_fetcher import PytdxFetcher
        from .tushare_fetcher import TushareFetcher
        from .yfinance_fetcher import YfinanceFetcher

        fetchers = [
            EfinanceFetcher(),
            AkshareFetcher(),
            TushareFetcher(),
            PytdxFetcher(),
            BaostockFetcher(),
            YfinanceFetcher(),
        ]

        # 按优先级排序
        self._fetchers = sorted(fetchers, key=lambda f: f.priority)

        priority_info = ", ".join([f"{f.name}(P{f.priority})" for f in self._fetchers])
        logger.info(f"已初始化 {len(self._fetchers)} 个数据源: {priority_info}")

    def add_fetcher(self, fetcher: BaseFetcher) -> None:
        """添加数据源并重新排序"""
        self._fetchers.append(fetcher)
        self._fetchers.sort(key=lambda f: f.priority)

    def get_daily_data(
        self,
        stock_code: str,
        start_date: str | None = None,
        end_date: str | None = None,
        days: int = 30,
    ) -> tuple[pd.DataFrame, str]:
        """
        获取日线数据（自动切换数据源）

        Args:
            stock_code: 股票代码
            start_date: 开始日期
            end_date: 结束日期
            days: 获取天数

        Returns:
            (数据, 成功的数据源名称)

        Raises:
            DataFetchError: 所有数据源都失败时抛出
        """
        errors = []

        for fetcher in self._fetchers:
            try:
                logger.info(f"尝试使用 [{fetcher.name}] 获取 {stock_code}...")
                df = fetcher.get_daily_data(
                    stock_code=stock_code,
                    start_date=start_date,
                    end_date=end_date,
                    days=days,
                )

                if df is not None and not df.empty:
                    logger.info(f"[{fetcher.name}] 成功获取 {stock_code}")
                    return df, fetcher.name

            except Exception as e:
                error_msg = f"[{fetcher.name}] 失败: {str(e)}"
                logger.warning(error_msg)
                errors.append(error_msg)
                continue

        # 所有数据源都失败
        error_summary = f"所有数据源获取 {stock_code} 失败:\n" + "\n".join(errors)
        logger.error(error_summary)
        raise DataFetchError(error_summary)

    def get_realtime_quote(self, stock_code: str):
        """
        获取实时行情数据（自动故障切换）

        Args:
            stock_code: 股票代码

        Returns:
            UnifiedRealtimeQuote 对象，失败返回 None
        """
        from stock_analyzer.config import get_config

        config = get_config()

        # 如果实时行情功能被禁用，直接返回 None
        if not config.realtime_quote.enable_realtime_quote:
            logger.debug(f"[实时行情] 功能已禁用，跳过 {stock_code}")
            return None

        # 美股单独处理
        if is_us_code(stock_code):
            for fetcher in self._fetchers:
                if fetcher.name == "YfinanceFetcher":
                    try:
                        quote = fetcher.get_realtime_quote(stock_code)
                        if quote is not None:
                            return quote
                    except Exception as e:
                        logger.warning(f"[实时行情] 美股 {stock_code} 获取失败: {e}")
                    break
            return None

        # 获取配置的数据源优先级
        source_priority = config.realtime_quote.realtime_source_priority.split(",")

        for source in source_priority:
            source = source.strip().lower()

            try:
                quote = self._try_get_realtime_by_source(stock_code, source)
                if quote is not None and hasattr(quote, "has_basic_data") and quote.has_basic_data():
                    logger.info(f"[实时行情] {stock_code} 成功获取 (来源: {source})")
                    return quote
            except Exception as e:
                logger.warning(f"[实时行情] [{source}] 失败: {e}")
                continue

        logger.warning(f"[实时行情] {stock_code} 所有数据源均失败")
        return None

    # 数据源映射配置: (source_key) -> (fetcher_name, extra_kwargs)
    _REALTIME_SOURCE_MAP: dict[str, tuple[str, dict[str, Any]]] = {
        "efinance": ("EfinanceFetcher", {}),
        "akshare_em": ("AkshareFetcher", {"source": "em"}),
        "akshare_sina": ("AkshareFetcher", {"source": "sina"}),
        "tencent": ("AkshareFetcher", {"source": "tencent"}),
        "akshare_qq": ("AkshareFetcher", {"source": "tencent"}),
        "tushare": ("TushareFetcher", {}),
    }

    def _try_get_realtime_by_source(self, stock_code: str, source: str):
        """尝试从指定数据源获取实时行情"""
        mapping = self._REALTIME_SOURCE_MAP.get(source)
        if not mapping:
            return None

        fetcher_name, extra_kwargs = mapping
        for fetcher in self._fetchers:
            if fetcher.name == fetcher_name:
                return fetcher.get_realtime_quote(stock_code, **extra_kwargs)
        return None

    def get_chip_distribution(self, stock_code: str):
        """
        获取筹码分布数据（带熔断和多数据源降级）

        Args:
            stock_code: 股票代码

        Returns:
            ChipDistribution 对象，失败返回 None
        """
        from stock_analyzer.config import get_config
        from stock_analyzer.infrastructure.data_sources.realtime_types import get_chip_circuit_breaker

        config = get_config()

        if not config.realtime_quote.enable_chip_distribution:
            return None

        circuit_breaker = get_chip_circuit_breaker()

        chip_sources = [
            ("AkshareFetcher", "akshare_chip"),
            ("TushareFetcher", "tushare_chip"),
            ("EfinanceFetcher", "efinance_chip"),
        ]

        for fetcher_name, source_key in chip_sources:
            if not circuit_breaker.is_available(source_key):
                continue

            try:
                for fetcher in self._fetchers:
                    if fetcher.name == fetcher_name:
                        chip = fetcher.get_chip_distribution(stock_code)
                        if chip is not None:
                            circuit_breaker.record_success(source_key)
                            return chip
                        break
            except Exception as e:
                logger.warning(f"[筹码分布] {fetcher_name} 获取 {stock_code} 失败: {e}")
                circuit_breaker.record_failure(source_key, str(e))
                continue

        return None

    def get_stock_name(self, stock_code: str) -> str | None:
        """
        获取股票中文名称

        Args:
            stock_code: 股票代码

        Returns:
            股票中文名称，失败返回 None
        """
        # 依次尝试各个数据源
        for fetcher in self._fetchers:
            if hasattr(fetcher, "get_stock_name"):
                try:
                    name = fetcher.get_stock_name(stock_code)
                    if name:
                        return name
                except Exception as e:
                    logger.debug(f"Failed to get stock name from {fetcher.name}: {e}")
                    continue

        return None

    def batch_get_stock_names(self, stock_codes: list[str]) -> dict[str, str]:
        """
        批量获取股票中文名称

        Args:
            stock_codes: 股票代码列表

        Returns:
            {股票代码: 股票名称} 字典
        """
        result = {}
        missing_codes = set(stock_codes)

        # 1. 尝试批量获取股票列表
        for fetcher in self._fetchers:
            if hasattr(fetcher, "get_stock_list") and missing_codes:
                try:
                    stock_list = fetcher.get_stock_list()
                    if stock_list is not None and not stock_list.empty:
                        for _, row in stock_list.iterrows():
                            code = row.get("code")
                            name = row.get("name")
                            if code and name and code in missing_codes:
                                result[code] = name
                                missing_codes.discard(code)

                        if not missing_codes:
                            break
                except Exception as e:
                    logger.debug(f"Failed to get stock names from {fetcher.name}: {e}")
                    continue

        # 2. 逐个获取剩余的
        for code in list(missing_codes):
            name = self.get_stock_name(code)
            if name:
                result[code] = name

        return result

    def get_main_indices(self) -> list[dict[str, Any]]:
        """获取主要指数实时行情"""
        for fetcher in self._fetchers:
            try:
                result = fetcher.get_main_indices()
                if result:
                    return result
            except Exception as e:
                logger.warning(f"[{fetcher.name}] get_main_indices 失败: {e}")
        return []

    def prefetch_realtime_quotes(self, stock_codes: list[str]) -> int:
        """
        批量预取实时行情数据

        策略：
        1. 检查优先级中是否包含全量拉取数据源（efinance/akshare_em）
        2. 如果自选股数量 >= 5 且使用全量数据源，则预取填充缓存

        Args:
            stock_codes: 待分析的股票代码列表

        Returns:
            预取的股票数量（0 表示跳过预取）
        """
        from stock_analyzer.config import get_config

        config = get_config()

        # 如果实时行情被禁用，跳过预取
        if not config.realtime_quote.enable_realtime_quote:
            return 0

        # 检查优先级中是否包含全量拉取数据源
        priority = config.realtime_quote.realtime_source_priority.lower()
        bulk_sources = ["efinance", "akshare_em", "tushare"]

        priority_list = [s.strip() for s in priority.split(",")]
        first_bulk_source_index = None
        for i, source in enumerate(priority_list):
            if source in bulk_sources:
                first_bulk_source_index = i
                break

        # 如果没有全量数据源，或者全量数据源排在第 3 位之后，跳过预取
        if first_bulk_source_index is None or first_bulk_source_index >= 2:
            return 0

        # 如果股票数量少于 5 个，不进行批量预取
        if len(stock_codes) < 5:
            return 0

        logger.info(f"[预取] 开始批量预取实时行情，共 {len(stock_codes)} 只股票...")

        # 尝试用第一只股票触发全量拉取
        try:
            first_code = stock_codes[0]
            quote = self.get_realtime_quote(first_code)

            if quote:
                logger.info("[预取] 批量预取完成，缓存已填充")
                return len(stock_codes)
            else:
                logger.warning("[预取] 批量预取失败，将使用逐个查询模式")
                return 0

        except Exception as e:
            logger.error(f"[预取] 批量预取异常: {e}")
            return 0

    def get_market_stats(self) -> dict[str, Any]:
        """获取市场涨跌统计"""
        for fetcher in self._fetchers:
            try:
                result = fetcher.get_market_stats()
                if result:
                    return result
            except Exception as e:
                logger.warning(f"[{fetcher.name}] get_market_stats 失败: {e}")
        return {}

    def get_sector_rankings(self, n: int = 5) -> tuple[list[dict], list[dict]]:
        """获取板块涨跌榜"""
        for fetcher in self._fetchers:
            try:
                result = fetcher.get_sector_rankings(n)
                if result:
                    return result
            except Exception as e:
                logger.warning(f"[{fetcher.name}] get_sector_rankings 失败: {e}")
        return ([], [])

    def get_base_info(self, stock_code: str) -> dict[str, Any] | None:
        """
        获取股票基本信息

        依次尝试各个数据源获取股票基本信息，包含市盈率、市净率、所处行业等

        Args:
            stock_code: 股票代码

        Returns:
            包含基本信息的字典，获取失败返回 None
        """
        for fetcher in self._fetchers:
            # 检查fetcher是否有get_base_info方法且可调用
            base_info_method = getattr(fetcher, "get_base_info", None)
            if base_info_method is not None and callable(base_info_method):
                try:
                    info = base_info_method(stock_code)
                    if info:
                        logger.info(f"[{fetcher.name}] 成功获取 {stock_code} 基本信息")
                        return info
                except Exception as e:
                    logger.debug(f"[{fetcher.name}] 获取 {stock_code} 基本信息失败: {e}")
                    continue

        logger.warning(f"所有数据源都无法获取 {stock_code} 的基本信息")
        return None

    @property
    def available_fetchers(self) -> list[str]:
        """返回可用数据源名称列表"""
        return [f.name for f in self._fetchers]
