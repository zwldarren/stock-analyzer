"""
===================================
AkshareFetcher - 主数据源 (Priority 1)
===================================

数据来源：
1. 东方财富爬虫（通过 akshare 库） - 默认数据源
2. 新浪财经接口 - 备选数据源
3. 腾讯财经接口 - 备选数据源

特点：免费、无需 Token、数据全面
风险：爬虫机制易被反爬封禁

防封禁策略：
1. 每次请求前随机休眠 2-5 秒
2. 随机轮换 User-Agent
3. 使用 tenacity 实现指数退避重试
4. 熔断器机制：连续失败后自动冷却

增强数据：
- 实时行情：量比、换手率、市盈率、市净率、总市值、流通市值
- 筹码分布：获利比例、平均成本、筹码集中度
"""

import asyncio
import logging
import random
from typing import Any, cast

import pandas as pd
from cachetools import TTLCache
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from ashare_analyzer.data.base import STANDARD_COLUMNS, USER_AGENTS, BaseFetcher
from ashare_analyzer.exceptions import DataFetchError, RateLimitError
from ashare_analyzer.infrastructure import AsyncRateLimiter, get_aiohttp_session
from ashare_analyzer.models import ChipDistribution, RealtimeSource, UnifiedRealtimeQuote
from ashare_analyzer.utils.stock_code import is_etf_code, is_hk_code, is_us_code

from .realtime_types import get_realtime_circuit_breaker, safe_float, safe_int

logger = logging.getLogger(__name__)

_realtime_cache: TTLCache[str, Any] = TTLCache(maxsize=1, ttl=1200)
_etf_realtime_cache: TTLCache[str, Any] = TTLCache(maxsize=1, ttl=1200)


class AkshareFetcher(BaseFetcher):
    """
    Akshare 数据源实现 (async)

    优先级：1（最高）
    数据来源：东方财富网爬虫

    关键策略：
    - 使用 AsyncRateLimiter 限流
    - 随机 User-Agent 轮换
    - 失败后指数退避重试（最多3次）
    """

    name = "AkshareFetcher"

    @property
    def priority(self) -> int:
        from ashare_analyzer.config import get_config

        return get_config().datasource.akshare_priority

    def __init__(self, rate_limiter: AsyncRateLimiter | None = None):
        super().__init__(rate_limiter=rate_limiter)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type((ConnectionError, TimeoutError)),
        before_sleep=before_sleep_log(cast(Any, logger), logging.WARNING),
    )
    async def _fetch_raw_data(self, stock_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        if is_us_code(stock_code):
            return await self._fetch_us_data(stock_code, start_date, end_date)
        elif is_hk_code(stock_code):
            return await self._fetch_hk_data(stock_code, start_date, end_date)
        elif is_etf_code(stock_code):
            return await self._fetch_etf_data(stock_code, start_date, end_date)
        else:
            return await self._fetch_stock_data(stock_code, start_date, end_date)

    async def _fetch_stock_data(self, stock_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        methods = [
            (self._fetch_stock_data_em, "东方财富"),
            (self._fetch_stock_data_sina, "新浪财经"),
            (self._fetch_stock_data_tx, "腾讯财经"),
        ]

        last_error = None

        for fetch_method, source_name in methods:
            try:
                logger.debug(f"[数据源] 尝试使用 {source_name} 获取 {stock_code}...")
                df = await fetch_method(stock_code, start_date, end_date)

                if df is not None and not df.empty:
                    logger.debug(f"[数据源] {source_name} 获取成功")
                    return df
            except Exception as e:
                last_error = e
                logger.warning(f"[数据源] {source_name} 获取失败: {e}")

        raise DataFetchError(f"Akshare 所有渠道获取失败: {last_error}")

    async def _fetch_stock_data_em(self, stock_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        import akshare as ak

        self._set_random_user_agent()
        await self._enforce_rate_limit()

        logger.debug(f"[API调用] ak.stock_zh_a_hist(symbol={stock_code}, ...)")

        try:
            import time as _time

            api_start = _time.time()

            def _call_api():
                return ak.stock_zh_a_hist(
                    symbol=stock_code,
                    period="daily",
                    start_date=start_date.replace("-", ""),
                    end_date=end_date.replace("-", ""),
                    adjust="qfq",
                )

            df = await asyncio.to_thread(_call_api)
            api_elapsed = _time.time() - api_start

            if df is not None and not df.empty:
                logger.debug(f"[API返回] ak.stock_zh_a_hist 成功: {len(df)} 行, 耗时 {api_elapsed:.2f}s")
                return df
            else:
                logger.warning("[API返回] ak.stock_zh_a_hist 返回空数据")
                return pd.DataFrame()

        except Exception as e:
            error_msg = str(e).lower()
            if any(keyword in error_msg for keyword in ["banned", "blocked", "频率", "rate", "限制"]):
                raise RateLimitError(f"Akshare(EM) 可能被限流: {e}") from e
            raise e

    async def _fetch_stock_data_sina(self, stock_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        import akshare as ak

        symbol = f"sh{stock_code}" if stock_code.startswith(("6", "5", "9")) else f"sz{stock_code}"

        await self._enforce_rate_limit()

        try:

            def _call_api():
                return ak.stock_zh_a_daily(
                    symbol=symbol,
                    start_date=start_date.replace("-", ""),
                    end_date=end_date.replace("-", ""),
                    adjust="qfq",
                )

            df = await asyncio.to_thread(_call_api)

            if df is not None and not df.empty:
                if "date" in df.columns:
                    df = df.rename(columns={"date": "日期"})

                rename_map = {
                    "open": "开盘",
                    "high": "最高",
                    "low": "最低",
                    "close": "收盘",
                    "volume": "成交量",
                    "amount": "成交额",
                }
                df = df.rename(columns=rename_map)

                if "收盘" in df.columns:
                    df["涨跌幅"] = df["收盘"].pct_change() * 100
                    df["涨跌幅"] = df["涨跌幅"].fillna(0)

                return df
            return pd.DataFrame()

        except Exception as e:
            raise e

    async def _fetch_stock_data_tx(self, stock_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        import akshare as ak

        symbol = f"sh{stock_code}" if stock_code.startswith(("6", "5", "9")) else f"sz{stock_code}"

        await self._enforce_rate_limit()

        try:

            def _call_api():
                return ak.stock_zh_a_hist_tx(
                    symbol=symbol,
                    start_date=start_date.replace("-", ""),
                    end_date=end_date.replace("-", ""),
                    adjust="qfq",
                )

            df = await asyncio.to_thread(_call_api)

            if df is not None and not df.empty:
                rename_map = {
                    "date": "日期",
                    "open": "开盘",
                    "high": "最高",
                    "low": "最低",
                    "close": "收盘",
                    "volume": "成交量",
                    "amount": "成交额",
                }
                df = df.rename(columns=rename_map)

                if "pct_chg" in df.columns:
                    df = df.rename(columns={"pct_chg": "涨跌幅"})
                elif "收盘" in df.columns:
                    df["涨跌幅"] = df["收盘"].pct_change() * 100
                    df["涨跌幅"] = df["涨跌幅"].fillna(0)

                return df
            return pd.DataFrame()

        except Exception as e:
            raise e

    async def _fetch_etf_data(self, stock_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        import akshare as ak

        self._set_random_user_agent()
        await self._enforce_rate_limit()

        logger.debug(
            f"[API调用] ak.fund_etf_hist_em(symbol={stock_code}, period=daily, "
            f"start_date={start_date.replace('-', '')}, end_date={end_date.replace('-', '')}, adjust=qfq)"
        )

        try:
            import time as _time

            api_start = _time.time()

            def _call_api():
                return ak.fund_etf_hist_em(
                    symbol=stock_code,
                    period="daily",
                    start_date=start_date.replace("-", ""),
                    end_date=end_date.replace("-", ""),
                    adjust="qfq",
                )

            df = await asyncio.to_thread(_call_api)
            api_elapsed = _time.time() - api_start

            if df is not None and not df.empty:
                logger.debug(f"[API返回] ak.fund_etf_hist_em 成功: 返回 {len(df)} 行数据, 耗时 {api_elapsed:.2f}s")
            else:
                logger.warning(f"[API返回] ak.fund_etf_hist_em 返回空数据, 耗时 {api_elapsed:.2f}s")

            return df

        except Exception as e:
            error_msg = str(e).lower()

            if any(keyword in error_msg for keyword in ["banned", "blocked", "频率", "rate", "限制"]):
                logger.warning(f"检测到可能被封禁: {e}")
                raise RateLimitError(f"Akshare 可能被限流: {e}") from e

            raise DataFetchError(f"Akshare 获取 ETF 数据失败: {e}") from e

    async def _fetch_us_data(self, stock_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        import akshare as ak

        self._set_random_user_agent()
        await self._enforce_rate_limit()

        symbol = stock_code.strip().upper()

        logger.debug(f"[API调用] ak.stock_us_daily(symbol={symbol}, adjust=qfq)")

        try:
            import time as _time

            api_start = _time.time()

            def _call_api():
                return ak.stock_us_daily(symbol=symbol, adjust="qfq")

            df = await asyncio.to_thread(_call_api)
            api_elapsed = _time.time() - api_start

            if df is not None and not df.empty:
                logger.debug(f"[API返回] ak.stock_us_daily 成功: 返回 {len(df)} 行数据, 耗时 {api_elapsed:.2f}s")

                df["date"] = pd.to_datetime(df["date"])
                start_dt = pd.to_datetime(start_date)
                end_dt = pd.to_datetime(end_date)
                df = df[(df["date"] >= start_dt) & (df["date"] <= end_dt)]

                rename_map = {
                    "date": "日期",
                    "open": "开盘",
                    "high": "最高",
                    "low": "最低",
                    "close": "收盘",
                    "volume": "成交量",
                }
                df = df.rename(columns=rename_map)

                if "收盘" in df.columns:
                    df["涨跌幅"] = df["收盘"].pct_change() * 100
                    df["涨跌幅"] = df["涨跌幅"].fillna(0)

                if "成交量" in df.columns and "收盘" in df.columns:
                    df["成交额"] = df["成交量"] * df["收盘"]
                else:
                    df["成交额"] = 0

                return df
            else:
                logger.warning(f"[API返回] ak.stock_us_daily 返回空数据, 耗时 {api_elapsed:.2f}s")
                return pd.DataFrame()

        except Exception as e:
            error_msg = str(e).lower()

            if any(keyword in error_msg for keyword in ["banned", "blocked", "频率", "rate", "限制"]):
                logger.warning(f"检测到可能被封禁: {e}")
                raise RateLimitError(f"Akshare 可能被限流: {e}") from e

            raise DataFetchError(f"Akshare 获取美股数据失败: {e}") from e

    async def _fetch_hk_data(self, stock_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        import akshare as ak

        self._set_random_user_agent()
        await self._enforce_rate_limit()

        code = stock_code.lower().replace("hk", "").zfill(5)

        logger.debug(
            f"[API调用] ak.stock_hk_hist(symbol={code}, period=daily, "
            f"start_date={start_date.replace('-', '')}, end_date={end_date.replace('-', '')}, adjust=qfq)"
        )

        try:
            import time as _time

            api_start = _time.time()

            def _call_api():
                return ak.stock_hk_hist(
                    symbol=code,
                    period="daily",
                    start_date=start_date.replace("-", ""),
                    end_date=end_date.replace("-", ""),
                    adjust="qfq",
                )

            df = await asyncio.to_thread(_call_api)
            api_elapsed = _time.time() - api_start

            if df is not None and not df.empty:
                logger.debug(f"[API返回] ak.stock_hk_hist 成功: 返回 {len(df)} 行数据, 耗时 {api_elapsed:.2f}s")
            else:
                logger.warning(f"[API返回] ak.stock_hk_hist 返回空数据, 耗时 {api_elapsed:.2f}s")

            return df

        except Exception as e:
            error_msg = str(e).lower()

            if any(keyword in error_msg for keyword in ["banned", "blocked", "频率", "rate", "限制"]):
                logger.warning(f"检测到可能被封禁: {e}")
                raise RateLimitError(f"Akshare 可能被限流: {e}") from e

            raise DataFetchError(f"Akshare 获取港股数据失败: {e}") from e

    def _normalize_data(self, df: pd.DataFrame, stock_code: str) -> pd.DataFrame:
        df = df.copy()

        column_mapping = {
            "日期": "date",
            "开盘": "open",
            "收盘": "close",
            "最高": "high",
            "最低": "low",
            "成交量": "volume",
            "成交额": "amount",
            "涨跌幅": "pct_chg",
        }

        df = df.rename(columns=column_mapping)
        df["code"] = stock_code

        keep_cols = ["code"] + STANDARD_COLUMNS
        existing_cols = [col for col in keep_cols if col in df.columns]
        df = df[existing_cols]

        df = df.dropna(how="all")

        key_cols = ["open", "high", "low", "close"]
        if all(col in df.columns for col in key_cols):
            before_len = len(df)
            df = df.dropna(subset=key_cols)
            if len(df) < before_len:
                logger.debug(f"[数据清洗] 移除了 {before_len - len(df)} 行包含缺失OHLC数据的记录")

        if "high" in df.columns and "low" in df.columns:
            mask_high_na = df["high"].isna()
            if mask_high_na.any():
                df.loc[mask_high_na, "high"] = df.loc[mask_high_na, ["open", "close"]].max(axis=1)
                logger.debug(f"[数据清洗] 填充了 {mask_high_na.sum()} 个缺失的 high 值")

            mask_low_na = df["low"].isna()
            if mask_low_na.any():
                df.loc[mask_low_na, "low"] = df.loc[mask_low_na, ["open", "close"]].min(axis=1)
                logger.debug(f"[数据清洗] 填充了 {mask_low_na.sum()} 个缺失的 low 值")

        if "volume" in df.columns:
            df["volume"] = df["volume"].fillna(0)

        if "amount" in df.columns:
            df["amount"] = df["amount"].fillna(0)

        df = df.reset_index(drop=True)

        return df

    async def get_realtime_quote(self, stock_code: str, **kwargs) -> UnifiedRealtimeQuote | None:
        source = kwargs.get("source", "em")
        circuit_breaker = get_realtime_circuit_breaker()
        source_key = f"akshare_{source}"

        if not circuit_breaker.is_available(source_key):
            logger.warning(f"[熔断] 数据源 {source_key} 处于熔断状态，跳过")
            return None

        if is_us_code(stock_code):
            logger.debug(f"[API跳过] {stock_code} 是美股，Akshare 不支持美股实时行情")
            return None
        elif is_hk_code(stock_code):
            return await self._get_hk_realtime_quote(stock_code)
        elif is_etf_code(stock_code):
            return await self._get_etf_realtime_quote(stock_code)
        else:
            if source == "sina":
                return await self._get_stock_realtime_quote_sina(stock_code)
            elif source == "tencent":
                return await self._get_stock_realtime_quote_tencent(stock_code)
            else:
                return await self._get_stock_realtime_quote_em(stock_code)

    async def _get_stock_realtime_quote_em(self, stock_code: str) -> UnifiedRealtimeQuote | None:
        import akshare as ak

        circuit_breaker = get_realtime_circuit_breaker()
        source_key = "akshare_em"

        try:
            cache_key = "realtime_data"
            if cache_key in _realtime_cache:
                df = _realtime_cache[cache_key]
                logger.debug("[缓存命中] A股实时行情(东财)")
            else:
                logger.debug("[缓存未命中] 触发全量刷新 A股实时行情(东财)")
                last_error: Exception | None = None
                df = None
                for attempt in range(1, 3):
                    try:
                        self._set_random_user_agent()
                        await self._enforce_rate_limit()

                        logger.debug(f"[API调用] ak.stock_zh_a_spot_em() 获取A股实时行情... (attempt {attempt}/2)")
                        import time as _time

                        api_start = _time.time()

                        def _call_api():
                            return ak.stock_zh_a_spot_em()

                        df = await asyncio.to_thread(_call_api)

                        api_elapsed = _time.time() - api_start
                        logger.debug(
                            f"[API返回] ak.stock_zh_a_spot_em 成功: 返回 {len(df)} 只股票, 耗时 {api_elapsed:.2f}s"
                        )
                        circuit_breaker.record_success(source_key)
                        break
                    except Exception as e:
                        last_error = e
                        logger.warning(f"[API错误] ak.stock_zh_a_spot_em 获取失败 (attempt {attempt}/2): {e}")
                        await asyncio.sleep(min(2**attempt, 5))

                if df is None:
                    logger.error(f"[API错误] ak.stock_zh_a_spot_em 最终失败: {last_error}")
                    circuit_breaker.record_failure(source_key, str(last_error))
                    df = pd.DataFrame()
                _realtime_cache[cache_key] = df
                logger.debug("[缓存更新] A股实时行情(东财) 缓存已刷新")

            if df is None or df.empty:
                logger.warning(f"[实时行情] A股实时行情数据为空，跳过 {stock_code}")
                return None

            row = df[df["代码"] == stock_code]
            if row.empty:
                logger.warning(f"[API返回] 未找到股票 {stock_code} 的实时行情")
                return None

            row = row.iloc[0]

            quote = UnifiedRealtimeQuote(
                code=stock_code,
                name=str(row.get("名称", "")),
                source=RealtimeSource.AKSHARE_EM,
                price=safe_float(row.get("最新价")),
                change_pct=safe_float(row.get("涨跌幅")),
                change_amount=safe_float(row.get("涨跌额")),
                volume=safe_int(row.get("成交量")),
                amount=safe_float(row.get("成交额")),
                volume_ratio=safe_float(row.get("量比")),
                turnover_rate=safe_float(row.get("换手率")),
                amplitude=safe_float(row.get("振幅")),
                open_price=safe_float(row.get("今开")),
                high=safe_float(row.get("最高")),
                low=safe_float(row.get("最低")),
                pe_ratio=safe_float(row.get("市盈率-动态")),
                pb_ratio=safe_float(row.get("市净率")),
                total_mv=safe_float(row.get("总市值")),
                circ_mv=safe_float(row.get("流通市值")),
                change_60d=safe_float(row.get("60日涨跌幅")),
                high_52w=safe_float(row.get("52周最高")),
                low_52w=safe_float(row.get("52周最低")),
            )

            logger.debug(
                f"[实时行情-东财] {stock_code} {quote.name}: 价格={quote.price}, 涨跌={quote.change_pct}%, "
                f"量比={quote.volume_ratio}, 换手率={quote.turnover_rate}%"
            )
            return quote

        except Exception as e:
            logger.error(f"[API错误] 获取 {stock_code} 实时行情(东财)失败: {e}")
            circuit_breaker.record_failure(source_key, str(e))
            return None

    async def _get_stock_realtime_quote_sina(self, stock_code: str) -> UnifiedRealtimeQuote | None:
        circuit_breaker = get_realtime_circuit_breaker()
        source_key = "akshare_sina"

        try:
            session = get_aiohttp_session()

            symbol = f"sh{stock_code}" if stock_code.startswith(("6", "5", "9")) else f"sz{stock_code}"

            url = f"http://hq.sinajs.cn/list={symbol}"
            headers = {
                "Referer": "http://finance.sina.com.cn",
                "User-Agent": random.choice(USER_AGENTS),
            }

            logger.debug(f"[API调用] 新浪财经接口获取 {stock_code} 实时行情...")

            await self._enforce_rate_limit()
            async with session.get(url, headers=headers) as response:
                if response.status != 200:
                    logger.warning(f"[API错误] 新浪接口返回状态码 {response.status}")
                    circuit_breaker.record_failure(source_key, f"HTTP {response.status}")
                    return None

                content = await response.text("gbk")

            if '=""' in content or not content:
                logger.warning(f"[API返回] 新浪接口未找到 {stock_code} 数据")
                return None

            data_start = content.find('"')
            data_end = content.rfind('"')
            if data_start == -1 or data_end == -1:
                logger.warning("[API返回] 新浪接口数据格式异常")
                circuit_breaker.record_failure(source_key, "数据格式异常")
                return None

            data_str = content[data_start + 1 : data_end]
            fields = data_str.split(",")

            if len(fields) < 32:
                logger.warning(f"[API返回] 新浪接口数据字段不足: {len(fields)}")
                return None

            circuit_breaker.record_success(source_key)

            price = safe_float(fields[3])
            pre_close = safe_float(fields[2])
            change_pct = None
            change_amount = None
            if price and pre_close and pre_close > 0:
                change_amount = price - pre_close
                change_pct = (change_amount / pre_close) * 100

            quote = UnifiedRealtimeQuote(
                code=stock_code,
                name=fields[0],
                source=RealtimeSource.AKSHARE_SINA,
                price=price,
                change_pct=change_pct,
                change_amount=change_amount,
                volume=safe_int(fields[8]),
                amount=safe_float(fields[9]),
                open_price=safe_float(fields[1]),
                high=safe_float(fields[4]),
                low=safe_float(fields[5]),
                pre_close=pre_close,
            )

            logger.debug(
                f"[实时行情-新浪] {stock_code} {quote.name}: 价格={quote.price}, 涨跌={quote.change_pct:.2f}%"
                if quote.change_pct
                else ""
            )
            return quote

        except Exception as e:
            logger.error(f"[API错误] 获取 {stock_code} 实时行情(新浪)失败: {e}")
            circuit_breaker.record_failure(source_key, str(e))
            return None

    async def _get_stock_realtime_quote_tencent(self, stock_code: str) -> UnifiedRealtimeQuote | None:
        circuit_breaker = get_realtime_circuit_breaker()
        source_key = "tencent"

        try:
            session = get_aiohttp_session()

            symbol = f"sh{stock_code}" if stock_code.startswith(("6", "5", "9")) else f"sz{stock_code}"

            url = f"http://qt.gtimg.cn/q={symbol}"
            headers = {"Referer": "http://finance.qq.com", "User-Agent": random.choice(USER_AGENTS)}

            logger.debug(f"[API调用] 腾讯财经接口获取 {stock_code} 实时行情...")

            await self._enforce_rate_limit()
            async with session.get(url, headers=headers) as response:
                if response.status != 200:
                    logger.warning(f"[API错误] 腾讯接口返回状态码 {response.status}")
                    circuit_breaker.record_failure(source_key, f"HTTP {response.status}")
                    return None

                content = await response.text("gbk")

            if '=""' in content or not content:
                logger.warning(f"[API返回] 腾讯接口未找到 {stock_code} 数据")
                return None

            data_start = content.find('"')
            data_end = content.rfind('"')
            if data_start == -1 or data_end == -1:
                logger.warning("[API返回] 腾讯接口数据格式异常")
                circuit_breaker.record_failure(source_key, "数据格式异常")
                return None

            data_str = content[data_start + 1 : data_end]
            fields = data_str.split("~")

            if len(fields) < 45:
                logger.warning(f"[API返回] 腾讯接口数据字段不足: {len(fields)}")
                return None

            circuit_breaker.record_success(source_key)

            quote = UnifiedRealtimeQuote(
                code=stock_code,
                name=fields[1] if len(fields) > 1 else "",
                source=RealtimeSource.TENCENT,
                price=safe_float(fields[3]),
                change_pct=safe_float(fields[32]),
                change_amount=safe_float(fields[31]) if len(fields) > 31 else None,
                volume=(safe_int(fields[6], default=0) or 0) * 100 if fields[6] else None,
                open_price=safe_float(fields[5]),
                high=safe_float(fields[34]) if len(fields) > 34 else None,
                low=safe_float(fields[35].split("/")[0])
                if len(fields) > 35 and "/" in str(fields[35])
                else safe_float(fields[35])
                if len(fields) > 35
                else None,
                pre_close=safe_float(fields[4]),
                turnover_rate=safe_float(fields[38]) if len(fields) > 38 else None,
                amplitude=safe_float(fields[43]) if len(fields) > 43 else None,
                volume_ratio=safe_float(fields[49]) if len(fields) > 49 else None,
                pe_ratio=safe_float(fields[39]) if len(fields) > 39 else None,
                pb_ratio=safe_float(fields[46]) if len(fields) > 46 else None,
                circ_mv=(safe_float(fields[44], default=0) or 0) * 100000000
                if len(fields) > 44 and fields[44]
                else None,
                total_mv=(safe_float(fields[45], default=0) or 0) * 100000000
                if len(fields) > 45 and fields[45]
                else None,
            )

            logger.debug(
                f"[实时行情-腾讯] {stock_code} {quote.name}: 价格={quote.price}, "
                f"涨跌={quote.change_pct}%, 量比={quote.volume_ratio}, 换手率={quote.turnover_rate}%, "
                f"PE={quote.pe_ratio}, PB={quote.pb_ratio}"
            )
            return quote

        except Exception as e:
            logger.error(f"[API错误] 获取 {stock_code} 实时行情(腾讯)失败: {e}")
            circuit_breaker.record_failure(source_key, str(e))
            return None

    async def _get_etf_realtime_quote(self, stock_code: str) -> UnifiedRealtimeQuote | None:
        import akshare as ak

        circuit_breaker = get_realtime_circuit_breaker()
        source_key = "akshare_etf"

        try:
            cache_key = "etf_realtime_data"
            if cache_key in _etf_realtime_cache:
                df = _etf_realtime_cache[cache_key]
                logger.debug("[缓存命中] 使用缓存的ETF实时行情数据")
            else:
                last_error: Exception | None = None
                df = None
                for attempt in range(1, 3):
                    try:
                        self._set_random_user_agent()
                        await self._enforce_rate_limit()

                        logger.debug(f"[API调用] ak.fund_etf_spot_em() 获取ETF实时行情... (attempt {attempt}/2)")
                        import time as _time

                        api_start = _time.time()

                        def _call_api():
                            return ak.fund_etf_spot_em()

                        df = await asyncio.to_thread(_call_api)

                        api_elapsed = _time.time() - api_start
                        logger.debug(
                            f"[API返回] ak.fund_etf_spot_em 成功: 返回 {len(df)} 只ETF, 耗时 {api_elapsed:.2f}s"
                        )
                        circuit_breaker.record_success(source_key)
                        break
                    except Exception as e:
                        last_error = e
                        logger.warning(f"[API错误] ak.fund_etf_spot_em 获取失败 (attempt {attempt}/2): {e}")
                        await asyncio.sleep(min(2**attempt, 5))

                if df is None:
                    logger.error(f"[API错误] ak.fund_etf_spot_em 最终失败: {last_error}")
                    circuit_breaker.record_failure(source_key, str(last_error))
                    df = pd.DataFrame()
                _etf_realtime_cache[cache_key] = df

            if df is None or df.empty:
                logger.warning(f"[实时行情] ETF实时行情数据为空，跳过 {stock_code}")
                return None

            row = df[df["代码"] == stock_code]
            if row.empty:
                logger.warning(f"[API返回] 未找到 ETF {stock_code} 的实时行情")
                return None

            row = row.iloc[0]

            quote = UnifiedRealtimeQuote(
                code=stock_code,
                name=str(row.get("名称", "")),
                source=RealtimeSource.AKSHARE_EM,
                price=safe_float(row.get("最新价")),
                change_pct=safe_float(row.get("涨跌幅")),
                change_amount=safe_float(row.get("涨跌额")),
                volume=safe_int(row.get("成交量")),
                amount=safe_float(row.get("成交额")),
                volume_ratio=safe_float(row.get("量比")),
                turnover_rate=safe_float(row.get("换手率")),
                amplitude=safe_float(row.get("振幅")),
                open_price=safe_float(row.get("今开")),
                high=safe_float(row.get("最高")),
                low=safe_float(row.get("最低")),
                total_mv=safe_float(row.get("总市值")),
                circ_mv=safe_float(row.get("流通市值")),
                high_52w=safe_float(row.get("52周最高")),
                low_52w=safe_float(row.get("52周最低")),
            )

            logger.debug(
                f"[ETF实时行情] {stock_code} {quote.name}: 价格={quote.price}, 涨跌={quote.change_pct}%, "
                f"换手率={quote.turnover_rate}%"
            )
            return quote

        except Exception as e:
            logger.error(f"[API错误] 获取 ETF {stock_code} 实时行情失败: {e}")
            circuit_breaker.record_failure(source_key, str(e))
            return None

    async def _get_hk_realtime_quote(self, stock_code: str) -> UnifiedRealtimeQuote | None:
        import akshare as ak

        circuit_breaker = get_realtime_circuit_breaker()
        source_key = "akshare_hk"

        try:
            self._set_random_user_agent()
            await self._enforce_rate_limit()

            code = stock_code.lower().replace("hk", "").zfill(5)

            logger.debug("[API调用] ak.stock_hk_spot_em() 获取港股实时行情...")
            import time as _time

            api_start = _time.time()

            def _call_api():
                return ak.stock_hk_spot_em()

            df = await asyncio.to_thread(_call_api)

            api_elapsed = _time.time() - api_start
            logger.debug(f"[API返回] ak.stock_hk_spot_em 成功: 返回 {len(df)} 只港股, 耗时 {api_elapsed:.2f}s")
            circuit_breaker.record_success(source_key)

            row = df[df["代码"] == code]
            if row.empty:
                logger.warning(f"[API返回] 未找到港股 {code} 的实时行情")
                return None

            row = row.iloc[0]

            quote = UnifiedRealtimeQuote(
                code=stock_code,
                name=str(row.get("名称", "")),
                source=RealtimeSource.AKSHARE_EM,
                price=safe_float(row.get("最新价")),
                change_pct=safe_float(row.get("涨跌幅")),
                change_amount=safe_float(row.get("涨跌额")),
                volume=safe_int(row.get("成交量")),
                amount=safe_float(row.get("成交额")),
                volume_ratio=safe_float(row.get("量比")),
                turnover_rate=safe_float(row.get("换手率")),
                amplitude=safe_float(row.get("振幅")),
                pe_ratio=safe_float(row.get("市盈率")),
                pb_ratio=safe_float(row.get("市净率")),
                total_mv=safe_float(row.get("总市值")),
                circ_mv=safe_float(row.get("流通市值")),
                high_52w=safe_float(row.get("52周最高")),
                low_52w=safe_float(row.get("52周最低")),
            )

            logger.debug(
                f"[港股实时行情] {stock_code} {quote.name}: 价格={quote.price}, 涨跌={quote.change_pct}%, "
                f"换手率={quote.turnover_rate}%"
            )
            return quote

        except Exception as e:
            logger.error(f"[API错误] 获取港股 {stock_code} 实时行情失败: {e}")
            circuit_breaker.record_failure(source_key, str(e))
            return None

    async def get_chip_distribution(self, stock_code: str) -> ChipDistribution | None:
        import akshare as ak

        if is_us_code(stock_code):
            logger.debug(f"[API跳过] {stock_code} 是美股，无筹码分布数据")
            return None

        if is_etf_code(stock_code):
            logger.debug(f"[API跳过] {stock_code} 是 ETF/指数，无筹码分布数据")
            return None

        try:
            self._set_random_user_agent()
            await self._enforce_rate_limit()

            logger.debug(f"[API调用] ak.stock_cyq_em(symbol={stock_code}) 获取筹码分布...")
            import time as _time

            api_start = _time.time()

            def _call_api():
                return ak.stock_cyq_em(symbol=stock_code)

            df = await asyncio.to_thread(_call_api)

            api_elapsed = _time.time() - api_start

            if df.empty:
                logger.warning(f"[API返回] ak.stock_cyq_em 返回空数据, 耗时 {api_elapsed:.2f}s")
                return None

            logger.debug(f"[API返回] ak.stock_cyq_em 成功: 返回 {len(df)} 天数据, 耗时 {api_elapsed:.2f}s")

            latest = df.iloc[-1]

            profit_ratio_raw = safe_float(latest.get("获利比例"), default=0.0) or 0.0
            concentration_90_raw = safe_float(latest.get("90集中度"), default=0.0) or 0.0
            concentration_70_raw = safe_float(latest.get("70集中度"), default=0.0) or 0.0

            chip = ChipDistribution(
                code=stock_code,
                date=str(latest.get("日期", "")),
                profit_ratio=profit_ratio_raw / 100.0 if profit_ratio_raw > 1.0 else profit_ratio_raw,
                avg_cost=safe_float(latest.get("平均成本"), default=0.0) or 0.0,
                cost_90_low=safe_float(latest.get("90成本-低"), default=0.0) or 0.0,
                cost_90_high=safe_float(latest.get("90成本-高"), default=0.0) or 0.0,
                concentration_90=concentration_90_raw / 100.0 if concentration_90_raw > 1.0 else concentration_90_raw,
                cost_70_low=safe_float(latest.get("70成本-低"), default=0.0) or 0.0,
                cost_70_high=safe_float(latest.get("70成本-高"), default=0.0) or 0.0,
                concentration_70=concentration_70_raw / 100.0 if concentration_70_raw > 1.0 else concentration_70_raw,
            )

            logger.debug(
                f"[筹码分布] {stock_code} 日期={chip.date}: 获利比例={chip.profit_ratio:.1%}, "
                f"平均成本={chip.avg_cost}, 90%集中度={chip.concentration_90:.2%}, "
                f"70%集中度={chip.concentration_70:.2%}"
            )
            return chip

        except Exception as e:
            logger.error(f"[API错误] 获取 {stock_code} 筹码分布失败: {e}")
            return None

    async def get_enhanced_data(self, stock_code: str, days: int = 60) -> dict[str, Any]:
        result = {
            "code": stock_code,
            "daily_data": None,
            "realtime_quote": None,
            "chip_distribution": None,
        }

        try:
            df = await self.get_daily_data(stock_code, days=days)
            result["daily_data"] = df
        except Exception as e:
            logger.error(f"获取 {stock_code} 日线数据失败: {e}")

        result["realtime_quote"] = await self.get_realtime_quote(stock_code)
        result["chip_distribution"] = await self.get_chip_distribution(stock_code)

        return result

    async def get_main_indices(self) -> list[dict[str, Any]] | None:
        import akshare as ak

        indices_map = {
            "sh000001": "上证指数",
            "sz399001": "深证成指",
            "sz399006": "创业板指",
            "sh000688": "科创50",
            "sh000016": "上证50",
            "sh000300": "沪深300",
        }

        try:
            self._set_random_user_agent()
            await self._enforce_rate_limit()

            def _call_api():
                return ak.stock_zh_index_spot_sina()

            df = await asyncio.to_thread(_call_api)

            results = []
            if df is not None and not df.empty:
                for code, name in indices_map.items():
                    row = df[df["代码"] == code]
                    if row.empty:
                        row = df[df["代码"].str.contains(code)]

                    if not row.empty:
                        row = row.iloc[0]
                        current = safe_float(row.get("最新价", 0), default=0.0) or 0.0
                        prev_close = safe_float(row.get("昨收", 0), default=0.0) or 0.0
                        high = safe_float(row.get("最高", 0), default=0.0) or 0.0
                        low = safe_float(row.get("最低", 0), default=0.0) or 0.0

                        amplitude = 0.0
                        if prev_close > 0:
                            amplitude = (high - low) / prev_close * 100

                        results.append(
                            {
                                "code": code,
                                "name": name,
                                "current": current,
                                "change": safe_float(row.get("涨跌额", 0)),
                                "change_pct": safe_float(row.get("涨跌幅", 0)),
                                "open": safe_float(row.get("今开", 0)),
                                "high": high,
                                "low": low,
                                "prev_close": prev_close,
                                "volume": safe_float(row.get("成交量", 0)),
                                "amount": safe_float(row.get("成交额", 0)),
                                "amplitude": amplitude,
                            }
                        )
            return results

        except Exception as e:
            logger.error(f"[Akshare] 获取指数行情失败: {e}")
            return None

    async def get_market_stats(self) -> dict[str, Any] | None:
        import akshare as ak

        try:
            self._set_random_user_agent()
            await self._enforce_rate_limit()

            logger.debug("[API调用] ak.stock_zh_a_spot_em() 获取市场统计...")

            def _call_api():
                return ak.stock_zh_a_spot_em()

            df = await asyncio.to_thread(_call_api)

            if df is not None and not df.empty:
                return self._calc_market_stats(df, change_col="涨跌幅", amount_col="成交额")
        except Exception as e:
            logger.warning(f"[Akshare] 东财接口获取市场统计失败: {e}，尝试新浪接口")

        try:
            self._set_random_user_agent()
            await self._enforce_rate_limit()

            logger.debug("[API调用] ak.stock_zh_a_spot() 获取市场统计(新浪)...")

            def _call_api():
                return ak.stock_zh_a_spot()

            df = await asyncio.to_thread(_call_api)

            if df is not None and not df.empty:
                change_col = None
                for col in ["change_percent", "changepercent", "涨跌幅", "trade_ratio"]:
                    if col in df.columns:
                        change_col = col
                        break

                amount_col = None
                for col in ["amount", "成交额", "trade_amount"]:
                    if col in df.columns:
                        amount_col = col
                        break

                if change_col:
                    return self._calc_market_stats(df, change_col=change_col, amount_col=amount_col)
        except Exception as e:
            logger.error(f"[Akshare] 新浪接口获取市场统计也失败: {e}")

        return None

    def _calc_market_stats(
        self,
        df: pd.DataFrame,
        change_col: str,
        amount_col: str | None = None,
    ) -> dict[str, Any] | None:
        if change_col not in df.columns:
            return None

        df[change_col] = pd.to_numeric(df[change_col], errors="coerce")
        stats = {
            "up_count": len(df[df[change_col] > 0]),
            "down_count": len(df[df[change_col] < 0]),
            "flat_count": len(df[df[change_col] == 0]),
            "limit_up_count": len(df[df[change_col] >= 9.9]),
            "limit_down_count": len(df[df[change_col] <= -9.9]),
            "total_amount": 0.0,
        }
        if amount_col and amount_col in df.columns:
            df[amount_col] = pd.to_numeric(df[amount_col], errors="coerce")
            stats["total_amount"] = df[amount_col].sum() / 1e8
        return stats

    async def get_sector_rankings(self, n: int = 5) -> tuple[list[dict], list[dict]] | None:
        import akshare as ak

        try:
            self._set_random_user_agent()
            await self._enforce_rate_limit()

            logger.debug("[API调用] ak.stock_board_industry_name_em() 获取板块排行...")

            def _call_api():
                return ak.stock_board_industry_name_em()

            df = await asyncio.to_thread(_call_api)

            if df is not None and not df.empty:
                change_col = "涨跌幅"
                if change_col in df.columns:
                    df[change_col] = pd.to_numeric(df[change_col], errors="coerce")
                    df = df.dropna(subset=[change_col])

                    top = df.nlargest(n, change_col)
                    top_sectors = [
                        {"name": row["板块名称"], "change_pct": row[change_col]} for _, row in top.iterrows()
                    ]

                    bottom = df.nsmallest(n, change_col)
                    bottom_sectors = [
                        {"name": row["板块名称"], "change_pct": row[change_col]} for _, row in bottom.iterrows()
                    ]

                    return top_sectors, bottom_sectors
        except Exception as e:
            logger.warning(f"[Akshare] 东财接口获取板块排行失败: {e}，尝试新浪接口")

        try:
            self._set_random_user_agent()
            await self._enforce_rate_limit()

            logger.debug("[API调用] ak.stock_sector_spot() 获取板块排行(新浪)...")

            def _call_api():
                return ak.stock_sector_spot(indicator="新浪行业")

            df = await asyncio.to_thread(_call_api)

            if df is None or df.empty:
                return None

            change_col = None
            for col in ["涨跌幅", "change_pct", "涨幅"]:
                if col in df.columns:
                    change_col = col
                    break

            name_col = None
            for col in ["板块", "板块名称", "label", "name"]:
                if col in df.columns:
                    name_col = col
                    break

            if not change_col or not name_col:
                return None

            df[change_col] = pd.to_numeric(df[change_col], errors="coerce")
            df = df.dropna(subset=[change_col])
            top = df.nlargest(n, change_col)
            bottom = df.nsmallest(n, change_col)
            top_sectors = [
                {"name": str(row[name_col]), "change_pct": float(row[change_col])} for _, row in top.iterrows()
            ]
            bottom_sectors = [
                {"name": str(row[name_col]), "change_pct": float(row[change_col])} for _, row in bottom.iterrows()
            ]
            return top_sectors, bottom_sectors
        except Exception as e:
            logger.error(f"[Akshare] 新浪接口获取板块排行也失败: {e}")
            return None
