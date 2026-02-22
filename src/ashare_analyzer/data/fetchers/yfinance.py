"""
===================================
YfinanceFetcher - 兜底数据源 (Priority 4)
===================================

数据来源：Yahoo Finance（通过 yfinance 库）
特点：国际数据源、可能有延迟或缺失
定位：当所有国内数据源都失败时的最后保障

关键策略：
1. 自动将 A 股代码转换为 yfinance 格式（.SS / .SZ）
2. 处理 Yahoo Finance 的数据格式差异
3. 失败后指数退避重试
"""

import asyncio
import logging
from typing import Any, cast

import pandas as pd
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from ashare_analyzer.data.base import STANDARD_COLUMNS, BaseFetcher, DataFetchError
from ashare_analyzer.infrastructure import AsyncRateLimiter
from ashare_analyzer.models import RealtimeSource, UnifiedRealtimeQuote
from ashare_analyzer.utils.stock_code import StockType, convert_to_provider_format, detect_stock_type

logger = logging.getLogger(__name__)


class YfinanceFetcher(BaseFetcher):
    """
    Yahoo Finance 数据源实现 (async)

    优先级：4（最低，作为兜底）
    数据来源：Yahoo Finance

    关键策略：
    - 自动转换股票代码格式
    - 处理时区和数据格式差异
    - 失败后指数退避重试

    注意事项：
    - A 股数据可能有延迟
    - 某些股票可能无数据
    - 数据精度可能与国内源略有差异
    """

    name = "YfinanceFetcher"

    @property
    def priority(self) -> int:
        from ashare_analyzer.config import get_config

        return get_config().datasource.yfinance_priority

    def __init__(self, rate_limiter: AsyncRateLimiter | None = None):
        super().__init__(rate_limiter=rate_limiter)

    def _convert_stock_code(self, stock_code: str) -> str:
        code = stock_code.strip().upper()

        if ".SS" in code or ".SZ" in code or ".HK" in code:
            return code

        return convert_to_provider_format(stock_code, "yfinance")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type((ConnectionError, TimeoutError)),
        before_sleep=before_sleep_log(cast(Any, logger), logging.WARNING),
    )
    async def _fetch_raw_data(self, stock_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        import yfinance as yf

        yf_code = self._convert_stock_code(stock_code)

        logger.debug(f"调用 yfinance.download({yf_code}, {start_date}, {end_date})")

        try:
            await self._enforce_rate_limit()

            def _call_api():
                return yf.download(
                    tickers=yf_code,
                    start=start_date,
                    end=end_date,
                    progress=False,
                    auto_adjust=True,
                )

            df = await asyncio.to_thread(_call_api)

            if df.empty:
                raise DataFetchError(f"Yahoo Finance 未查询到 {stock_code} 的数据")

            return df

        except Exception as e:
            if isinstance(e, DataFetchError):
                raise
            raise DataFetchError(f"Yahoo Finance 获取数据失败: {e}") from e

    def _normalize_data(self, df: pd.DataFrame, stock_code: str) -> pd.DataFrame:
        df = df.copy()

        if isinstance(df.columns, pd.MultiIndex):
            logger.debug("检测到 MultiIndex 列名，进行扁平化处理")
            df.columns = df.columns.get_level_values(0)

        df = df.reset_index()

        column_mapping = {
            "Date": "date",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        }

        df = df.rename(columns=column_mapping)

        if "close" in df.columns:
            df["pct_chg"] = df["close"].pct_change() * 100
            df["pct_chg"] = df["pct_chg"].fillna(0).round(2)

        if "volume" in df.columns and "close" in df.columns:
            df["amount"] = df["volume"] * df["close"]
        else:
            df["amount"] = 0

        df["code"] = stock_code

        keep_cols = ["code"] + STANDARD_COLUMNS
        existing_cols = [col for col in keep_cols if col in df.columns]
        df = df[existing_cols]

        return df

    async def get_main_indices(self) -> list[dict[str, Any]] | None:
        import yfinance as yf

        yf_mapping = {
            "sh000001": ("000001.SS", "上证指数"),
            "sz399001": ("399001.SZ", "深证成指"),
            "sz399006": ("399006.SZ", "创业板指"),
            "sh000688": ("000688.SS", "科创50"),
            "sh000016": ("000016.SS", "上证50"),
            "sh000300": ("000300.SS", "沪深300"),
        }

        results = []
        try:
            for ak_code, (yf_code, name) in yf_mapping.items():
                try:
                    await self._enforce_rate_limit()

                    def _get_ticker_history(code=yf_code):
                        ticker = yf.Ticker(code)
                        return ticker.history(period="2d")

                    hist = await asyncio.to_thread(_get_ticker_history)

                    if hist.empty:
                        continue

                    today = hist.iloc[-1]
                    prev = hist.iloc[-2] if len(hist) > 1 else today

                    price = float(today["Close"])
                    prev_close = float(prev["Close"])
                    change = price - prev_close
                    change_pct = (change / prev_close) * 100 if prev_close else 0

                    high = float(today["High"])
                    low = float(today["Low"])
                    amplitude = ((high - low) / prev_close * 100) if prev_close else 0

                    results.append(
                        {
                            "code": ak_code,
                            "name": name,
                            "current": price,
                            "change": change,
                            "change_pct": change_pct,
                            "open": float(today["Open"]),
                            "high": high,
                            "low": low,
                            "prev_close": prev_close,
                            "volume": float(today["Volume"]),
                            "amount": 0.0,
                            "amplitude": amplitude,
                        }
                    )
                    logger.debug(f"[Yfinance] 获取指数 {name} 成功")

                except Exception as e:
                    logger.warning(f"[Yfinance] 获取指数 {name} 失败: {e}")
                    continue

            if results:
                logger.info(f"[Yfinance] 成功获取 {len(results)} 个指数行情")
                return results

        except Exception as e:
            logger.error(f"[Yfinance] 获取指数行情失败: {e}")

        return None

    def _is_us_stock(self, stock_code: str) -> bool:
        return detect_stock_type(stock_code) == StockType.US

    async def get_realtime_quote(self, stock_code: str, **kwargs) -> UnifiedRealtimeQuote | None:
        import yfinance as yf

        if not self._is_us_stock(stock_code):
            logger.debug(f"[Yfinance] {stock_code} 不是美股，跳过")
            return None

        try:
            symbol = stock_code.strip().upper()
            logger.debug(f"[Yfinance] 获取美股 {symbol} 实时行情")

            await self._enforce_rate_limit()

            ticker = yf.Ticker(symbol)

            try:

                def _get_fast_info(ticker=ticker):
                    return ticker.fast_info

                info = await asyncio.to_thread(_get_fast_info)

                if info is None:
                    raise DataFetchError("fast_info is None")

                price = getattr(info, "lastPrice", None) or getattr(info, "last_price", None)
                prev_close = getattr(info, "previousClose", None) or getattr(info, "previous_close", None)
                open_price = getattr(info, "open", None)
                high = getattr(info, "dayHigh", None) or getattr(info, "day_high", None)
                low = getattr(info, "dayLow", None) or getattr(info, "day_low", None)
                volume = getattr(info, "lastVolume", None) or getattr(info, "last_volume", None)
                market_cap = getattr(info, "marketCap", None) or getattr(info, "market_cap", None)

            except Exception:

                def _get_history(ticker=ticker):
                    return ticker.history(period="2d")

                hist = await asyncio.to_thread(_get_history)

                if hist.empty:
                    logger.warning(f"[Yfinance] 无法获取 {symbol} 的数据")
                    return None

                today = hist.iloc[-1]
                prev = hist.iloc[-2] if len(hist) > 1 else today

                price = float(today["Close"])
                prev_close = float(prev["Close"])
                open_price = float(today["Open"])
                high = float(today["High"])
                low = float(today["Low"])
                volume = int(today["Volume"])
                market_cap = None

            change_amount = None
            change_pct = None
            if price is not None and prev_close is not None and prev_close > 0:
                change_amount = price - prev_close
                change_pct = (change_amount / prev_close) * 100

            amplitude = None
            if high is not None and low is not None and prev_close is not None and prev_close > 0:
                amplitude = ((high - low) / prev_close) * 100

            try:

                def _get_info(ticker=ticker):
                    return ticker.info

                info_dict = await asyncio.to_thread(_get_info)
                name = info_dict.get("shortName", "") or info_dict.get("longName", "") or symbol
            except Exception:
                logger.debug(f"[Yfinance] Failed to get name for {symbol}")
                name = symbol

            quote = UnifiedRealtimeQuote(
                code=symbol,
                name=name,
                source=RealtimeSource.FALLBACK,
                price=price,
                change_pct=round(change_pct, 2) if change_pct is not None else None,
                change_amount=round(change_amount, 4) if change_amount is not None else None,
                volume=volume,
                amount=None,
                volume_ratio=None,
                turnover_rate=None,
                amplitude=round(amplitude, 2) if amplitude is not None else None,
                open_price=open_price,
                high=high,
                low=low,
                pre_close=prev_close,
                pe_ratio=None,
                pb_ratio=None,
                total_mv=market_cap,
                circ_mv=None,
            )

            logger.info(f"[Yfinance] 获取美股 {symbol} 实时行情成功: 价格={price}")
            return quote

        except Exception as e:
            logger.warning(f"[Yfinance] 获取美股 {stock_code} 实时行情失败: {e}")
            return None
