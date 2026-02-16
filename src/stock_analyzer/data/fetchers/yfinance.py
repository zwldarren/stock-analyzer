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

import logging
from typing import Any

import pandas as pd
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from stock_analyzer.data.base import STANDARD_COLUMNS, BaseFetcher, DataFetchError
from stock_analyzer.models import RealtimeSource, UnifiedRealtimeQuote
from stock_analyzer.utils.stock_code import StockType, convert_to_provider_format, detect_stock_type

logger = logging.getLogger(__name__)


class YfinanceFetcher(BaseFetcher):
    """
    Yahoo Finance 数据源实现

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
        from stock_analyzer.config import get_config

        return get_config().datasource.yfinance_priority

    def __init__(self):
        """初始化 YfinanceFetcher"""
        pass

    def _convert_stock_code(self, stock_code: str) -> str:
        """
        Convert stock code to Yahoo Finance format.

        Yahoo Finance code format:
        - A-share Shanghai: 600519.SS
        - A-share Shenzhen: 000001.SZ
        - HK stocks: 0700.HK
        - US stocks: AAPL (no suffix)

        Args:
            stock_code: Original code (e.g., '600519', 'hk00700', 'AAPL')

        Returns:
            Yahoo Finance formatted code

        Examples:
            >>> fetcher._convert_stock_code('600519')
            '600519.SS'
            >>> fetcher._convert_stock_code('hk00700')
            '0700.HK'
            >>> fetcher._convert_stock_code('AAPL')
            'AAPL'
        """
        code = stock_code.strip().upper()

        # Already has suffix
        if ".SS" in code or ".SZ" in code or ".HK" in code:
            return code

        # Use unified conversion function
        return convert_to_provider_format(stock_code, "yfinance")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type((ConnectionError, TimeoutError)),
        before_sleep=before_sleep_log(logger, logging.WARNING),  # type: ignore[arg-type]
    )
    def _fetch_raw_data(self, stock_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        从 Yahoo Finance 获取原始数据

        使用 yfinance.download() 获取历史数据

        流程：
        1. 转换股票代码格式
        2. 调用 yfinance API
        3. 处理返回数据
        """
        import yfinance as yf

        # 转换代码格式
        yf_code = self._convert_stock_code(stock_code)

        logger.debug(f"调用 yfinance.download({yf_code}, {start_date}, {end_date})")

        try:
            # 使用 yfinance 下载数据
            df = yf.download(
                tickers=yf_code,
                start=start_date,
                end=end_date,
                progress=False,  # 禁止进度条
                auto_adjust=True,  # 自动调整价格（复权）
            )

            if df.empty:
                raise DataFetchError(f"Yahoo Finance 未查询到 {stock_code} 的数据")

            return df

        except Exception as e:
            if isinstance(e, DataFetchError):
                raise
            raise DataFetchError(f"Yahoo Finance 获取数据失败: {e}") from e

    def _normalize_data(self, df: pd.DataFrame, stock_code: str) -> pd.DataFrame:
        """
        标准化 Yahoo Finance 数据

        yfinance 返回的列名：
        Open, High, Low, Close, Volume（索引是日期）

        注意：新版 yfinance 返回 MultiIndex 列名，如 ('Close', 'AMD')
        需要先扁平化列名再进行处理

        需要映射到标准列名：
        date, open, high, low, close, volume, amount, pct_chg
        """
        df = df.copy()

        # 处理 MultiIndex 列名（新版 yfinance 返回格式）
        # 例如: ('Close', 'AMD') -> 'Close'
        if isinstance(df.columns, pd.MultiIndex):
            logger.debug("检测到 MultiIndex 列名，进行扁平化处理")
            # 取第一级列名（Price level: Close, High, Low, etc.）
            df.columns = df.columns.get_level_values(0)

        # 重置索引，将日期从索引变为列
        df = df.reset_index()

        # 列名映射（yfinance 使用首字母大写）
        column_mapping = {
            "Date": "date",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        }

        df = df.rename(columns=column_mapping)

        # 计算涨跌幅（因为 yfinance 不直接提供）
        if "close" in df.columns:
            df["pct_chg"] = df["close"].pct_change() * 100
            df["pct_chg"] = df["pct_chg"].fillna(0).round(2)

        # 计算成交额（yfinance 不提供，使用估算值）
        # 成交额 ≈ 成交量 * 平均价格
        if "volume" in df.columns and "close" in df.columns:
            df["amount"] = df["volume"] * df["close"]
        else:
            df["amount"] = 0

        # 添加股票代码列
        df["code"] = stock_code

        # 只保留需要的列
        keep_cols = ["code"] + STANDARD_COLUMNS
        existing_cols = [col for col in keep_cols if col in df.columns]
        df = df[existing_cols]

        return df

    def get_main_indices(self) -> list[dict[str, Any]] | None:
        """
        获取主要指数行情 (Yahoo Finance)
        """
        import yfinance as yf

        # 映射关系：akshare代码 -> (yfinance代码, 名称)
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
                    ticker = yf.Ticker(yf_code)
                    # 获取最近2天数据以计算涨跌
                    hist = ticker.history(period="2d")
                    if hist.empty:
                        continue

                    today = hist.iloc[-1]
                    prev = hist.iloc[-2] if len(hist) > 1 else today

                    price = float(today["Close"])
                    prev_close = float(prev["Close"])
                    change = price - prev_close
                    change_pct = (change / prev_close) * 100 if prev_close else 0

                    # 振幅
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
                            "amount": 0.0,  # Yahoo Finance 可能不提供准确的成交额
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
        """
        Check if code is a US stock.

        US stock code rules:
        - 1-5 uppercase letters, e.g., 'AAPL', 'TSLA'
        - May contain '.', e.g., 'BRK.B'
        """
        return detect_stock_type(stock_code) == StockType.US

    def get_realtime_quote(self, stock_code: str, **kwargs) -> UnifiedRealtimeQuote | None:
        """
        获取美股实时行情数据

        数据来源：yfinance Ticker.info

        Args:
            stock_code: 美股代码，如 'AMD', 'AAPL', 'TSLA'

        Returns:
            UnifiedRealtimeQuote 对象，获取失败返回 None
        """
        import yfinance as yf

        # 仅处理美股
        if not self._is_us_stock(stock_code):
            logger.debug(f"[Yfinance] {stock_code} 不是美股，跳过")
            return None

        try:
            symbol = stock_code.strip().upper()
            logger.debug(f"[Yfinance] 获取美股 {symbol} 实时行情")

            ticker = yf.Ticker(symbol)

            # 尝试获取 fast_info（更快，但字段较少）
            try:
                info = ticker.fast_info
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
                # 回退到 history 方法获取最新数据
                logger.debug(f"[Yfinance] fast_info failed for {symbol}, falling back to history method")
                hist = ticker.history(period="2d")
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

            # 计算涨跌幅
            change_amount = None
            change_pct = None
            if price is not None and prev_close is not None and prev_close > 0:
                change_amount = price - prev_close
                change_pct = (change_amount / prev_close) * 100

            # 计算振幅
            amplitude = None
            if high is not None and low is not None and prev_close is not None and prev_close > 0:
                amplitude = ((high - low) / prev_close) * 100

            # 获取股票名称
            try:
                name = ticker.info.get("shortName", "") or ticker.info.get("longName", "") or symbol
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
                amount=None,  # yfinance 不直接提供成交额
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
