"""
===================================
EfinanceFetcher - 优先数据源 (Priority 0)
===================================

数据来源：东方财富爬虫（通过 efinance 库）
特点：免费、无需 Token、数据全面、API 简洁
仓库：https://github.com/Micro-sheep/efinance

与 AkshareFetcher 类似，但 efinance 库：
1. API 更简洁易用
2. 支持批量获取数据
3. 更稳定的接口封装

防封禁策略：
1. 使用 AsyncRateLimiter 限流
2. 随机轮换 User-Agent
3. 使用 tenacity 实现指数退避重试
4. 熔断器机制：连续失败后自动冷却
"""

import asyncio
import logging
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

from ashare_analyzer.data.base import STANDARD_COLUMNS, BaseFetcher
from ashare_analyzer.exceptions import DataFetchError, RateLimitError
from ashare_analyzer.infrastructure import AsyncRateLimiter
from ashare_analyzer.models import RealtimeSource, UnifiedRealtimeQuote
from ashare_analyzer.utils.stock_code import is_etf_code, is_us_code

from .realtime_types import get_realtime_circuit_breaker, safe_float, safe_int

logger = logging.getLogger(__name__)

_realtime_cache: TTLCache[str, Any] = TTLCache(maxsize=1, ttl=1200)
_etf_realtime_cache: TTLCache[str, Any] = TTLCache(maxsize=1, ttl=1200)


class EfinanceFetcher(BaseFetcher):
    """
    Efinance 数据源实现 (async)

    优先级：0（最高，优先于 AkshareFetcher）
    数据来源：东方财富网（通过 efinance 库封装）
    仓库：https://github.com/Micro-sheep/efinance

    主要 API：
    - ef.stock.get_quote_history(): 获取历史 K 线数据
    - ef.stock.get_base_info(): 获取股票基本信息
    - ef.stock.get_realtime_quotes(): 获取实时行情

    关键策略：
    - 使用 AsyncRateLimiter 限流
    - 随机 User-Agent 轮换
    - 失败后指数退避重试（最多5次）
    """

    name = "EfinanceFetcher"

    @property
    def priority(self) -> int:
        from ashare_analyzer.config import get_config

        return get_config().datasource.efinance_priority

    def __init__(self, rate_limiter: AsyncRateLimiter | None = None):
        super().__init__(rate_limiter=rate_limiter)

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=4, max=60),
        retry=retry_if_exception_type((ConnectionError, TimeoutError, Exception)),
        before_sleep=before_sleep_log(cast(Any, logger), logging.WARNING),
    )
    async def _fetch_raw_data(self, stock_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        if is_us_code(stock_code):
            raise DataFetchError(f"EfinanceFetcher 不支持美股 {stock_code}，请使用 AkshareFetcher 或 YfinanceFetcher")

        if is_etf_code(stock_code):
            return await self._fetch_etf_data(stock_code, start_date, end_date)
        else:
            return await self._fetch_stock_data(stock_code, start_date, end_date)

    async def _fetch_stock_data(self, stock_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        import efinance as ef

        self._set_random_user_agent()
        await self._enforce_rate_limit()

        beg_date = start_date.replace("-", "")
        end_date_fmt = end_date.replace("-", "")

        logger.debug(
            f"[API调用] ef.stock.get_quote_history(stock_codes={stock_code}, "
            f"beg={beg_date}, end={end_date_fmt}, klt=101, fqt=1)"
        )

        try:
            import time as _time

            api_start = _time.time()

            def _call_api():
                return ef.stock.get_quote_history(
                    stock_codes=stock_code,
                    beg=beg_date,
                    end=end_date_fmt,
                    klt=101,
                    fqt=1,
                )

            df = await asyncio.to_thread(_call_api)

            api_elapsed = _time.time() - api_start

            if df is not None and isinstance(df, pd.DataFrame) and not df.empty:
                logger.debug(
                    f"[API返回] ef.stock.get_quote_history 成功: 返回 {len(df)} 行数据, 耗时 {api_elapsed:.2f}s"
                )
            else:
                logger.warning(f"[API返回] ef.stock.get_quote_history 返回空数据, 耗时 {api_elapsed:.2f}s")

            if isinstance(df, dict):
                raise DataFetchError(f"efinance returned dict instead of DataFrame for {stock_code}")
            return df

        except Exception as e:
            error_msg = str(e).lower()

            if any(keyword in error_msg for keyword in ["banned", "blocked", "频率", "rate", "限制"]):
                logger.warning(f"检测到可能被封禁: {e}")
                raise RateLimitError(f"efinance 可能被限流: {e}") from e

            raise DataFetchError(f"efinance 获取数据失败: {e}") from e

    async def _fetch_etf_data(self, stock_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        import efinance as ef

        self._set_random_user_agent()
        await self._enforce_rate_limit()

        logger.debug(f"[API调用] ef.fund.get_quote_history(fund_code={stock_code})")

        try:
            import time as _time

            api_start = _time.time()

            def _call_api():
                return ef.fund.get_quote_history(fund_code=stock_code)

            df = await asyncio.to_thread(_call_api)

            if df is not None and not df.empty and "日期" in df.columns:
                mask = (df["日期"] >= start_date) & (df["日期"] <= end_date)
                df = df[mask].copy()

            api_elapsed = _time.time() - api_start

            if df is not None and not df.empty:
                logger.debug(
                    f"[API返回] ef.fund.get_quote_history 成功: 返回 {len(df)} 行数据, 耗时 {api_elapsed:.2f}s"
                )
            else:
                logger.warning(f"[API返回] ef.fund.get_quote_history 返回空数据, 耗时 {api_elapsed:.2f}s")

            return df

        except Exception as e:
            error_msg = str(e).lower()

            if any(keyword in error_msg for keyword in ["banned", "blocked", "频率", "rate", "限制"]):
                logger.warning(f"检测到可能被封禁: {e}")
                raise RateLimitError(f"efinance 可能被限流: {e}") from e

            raise DataFetchError(f"efinance 获取 ETF 数据失败: {e}") from e

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
            "股票代码": "code",
            "股票名称": "name",
            "基金代码": "code",
            "基金名称": "name",
            "单位净值": "close",
        }

        df = df.rename(columns=column_mapping)

        if "close" in df.columns and "open" not in df.columns:
            df["open"] = df["close"]
            df["high"] = df["close"]
            df["low"] = df["close"]

        if "volume" not in df.columns:
            df["volume"] = 0
        if "amount" not in df.columns:
            df["amount"] = 0

        if "code" not in df.columns:
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
        if is_etf_code(stock_code):
            return await self._get_etf_realtime_quote(stock_code)

        import efinance as ef

        circuit_breaker = get_realtime_circuit_breaker()
        source_key = "efinance"

        if not circuit_breaker.is_available(source_key):
            logger.warning(f"[熔断] 数据源 {source_key} 处于熔断状态，跳过")
            return None

        try:
            cache_key = "realtime_data"
            if cache_key in _realtime_cache:
                df = _realtime_cache[cache_key]
                logger.debug("[缓存命中] 实时行情(efinance)")
            else:
                logger.debug("[缓存未命中] 触发全量刷新 实时行情(efinance)")
                self._set_random_user_agent()
                await self._enforce_rate_limit()

                logger.debug("[API调用] ef.stock.get_realtime_quotes() 获取实时行情...")
                import time as _time

                api_start = _time.time()

                def _call_api():
                    return ef.stock.get_realtime_quotes()

                df = await asyncio.to_thread(_call_api)

                api_elapsed = _time.time() - api_start
                logger.debug(
                    f"[API返回] ef.stock.get_realtime_quotes 成功: 返回 {len(df)} 只股票, 耗时 {api_elapsed:.2f}s"
                )
                circuit_breaker.record_success(source_key)

                _realtime_cache[cache_key] = df
                logger.debug("[缓存更新] 实时行情(efinance) 缓存已刷新")

            code_col = "股票代码" if "股票代码" in df.columns else "code"
            row = df[df[code_col] == stock_code]
            if row.empty:
                logger.warning(f"[API返回] 未找到股票 {stock_code} 的实时行情")
                return None

            row = row.iloc[0]

            name_col = "股票名称" if "股票名称" in df.columns else "name"
            price_col = "最新价" if "最新价" in df.columns else "price"
            pct_col = "涨跌幅" if "涨跌幅" in df.columns else "pct_chg"
            chg_col = "涨跌额" if "涨跌额" in df.columns else "change"
            vol_col = "成交量" if "成交量" in df.columns else "volume"
            amt_col = "成交额" if "成交额" in df.columns else "amount"
            turn_col = "换手率" if "换手率" in df.columns else "turnover_rate"
            amp_col = "振幅" if "振幅" in df.columns else "amplitude"
            high_col = "最高" if "最高" in df.columns else "high"
            low_col = "最低" if "最低" in df.columns else "low"
            open_col = "开盘" if "开盘" in df.columns else "open"
            vol_ratio_col = "量比" if "量比" in df.columns else "volume_ratio"
            pe_col = (
                "动态市盈率" if "动态市盈率" in df.columns else ("市盈率" if "市盈率" in df.columns else "pe_ratio")
            )
            pb_col = "市净率" if "市净率" in df.columns else "pb_ratio"
            total_mv_col = "总市值" if "总市值" in df.columns else "total_mv"
            circ_mv_col = "流通市值" if "流通市值" in df.columns else "circ_mv"

            quote = UnifiedRealtimeQuote(
                code=stock_code,
                name=str(row.get(name_col, "")),
                source=RealtimeSource.EFINANCE,
                price=safe_float(row.get(price_col)),
                change_pct=safe_float(row.get(pct_col)),
                change_amount=safe_float(row.get(chg_col)),
                volume=safe_int(row.get(vol_col)),
                amount=safe_float(row.get(amt_col)),
                turnover_rate=safe_float(row.get(turn_col)),
                amplitude=safe_float(row.get(amp_col)),
                high=safe_float(row.get(high_col)),
                low=safe_float(row.get(low_col)),
                open_price=safe_float(row.get(open_col)),
                volume_ratio=safe_float(row.get(vol_ratio_col)),
                pe_ratio=safe_float(row.get(pe_col)),
                pb_ratio=safe_float(row.get(pb_col)),
                total_mv=safe_float(row.get(total_mv_col)),
                circ_mv=safe_float(row.get(circ_mv_col)),
            )

            logger.debug(
                f"[实时行情-efinance] {stock_code} {quote.name}: 价格={quote.price}, 涨跌={quote.change_pct}%, "
                f"量比={quote.volume_ratio}, 换手率={quote.turnover_rate}%"
            )
            return quote

        except Exception as e:
            logger.error(f"[API错误] 获取 {stock_code} 实时行情(efinance)失败: {e}")
            circuit_breaker.record_failure(source_key, str(e))
            return None

    async def _get_etf_realtime_quote(self, stock_code: str) -> UnifiedRealtimeQuote | None:
        import efinance as ef

        circuit_breaker = get_realtime_circuit_breaker()
        source_key = "efinance_etf"

        if not circuit_breaker.is_available(source_key):
            logger.warning(f"[熔断] 数据源 {source_key} 处于熔断状态，跳过")
            return None

        try:
            cache_key = "etf_realtime_data"
            if cache_key in _etf_realtime_cache:
                df = _etf_realtime_cache[cache_key]
                logger.debug("[缓存命中] ETF实时行情(efinance)")
            else:
                self._set_random_user_agent()
                await self._enforce_rate_limit()

                logger.debug("[API调用] ef.stock.get_realtime_quotes(['ETF']) 获取ETF实时行情...")
                import time as _time

                api_start = _time.time()

                def _call_api():
                    return ef.stock.get_realtime_quotes(["ETF"])

                df = await asyncio.to_thread(_call_api)
                api_elapsed = _time.time() - api_start

                if df is not None and not df.empty:
                    logger.debug(f"[API返回] ETF 实时行情成功: {len(df)} 条, 耗时 {api_elapsed:.2f}s")
                    circuit_breaker.record_success(source_key)
                else:
                    logger.warning(f"[API返回] ETF 实时行情为空, 耗时 {api_elapsed:.2f}s")
                    df = pd.DataFrame()

                _etf_realtime_cache[cache_key] = df

            if df is None or df.empty:
                logger.warning(f"[实时行情] ETF实时行情数据为空(efinance)，跳过 {stock_code}")
                return None

            code_col = "股票代码" if "股票代码" in df.columns else "code"
            code_series = df[code_col].astype(str).str.zfill(6)
            target_code = str(stock_code).strip().zfill(6)
            row = df[code_series == target_code]
            if row.empty:
                logger.warning(f"[API返回] 未找到 ETF {stock_code} 的实时行情(efinance)")
                return None

            row = row.iloc[0]
            name_col = "股票名称" if "股票名称" in df.columns else "name"
            price_col = "最新价" if "最新价" in df.columns else "price"
            pct_col = "涨跌幅" if "涨跌幅" in df.columns else "pct_chg"
            chg_col = "涨跌额" if "涨跌额" in df.columns else "change"
            vol_col = "成交量" if "成交量" in df.columns else "volume"
            amt_col = "成交额" if "成交额" in df.columns else "amount"
            turn_col = "换手率" if "换手率" in df.columns else "turnover_rate"
            amp_col = "振幅" if "振幅" in df.columns else "amplitude"
            high_col = "最高" if "最高" in df.columns else "high"
            low_col = "最低" if "最低" in df.columns else "low"
            open_col = "开盘" if "开盘" in df.columns else "open"

            quote = UnifiedRealtimeQuote(
                code=target_code,
                name=str(row.get(name_col, "")),
                source=RealtimeSource.EFINANCE,
                price=safe_float(row.get(price_col)),
                change_pct=safe_float(row.get(pct_col)),
                change_amount=safe_float(row.get(chg_col)),
                volume=safe_int(row.get(vol_col)),
                amount=safe_float(row.get(amt_col)),
                turnover_rate=safe_float(row.get(turn_col)),
                amplitude=safe_float(row.get(amp_col)),
                high=safe_float(row.get(high_col)),
                low=safe_float(row.get(low_col)),
                open_price=safe_float(row.get(open_col)),
            )

            logger.debug(
                f"[ETF实时行情-efinance] {target_code} {quote.name}: "
                f"价格={quote.price}, 涨跌={quote.change_pct}%, 换手率={quote.turnover_rate}%"
            )
            return quote
        except Exception as e:
            logger.error(f"[API错误] 获取 ETF {stock_code} 实时行情(efinance)失败: {e}")
            circuit_breaker.record_failure(source_key, str(e))
            return None

    async def get_main_indices(self) -> list[dict[str, Any]] | None:
        import efinance as ef

        indices_map = {
            "000001": ("上证指数", "sh000001"),
            "399001": ("深证成指", "sz399001"),
            "399006": ("创业板指", "sz399006"),
            "000688": ("科创50", "sh000688"),
            "000016": ("上证50", "sh000016"),
            "000300": ("沪深300", "sh000300"),
        }

        try:
            self._set_random_user_agent()
            await self._enforce_rate_limit()

            logger.debug("[API调用] ef.stock.get_realtime_quotes(['沪深系列指数']) 获取指数行情...")
            import time as _time

            api_start = _time.time()

            def _call_api():
                return ef.stock.get_realtime_quotes(["沪深系列指数"])

            df = await asyncio.to_thread(_call_api)
            api_elapsed = _time.time() - api_start

            if df is None or df.empty:
                logger.warning(f"[API返回] 指数行情为空, 耗时 {api_elapsed:.2f}s")
                return None

            logger.debug(f"[API返回] 指数行情成功: {len(df)} 条, 耗时 {api_elapsed:.2f}s")
            code_col = "股票代码" if "股票代码" in df.columns else "code"
            code_series = df[code_col].astype(str).str.zfill(6)

            results: list[dict[str, Any]] = []
            for code, (name, full_code) in indices_map.items():
                row = df[code_series == code]
                if row.empty:
                    continue
                item = row.iloc[0]

                price_col = "最新价" if "最新价" in df.columns else "price"
                pct_col = "涨跌幅" if "涨跌幅" in df.columns else "pct_chg"
                chg_col = "涨跌额" if "涨跌额" in df.columns else "change"
                open_col = "开盘" if "开盘" in df.columns else "open"
                high_col = "最高" if "最高" in df.columns else "high"
                low_col = "最低" if "最低" in df.columns else "low"
                vol_col = "成交量" if "成交量" in df.columns else "volume"
                amt_col = "成交额" if "成交额" in df.columns else "amount"
                amp_col = "振幅" if "振幅" in df.columns else "amplitude"

                current = safe_float(item.get(price_col, 0))
                change_amount = safe_float(item.get(chg_col, 0))

                results.append(
                    {
                        "code": full_code,
                        "name": name,
                        "current": current,
                        "change": change_amount,
                        "change_pct": safe_float(item.get(pct_col, 0)),
                        "open": safe_float(item.get(open_col, 0)),
                        "high": safe_float(item.get(high_col, 0)),
                        "low": safe_float(item.get(low_col, 0)),
                        "prev_close": (current - change_amount) if (current and change_amount) else 0,
                        "volume": safe_float(item.get(vol_col, 0)),
                        "amount": safe_float(item.get(amt_col, 0)),
                        "amplitude": safe_float(item.get(amp_col, 0)),
                    }
                )

            if results:
                logger.info(f"[efinance] 获取到 {len(results)} 个指数行情")
            return results if results else None
        except Exception as e:
            logger.error(f"[efinance] 获取指数行情失败: {e}")
            return None

    async def get_market_stats(self) -> dict[str, Any] | None:
        import efinance as ef

        try:
            self._set_random_user_agent()
            await self._enforce_rate_limit()

            cache_key = "realtime_data"
            df = _realtime_cache.get(cache_key)

            if df is None:
                logger.debug("[API调用] ef.stock.get_realtime_quotes() 获取市场统计...")

                def _call_api():
                    return ef.stock.get_realtime_quotes()

                df = await asyncio.to_thread(_call_api)
                _realtime_cache[cache_key] = df

            if df is None or df.empty:
                logger.warning("[API返回] 市场统计数据为空")
                return None

            change_col = "涨跌幅" if "涨跌幅" in df.columns else "pct_chg"
            amount_col = "成交额" if "成交额" in df.columns else "amount"
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
            if amount_col in df.columns:
                df[amount_col] = pd.to_numeric(df[amount_col], errors="coerce")
                stats["total_amount"] = df[amount_col].sum() / 1e8
            return stats
        except Exception as e:
            logger.error(f"[efinance] 获取市场统计失败: {e}")
            return None

    async def get_sector_rankings(self, n: int = 5) -> tuple[list[dict], list[dict]] | None:
        import efinance as ef

        try:
            self._set_random_user_agent()
            await self._enforce_rate_limit()

            logger.debug("[API调用] ef.stock.get_realtime_quotes(['行业板块']) 获取板块行情...")

            def _call_api():
                return ef.stock.get_realtime_quotes(["行业板块"])

            df = await asyncio.to_thread(_call_api)
            if df is None or df.empty:
                logger.warning("[efinance] 板块行情数据为空")
                return None

            change_col = "涨跌幅" if "涨跌幅" in df.columns else "pct_chg"
            name_col = "股票名称" if "股票名称" in df.columns else "name"
            if change_col not in df.columns or name_col not in df.columns:
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
            logger.error(f"[efinance] 获取板块排行失败: {e}")
            return None

    async def get_base_info(self, stock_code: str) -> dict[str, Any] | None:
        import efinance as ef

        try:
            self._set_random_user_agent()
            await self._enforce_rate_limit()

            logger.debug(f"[API调用] ef.stock.get_base_info(stock_codes={stock_code}) 获取基本信息...")
            import time as _time

            api_start = _time.time()

            def _call_api():
                return ef.stock.get_base_info(stock_code)

            info = await asyncio.to_thread(_call_api)

            api_elapsed = _time.time() - api_start
            logger.debug(f"[API返回] ef.stock.get_base_info 成功, 耗时 {api_elapsed:.2f}s")

            if info is None:
                logger.warning(f"[API返回] 未获取到 {stock_code} 的基本信息")
                return None

            if isinstance(info, pd.Series):
                return info.to_dict()
            elif isinstance(info, pd.DataFrame) and not info.empty:
                return info.iloc[0].to_dict()

            return None

        except Exception as e:
            logger.error(f"[API错误] 获取 {stock_code} 基本信息失败: {e}")
            return None

    async def get_belong_board(self, stock_code: str) -> pd.DataFrame | None:
        import efinance as ef

        try:
            self._set_random_user_agent()
            await self._enforce_rate_limit()

            logger.debug(f"[API调用] ef.stock.get_belong_board(stock_code={stock_code}) 获取所属板块...")
            import time as _time

            api_start = _time.time()

            def _call_api():
                return ef.stock.get_belong_board(stock_code)

            df = await asyncio.to_thread(_call_api)

            api_elapsed = _time.time() - api_start

            if df is not None and not df.empty:
                logger.debug(
                    f"[API返回] ef.stock.get_belong_board 成功: 返回 {len(df)} 个板块, 耗时 {api_elapsed:.2f}s"
                )
            return df

        except Exception as e:
            logger.error(f"[API错误] 获取 {stock_code} 所属板块失败: {e}")
            return None

    async def get_enhanced_data(self, stock_code: str, days: int = 60) -> dict[str, Any]:
        result = {
            "code": stock_code,
            "daily_data": None,
            "realtime_quote": None,
            "base_info": None,
            "belong_board": None,
        }

        try:
            df = await self.get_daily_data(stock_code, days=days)
            result["daily_data"] = df
        except Exception as e:
            logger.error(f"获取 {stock_code} 日线数据失败: {e}")

        result["realtime_quote"] = await self.get_realtime_quote(stock_code)
        result["base_info"] = await self.get_base_info(stock_code)
        result["belong_board"] = await self.get_belong_board(stock_code)

        return result
