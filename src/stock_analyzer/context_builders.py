"""
Context builders for stock analysis.

Breaks down the monolithic _build_analysis_context() into focused functions.
Each builder is responsible for a specific domain of context data.
"""

import logging
from typing import TYPE_CHECKING, Any

import pandas as pd

from stock_analyzer.technical_indicators import calculate_all_indicators

if TYPE_CHECKING:
    from stock_analyzer.models import (
        UnifiedRealtimeQuote,
    )

logger = logging.getLogger(__name__)


def _get_default_technical_indicators(current_price: float = 0.0, ma20: float = 0.0) -> dict[str, float]:
    """Return default technical indicator values when calculation fails or data is insufficient.

    Uses reasonable estimates based on available price data to avoid showing zeros.
    """
    # Use current price or MA20 as base for estimates
    base_price = current_price if current_price > 0 else ma20

    # Estimate ATR as a reasonable percentage of price (typical daily volatility ~1-2%)
    estimated_atr = base_price * 0.015 if base_price > 0 else 0.0

    # Estimate Bollinger Bands based on typical 2% std dev around MA20
    bb_middle = ma20 if ma20 > 0 else current_price
    bb_width = bb_middle * 0.04 if bb_middle > 0 else 0.0  # 4% total width (2% each side)

    logger.debug(
        f"[技术指标] 使用默认值计算 - 当前价格: {current_price:.2f}, MA20: {ma20:.2f}, "
        f"布林中轨: {bb_middle:.2f}, ATR估计: {estimated_atr:.4f}"
    )

    return {
        "adx": 20.0,
        "rsi_14": 50.0,
        "rsi_28": 50.0,
        "macd": 0.0,
        "macd_signal": 0.0,
        "macd_hist": 0.0,
        "bb_upper": round(bb_middle + bb_width, 2) if bb_middle > 0 else 0.0,
        "bb_middle": round(bb_middle, 2),
        "bb_lower": round(bb_middle - bb_width, 2) if bb_middle > 0 else 0.0,
        "bb_position": 0.5,
        "atr": round(estimated_atr, 4),
        "atr_ratio": round(estimated_atr / base_price * 100, 4) if base_price > 0 else 0.0,
        "stochastic_k": 50.0,
        "stochastic_d": 50.0,
    }


def build_basic_context(
    stock_code: str,
    stock_name: str,
    daily_data: pd.DataFrame | None,
    realtime_quote: "UnifiedRealtimeQuote | None",
) -> dict[str, Any]:
    """Build basic context with stock info and raw data."""
    context: dict[str, Any] = {
        "code": stock_code,
        "stock_name": stock_name,
        "raw_data": daily_data.to_dict("records") if daily_data is not None and hasattr(daily_data, "to_dict") else [],
    }

    if realtime_quote:
        context["realtime"] = {
            "name": realtime_quote.name,
            "price": realtime_quote.price,
            "change_amount": realtime_quote.change_amount,
            "volume_ratio": realtime_quote.volume_ratio,
            "turnover_rate": realtime_quote.turnover_rate,
            "pe_ratio": realtime_quote.pe_ratio,
            "pb_ratio": realtime_quote.pb_ratio,
            "source": realtime_quote.source.value if realtime_quote.source else None,
        }

    return context


def build_technical_context(daily_data: pd.DataFrame | None) -> dict[str, Any]:
    """Build technical analysis context with indicators."""
    context: dict[str, Any] = {}

    if daily_data is None or daily_data.empty:
        return context

    # Calculate moving averages
    if "close" in daily_data.columns:
        daily_data["ma5"] = daily_data["close"].rolling(window=5, min_periods=1).mean()
        daily_data["ma10"] = daily_data["close"].rolling(window=10, min_periods=1).mean()
        daily_data["ma20"] = daily_data["close"].rolling(window=20, min_periods=1).mean()
        daily_data["ma60"] = daily_data["close"].rolling(window=60, min_periods=1).mean()
        daily_data["ma120"] = daily_data["close"].rolling(window=120, min_periods=1).mean()

    # Calculate volume ratio
    if "volume" in daily_data.columns:
        daily_data["volume_ratio"] = daily_data["volume"] / daily_data["volume"].rolling(window=5, min_periods=1).mean()

    # Add today's data
    latest = daily_data.iloc[-1]
    date_value = latest.get("date") or latest.get("trade_date", "")
    context["date"] = str(date_value) if date_value else ""

    context["today"] = {
        "date": str(date_value) if date_value else "",
        "open": latest.get("open"),
        "high": latest.get("high"),
        "low": latest.get("low"),
        "close": latest.get("close"),
        "volume": latest.get("volume"),
        "amount": latest.get("amount"),
        "pct_chg": latest.get("pct_chg"),
        "ma5": latest.get("ma5"),
        "ma10": latest.get("ma10"),
        "ma20": latest.get("ma20"),
        "volume_ratio": latest.get("volume_ratio"),
    }

    # Build technical indicators and add to context["today"]
    # This is needed because technical_agent expects indicators in the "today" dict
    technical_indicators = build_technical_indicators(daily_data)
    if technical_indicators:
        context["today"].update(technical_indicators)
        # Also store in context["technical_data"] for other consumers
        context["technical_data"] = technical_indicators

    # Calculate MA status
    close = latest.get("close") or 0
    ma5 = latest.get("ma5") or 0
    ma10 = latest.get("ma10") or 0
    ma20 = latest.get("ma20") or 0

    if close > ma5 > ma10 > ma20 > 0:
        context["ma_status"] = "多头排列 📈"
    elif close < ma5 < ma10 < ma20 and ma20 > 0:
        context["ma_status"] = "空头排列 📉"
    elif close > ma5 and ma5 > ma10:
        context["ma_status"] = "短期向好 🔼"
    elif close < ma5 and ma5 < ma10:
        context["ma_status"] = "短期走弱 🔽"
    else:
        context["ma_status"] = "震荡整理 ↔️"

    # Add yesterday's data
    if len(daily_data) > 1:
        prev = daily_data.iloc[-2]
        context["yesterday"] = {
            "close": prev.get("close"),
            "volume": prev.get("volume"),
        }
        # Calculate changes
        prev_close = prev.get("close") or 0
        prev_volume = prev.get("volume") or 0
        if prev_close > 0:
            context["price_change_ratio"] = round(((latest.get("close") or 0) - prev_close) / prev_close * 100, 2)
        if prev_volume > 0:
            context["volume_change_ratio"] = round(((latest.get("volume") or 0) / prev_volume), 2)

    return context


def build_technical_indicators(daily_data: pd.DataFrame | None) -> dict[str, Any]:
    """Build detailed technical indicators including RSI, MACD, Bollinger Bands, ATR, ADX, Stochastic."""
    if daily_data is None or daily_data.empty or "close" not in daily_data.columns:
        return {}

    technical_data: dict[str, Any] = {}
    technical_data["current_price"] = float(daily_data["close"].iloc[-1])
    technical_data["ma20"] = float(daily_data["ma20"].iloc[-1]) if "ma20" in daily_data.columns else 0
    technical_data["ma60"] = float(daily_data["ma60"].iloc[-1]) if "ma60" in daily_data.columns else 0
    technical_data["ma120"] = float(daily_data["ma120"].iloc[-1]) if "ma120" in daily_data.columns else 0

    # MA alignment detection
    current = technical_data["current_price"]
    ma20 = technical_data["ma20"]
    ma60 = technical_data["ma60"]
    ma120 = technical_data["ma120"]

    if current > ma20 > ma60 > ma120 and ma120 > 0:
        technical_data["ma_alignment"] = "bullish"
    elif current > ma20 > ma60:
        technical_data["ma_alignment"] = "medium_bullish"
    elif current > ma20:
        technical_data["ma_alignment"] = "short_bullish"
    else:
        technical_data["ma_alignment"] = "neutral"

    # Volume momentum
    if "volume" in daily_data.columns:
        avg_volume = daily_data["volume"].rolling(window=20, min_periods=1).mean()
        current_volume = daily_data["volume"].iloc[-1]
        avg_vol = avg_volume.iloc[-1] if len(avg_volume) > 0 else 1
        technical_data["volume_momentum"] = float((current_volume / avg_vol * 100) if avg_vol > 0 else 100)

        # Price-volume correlation
        if len(daily_data) >= 5:
            recent = daily_data.tail(5)
            price_change = recent["close"].iloc[-1] - recent["close"].iloc[0]
            volume_avg = recent["volume"].mean()
            technical_data["price_volume_correlation"] = (
                0.5 if price_change > 0 and recent["volume"].iloc[-1] > volume_avg else 0.3
            )

    # Calculate advanced technical indicators using the new module
    if all(col in daily_data.columns for col in ["high", "low", "close"]):
        # Fix: Filter out rows with NaN in any of the required columns to ensure consistent data
        valid_data = daily_data[["high", "low", "close"]].dropna()

        logger.debug(
            f"[技术指标] 原始数据: {len(daily_data)} 行, 有效数据(去除NaN): {len(valid_data)} 行, 需要 >= 30 行才能计算"
        )

        # Only calculate if we have enough valid data
        if len(valid_data) >= 30:
            high_prices = valid_data["high"].tolist()
            low_prices = valid_data["low"].tolist()
            close_prices = valid_data["close"].tolist()

            logger.debug(
                f"[技术指标] 开始计算 - high_prices: {len(high_prices)} 条, "
                f"low_prices: {len(low_prices)} 条, close_prices: {len(close_prices)} 条"
            )
            try:
                advanced_indicators = calculate_all_indicators(
                    high_prices=high_prices,
                    low_prices=low_prices,
                    close_prices=close_prices,
                )

                # RSI
                technical_data["rsi_14"] = advanced_indicators["rsi_14"]
                technical_data["rsi_28"] = advanced_indicators["rsi_28"]

                # MACD
                technical_data["macd"] = advanced_indicators["macd"]
                technical_data["macd_signal"] = advanced_indicators["macd_signal"]
                technical_data["macd_hist"] = advanced_indicators["macd_hist"]

                # Bollinger Bands
                technical_data["bb_upper"] = advanced_indicators["bb_upper"]
                technical_data["bb_middle"] = advanced_indicators["bb_middle"]
                technical_data["bb_lower"] = advanced_indicators["bb_lower"]
                technical_data["bb_position"] = advanced_indicators["bb_position"]

                # ATR
                technical_data["atr"] = advanced_indicators["atr"]
                technical_data["atr_ratio"] = advanced_indicators["atr_ratio"]

                # ADX
                technical_data["adx"] = advanced_indicators["adx"]

                # Stochastic
                technical_data["stochastic_k"] = advanced_indicators["stochastic_k"]
                technical_data["stochastic_d"] = advanced_indicators["stochastic_d"]

            except Exception as e:
                logger.warning(f"Failed to calculate advanced indicators: {e}")
                ma20 = technical_data.get("ma20", 0)
                technical_data.update(_get_default_technical_indicators(technical_data["current_price"], ma20))
        else:
            # Not enough data for advanced indicators
            logger.warning(
                f"[技术指标] 数据不足，无法计算高级指标 - 需要 >= 30 行有效数据，"
                f"实际只有 {len(valid_data)} 行，将使用默认值"
            )
            ma20 = technical_data.get("ma20", 0)
            technical_data.update(_get_default_technical_indicators(technical_data["current_price"], ma20))
    else:
        # Missing high/low data, use defaults
        ma20 = technical_data.get("ma20", 0)
        technical_data.update(_get_default_technical_indicators(technical_data["current_price"], ma20))

    # OBV trend
    if len(daily_data) >= 2 and "close" in daily_data.columns and "volume" in daily_data.columns:
        recent = daily_data.tail(5)
        price_up_days = sum(1 for i in range(1, len(recent)) if recent["close"].iloc[i] > recent["close"].iloc[i - 1])
        technical_data["obv_trend"] = "up" if price_up_days >= 3 else "neutral"

    # Up volume ratio
    if "close" in daily_data.columns and "volume" in daily_data.columns and len(daily_data) >= 5:
        recent = daily_data.tail(5)
        up_volume = 0
        down_volume = 0
        for i in range(1, len(recent)):
            if recent["close"].iloc[i] > recent["close"].iloc[i - 1]:
                up_volume += recent["volume"].iloc[i]
            else:
                down_volume += recent["volume"].iloc[i]
        total_volume = up_volume + down_volume
        technical_data["up_volume_ratio"] = float(up_volume / total_volume if total_volume > 0 else 0.5)

    # Volume spike detection
    if "volume" in daily_data.columns and len(daily_data) >= 2:
        avg_vol = daily_data["volume"].rolling(window=20, min_periods=1).mean().iloc[-1]
        current_vol = daily_data["volume"].iloc[-1]
        technical_data["volume_spike"] = current_vol > avg_vol * 1.5 if avg_vol > 0 else False

    return technical_data


def build_valuation_context(
    realtime_quote: "UnifiedRealtimeQuote | None",
    daily_data: pd.DataFrame | None,
    current_price: float,
    data_service: Any,
    stock_code: str,
) -> dict[str, Any]:
    """Build valuation context with PE/PB and industry comparison."""
    valuation_data: dict[str, Any] = {}

    if realtime_quote:
        pe = realtime_quote.pe_ratio
        pb = realtime_quote.pb_ratio
        if pe is not None and pe > 0:
            valuation_data["pe_ratio"] = float(pe)
        if pb is not None and pb > 0:
            valuation_data["pb_ratio"] = float(pb)

    # Get industry data
    industry_name = None
    try:
        industry_info = data_service.get_stock_industry(stock_code)
        if industry_info and industry_info.get("industry"):
            industry_name = industry_info["industry"]
    except Exception as e:
        logger.debug(f"Failed to get industry info for {stock_code}: {e}")

    if industry_name:
        try:
            industry_valuation = data_service.get_industry_valuation(industry_name)
            if industry_valuation:
                valuation_data["industry_pe"] = industry_valuation.get("avg_pe_ttm", 0)
                valuation_data["industry_pb"] = industry_valuation.get("avg_pb", 0)
                valuation_data["industry_name"] = industry_valuation.get("industry_name", industry_name)
        except Exception as e:
            logger.debug(f"Failed to get industry valuation for {industry_name}: {e}")

    # Calculate EPS and BVPS with validation
    pe_ratio = valuation_data.get("pe_ratio", 0)
    pb_ratio = valuation_data.get("pb_ratio", 0)

    if current_price > 0 and pe_ratio > 0 and 5 <= pe_ratio <= 100:
        valuation_data["eps"] = round(current_price / pe_ratio, 2)

    if current_price > 0 and pb_ratio > 0 and 0.5 <= pb_ratio <= 20:
        valuation_data["book_value_per_share"] = round(current_price / pb_ratio, 2)

    # Calculate industry deviation
    industry_pb = valuation_data.get("industry_pb", 0)
    if pb_ratio > 0 and industry_pb > 0:
        valuation_data["pb_deviation_from_industry"] = round((pb_ratio - industry_pb) / industry_pb * 100, 2)

    return valuation_data


def build_financial_context(
    realtime_quote: "UnifiedRealtimeQuote | None",
    daily_data: pd.DataFrame | None,
) -> dict[str, Any]:
    """Build financial metrics context."""
    financial_data: dict[str, Any] = {}

    if realtime_quote:
        pe = realtime_quote.pe_ratio
        pb = realtime_quote.pb_ratio
        if pe is not None:
            financial_data["pe_ratio"] = float(pe)
        if pb is not None:
            financial_data["pb_ratio"] = float(pb)

    if daily_data is not None and not daily_data.empty and "close" in daily_data.columns:
        returns = daily_data["close"].pct_change().dropna()
        if len(returns) > 0:
            financial_data["volatility"] = round(float(returns.std() * 100), 2)

        if len(daily_data) >= 20:
            price_20d_ago = daily_data["close"].iloc[-20]
            current = daily_data["close"].iloc[-1]
            if price_20d_ago > 0:
                financial_data["price_momentum_20d"] = round((current - price_20d_ago) / price_20d_ago * 100, 2)

    return financial_data


def build_growth_context(
    daily_data: pd.DataFrame | None,
    realtime_quote: "UnifiedRealtimeQuote | None",
) -> dict[str, Any]:
    """Build growth metrics context."""
    growth_data: dict[str, Any] = {}

    if daily_data is None or daily_data.empty or "close" not in daily_data.columns:
        return growth_data

    closes = daily_data["close"].tolist()
    if len(closes) >= 20:
        current_price = closes[-1]

        if len(closes) >= 20 and closes[-20] > 0:
            ret_1m = (current_price - closes[-20]) / closes[-20] * 100
            growth_data["price_momentum_1m"] = round(ret_1m, 2)

        if len(closes) >= 60 and closes[-60] > 0:
            ret_3m = (current_price - closes[-60]) / closes[-60] * 100
            growth_data["revenue_cagr"] = round(ret_3m, 2)
        elif len(closes) >= 40:
            ret_est = (current_price - closes[-40]) / closes[-40] * 100
            growth_data["revenue_cagr"] = round(ret_est * 6.3, 2)

    if realtime_quote:
        forward_pe = realtime_quote.pe_ratio
        if forward_pe is not None and forward_pe > 0:
            growth_data["forward_pe"] = float(forward_pe)

    return growth_data


def build_market_context(daily_data: pd.DataFrame | None) -> dict[str, Any]:
    """Build market context with relative strength."""
    market_data: dict[str, Any] = {}

    if daily_data is None or daily_data.empty or "close" not in daily_data.columns:
        return market_data

    closes = daily_data["close"].tolist()
    if len(closes) >= 20:
        current = closes[-1]
        past_20d = closes[-20]
        if past_20d > 0:
            stock_return = (current - past_20d) / past_20d
            market_data["relative_strength_ratio"] = round(1 + stock_return, 2)

            if stock_return > 0.1:
                market_data["rs_trend"] = "improving"
            elif stock_return < -0.1:
                market_data["rs_trend"] = "declining"
            else:
                market_data["rs_trend"] = "neutral"

    return market_data


def build_chip_context(chip_data: Any) -> dict[str, Any] | None:
    """Build chip distribution context."""
    if not chip_data:
        return None

    return {
        "profit_ratio": chip_data.profit_ratio,
        "avg_cost": chip_data.avg_cost,
        "concentration_90": chip_data.concentration_90,
        "concentration_70": chip_data.concentration_70,
    }


def build_price_data(daily_data: pd.DataFrame | None) -> dict[str, Any] | None:
    """Build price data for momentum analysis."""
    if daily_data is None or daily_data.empty or "close" not in daily_data.columns:
        return None

    closes = daily_data["close"].tolist()
    volumes = daily_data["volume"].tolist() if "volume" in daily_data.columns else []
    return {"close": closes, "volume": volumes}


def get_current_price(realtime_quote: "UnifiedRealtimeQuote | None", daily_data: pd.DataFrame | None) -> float:
    """Get current price from realtime quote or daily data."""
    if realtime_quote and realtime_quote.price is not None:
        return float(realtime_quote.price)
    if daily_data is not None and not daily_data.empty and "close" in daily_data.columns:
        return float(daily_data["close"].iloc[-1])
    return 0.0
