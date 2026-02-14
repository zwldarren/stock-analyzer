"""
Technical Indicators Calculation Module

Implements professional-grade technical indicators for stock analysis.
All calculations follow standard formulas used in quantitative finance.

Indicators Implemented:
- RSI (Relative Strength Index)
- MACD (Moving Average Convergence Divergence)
- Bollinger Bands
- ATR (Average True Range)
- ADX (Average Directional Index)
- Stochastic Oscillator
"""

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def calculate_rsi(close_prices: list[float] | pd.Series, period: int = 14) -> float:
    """
    Calculate RSI (Relative Strength Index).

    RSI = 100 - (100 / (1 + RS))
    RS = Average Gain / Average Loss

    Args:
        close_prices: List of closing prices
        period: RSI period (default 14)

    Returns:
        RSI value (0-100), returns 50.0 if insufficient data
    """
    if len(close_prices) < period + 1:
        return 50.0

    try:
        prices = pd.Series(close_prices)
        delta = prices.diff()

        # Separate gains and losses
        gains = delta.where(delta > 0, 0.0)
        losses = (-delta).where(delta < 0, 0.0)

        # Calculate average gains and losses using EMA
        avg_gain = gains.ewm(com=period - 1, min_periods=period).mean()
        avg_loss = losses.ewm(com=period - 1, min_periods=period).mean()

        # Calculate RS and RSI
        rs = avg_gain / avg_loss.replace(0, np.inf)
        rsi = 100 - (100 / (1 + rs))

        return round(float(rsi.iloc[-1]), 2)

    except Exception as e:
        logger.debug(f"RSI calculation failed: {e}")
        return 50.0


def calculate_macd(
    close_prices: list[float] | pd.Series,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
) -> dict[str, float]:
    """
    Calculate MACD (Moving Average Convergence Divergence).

    MACD Line = EMA(fast) - EMA(slow)
    Signal Line = EMA(MACD Line, signal_period)
    Histogram = MACD Line - Signal Line

    Args:
        close_prices: List of closing prices
        fast_period: Fast EMA period (default 12)
        slow_period: Slow EMA period (default 26)
        signal_period: Signal line period (default 9)

    Returns:
        Dict with macd, signal, histogram values
    """
    default_result = {"macd": 0.0, "signal": 0.0, "histogram": 0.0}

    # Use pandas EMA which can work with limited data
    # We only need enough data for the slow EMA to stabilize
    if len(close_prices) < slow_period:
        return default_result

    try:
        prices = pd.Series(close_prices)

        # Calculate EMAs
        ema_fast = prices.ewm(span=fast_period, adjust=False).mean()
        ema_slow = prices.ewm(span=slow_period, adjust=False).mean()

        # MACD line
        macd_line = ema_fast - ema_slow

        # Signal line
        signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()

        # Histogram
        histogram = macd_line - signal_line

        return {
            "macd": round(float(macd_line.iloc[-1]), 4),
            "signal": round(float(signal_line.iloc[-1]), 4),
            "histogram": round(float(histogram.iloc[-1]), 4),
        }

    except Exception as e:
        logger.debug(f"MACD calculation failed: {e}")
        return default_result


def calculate_bollinger_bands(
    close_prices: list[float] | pd.Series,
    period: int = 20,
    std_dev: float = 2.0,
) -> dict[str, float]:
    """
    Calculate Bollinger Bands.

    Middle Band = SMA(period)
    Upper Band = Middle + (std_dev * Standard Deviation)
    Lower Band = Middle - (std_dev * Standard Deviation)

    Args:
        close_prices: List of closing prices
        period: SMA period (default 20)
        std_dev: Number of standard deviations (default 2.0)

    Returns:
        Dict with upper, middle, lower bands and bb_position (0-1)
    """
    default_result = {"upper": 0.0, "middle": 0.0, "lower": 0.0, "bb_position": 0.5}

    if len(close_prices) < period:
        return default_result

    try:
        prices = pd.Series(close_prices)

        # Calculate middle band (SMA)
        middle = prices.rolling(window=period).mean()

        # Calculate standard deviation
        std = prices.rolling(window=period).std()

        # Calculate bands
        upper = middle + (std_dev * std)
        lower = middle - (std_dev * std)

        # Calculate position within bands (0 = at lower, 1 = at upper)
        current_price = prices.iloc[-1]
        upper_val = upper.iloc[-1]
        lower_val = lower.iloc[-1]

        bb_position = (current_price - lower_val) / (upper_val - lower_val) if upper_val != lower_val else 0.5

        # Clamp to reasonable range
        bb_position = max(0.0, min(1.0, float(bb_position)))

        return {
            "upper": round(float(upper_val), 2),
            "middle": round(float(middle.iloc[-1]), 2),
            "lower": round(float(lower_val), 2),
            "bb_position": round(bb_position, 3),
        }

    except Exception as e:
        logger.debug(f"Bollinger Bands calculation failed: {e}")
        return default_result


def calculate_atr(
    high_prices: list[float] | pd.Series,
    low_prices: list[float] | pd.Series,
    close_prices: list[float] | pd.Series,
    period: int = 14,
) -> dict[str, float]:
    """
    Calculate ATR (Average True Range).

    True Range = max(H-L, |H-PrevC|, |L-PrevC|)
    ATR = SMA(True Range, period)

    Args:
        high_prices: List of high prices
        low_prices: List of low prices
        close_prices: List of closing prices
        period: ATR period (default 14)

    Returns:
        Dict with atr and atr_ratio (ATR / current_price)
    """
    default_result = {"atr": 0.0, "atr_ratio": 0.0}

    if len(high_prices) < period + 1:
        return default_result

    try:
        high = pd.Series(high_prices)
        low = pd.Series(low_prices)
        close = pd.Series(close_prices)

        # Calculate True Range
        prev_close = close.shift(1)

        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()

        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # Calculate ATR using EMA (Wilder's method)
        atr = true_range.ewm(alpha=1 / period, min_periods=period).mean()

        current_price = close.iloc[-1]
        atr_value = atr.iloc[-1]

        atr_ratio = atr_value / current_price if current_price > 0 else 0.0

        return {
            "atr": round(float(atr_value), 4),
            "atr_ratio": round(float(atr_ratio), 4),
        }

    except Exception as e:
        logger.debug(f"ATR calculation failed: {e}")
        return default_result


def calculate_adx(
    high_prices: list[float] | pd.Series,
    low_prices: list[float] | pd.Series,
    close_prices: list[float] | pd.Series,
    period: int = 14,
) -> float:
    """
    Calculate ADX (Average Directional Index).

    ADX measures trend strength (not direction).
    ADX > 25 indicates a strong trend.
    ADX < 20 indicates a weak or no trend.

    Args:
        high_prices: List of high prices
        low_prices: List of low prices
        close_prices: List of closing prices
        period: ADX period (default 14)

    Returns:
        ADX value (0-100), returns 20.0 if insufficient data
    """
    if len(high_prices) < period * 2:
        return 20.0

    try:
        high = pd.Series(high_prices)
        low = pd.Series(low_prices)
        close = pd.Series(close_prices)

        # Calculate +DM and -DM
        up_move = high.diff()
        down_move = -low.diff()

        plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
        minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)

        # Calculate True Range
        prev_close = close.shift(1)
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # Smooth the values using Wilder's EMA
        atr = true_range.ewm(alpha=1 / period, min_periods=period).mean()
        plus_di = 100 * (plus_dm.ewm(alpha=1 / period, min_periods=period).mean() / atr)
        minus_di = 100 * (minus_dm.ewm(alpha=1 / period, min_periods=period).mean() / atr)

        # Calculate DX
        dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.inf))

        # Calculate ADX (smoothed DX)
        adx = dx.ewm(alpha=1 / period, min_periods=period).mean()

        return round(float(adx.iloc[-1]), 2)

    except Exception as e:
        logger.debug(f"ADX calculation failed: {e}")
        return 20.0


def calculate_stochastic(
    high_prices: list[float] | pd.Series,
    low_prices: list[float] | pd.Series,
    close_prices: list[float] | pd.Series,
    k_period: int = 14,
    d_period: int = 3,
) -> dict[str, float]:
    """
    Calculate Stochastic Oscillator.

    %K = (Current Close - Lowest Low) / (Highest High - Lowest Low) * 100
    %D = SMA(%K, d_period)

    Args:
        high_prices: List of high prices
        low_prices: List of low prices
        close_prices: List of closing prices
        k_period: %K period (default 14)
        d_period: %D period (default 3)

    Returns:
        Dict with k and d values (0-100)
    """
    default_result = {"k": 50.0, "d": 50.0}

    if len(high_prices) < k_period + d_period:
        return default_result

    try:
        high = pd.Series(high_prices)
        low = pd.Series(low_prices)
        close = pd.Series(close_prices)

        # Calculate %K
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()

        k = 100 * (close - lowest_low) / (highest_high - lowest_low).replace(0, np.inf)

        # Calculate %D (smoothed %K)
        d = k.rolling(window=d_period).mean()

        # Handle inf values
        k_val = float(k.iloc[-1])
        if np.isinf(k_val) or np.isnan(k_val):
            k_val = 50.0

        d_val = float(d.iloc[-1])
        if np.isinf(d_val) or np.isnan(d_val):
            d_val = 50.0

        return {
            "k": round(max(0.0, min(100.0, k_val)), 2),
            "d": round(max(0.0, min(100.0, d_val)), 2),
        }

    except Exception as e:
        logger.debug(f"Stochastic calculation failed: {e}")
        return default_result


def calculate_all_indicators(
    high_prices: list[float],
    low_prices: list[float],
    close_prices: list[float],
    volume: list[float] | None = None,
) -> dict[str, Any]:
    """
    Calculate all technical indicators at once.

    Args:
        high_prices: List of high prices
        low_prices: List of low prices
        close_prices: List of closing prices
        volume: Optional list of volume data

    Returns:
        Dict with all calculated indicators
    """
    result = {
        # RSI
        "rsi_14": calculate_rsi(close_prices, 14),
        "rsi_28": calculate_rsi(close_prices, 28),
        # MACD
        "macd": 0.0,
        "macd_signal": 0.0,
        "macd_hist": 0.0,
        # Bollinger Bands
        "bb_upper": 0.0,
        "bb_middle": 0.0,
        "bb_lower": 0.0,
        "bb_position": 0.5,
        # ATR
        "atr": 0.0,
        "atr_ratio": 0.0,
        # ADX
        "adx": 20.0,
        # Stochastic
        "stochastic_k": 50.0,
        "stochastic_d": 50.0,
    }

    try:
        # MACD
        macd_result = calculate_macd(close_prices)
        result["macd"] = macd_result["macd"]
        result["macd_signal"] = macd_result["signal"]
        result["macd_hist"] = macd_result["histogram"]

        # Bollinger Bands
        bb_result = calculate_bollinger_bands(close_prices)
        result["bb_upper"] = bb_result["upper"]
        result["bb_middle"] = bb_result["middle"]
        result["bb_lower"] = bb_result["lower"]
        result["bb_position"] = bb_result["bb_position"]

        # ATR
        atr_result = calculate_atr(high_prices, low_prices, close_prices)
        result["atr"] = atr_result["atr"]
        result["atr_ratio"] = atr_result["atr_ratio"]

        # ADX
        result["adx"] = calculate_adx(high_prices, low_prices, close_prices)

        # Stochastic
        stoch_result = calculate_stochastic(high_prices, low_prices, close_prices)
        result["stochastic_k"] = stoch_result["k"]
        result["stochastic_d"] = stoch_result["d"]

    except Exception as e:
        logger.warning(f"Error calculating indicators: {e}")

    return result


def interpret_rsi(rsi: float) -> str:
    """Interpret RSI value."""
    if rsi >= 80:
        return "severely_overbought"
    elif rsi >= 70:
        return "overbought"
    elif rsi >= 50:
        return "bullish"
    elif rsi >= 30:
        return "bearish"
    elif rsi >= 20:
        return "oversold"
    else:
        return "severely_oversold"


def interpret_stochastic(k: float, d: float) -> str:
    """Interpret Stochastic oscillator."""
    if k >= 80 and d >= 80:
        return "overbought"
    elif k <= 20 and d <= 20:
        return "oversold"
    elif k > d:
        return "bullish_crossover"
    elif k < d:
        return "bearish_crossover"
    else:
        return "neutral"


def interpret_macd(macd: float, signal: float, histogram: float) -> str:
    """Interpret MACD indicator."""
    if histogram > 0:
        if macd > signal:
            return "bullish"
        else:
            return "bullish_weakening"
    else:
        if macd < signal:
            return "bearish"
        else:
            return "bearish_weakening"


def interpret_adx(adx: float) -> str:
    """Interpret ADX trend strength."""
    if adx >= 50:
        return "extremely_strong_trend"
    elif adx >= 40:
        return "very_strong_trend"
    elif adx >= 25:
        return "strong_trend"
    elif adx >= 20:
        return "developing_trend"
    else:
        return "weak_or_no_trend"
