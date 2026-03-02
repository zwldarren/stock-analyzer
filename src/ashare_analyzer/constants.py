"""
Business constants for stock analyzer.

Contains signal types, emoji mappings, and other core business constants.
"""

# ========== Signal Type Constants ==========
# Unified signal type definitions, consistent with SignalType enum
# Supports three core signals: BUY, HOLD, SELL

SIGNAL_BUY = "buy"
SIGNAL_HOLD = "hold"
SIGNAL_SELL = "sell"

# Signal type to display name mapping
SIGNAL_DISPLAY_NAMES = {
    SIGNAL_BUY: "买入",
    SIGNAL_HOLD: "持有",
    SIGNAL_SELL: "卖出",
}

# ========== Emoji Constants ==========
# Unified emoji mapping for consistent usage across the project

# Signal type emoji mapping
SIGNAL_EMOJI_MAP: dict[str, str] = {
    SIGNAL_BUY: "🟢",
    SIGNAL_HOLD: "🟡",
    SIGNAL_SELL: "🔴",
}

# Alert type emoji mapping
ALERT_TYPE_EMOJI_MAP: dict[str, str] = {
    "info": "ℹ️",
    "warning": "⚠️",
    "error": "❌",
    "success": "✅",
}

# Report related emoji
REPORT_EMOJI = {
    "title": "🎯",
    "dashboard": "📊",
    "market": "📈",
    "ai": "🤖",
    "money": "💰",
    "risk": "⚠️",
}


def get_signal_emoji(signal: str) -> str:
    """
    Get emoji for signal type.

    Args:
        signal: Signal type (buy/sell/hold), case insensitive

    Returns:
        Corresponding emoji string
    """
    normalized = signal.lower().strip()
    return SIGNAL_EMOJI_MAP.get(normalized, "⚪")


def get_alert_emoji(alert_type: str) -> str:
    """
    Get emoji for alert type.

    Args:
        alert_type: Alert type (info/warning/error/success), case insensitive

    Returns:
        Corresponding emoji string
    """
    return ALERT_TYPE_EMOJI_MAP.get(alert_type.lower(), "📢")


def normalize_signal(signal: str) -> str:
    """
    Normalize signal type string.

    Converts various signal representations to standard lowercase form:
    - "BUY", "Buy", "buy" -> "buy"
    - "SELL", "Sell", "sell" -> "sell"
    - "HOLD", "Hold", "hold" -> "hold"

    Args:
        signal: Signal type string

    Returns:
        Normalized signal type ("buy"/"sell"/"hold")
    """
    normalized = signal.lower().strip()
    # Ensure it's a valid signal type
    if normalized not in SIGNAL_EMOJI_MAP:
        normalized = SIGNAL_HOLD  # Default to hold
    return normalized


def get_signal_display_name(signal: str) -> str:
    """
    Get Chinese display name for signal.

    Args:
        signal: Signal type (buy/sell/hold)

    Returns:
        Chinese display name (买入/卖出/持有)
    """
    normalized = normalize_signal(signal)
    return SIGNAL_DISPLAY_NAMES.get(normalized, "持有")


# ========== A-Share Valuation Parameters ==========
# Parameters adjusted for A-share market characteristics

# Market premium: A-shares typically trade at higher multiples than US stocks
A_SHARE_MARKET_PREMIUM = 2.0

# Risk-free rate: 10-year Chinese government bond yield ~3%
A_SHARE_RISK_FREE_RATE = 0.03

# Risk premium: Additional return required for A-share risk
A_SHARE_RISK_PREMIUM = 0.05

# Default valuation multiples when no data available
DEFAULT_PB = 2.0  # Default P/B ratio
DEFAULT_PS = 2.0  # Default P/S ratio

# ========== A-Share Valuation Scoring Thresholds ==========
# Thresholds for value scoring in StyleAgent (A-share adjusted)
# A-share market has higher average valuations:
# - Average PE ~20-30 (vs US 15-20)
# - Average PB ~2-3 (vs US 1.5-2)

# P/E scoring thresholds (scores: 3/2/1 points)
A_SHARE_PE_EXCELLENT = 20  # < 20: 3 points (undervalued)
A_SHARE_PE_GOOD = 25  # < 25: 2 points (fair value)
A_SHARE_PE_ACCEPTABLE = 35  # < 35: 1 point (slightly overvalued but acceptable)

# P/B scoring thresholds (scores: 2/1 points)
A_SHARE_PB_EXCELLENT = 2.5  # < 2.5: 2 points (undervalued)
A_SHARE_PB_GOOD = 3.5  # < 3.5: 1 point (fair value)

# Margin of safety thresholds (scores: 3/2 points)
A_SHARE_MOS_STRONG = 20  # > 20%: 3 points (strong margin)
A_SHARE_MOS_MODERATE = 10  # > 10%: 2 points (moderate margin)

# Graham number calculation multiplier for A-shares
# Traditional Graham formula uses 22.5 (15 PE x 1.5 PB)
# A-share adjusted uses lower multiplier for more conservative valuation
A_SHARE_GRAHAM_MULTIPLIER = 18.0  # Adjusted from 22.5 for A-share conservatism
