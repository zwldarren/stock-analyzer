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
