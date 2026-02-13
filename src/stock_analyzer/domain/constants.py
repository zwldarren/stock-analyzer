"""
业务常量定义

包含信号类型、emoji 映射等核心业务常量
"""

# ========== 信号类型常量 ==========
# 统一的信号类型定义，与 SignalType 枚举保持一致
# 支持三种核心信号：BUY（买入）、HOLD（持有）、SELL（卖出）

SIGNAL_BUY = "buy"
SIGNAL_HOLD = "hold"
SIGNAL_SELL = "sell"

# 信号类型到显示名称的映射
SIGNAL_DISPLAY_NAMES = {
    SIGNAL_BUY: "买入",
    SIGNAL_HOLD: "持有",
    SIGNAL_SELL: "卖出",
}

# ========== Emoji 常量 ==========
# 统一的 emoji 映射，确保整个项目使用一致的 emoji

# 信号类型对应的 emoji
SIGNAL_EMOJI_MAP: dict[str, str] = {
    SIGNAL_BUY: "🟢",
    SIGNAL_HOLD: "🟡",
    SIGNAL_SELL: "🔴",
}

# 通知类型对应的 emoji
ALERT_TYPE_EMOJI_MAP: dict[str, str] = {
    "info": "ℹ️",
    "warning": "⚠️",
    "error": "❌",
    "success": "✅",
}

# 报告相关的 emoji
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
    根据信号类型获取对应的 emoji

    Args:
        signal: 信号类型（buy/sell/hold），不区分大小写

    Returns:
        对应的 emoji 字符串
    """
    normalized = signal.lower().strip()
    return SIGNAL_EMOJI_MAP.get(normalized, "⚪")


def get_action_emoji(action: str) -> str:
    """
    根据操作类型获取对应的 emoji

    这是 get_signal_emoji 的别名，保持向后兼容

    Args:
        action: 操作类型（buy/sell/hold），不区分大小写

    Returns:
        对应的 emoji 字符串
    """
    return get_signal_emoji(action)


def get_alert_emoji(alert_type: str) -> str:
    """
    根据通知类型获取对应的 emoji

    Args:
        alert_type: 通知类型（info/warning/error/success），不区分大小写

    Returns:
        对应的 emoji 字符串
    """
    return ALERT_TYPE_EMOJI_MAP.get(alert_type.lower(), "📢")


def normalize_signal(signal: str) -> str:
    """
    标准化信号类型字符串

    将各种形式的信号表示统一为标准的小写形式：
    - "BUY", "Buy", "buy" -> "buy"
    - "SELL", "Sell", "sell" -> "sell"
    - "HOLD", "Hold", "hold" -> "hold"

    Args:
        signal: 信号类型字符串

    Returns:
        标准化的信号类型（"buy"/"sell"/"hold"）
    """
    normalized = signal.lower().strip()
    # 确保是有效的信号类型
    if normalized not in SIGNAL_EMOJI_MAP:
        normalized = SIGNAL_HOLD  # 默认为 hold
    return normalized


def get_signal_display_name(signal: str) -> str:
    """
    获取信号的中文显示名称

    Args:
        signal: 信号类型（buy/sell/hold）

    Returns:
        中文显示名称（买入/卖出/持有）
    """
    normalized = normalize_signal(signal)
    return SIGNAL_DISPLAY_NAMES.get(normalized, "持有")
