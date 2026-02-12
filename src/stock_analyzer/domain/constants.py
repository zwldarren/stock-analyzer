"""
业务常量定义

包含 emoji 常量等核心业务常量
"""

# ========== Emoji 常量 ==========
# 统一的 emoji 映射，确保整个项目使用一致的 emoji

# 操作类型对应的 emoji（英文）
ACTION_EMOJI_MAP: dict[str, str] = {
    "BUY": "🟢",
    "HOLD": "🟡",
    "SELL": "🔴",
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


def get_action_emoji(action: str) -> str:
    """根据操作类型获取对应的 emoji"""
    return ACTION_EMOJI_MAP.get(action, "⚪")


def get_signal_emoji(signal: str) -> str:
    """根据信号类型获取对应的 emoji"""
    return ACTION_EMOJI_MAP.get(signal.upper(), "⚪")


def get_alert_emoji(alert_type: str) -> str:
    """根据通知类型获取对应的 emoji"""
    return ALERT_TYPE_EMOJI_MAP.get(alert_type, "📢")
