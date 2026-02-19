"""
工具模块

提供通用的工具函数和配置。

子模块:
    - logging_config: 日志配置
    - stock_code: 股票代码处理工具
    - stock_name_resolver: 股票名称解析器
"""

from stock_analyzer.utils.logging_config import get_console, get_display, setup_logging
from stock_analyzer.utils.stock_code import (
    StockType,
    detect_stock_type,
    is_etf_code,
    is_hk_code,
    is_us_code,
)

__all__ = [
    "calculate_backoff_delay",
    "get_console",
    "get_display",
    "setup_logging",
    "StockType",
    "detect_stock_type",
    "is_us_code",
    "is_hk_code",
    "is_etf_code",
]


def calculate_backoff_delay(attempt: int, base_delay: float, max_delay: float = 60.0) -> float:
    """
    计算指数退避延迟时间

    使用指数退避算法计算重试延迟，避免频繁请求导致限流

    Args:
        attempt: 当前尝试次数（从0开始）
        base_delay: 基础延迟时间（秒）
        max_delay: 最大延迟时间（秒），默认60秒

    Returns:
        计算后的延迟时间（秒）

    Example:
        >>> calculate_backoff_delay(0, 2.0)
        2.0
        >>> calculate_backoff_delay(2, 2.0)
        8.0
        >>> calculate_backoff_delay(10, 2.0, max_delay=30.0)
        30.0
    """
    if attempt <= 0:
        return base_delay
    delay = base_delay * (2 ** (attempt - 1))
    return min(delay, max_delay)
