"""
领域层 - 核心业务逻辑和常量

该模块包含：
- 业务常量 (constants)
- 自定义异常 (exceptions)
- 股票名称解析 (stock_name_resolver)
"""

from stock_analyzer.domain.constants import (
    ALERT_TYPE_EMOJI_MAP,
    REPORT_EMOJI,
    SIGNAL_BUY,
    SIGNAL_EMOJI_MAP,
    SIGNAL_HOLD,
    SIGNAL_SELL,
    get_action_emoji,
    get_alert_emoji,
    get_signal_display_name,
    get_signal_emoji,
    normalize_signal,
)
from stock_analyzer.domain.exceptions import (
    AnalysisError,
    ConfigurationError,
    DataFetchError,
    DataSourceUnavailableError,
    NotificationError,
    RateLimitError,
    StockAnalyzerException,
    StorageError,
    ValidationError,
    handle_errors,
)
from stock_analyzer.domain.stock_name_resolver import (
    StockNameResolver,
    get_stock_name,
    get_stock_name_from_context,
)

__all__ = [
    # Exceptions
    "StockAnalyzerException",
    "DataFetchError",
    "RateLimitError",
    "DataSourceUnavailableError",
    "StorageError",
    "ValidationError",
    "AnalysisError",
    "NotificationError",
    "ConfigurationError",
    "handle_errors",
    # Constants
    "SIGNAL_BUY",
    "SIGNAL_SELL",
    "SIGNAL_HOLD",
    "SIGNAL_EMOJI_MAP",
    "ALERT_TYPE_EMOJI_MAP",
    "REPORT_EMOJI",
    # Functions
    "get_action_emoji",
    "get_alert_emoji",
    "get_signal_emoji",
    "normalize_signal",
    "get_signal_display_name",
    # Stock name resolver
    "StockNameResolver",
    "get_stock_name",
    "get_stock_name_from_context",
]
