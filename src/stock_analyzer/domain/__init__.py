"""
领域层 - 核心业务逻辑和常量

该模块包含：
- 业务常量 (constants)
- 自定义异常 (exceptions)
- 股票名称解析 (stock_name_resolver)
"""

from stock_analyzer.domain.constants import (
    get_action_emoji,
    get_alert_emoji,
    get_signal_emoji,
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
    "get_action_emoji",
    "get_alert_emoji",
    "get_signal_emoji",
    "StockNameResolver",
    "get_stock_name",
    "get_stock_name_from_context",
]
