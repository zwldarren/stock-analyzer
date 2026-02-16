"""
Exception hierarchy for stock analyzer.
"""

import logging
from collections.abc import Callable
from functools import wraps
from typing import Any, TypeVar

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


class StockAnalyzerException(Exception):
    """Base exception for stock analyzer."""

    def __init__(self, message: str, code: str | None = None):
        super().__init__(message)
        self.code = code
        self.message = message

    def __str__(self) -> str:
        if self.code:
            return f"[{self.code}] {self.message}"
        return self.message


class DataFetchError(StockAnalyzerException):
    """Data fetch failed."""

    def __init__(self, message: str = "数据获取失败", code: str | None = None):
        super().__init__(message, code)


class RateLimitError(DataFetchError):
    """Rate limit exceeded."""

    def __init__(self, message: str = "API请求频率超限，请稍后重试", code: str | None = None):
        super().__init__(message, code)


class DataSourceUnavailableError(DataFetchError):
    """Data source unavailable."""

    def __init__(self, message: str = "数据源暂时不可用", code: str | None = None):
        super().__init__(message, code)


class StorageError(StockAnalyzerException):
    """Storage operation failed."""

    def __init__(self, message: str = "数据存储失败", code: str | None = None):
        super().__init__(message, code)


class ValidationError(StockAnalyzerException):
    """Data validation failed."""

    def __init__(self, message: str = "数据验证失败", code: str | None = None):
        super().__init__(message, code)


class AnalysisError(StockAnalyzerException):
    """Analysis process failed."""

    def __init__(self, message: str = "分析过程出错", code: str | None = None):
        super().__init__(message, code)


class NotificationError(StockAnalyzerException):
    """Notification send failed."""

    def __init__(self, message: str = "通知发送失败", code: str | None = None):
        super().__init__(message, code)


class ConfigurationError(StockAnalyzerException):
    """Configuration error."""

    def __init__(self, message: str = "配置错误", code: str | None = None):
        super().__init__(message, code)


def handle_errors(
    error_message: str,
    default_return: Any = None,
    raise_on: tuple[type[Exception], ...] = (Exception,),
    log_level: str = "error",
) -> Callable[[F], F]:
    """
    Error handling decorator.

    Unified handling of function exceptions, logging and returning default value.

    Args:
        error_message: Error message prefix
        default_return: Default return value on exception
        raise_on: Exception types to re-raise
        log_level: Log level (debug, info, warning, error)

    Example:
        @handle_errors("获取数据失败", default_return=None)
        def fetch_data(code: str) -> dict | None:
            return api.get_data(code)
    """

    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return func(*args, **kwargs)
            except raise_on:
                # Re-raise specified exceptions
                raise
            except Exception as e:
                # Log the error
                msg = f"{error_message}: {e}"
                if log_level == "debug":
                    logger.debug(msg)
                elif log_level == "info":
                    logger.info(msg)
                elif log_level == "warning":
                    logger.warning(msg)
                else:
                    logger.error(msg)
                return default_return

        return wrapper  # type: ignore[return-value]

    return decorator


def safe_execute(func: Callable[..., Any], default_return: Any = None, *args: Any, **kwargs: Any) -> Any:
    """
    Execute function safely with default return on error.

    Args:
        func: Function to execute
        default_return: Default return value on exception
        *args: Positional arguments for func
        **kwargs: Keyword arguments for func

    Returns:
        Function result or default_return on exception
    """
    try:
        return func(*args, **kwargs)
    except Exception:
        return default_return
