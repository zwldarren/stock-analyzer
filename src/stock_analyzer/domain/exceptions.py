"""领域层异常定义"""

import logging
from collections.abc import Callable
from functools import wraps
from typing import Any, TypeVar

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


class StockAnalyzerException(Exception):
    """基础异常类"""

    def __init__(self, message: str, code: str | None = None):
        super().__init__(message)
        self.code = code
        self.message = message

    def __str__(self) -> str:
        if self.code:
            return f"[{self.code}] {self.message}"
        return self.message


class DataFetchError(StockAnalyzerException):
    """数据获取失败"""

    pass


class RateLimitError(DataFetchError):
    """API 速率限制异常"""

    pass


class DataSourceUnavailableError(DataFetchError):
    """数据源不可用异常"""

    pass


class StorageError(StockAnalyzerException):
    """数据存储失败"""

    pass


class ValidationError(StockAnalyzerException):
    """数据验证失败"""

    pass


class AnalysisError(StockAnalyzerException):
    """分析过程错误"""

    pass


class NotificationError(StockAnalyzerException):
    """通知发送失败"""

    pass


class ConfigurationError(StockAnalyzerException):
    """配置错误"""

    pass


def handle_errors(
    error_message: str,
    default_return: Any = None,
    raise_on: tuple[type[Exception], ...] = (Exception,),
    log_level: str = "error",
) -> Callable[[F], F]:
    """错误处理装饰器

    统一处理函数异常，记录日志并返回默认值

    Args:
        error_message: 错误消息前缀
        default_return: 发生异常时的默认返回值
        raise_on: 需要重新抛出的异常类型
        log_level: 日志级别 (debug, info, warning, error)

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
                # 重新抛出指定的异常
                raise
            except Exception as e:
                # 记录日志
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
