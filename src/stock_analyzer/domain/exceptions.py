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
    """数据获取失败异常"""

    def __init__(self, message: str = "数据获取失败", code: str | None = None):
        super().__init__(message, code)


class RateLimitError(DataFetchError):
    """API 速率限制异常，当API请求频率超过限制时抛出"""

    def __init__(self, message: str = "API请求频率超限，请稍后重试", code: str | None = None):
        super().__init__(message, code)


class DataSourceUnavailableError(DataFetchError):
    """数据源不可用异常，当指定的数据源暂时无法访问时抛出"""

    def __init__(self, message: str = "数据源暂时不可用", code: str | None = None):
        super().__init__(message, code)


class StorageError(StockAnalyzerException):
    """数据存储失败异常，当数据库操作或文件写入失败时抛出"""

    def __init__(self, message: str = "数据存储失败", code: str | None = None):
        super().__init__(message, code)


class ValidationError(StockAnalyzerException):
    """数据验证失败异常，当输入数据不符合预期格式或范围时抛出"""

    def __init__(self, message: str = "数据验证失败", code: str | None = None):
        super().__init__(message, code)


class AnalysisError(StockAnalyzerException):
    """分析过程错误异常，当AI分析或计算过程中发生错误时抛出"""

    def __init__(self, message: str = "分析过程出错", code: str | None = None):
        super().__init__(message, code)


class NotificationError(StockAnalyzerException):
    """通知发送失败异常，当消息推送失败时抛出"""

    def __init__(self, message: str = "通知发送失败", code: str | None = None):
        super().__init__(message, code)


class ConfigurationError(StockAnalyzerException):
    """配置错误异常，当配置项缺失或无效时抛出"""

    def __init__(self, message: str = "配置错误", code: str | None = None):
        super().__init__(message, code)


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
