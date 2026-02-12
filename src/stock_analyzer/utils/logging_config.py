"""日志配置模块"""

import sys
from datetime import datetime
from pathlib import Path

from loguru import logger


def setup_logging(
    debug: bool = False,
    log_dir: str = "./logs",
    json_format: bool = False,
) -> None:
    """配置 Loguru 日志系统

    Args:
        debug: 是否启用调试模式
        log_dir: 日志文件保存目录
        json_format: 是否使用 JSON 格式（便于日志收集系统解析）
    """
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    today_str = datetime.now().strftime("%Y%m%d")
    log_file = log_path / f"stock_analysis_{today_str}.log"
    debug_log_file = log_path / f"stock_analysis_debug_{today_str}.log"

    # 移除默认的 stderr handler
    logger.remove()

    # 控制台 Handler - 彩色输出
    console_level = "DEBUG" if debug else "INFO"

    def shorten_name(record):
        """缩短模块路径，移除 stock_analyzer 前缀（但保留至少两级）"""
        name = record["name"]
        parts = name.split(".")
        if len(parts) > 2 and parts[0] == "stock_analyzer":
            record["extra"]["short_name"] = ".".join(parts[1:])
        else:
            record["extra"]["short_name"] = name
        return True

    console_format = (
        "<green>{time:HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{extra[short_name]}</cyan>:<cyan>{line}</cyan> | "
        "<level>{message}</level>"
    )
    logger.add(
        sys.stdout,
        level=console_level,
        format=console_format,
        colorize=True,
        enqueue=True,
        filter=shorten_name,
    )

    # 常规日志文件 - INFO 级别及以上
    if json_format:
        file_format = "{extra[json]}"
    else:
        file_format = "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {extra[short_name]}:{line} | {message}"

    logger.add(
        str(log_file),
        level="INFO",
        format=file_format,
        rotation="00:00",  # 每天午夜轮转
        retention="30 days",  # 保留30天
        encoding="utf-8",
        enqueue=True,
        delay=True,  # 延迟打开文件
        filter=shorten_name,
    )

    # 调试日志文件 - DEBUG 级别及以上
    # 调试日志保留完整路径以便调试
    debug_format = "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}"
    logger.add(
        str(debug_log_file),
        level="DEBUG",
        format=debug_format,
        rotation="00:00",
        retention="7 days",  # 调试日志保留7天
        encoding="utf-8",
        enqueue=True,
        delay=True,
    )

    # 拦截标准库的 logging，使其输出到 loguru
    _intercept_standard_logging()

    # 降低第三方库日志级别
    _suppress_noisy_loggers()

    logger.info(f"日志系统初始化完成，日志目录: {log_path.absolute()}")
    logger.info(f"常规日志: {log_file}")
    logger.info(f"调试日志: {debug_log_file}")


def _intercept_standard_logging() -> None:
    """拦截 Python 标准库的 logging，重定向到 loguru"""
    import logging

    class InterceptHandler(logging.Handler):
        """将标准库日志转发到 loguru"""

        def emit(self, record: logging.LogRecord) -> None:
            try:
                level = logger.level(record.levelname).name
            except ValueError:
                level = record.levelno

            frame, depth = sys._getframe(6), 6
            while frame and frame.f_code.co_filename == logging.__file__:
                frame = frame.f_back
                depth += 1

            logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())

    # 配置根 logger 使用拦截 handler
    logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)

    # 替换所有现有 logger 的 handlers
    for name in logging.root.manager.loggerDict:
        logging_logger = logging.getLogger(name)
        logging_logger.handlers = [InterceptHandler()]
        logging_logger.propagate = False


def _suppress_noisy_loggers() -> None:
    """降低嘈杂的第三方库日志级别"""
    import logging

    noisy_loggers = [
        "urllib3",
        "urllib3.connectionpool",
        "sqlalchemy",
        "sqlalchemy.engine",
        "google",
        "google.auth",
        "httpx",
        "httpx._client",
        "httpcore",
        "httpcore.connection",
        "asyncio",
        "discord",
        "discord.client",
        "websockets",
        "websockets.client",
        # litellm 相关 - 全面抑制
        "litellm",
        "litellm.llms",
        "litellm.utils",
        "litellm.cost_calculator",
        "litellm.router",
        "litellm.proxy",
        "litellm.caching",
        "litellm.main",
        "litellm.litellm_core_utils",
        "LiteLLM",
        "LiteLLM Proxy",
        "LiteLLM Router",
    ]

    for name in noisy_loggers:
        logging.getLogger(name).setLevel(logging.WARNING)
