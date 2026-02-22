"""日志配置模块"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.logging import RichHandler
from rich.theme import Theme

_console: Console | None = None
_display: Any = None  # RichConsoleDisplay instance
_live_display: Any = None
_log_buffer: list[tuple[str, str]] = []


def get_console() -> Console:
    """获取全局 Console 实例，确保统一使用"""
    global _console
    if _console is None:
        custom_theme = Theme(
            {
                "logging.level.info": "green",
                "logging.level.warning": "yellow",
                "logging.level.error": "red bold",
                "logging.level.debug": "dim",
                "banner": "bold cyan",
                "banner.dim": "dim cyan",
            }
        )
        _console = Console(theme=custom_theme, force_terminal=True)
    return _console


def get_display() -> Any:
    """Get or create the global display instance."""
    global _display
    if _display is None:
        from ashare_analyzer.utils.console_display import RichConsoleDisplay

        _display = RichConsoleDisplay()
    return _display


def set_live_display(live: Any) -> None:
    """设置 Live 显示对象，用于协调日志输出"""
    global _live_display
    _live_display = live


def clear_live_display() -> None:
    """清除 Live 显示对象"""
    global _live_display, _log_buffer
    _live_display = None
    # 清空缓冲区
    _log_buffer.clear()


def get_buffered_logs() -> list[tuple[str, str]]:
    """获取缓冲的日志消息并清空缓冲区"""
    global _log_buffer
    logs = _log_buffer.copy()
    _log_buffer.clear()
    return logs


class LiveAwareRichHandler(RichHandler):
    """
    自定义 RichHandler，能够感知 Live 显示并正确输出日志。

    当 Live 显示（如进度条）激活时，将日志消息缓冲，由 Live 显示负责渲染。
    当没有 Live 显示时，直接输出到控制台。
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._buffer_logs = False

    def emit(self, record: logging.LogRecord) -> None:
        """处理日志记录"""
        global _live_display, _log_buffer

        # 格式化消息
        message = self.format(record)
        level_name = record.levelname

        if _live_display is not None and _live_display.is_started:
            _log_buffer.append((level_name, message))
            return
        super().emit(record)


def setup_logging(
    debug: bool = False,
    log_dir: str = "./logs",
) -> None:
    """配置日志系统

    Args:
        debug: 是否启用调试模式
        log_dir: 日志文件保存目录
    """
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    debug_log_file = log_path / f"stock_analysis_debug_{timestamp}.log"

    # 获取根 logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)

    # 清除现有 handlers（避免重复）
    root_logger.handlers.clear()

    # 控制台 Handler
    console = get_console()
    console_handler = LiveAwareRichHandler(
        console=console,
        show_time=True,
        show_path=debug,
        rich_tracebacks=True,
        tracebacks_show_locals=debug,
        show_level=True,
        omit_repeated_times=True,
        markup=True,
    )
    console_level = logging.DEBUG if debug else logging.INFO
    console_handler.setLevel(console_level)
    # 使用更简洁的格式，Rich 会自动添加时间
    console_handler.setFormatter(logging.Formatter("%(message)s"))
    root_logger.addHandler(console_handler)

    # 文件 Handler - 调试日志 (DEBUG 及以上)，仅在 debug 模式启用
    if debug:
        debug_handler = logging.FileHandler(debug_log_file, encoding="utf-8")
        debug_handler.setLevel(logging.DEBUG)
        debug_handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d | %(message)s")
        )
        root_logger.addHandler(debug_handler)
        # 输出初始化信息到调试日志
        logger = logging.getLogger(__name__)
        logger.debug(f"日志系统初始化完成，日志目录: {log_path.absolute()}")
        logger.debug(f"调试日志: {debug_log_file}")

    # 降低第三方库日志级别
    _suppress_noisy_loggers()


def _suppress_noisy_loggers() -> None:
    """降低嘈杂的第三方库日志级别"""
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
        "litellm",
        "litellm.llms",
        "litellm.utils",
        "litellm.cost_calculator",
        "litellm.router",
        "litellm.proxy",
        "litellm.caching",
        "litellm.main",
        "litellm.litellm_core_utils",
    ]

    for name in noisy_loggers:
        logging.getLogger(name).setLevel(logging.WARNING)
