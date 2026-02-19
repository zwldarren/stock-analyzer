"""
Rich Console Display Module

Provides Rich-based terminal display for analysis progress and reports.
"""

from typing import TYPE_CHECKING

from rich.console import Console
from rich.live import Live
from rich.progress import Progress, SpinnerColumn, TaskID, TextColumn, TimeElapsedColumn
from rich.rule import Rule
from rich.table import Table

from stock_analyzer.constants import get_signal_emoji
from stock_analyzer.models import AnalysisResult

if TYPE_CHECKING:
    pass

from stock_analyzer.utils.logging_config import (
    clear_live_display,
    get_console,
    set_live_display,
)

__all__ = ["RichConsoleDisplay"]


class RichConsoleDisplay:
    """
    Rich-based terminal display manager.
    """

    def __init__(self, console: Console | None = None):
        self.console = console or get_console()
        self._live: Live | None = None
        self._progress: Progress | None = None
        self._stock_tasks: dict[str, TaskID] = {}
        self._stock_codes: list[str] = []

    def start_analysis(self, stock_codes: list[str]) -> None:
        """Initialize progress display for analysis using Live."""
        self._stock_codes = stock_codes

        # 创建进度条
        self._progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            TimeElapsedColumn(),
            console=self.console,
        )

        # 为每只股票创建任务
        for code in stock_codes:
            task_id = self._progress.add_task(f"○ {code} 等待中", total=None)
            self._stock_tasks[code] = task_id

        self._live = Live(
            self._progress,
            console=self.console,
            refresh_per_second=4,
            screen=False,
        )
        self._live.start()

        # 通知日志系统 Live 已启动
        set_live_display(self._live)

    def finish_analysis(self) -> None:
        """Stop progress display."""
        # 通知日志系统 Live 已停止
        clear_live_display()

        if self._live:
            self._live.stop()
            self._live = None

        if self._progress:
            self._progress.stop()
            self._progress = None

        self._stock_tasks.clear()
        self._stock_codes.clear()

    def update_stock_progress(self, code: str, status: str, name: str = "") -> None:
        """Update stock progress status.

        Args:
            code: Stock code
            status: One of 'waiting', 'analyzing', 'completed', 'error'
            name: Stock name (optional, for display)
        """
        if not self._progress or code not in self._stock_tasks:
            return

        task_id = self._stock_tasks[code]
        display_name = f"{name}({code})" if name else code

        status_map = {
            "waiting": ("○", "等待中", "dim"),
            "analyzing": ("◌", "分析中...", "cyan"),
            "completed": ("✓", "完成", "green"),
            "error": ("✗", "失败", "red"),
        }
        symbol, text, style = status_map.get(status, ("○", "", "white"))
        self._progress.update(task_id, description=f"[{style}]{symbol} {display_name} {text}[/{style}]")

    def start_agent(self, agent_name: str) -> None:
        """Display agent execution start - 简化版本不显示 Agent 进度."""
        pass

    def complete_agent(self, agent_name: str, signal: str, confidence: int, error: str | None = None) -> None:
        """Display agent completion with result - 简化版本不显示 Agent 进度."""
        pass

    def show_stock_result(self, result: AnalysisResult) -> None:
        """Display single stock analysis result summary."""
        action = result.final_action or "HOLD"
        emoji = get_signal_emoji(action)
        self.console.print(f"  {emoji} {result.name}({result.code}): {action} | 置信度 {result.sentiment_score}%")

    def show_final_report(self, results: list[AnalysisResult]) -> None:
        """Display final report with overview table and per-stock agent details."""
        if not results:
            self.console.print("[yellow]无分析结果[/yellow]")
            return

        # 按情感分数降序排序
        sorted_results = sorted(results, key=lambda x: x.sentiment_score, reverse=True)

        self.console.print()

        # 1. 概览表
        self._render_overview_table(sorted_results)

        # 2. 每只股票的 Agent 详情
        for result in sorted_results:
            self.console.print()
            self._render_stock_detail(result)

    def _render_overview_table(self, results: list[AnalysisResult]) -> None:
        """Render decision overview table."""
        table = Table(
            title="📊 今日决策概览",
            show_header=True,
            header_style="bold cyan",
            border_style="dim",
            title_style="bold",
            expand=False,
        )
        table.add_column("股票", style="white", no_wrap=True)
        table.add_column("决策", justify="center", no_wrap=True)
        table.add_column("置信度", justify="right", no_wrap=True)
        table.add_column("仓位", justify="right", no_wrap=True)
        table.add_column("趋势预测", no_wrap=True)

        for r in results:
            action = r.final_action or "HOLD"
            emoji = get_signal_emoji(action)
            position = f"{r.position_ratio * 100:.0f}%" if r.position_ratio else "-"
            trend = r.trend_prediction or "-"

            # 根据操作设置颜色
            action_style = {"BUY": "green", "SELL": "red", "HOLD": "yellow"}.get(action, "white")

            table.add_row(
                f"{r.name}({r.code})",
                f"[{action_style}]{emoji} {action}[/]",
                f"{r.sentiment_score}%",
                position,
                trend,
            )

        self.console.print(table)

    def _render_stock_detail(self, result: AnalysisResult) -> None:
        """Render agent details for a single stock."""
        action = result.final_action or "HOLD"
        emoji = get_signal_emoji(action)
        action_style = {"BUY": "green", "SELL": "red", "HOLD": "yellow"}.get(action, "white")

        # 标题用 Rule 分隔线，更统一
        position_str = f" | 仓位: {result.position_ratio * 100:.0f}%" if result.position_ratio else ""
        title_text = (
            f"{emoji} {result.name} ({result.code})"
            f" — [{action_style}]{action}[/] | 置信度: {result.sentiment_score}%{position_str}"
        )
        self.console.print(Rule(title_text, style="dim"))

        # Agent 共识表
        dashboard = result.dashboard or {}
        agent_consensus = dashboard.get("agent_consensus", {})
        agent_signals = agent_consensus.get("signals", {})
        agent_confidences = agent_consensus.get("confidences", {})
        agent_reasonings = agent_consensus.get("reasonings", {})

        if agent_signals:
            table = Table(
                show_header=True,
                header_style="bold dim",
                border_style="dim",
                expand=True,
                pad_edge=False,
            )
            table.add_column("Agent", style="cyan", width=20, no_wrap=True)
            table.add_column("信号", justify="center", width=10, no_wrap=True)
            table.add_column("置信度", justify="right", width=8, no_wrap=True)
            table.add_column("理由", ratio=1, overflow="fold")

            for agent_name, signal in agent_signals.items():
                confidence = agent_confidences.get(agent_name, 0)
                reasoning = agent_reasonings.get(agent_name, "")
                signal_emoji = get_signal_emoji(signal)
                signal_style = {"buy": "green", "sell": "red", "hold": "yellow"}.get(signal.lower(), "white")

                table.add_row(
                    agent_name,
                    f"[{signal_style}]{signal_emoji} {signal}[/]",
                    f"{confidence}%",
                    reasoning,
                )

            self.console.print(table)

        # 关键因素和风险提示
        extras = []
        key_factors = dashboard.get("key_factors", [])
        if key_factors:
            extras.append(f"🔑 关键因子: {', '.join(key_factors[:3])}")
        if result.risk_warning:
            extras.append(f"⚠️  风险: {result.risk_warning}")

        if extras:
            self.console.print(" | ".join(extras))
