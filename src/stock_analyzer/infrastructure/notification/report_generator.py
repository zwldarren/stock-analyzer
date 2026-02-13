"""
Report Generator

Simplified report generation for multi-agent analysis results.
"""

import logging
from datetime import datetime

from stock_analyzer.domain.constants import (
    REPORT_EMOJI,
    get_action_emoji,
    get_signal_emoji,
)
from stock_analyzer.domain.models import AnalysisResult

logger = logging.getLogger(__name__)


class ReportGenerator:
    """Simplified report generator for AI decision reports."""

    @staticmethod
    def generate_dashboard_report(results: list[AnalysisResult], report_date: str | None = None) -> str:
        """
        Generate AI decision report in dashboard format.

        Args:
            results: List of analysis results
            report_date: Report date (default: today)

        Returns:
            Markdown formatted report
        """
        if report_date is None:
            report_date = datetime.now().strftime("%Y-%m-%d")

        # Sort by sentiment score (descending)
        sorted_results = sorted(results, key=lambda x: x.sentiment_score, reverse=True)

        # Count actions
        action_counts = {"BUY": 0, "HOLD": 0, "SELL": 0}
        for r in results:
            action = r.final_action or "HOLD"
            if action in action_counts:
                action_counts[action] += 1

        lines = [
            f"# {REPORT_EMOJI['title']} {report_date} AI投资决策报告",
            "",
            f"> 共分析 **{len(results)}** 只股票 | "
            f"{get_action_emoji('BUY')}买入:{action_counts['BUY']} "
            f"{get_action_emoji('HOLD')}持有:{action_counts['HOLD']} "
            f"{get_action_emoji('SELL')}卖出:{action_counts['SELL']}",
            "",
            "---",
            "",
        ]

        # Decision overview
        if results:
            lines.extend([f"## {REPORT_EMOJI['dashboard']} 今日决策概览", ""])
            for r in sorted_results:
                action = r.final_action or "HOLD"
                emoji = get_action_emoji(action)
                position_ratio = r.position_ratio or 0
                ratio_str = f" | 仓位{position_ratio * 100:.0f}%" if position_ratio > 0 else ""
                lines.append(f"{emoji} **{r.name}({r.code})**: **{action}** | 置信度{r.sentiment_score}%{ratio_str}")
            lines.extend(["", "---", ""])

        # Individual stock details
        for result in sorted_results:
            lines.extend(ReportGenerator._generate_stock_section(result))

        # Footer
        lines.extend(["", f"*报告生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*"])

        return "\n".join(lines)

    @staticmethod
    def _generate_stock_section(result: AnalysisResult) -> list[str]:
        """Generate report section for a single stock."""
        action = result.final_action or "HOLD"
        action_emoji = get_action_emoji(action)
        stock_name = result.name if result.name and not result.name.startswith("股票") else f"股票{result.code}"

        lines = [
            f"## {action_emoji} {stock_name} ({result.code})",
            "",
            f"### {REPORT_EMOJI['title']} 最终决策",
            "",
            f"**{action_emoji} {action}** | 置信度: {result.sentiment_score}%",
            "",
        ]

        # Position ratio
        position_ratio = result.position_ratio or 0
        if position_ratio > 0:
            lines.append(f"{REPORT_EMOJI['money']} **建议仓位**: {position_ratio * 100:.0f}%")
            lines.append("")

        # Decision reasoning
        decision_reason = result.decision_reasoning or ""
        if decision_reason:
            lines.extend([f"**决策理由**: {decision_reason}", ""])

        # Agent consensus with detailed breakdown
        dashboard = result.dashboard or {}
        agent_consensus = dashboard.get("agent_consensus", {})
        agent_signals = agent_consensus.get("signals", {})
        agent_confidences = agent_consensus.get("confidences", {})
        agent_reasonings = agent_consensus.get("reasonings", {})

        if agent_signals:
            lines.extend([f"### {REPORT_EMOJI['ai']} Agent 共识分析", ""])
            lines.append("| Agent | 信号 | 置信度 | 原因 |")
            lines.append("|-------|------|--------|------|")

            for agent_name, signal in agent_signals.items():
                confidence = agent_confidences.get(agent_name, 0)
                reasoning = agent_reasonings.get(agent_name, "")
                signal_emoji = get_signal_emoji(signal)
                lines.append(f"| {agent_name} | {signal_emoji} {signal} | {confidence}% | {reasoning} |")

            # Consensus level
            consensus_level = agent_consensus.get("consensus_level", "N/A")
            lines.append("")
            lines.append(f"**共识度**: {consensus_level}")
            lines.append("")

        # Market snapshot
        snapshot = result.market_snapshot
        if snapshot:
            # Safely get values, ensuring they're strings without markdown
            close = str(snapshot.get("close", "N/A")).replace("**", "").replace("*", "")
            prev_close = str(snapshot.get("prev_close", "N/A")).replace("**", "").replace("*", "")
            open_price = str(snapshot.get("open", "N/A")).replace("**", "").replace("*", "")
            high = str(snapshot.get("high", "N/A")).replace("**", "").replace("*", "")
            low = str(snapshot.get("low", "N/A")).replace("**", "").replace("*", "")
            pct_chg = str(snapshot.get("pct_chg", "N/A")).replace("**", "").replace("*", "")

            lines.extend(
                [
                    f"### {REPORT_EMOJI['market']} 当日行情",
                    "",
                    f"**收盘**: {close} | **昨收**: {prev_close} | **开盘**: {open_price}",
                    f"**最高**: {high} | **最低**: {low} | **涨跌幅**: {pct_chg}",
                    "",
                ]
            )

            if "price" in snapshot:
                price = str(snapshot.get("price", "N/A")).replace("**", "").replace("*", "")
                volume_ratio = str(snapshot.get("volume_ratio", "N/A")).replace("**", "").replace("*", "")
                turnover_rate = str(snapshot.get("turnover_rate", "N/A")).replace("**", "").replace("*", "")
                lines.append(f"**当前价**: {price} | **量比**: {volume_ratio} | **换手率**: {turnover_rate}%")
                lines.append("")

        # Risk warning
        risk_warning = result.risk_warning or ""
        if risk_warning:
            lines.extend([f"{REPORT_EMOJI['risk']} **风险提示**: {risk_warning}", ""])

        lines.extend(["---", ""])
        return lines

    @staticmethod
    def generate_single_stock_report(result: AnalysisResult) -> str:
        """Generate single stock report for notifications."""
        report_date = datetime.now().strftime("%Y-%m-%d %H:%M")
        action = result.final_action or "HOLD"
        action_emoji = get_action_emoji(action)
        stock_name = result.name if result.name and not result.name.startswith("股票") else f"股票{result.code}"

        lines = [
            f"## {action_emoji} {stock_name} ({result.code})",
            "",
            f"> {report_date} | 置信度: **{result.sentiment_score}%** | {result.trend_prediction}",
            "",
            f"**{action_emoji} {action}**",
            "",
        ]

        # Decision reasoning
        decision_reason = result.decision_reasoning or ""
        if decision_reason:
            lines.extend([f"**决策理由**: {decision_reason}", ""])

        # Position ratio
        position_ratio = result.position_ratio or 0
        if position_ratio > 0:
            lines.append(f"{REPORT_EMOJI['money']} **建议仓位**: {position_ratio * 100:.0f}%")
            lines.append("")

        # Agent consensus with detailed breakdown
        dashboard = result.dashboard or {}
        agent_consensus = dashboard.get("agent_consensus", {})
        agent_signals = agent_consensus.get("signals", {})
        agent_confidences = agent_consensus.get("confidences", {})
        agent_reasonings = agent_consensus.get("reasonings", {})

        if agent_signals:
            lines.extend([f"### {REPORT_EMOJI['ai']} Agent 共识", ""])
            for agent_name, signal in agent_signals.items():
                confidence = agent_confidences.get(agent_name, 0)
                reasoning = agent_reasonings.get(agent_name, "")
                signal_emoji = get_signal_emoji(signal)
                lines.append(f"- **{agent_name}**: {signal_emoji} {signal} (置信度{confidence}%)")
                if reasoning:
                    lines.append(f"  - {reasoning}")
            lines.append("")

            # Consensus level
            consensus_level = agent_consensus.get("consensus_level", "N/A")
            lines.append(f"**共识度**: {consensus_level}")
            lines.append("")

        # Risk warning
        risk_warning = result.risk_warning or ""
        if risk_warning:
            lines.extend([f"{REPORT_EMOJI['risk']} **风险提示**: {risk_warning}", ""])

        lines.extend(["---", "*AI生成，仅供参考，不构成投资建议*"])
        return "\n".join(lines)
