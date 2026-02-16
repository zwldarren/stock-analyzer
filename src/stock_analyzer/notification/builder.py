"""
通知消息构建器

提供便捷的消息构建方法
"""

from stock_analyzer.constants import (
    REPORT_EMOJI,
    get_alert_emoji,
    get_signal_emoji,
)
from stock_analyzer.models import AnalysisResult


class NotificationBuilder:
    """通知消息构建器"""

    @staticmethod
    def build_simple_alert(title: str, content: str, alert_type: str = "info") -> str:
        """
        构建简单的提醒消息

        Args:
            title: 标题
            content: 内容
            alert_type: 类型（info, warning, error, success）
        """
        emoji = get_alert_emoji(alert_type)

        return f"{emoji} **{title}**\n\n{content}"

    @staticmethod
    def build_stock_summary(results: list[AnalysisResult]) -> str:
        """
        构建股票摘要（简短版）

        适用于快速通知
        """
        lines = [f"{REPORT_EMOJI['dashboard']} **今日自选股摘要**", ""]

        for r in sorted(results, key=lambda x: x.sentiment_score, reverse=True):
            action = r.final_action or "HOLD"
            emoji = get_signal_emoji(action)
            lines.append(f"{emoji} {r.name}({r.code}): {action} | 评分 {r.sentiment_score}")

        return "\n".join(lines)
