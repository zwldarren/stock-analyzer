"""
通知服务主类

提供统一的接口来管理所有通知渠道
"""

import asyncio
import logging
from pathlib import Path
from typing import Any

from stock_analyzer.config import get_config
from stock_analyzer.exceptions import NotificationError
from stock_analyzer.models import AnalysisResult
from stock_analyzer.notification.context import MessageContext

from .base import ChannelDetector, NotificationChannel
from .email import EmailChannel
from .report_generator import ReportGenerator
from .telegram import TelegramChannel

logger = logging.getLogger(__name__)


class NotificationService:
    """
    通知服务

    职责：
    1. 生成 Markdown 格式的分析日报
    2. 向所有已配置的渠道推送消息
    3. 支持本地保存日报
    4. 支持上下文感知回复（通过 MessageContext）

    解耦说明：
    - 不再直接依赖 BotMessage，而是通过 MessageContext 抽象
    - 使用 message_adapter.adapt_bot_message() 进行转换
    """

    def __init__(self, context: MessageContext | None = None):
        """
        初始化通知服务

        Args:
            context: 消息上下文（用于上下文回复，可选）
        """
        self._settings = get_config()
        self._context = context
        self._channels: dict[NotificationChannel, Any] = {}

        self._init_channels()

        self._available_channels = self._detect_available_channels()

        if not self._available_channels:
            logger.warning("未配置有效的通知渠道，将不发送推送通知")
        else:
            channel_names = [ChannelDetector.get_channel_name(ch) for ch in self._available_channels]
            logger.info(f"已配置 {len(channel_names)} 个通知渠道：{', '.join(channel_names)}")

    def _init_channels(self) -> None:
        """初始化各通知渠道"""
        settings = self._settings
        nc = settings.notification_channel

        if nc.telegram_bot_token and nc.telegram_chat_id:
            self._channels[NotificationChannel.TELEGRAM] = TelegramChannel(
                {
                    "bot_token": nc.telegram_bot_token,
                    "chat_id": nc.telegram_chat_id,
                    "message_thread_id": nc.telegram_message_thread_id,
                }
            )

        if nc.email_sender and nc.email_password:
            receivers = nc.email_receivers or []
            if not receivers:
                receivers = [nc.email_sender]
            self._channels[NotificationChannel.EMAIL] = EmailChannel(
                {
                    "sender": nc.email_sender,
                    "sender_name": nc.email_sender_name,
                    "password": nc.email_password,
                    "receivers": receivers,
                }
            )

        if nc.discord_webhook_url:
            try:
                from .discord import DiscordChannel

                self._channels[NotificationChannel.DISCORD] = DiscordChannel({"webhook_url": nc.discord_webhook_url})
            except ImportError:
                logger.warning("Discord channel not available - requires additional dependencies")

        if nc.custom_webhook_urls:
            try:
                from .custom import CustomWebhookChannel

                self._channels[NotificationChannel.CUSTOM] = CustomWebhookChannel(
                    {
                        "webhook_urls": nc.custom_webhook_urls,
                        "bearer_token": nc.custom_webhook_bearer_token,
                    }
                )
            except ImportError:
                logger.warning("Custom Webhook channel not available")

    def _detect_available_channels(self) -> list[NotificationChannel]:
        """检测所有已配置的渠道"""
        available = []
        for channel_type, channel in self._channels.items():
            if channel.is_available():
                available.append(channel_type)
        return available

    def is_available(self) -> bool:
        """检查通知服务是否可用"""
        return len(self._available_channels) > 0

    def get_available_channels(self) -> list[NotificationChannel]:
        """获取所有已配置的渠道"""
        return self._available_channels

    def get_channel_names(self) -> str:
        """获取所有已配置渠道的名称"""
        names = [ChannelDetector.get_channel_name(ch) for ch in self._available_channels]
        return ", ".join(names)

    def generate_daily_report(self, results: list[AnalysisResult], report_date: str | None = None) -> str:
        """生成日报"""
        return ReportGenerator.generate_dashboard_report(results, report_date)

    def generate_dashboard_report(self, results: list[AnalysisResult], report_date: str | None = None) -> str:
        """生成决策仪表盘报告"""
        return ReportGenerator.generate_dashboard_report(results, report_date)

    def generate_single_stock_report(self, result: AnalysisResult) -> str:
        """生成单股报告"""
        return ReportGenerator.generate_single_stock_report(result)

    async def send(self, content: str, **kwargs: Any) -> bool:
        """
        统一发送接口 - 向所有已配置的渠道发送（异步并行）

        Args:
            content: 消息内容（Markdown 格式）
            **kwargs: 额外参数

        Returns:
            是否至少有一个渠道发送成功
        """
        if not self._available_channels:
            logger.warning("通知服务不可用，跳过推送")
            return False

        logger.info(f"正在向 {len(self._available_channels)} 个渠道发送通知：{self.get_channel_names()}")

        tasks = []
        for channel_type in self._available_channels:
            channel = self._channels[channel_type]
            tasks.append(self._send_to_channel_safe(channel, channel_type, content, **kwargs))

        results = await asyncio.gather(*tasks)

        success_count = sum(1 for r in results if r is True)
        fail_count = len(results) - success_count

        logger.info(f"通知发送完成：成功 {success_count} 个，失败 {fail_count} 个")
        return success_count > 0

    async def _send_to_channel_safe(
        self, channel: Any, channel_type: NotificationChannel, content: str, **kwargs: Any
    ) -> bool:
        """安全发送到单个渠道"""
        channel_name = ChannelDetector.get_channel_name(channel_type)
        try:
            result = await channel.send(content, **kwargs)
            return result
        except NotificationError as e:
            logger.error(f"{channel_name} 发送失败: {e}")
            return False
        except Exception as e:
            logger.error(f"{channel_name} 发送异常: {e}")
            return False

    def save_report_to_file(self, content: str, filename: str | None = None) -> str:
        """
        保存日报到本地文件

        Args:
            content: 日报内容
            filename: 文件名（可选，默认按日期生成）

        Returns:
            保存的文件路径
        """
        if filename is None:
            from datetime import datetime

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"report_{timestamp}.md"

        reports_dir = Path(self._settings.reports_dir)
        reports_dir.mkdir(parents=True, exist_ok=True)

        filepath = reports_dir / filename

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)

        logger.info(f"日报已保存到: {filepath}")
        return str(filepath)

    async def send_to_context(self, content: str) -> bool:
        """发送报告到上下文（用于机器人回复，异步）

        根据MessageContext中的平台信息，将消息发送到对应的渠道。
        如果context未设置或平台不支持，则返回False。

        Args:
            content: 消息内容

        Returns:
            是否发送成功
        """
        if not self._context or not self._context.platform:
            logger.debug("No context or platform specified, skipping context-based sending")
            return False

        platform = self._context.platform.lower()

        platform_channel_map = {
            "telegram": NotificationChannel.TELEGRAM,
            "email": NotificationChannel.EMAIL,
            "discord": NotificationChannel.DISCORD,
            "custom": NotificationChannel.CUSTOM,
        }

        channel = platform_channel_map.get(platform)
        if not channel:
            logger.warning(f"Unsupported platform for context sending: {platform}")
            return False

        if channel not in self._available_channels:
            logger.warning(f"Channel {channel} not available for context sending")
            return False

        try:
            logger.info(f"正在发送上下文回复到 {platform}")
            return await self._send_to_channel(channel, content)
        except NotificationError as e:
            logger.error(f"发送上下文回复到 {platform} 失败: {e}")
            return False

    async def _send_to_channel(self, channel: NotificationChannel, content: str) -> bool:
        """发送到指定渠道（异步）"""
        if channel not in self._channels:
            return False
        try:
            return await self._channels[channel].send(content)
        except NotificationError as e:
            logger.error(f"发送到 {channel} 失败: {e}")
            return False


def get_notification_service(context: MessageContext | None = None) -> NotificationService:
    """获取通知服务实例

    Args:
        context: 消息上下文（可选），用于上下文感知回复

    Returns:
        NotificationService 实例
    """
    return NotificationService(context=context)


async def send_daily_report(results: list[AnalysisResult]) -> bool:
    """
    发送每日报告的快捷方式（异步）

    自动识别渠道并推送
    """
    service = get_notification_service()

    report = service.generate_dashboard_report(results)

    service.save_report_to_file(report)

    return await service.send(report)
