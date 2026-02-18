"""
自定义 Webhook 通知渠道
"""

import asyncio
import logging
from typing import Any

from stock_analyzer.infrastructure import get_aiohttp_session
from stock_analyzer.notification.base import NotificationChannel, NotificationChannelBase

logger = logging.getLogger(__name__)


class CustomWebhookChannel(NotificationChannelBase):
    """自定义 Webhook 通知渠道"""

    def __init__(self, config: dict[str, Any]):
        self.webhook_urls: list[str] = []
        self.bearer_token: str | None = None
        super().__init__(config)

    def _validate_config(self) -> None:
        """验证配置"""
        self.webhook_urls = self.config.get("webhook_urls", [])
        self.bearer_token = self.config.get("bearer_token")

    def is_available(self) -> bool:
        """检查配置是否完整"""
        return bool(self.webhook_urls)

    @property
    def channel_type(self) -> NotificationChannel:
        return NotificationChannel.CUSTOM

    async def send(self, content: str, **kwargs: Any) -> bool:
        """
        发送消息到自定义 Webhook（异步）

        Args:
            content: Markdown 格式的消息内容

        Returns:
            是否发送成功
        """
        if not self.webhook_urls:
            logger.warning("自定义 Webhook 未配置，跳过推送")
            return False

        tasks = [self._send_to_webhook(url, content) for url in self.webhook_urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        success_count = sum(1 for r in results if r is True)
        total_count = len(self.webhook_urls)

        logger.info(f"自定义 Webhook 发送完成: {success_count}/{total_count} 成功")
        return success_count > 0

    async def _send_to_webhook(self, webhook_url: str, content: str) -> bool:
        """发送到单个 Webhook（异步）"""
        session = get_aiohttp_session()
        headers = {"Content-Type": "application/json"}

        if self.bearer_token:
            headers["Authorization"] = f"Bearer {self.bearer_token}"

        payload = {
            "content": content,
            "format": "markdown",
        }

        async with session.post(webhook_url, json=payload, headers=headers) as response:
            if response.status in (200, 201, 204):
                return True
            else:
                text = await response.text()
                logger.error(f"自定义 Webhook 请求失败: HTTP {response.status}, {text}")
                return False
