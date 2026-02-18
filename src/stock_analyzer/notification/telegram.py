"""
Telegram 通知渠道
"""

import logging
import re
from typing import Any

from stock_analyzer.infrastructure import get_aiohttp_session
from stock_analyzer.notification.base import NotificationChannel, NotificationChannelBase

logger = logging.getLogger(__name__)


class TelegramChannel(NotificationChannelBase):
    """Telegram Bot 通知渠道"""

    def __init__(self, config: dict[str, Any]):
        self.bot_token: str | None = None
        self.chat_id: str | None = None
        self.message_thread_id: str | None = None
        super().__init__(config)

    def _validate_config(self) -> None:
        """验证配置"""
        self.bot_token = self.config.get("bot_token")
        self.chat_id = self.config.get("chat_id")
        self.message_thread_id = self.config.get("message_thread_id")

    def is_available(self) -> bool:
        """检查配置是否完整"""
        return bool(self.bot_token and self.chat_id)

    @property
    def channel_type(self) -> NotificationChannel:
        return NotificationChannel.TELEGRAM

    async def send(self, content: str, **kwargs: Any) -> bool:
        """
        发送消息到 Telegram（异步）

        Args:
            content: Markdown 格式的消息内容

        Returns:
            是否发送成功
        """
        if not self.bot_token or not self.chat_id:
            logger.warning("Telegram 配置不完整，跳过推送")
            return False

        try:
            api_url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
            max_length = 4096

            if len(content) <= max_length:
                return await self._send_message(api_url, content)
            else:
                return await self._send_chunked(api_url, content, max_length)

        except Exception as e:
            logger.error(f"发送 Telegram 消息失败: {e}")
            return False

    async def _send_message(self, api_url: str, text: str) -> bool:
        """发送单条消息（异步）"""
        session = get_aiohttp_session()
        telegram_text = self._convert_to_telegram_markdown(text)

        payload = {
            "chat_id": self.chat_id,
            "text": telegram_text,
            "parse_mode": "Markdown",
            "disable_web_page_preview": True,
        }

        if self.message_thread_id:
            payload["message_thread_id"] = self.message_thread_id

        async with session.post(api_url, json=payload) as response:
            if response.status == 200:
                result = await response.json()
                if result.get("ok"):
                    logger.info("Telegram 消息发送成功")
                    return True
                else:
                    error_desc = result.get("description", "未知错误")
                    logger.error(f"Telegram 返回错误: {error_desc}")

                    if "parse" in error_desc.lower() or "markdown" in error_desc.lower():
                        logger.info("尝试使用纯文本格式重新发送...")
                        payload["text"] = text
                        payload.pop("parse_mode", None)

                        async with session.post(api_url, json=payload) as retry_response:
                            if retry_response.status == 200:
                                retry_result = await retry_response.json()
                                if retry_result.get("ok"):
                                    logger.info("Telegram 消息发送成功（纯文本）")
                                    return True
                    return False
            else:
                logger.error(f"Telegram 请求失败: HTTP {response.status}")
                return False

    async def _send_chunked(self, api_url: str, content: str, max_length: int) -> bool:
        """分段发送长消息（异步）"""
        sections = content.split("\n---\n")

        current_chunk = []
        current_length = 0
        all_success = True
        chunk_index = 1

        for section in sections:
            section_length = len(section) + 5

            if current_length + section_length > max_length:
                if current_chunk:
                    chunk_content = "\n---\n".join(current_chunk)
                    logger.info(f"发送 Telegram 消息块 {chunk_index}...")
                    if not await self._send_message(api_url, chunk_content):
                        all_success = False
                    chunk_index += 1

                current_chunk = [section]
                current_length = section_length
            else:
                current_chunk.append(section)
                current_length += section_length

        if current_chunk:
            chunk_content = "\n---\n".join(current_chunk)
            logger.info(f"发送 Telegram 消息块 {chunk_index}...")
            if not await self._send_message(api_url, chunk_content):
                all_success = False

        return all_success

    def _convert_to_telegram_markdown(self, text: str) -> str:
        """将标准 Markdown 转换为 Telegram 支持的格式"""
        result = text

        result = re.sub(r"^#{1,6}\s+", "", result, flags=re.MULTILINE)

        result = re.sub(r"\*\*(.+?)\*\*", r"*\1*", result)

        for char in ["[", "]", "(", ")"]:
            result = result.replace(char, f"\\{char}")

        return result
