"""
Discord 通知渠道
"""

import asyncio
import logging
from typing import Any

from stock_analyzer.exceptions import NotificationError
from stock_analyzer.infrastructure import get_aiohttp_session
from stock_analyzer.notification.base import NotificationChannel, NotificationChannelBase

logger = logging.getLogger(__name__)


class DiscordChannel(NotificationChannelBase):
    """Discord Webhook 通知渠道"""

    MAX_LENGTH = 2000

    def __init__(self, config: dict[str, Any]):
        self.webhook_url: str | None = None
        super().__init__(config)

    def _validate_config(self) -> None:
        """验证配置"""
        self.webhook_url = self.config.get("webhook_url")

    def is_available(self) -> bool:
        """检查配置是否完整"""
        return bool(self.webhook_url)

    @property
    def channel_type(self) -> NotificationChannel:
        return NotificationChannel.DISCORD

    async def send(self, content: str, **kwargs: Any) -> bool:
        """
        发送消息到 Discord（异步）

        Args:
            content: Markdown 格式的消息内容

        Returns:
            是否发送成功
        """
        if not self.webhook_url:
            logger.warning("Discord Webhook 未配置，跳过推送")
            return False

        if len(content) > self.MAX_LENGTH:
            logger.info(f"Discord 消息内容超长({len(content)}字符)，将分批发送")
            return await self._send_chunked(content)

        try:
            return await self._send_message(content)
        except NotificationError as e:
            logger.error(f"发送 Discord 消息失败: {e}")
            return False

    async def _send_chunked(self, content: str) -> bool:
        """分批发送长消息（异步）"""
        if "\n## " in content:
            parts = content.split("\n## ")
            sections = [parts[0]] + [f"## {p}" for p in parts[1:]]
            separator = "\n"
        elif "\n---\n" in content:
            sections = content.split("\n---\n")
            separator = "\n---\n"
        elif "\n### " in content:
            parts = content.split("\n### ")
            sections = [parts[0]] + [f"### {p}" for p in parts[1:]]
            separator = "\n"
        else:
            return await self._send_force_chunked(content)

        chunks = []
        current_chunk = []
        current_length = 0
        separator_len = len(separator)

        for section in sections:
            section_len = len(section) + separator_len

            if section_len > self.MAX_LENGTH - 100:
                if current_chunk:
                    chunks.append(separator.join(current_chunk))
                    current_chunk = []
                    current_length = 0

                sub_chunks = self._split_large_section(section)
                chunks.extend(sub_chunks)
                continue

            if current_length + section_len > self.MAX_LENGTH - 100:
                if current_chunk:
                    chunks.append(separator.join(current_chunk))
                current_chunk = [section]
                current_length = len(section)
            else:
                current_chunk.append(section)
                current_length += section_len

        if current_chunk:
            chunks.append(separator.join(current_chunk))

        total_chunks = len(chunks)
        success_count = 0

        logger.info(f"Discord 分批发送：共 {total_chunks} 批")

        for i, chunk in enumerate(chunks):
            if total_chunks > 1:
                page_marker = f"\n\n📄 ({i + 1}/{total_chunks})"
                if len(chunk) + len(page_marker) > self.MAX_LENGTH:
                    chunk = chunk[: self.MAX_LENGTH - len(page_marker) - 10] + page_marker
                else:
                    chunk = chunk + page_marker

            try:
                if await self._send_message(chunk):
                    success_count += 1
                    logger.info(f"Discord 第 {i + 1}/{total_chunks} 批发送成功")
                else:
                    logger.error(f"Discord 第 {i + 1}/{total_chunks} 批发送失败")
            except NotificationError as e:
                logger.error(f"Discord 第 {i + 1}/{total_chunks} 批发送异常: {e}")

            if i < total_chunks - 1:
                await asyncio.sleep(1.5)

        return success_count == total_chunks

    def _split_large_section(self, section: str, max_size: int = 1800) -> list[str]:
        """将大段落智能分割成多个小段落"""
        lines = section.split("\n")
        chunks = []
        current_chunk = []
        current_length = 0

        for line in lines:
            line_len = len(line) + 1

            if line_len > max_size:
                if current_chunk:
                    chunks.append("\n".join(current_chunk))
                    current_chunk = []
                    current_length = 0
                chunks.append(line[: max_size - 20] + "...(截断)")
                continue

            if current_length + line_len > max_size:
                if current_chunk:
                    chunks.append("\n".join(current_chunk))
                current_chunk = [line]
                current_length = line_len
            else:
                current_chunk.append(line)
                current_length += line_len

        if current_chunk:
            chunks.append("\n".join(current_chunk))

        return chunks if chunks else [section[:max_size]]

    async def _send_force_chunked(self, content: str) -> bool:
        """强制按行分割发送（异步）"""
        chunks = []
        current_chunk = ""
        lines = content.split("\n")

        for line in lines:
            test_chunk = current_chunk + ("\n" if current_chunk else "") + line
            if len(test_chunk) > self.MAX_LENGTH - 50:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = line
            else:
                current_chunk = test_chunk

        if current_chunk:
            chunks.append(current_chunk)

        total_chunks = len(chunks)
        success_count = 0

        logger.info(f"Discord 强制分批发送：共 {total_chunks} 批")

        for i, chunk in enumerate(chunks):
            page_marker = f"\n\n📄 ({i + 1}/{total_chunks})" if total_chunks > 1 else ""
            final_chunk = chunk + page_marker if len(chunk) + len(page_marker) <= self.MAX_LENGTH else chunk

            try:
                if await self._send_message(final_chunk):
                    success_count += 1
            except NotificationError as e:
                logger.error(f"Discord 第 {i + 1}/{total_chunks} 批发送异常: {e}")

            if i < total_chunks - 1:
                await asyncio.sleep(1)

        return success_count == total_chunks

    async def _send_message(self, content: str) -> bool:
        """发送单条 Discord 消息（异步）"""
        if not self.webhook_url:
            return False

        session = get_aiohttp_session()
        payload = {"content": content}

        async with session.post(
            self.webhook_url,
            json=payload,
            headers={"Content-Type": "application/json"},
        ) as response:
            if response.status in (200, 204):
                return True
            else:
                text = await response.text()
                logger.error(f"Discord 请求失败: HTTP {response.status}, {text}")
                return False
