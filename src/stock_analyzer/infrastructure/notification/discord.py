"""
Discord 通知渠道
"""

import logging
import time
from typing import Any

import httpx

from stock_analyzer.domain.exceptions import NotificationError
from stock_analyzer.infrastructure.notification.base import NotificationChannel, NotificationChannelBase

logger = logging.getLogger(__name__)


class DiscordChannel(NotificationChannelBase):
    """Discord Webhook 通知渠道"""

    MAX_LENGTH = 2000  # Discord 消息长度限制

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

    def send(self, content: str, **kwargs: Any) -> bool:
        """
        发送消息到 Discord

        Args:
            content: Markdown 格式的消息内容

        Returns:
            是否发送成功
        """
        if not self.webhook_url:
            logger.warning("Discord Webhook 未配置，跳过推送")
            return False

        # 检查长度，超长则分批发送
        if len(content) > self.MAX_LENGTH:
            logger.info(f"Discord 消息内容超长({len(content)}字符)，将分批发送")
            return self._send_chunked(content)

        try:
            return self._send_message(content)
        except NotificationError as e:
            logger.error(f"发送 Discord 消息失败: {e}")
            return False

    def _send_chunked(self, content: str) -> bool:
        """分批发送长消息"""
        # 智能分割：优先按 ## 二级标题（股票分隔），其次按 --- 分隔符，最后按 ### 标题
        if "\n## " in content:
            # 按股票章节分割，保留标题
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
            # 没有明显分隔符，强制按行分割
            return self._send_force_chunked(content)

        chunks = []
        current_chunk = []
        current_length = 0
        separator_len = len(separator)

        for section in sections:
            section_len = len(section) + separator_len

            # 单个段落就超长，需要进一步分割
            if section_len > self.MAX_LENGTH - 100:  # 预留空间给分页标记
                if current_chunk:
                    chunks.append(separator.join(current_chunk))
                    current_chunk = []
                    current_length = 0

                # 对超长段落进行智能分割
                sub_chunks = self._split_large_section(section)
                chunks.extend(sub_chunks)
                continue

            # 当前批次加上新段落会超限，先保存当前批次
            if current_length + section_len > self.MAX_LENGTH - 100:
                if current_chunk:
                    chunks.append(separator.join(current_chunk))
                current_chunk = [section]
                current_length = len(section)
            else:
                current_chunk.append(section)
                current_length += section_len

        # 添加最后一个批次
        if current_chunk:
            chunks.append(separator.join(current_chunk))

        # 分批发送
        total_chunks = len(chunks)
        success_count = 0

        logger.info(f"Discord 分批发送：共 {total_chunks} 批")

        for i, chunk in enumerate(chunks):
            # 添加分页标记
            if total_chunks > 1:
                page_marker = f"\n\n📄 ({i + 1}/{total_chunks})"
                # 确保加上标记后不超限
                if len(chunk) + len(page_marker) > self.MAX_LENGTH:
                    chunk = chunk[: self.MAX_LENGTH - len(page_marker) - 10] + page_marker
                else:
                    chunk = chunk + page_marker

            try:
                if self._send_message(chunk):
                    success_count += 1
                    logger.info(f"Discord 第 {i + 1}/{total_chunks} 批发送成功")
                else:
                    logger.error(f"Discord 第 {i + 1}/{total_chunks} 批发送失败")
            except NotificationError as e:
                logger.error(f"Discord 第 {i + 1}/{total_chunks} 批发送异常: {e}")

            # 批次间添加延迟，避免触发限制
            if i < total_chunks - 1:
                time.sleep(1.5)

        return success_count == total_chunks

    def _split_large_section(self, section: str, max_size: int = 1800) -> list[str]:
        """将大段落智能分割成多个小段落"""
        lines = section.split("\n")
        chunks = []
        current_chunk = []
        current_length = 0

        for line in lines:
            line_len = len(line) + 1  # +1 for newline

            # 单行就超长，直接截断
            if line_len > max_size:
                if current_chunk:
                    chunks.append("\n".join(current_chunk))
                    current_chunk = []
                    current_length = 0
                # 截断超长行
                chunks.append(line[: max_size - 20] + "...(截断)")
                continue

            # 当前块加上新行会超限
            if current_length + line_len > max_size:
                if current_chunk:
                    chunks.append("\n".join(current_chunk))
                current_chunk = [line]
                current_length = line_len
            else:
                current_chunk.append(line)
                current_length += line_len

        # 添加最后一个块
        if current_chunk:
            chunks.append("\n".join(current_chunk))

        return chunks if chunks else [section[:max_size]]

    def _send_force_chunked(self, content: str) -> bool:
        """强制按行分割发送"""
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
                if self._send_message(final_chunk):
                    success_count += 1
            except NotificationError as e:
                logger.error(f"Discord 第 {i + 1}/{total_chunks} 批发送异常: {e}")

            if i < total_chunks - 1:
                time.sleep(1)

        return success_count == total_chunks

    def _send_message(self, content: str) -> bool:
        """发送单条 Discord 消息"""
        if not self.webhook_url:
            return False

        payload = {
            "content": content,
        }

        response = httpx.post(
            self.webhook_url,
            json=payload,
            timeout=30,
            headers={"Content-Type": "application/json"},
        )

        if response.status_code in (200, 204):
            return True
        else:
            logger.error(f"Discord 请求失败: HTTP {response.status_code}, {response.text}")
            return False
