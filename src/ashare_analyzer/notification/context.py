"""
通知服务上下文模块

提供消息上下文抽象，解耦通知服务与具体消息模型的依赖。
"""

from dataclasses import dataclass
from typing import Protocol


@dataclass
class MessageContext:
    """消息上下文

    用于传递消息来源信息，支持通知服务的上下文感知回复。

    Attributes:
        platform: 平台标识（如 telegram, email, discord 等）
        user_id: 用户ID
        user_name: 用户名称
        chat_id: 聊天/群组ID
        message_id: 消息ID
        content: 消息内容
    """

    platform: str | None = None
    user_id: str | None = None
    user_name: str | None = None
    chat_id: str | None = None
    message_id: str | None = None
    content: str | None = None


class MessageContextProvider(Protocol):
    """消息上下文提供者协议

    实现此协议的对象可以提供消息上下文，用于通知服务。
    """

    def get_context(self) -> MessageContext:
        """获取消息上下文"""
        ...


def create_message_context(
    platform: str | None = None,
    user_id: str | None = None,
    user_name: str | None = None,
    chat_id: str | None = None,
    message_id: str | None = None,
    content: str | None = None,
) -> MessageContext:
    """创建消息上下文的工厂函数

    Args:
        platform: 平台标识
        user_id: 用户ID
        user_name: 用户名称
        chat_id: 聊天/群组ID
        message_id: 消息ID
        content: 消息内容

    Returns:
        MessageContext 实例
    """
    return MessageContext(
        platform=platform,
        user_id=user_id,
        user_name=user_name,
        chat_id=chat_id,
        message_id=message_id,
        content=content,
    )
