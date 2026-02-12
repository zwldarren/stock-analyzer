"""
Notification module.

Supported channels: Email, Telegram, Discord Webhook, Custom Webhook
"""

from stock_analyzer.infrastructure.notification.base import (
    NotificationChannel,
    NotificationChannelBase,
)
from stock_analyzer.infrastructure.notification.builder import NotificationBuilder
from stock_analyzer.infrastructure.notification.email import EmailChannel
from stock_analyzer.infrastructure.notification.report_generator import ReportGenerator
from stock_analyzer.infrastructure.notification.service import NotificationService
from stock_analyzer.infrastructure.notification.telegram import TelegramChannel

__all__ = [
    "NotificationService",
    "NotificationBuilder",
    "ReportGenerator",
    "NotificationChannel",
    "NotificationChannelBase",
    "TelegramChannel",
    "EmailChannel",
]
