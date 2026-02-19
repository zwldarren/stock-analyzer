"""
Notification module.

Supported channels: Email, Telegram, Discord Webhook, Custom Webhook
"""

from stock_analyzer.notification.base import (
    NotificationChannel,
    NotificationChannelBase,
)
from stock_analyzer.notification.builder import NotificationBuilder
from stock_analyzer.notification.email import EmailChannel
from stock_analyzer.notification.report_generator import ReportGenerator
from stock_analyzer.notification.service import NotificationService, get_notification_service
from stock_analyzer.notification.telegram import TelegramChannel

__all__ = [
    "NotificationService",
    "get_notification_service",
    "NotificationBuilder",
    "ReportGenerator",
    "NotificationChannel",
    "NotificationChannelBase",
    "TelegramChannel",
    "EmailChannel",
]
