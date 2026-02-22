"""
Notification module.

Supported channels: Email, Telegram, Discord Webhook, Custom Webhook
"""

from ashare_analyzer.notification.base import (
    NotificationChannel,
    NotificationChannelBase,
)
from ashare_analyzer.notification.builder import NotificationBuilder
from ashare_analyzer.notification.email import EmailChannel
from ashare_analyzer.notification.report_generator import ReportGenerator
from ashare_analyzer.notification.service import NotificationService, get_notification_service
from ashare_analyzer.notification.telegram import TelegramChannel

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
