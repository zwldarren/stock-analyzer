"""
邮件通知渠道
"""

import logging
import smtplib
from datetime import datetime
from email.header import Header
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.utils import formataddr
from typing import Any

import markdown2

from stock_analyzer.notification.base import NotificationChannel, NotificationChannelBase

logger = logging.getLogger(__name__)

# SMTP 服务器配置
SMTP_CONFIGS = {
    "qq.com": {"server": "smtp.qq.com", "port": 465, "ssl": True},
    "foxmail.com": {"server": "smtp.qq.com", "port": 465, "ssl": True},
    "163.com": {"server": "smtp.163.com", "port": 465, "ssl": True},
    "126.com": {"server": "smtp.126.com", "port": 465, "ssl": True},
    "gmail.com": {"server": "smtp.gmail.com", "port": 587, "ssl": False},
    "outlook.com": {"server": "smtp-mail.outlook.com", "port": 587, "ssl": False},
    "hotmail.com": {"server": "smtp-mail.outlook.com", "port": 587, "ssl": False},
    "live.com": {"server": "smtp-mail.outlook.com", "port": 587, "ssl": False},
    "sina.com": {"server": "smtp.sina.com", "port": 465, "ssl": True},
    "sohu.com": {"server": "smtp.sohu.com", "port": 465, "ssl": True},
    "aliyun.com": {"server": "smtp.aliyun.com", "port": 465, "ssl": True},
    "139.com": {"server": "smtp.139.com", "port": 465, "ssl": True},
}


class EmailChannel(NotificationChannelBase):
    """邮件 SMTP 通知渠道"""

    def __init__(self, config: dict[str, Any]):
        self.sender: str | None = None
        self.sender_name: str = "stock_analyzer股票分析助手"
        self.password: str | None = None
        self.receivers: list[str] = []
        super().__init__(config)

    def _validate_config(self) -> None:
        """验证配置"""
        self.sender = self.config.get("sender")
        self.sender_name = self.config.get("sender_name", "stock_analyzer股票分析助手")
        self.password = self.config.get("password")
        self.receivers = self.config.get("receivers", [])
        if not self.receivers and self.sender:
            self.receivers = [self.sender]

    def is_available(self) -> bool:
        """检查配置是否完整"""
        return bool(self.sender and self.password)

    @property
    def channel_type(self) -> NotificationChannel:
        return NotificationChannel.EMAIL

    def send(self, content: str, **kwargs: Any) -> bool:
        """
        发送邮件

        Args:
            content: Markdown 格式的邮件内容
            **kwargs: 可包含 subject 参数指定主题

        Returns:
            是否发送成功
        """
        if not self.sender or not self.password:
            logger.warning("邮件配置不完整，跳过推送")
            return False

        subject = kwargs.get("subject")
        if subject is None:
            date_str = datetime.now().strftime("%Y-%m-%d")
            subject = f"📈 股票智能分析报告 - {date_str}"

        try:
            # 将 Markdown 转换为 HTML
            html_content = self._markdown_to_html(content)

            # 构建邮件
            msg = MIMEMultipart("alternative")
            msg["Subject"] = str(Header(subject, "utf-8"))
            msg["From"] = formataddr((self.sender_name, self.sender))
            msg["To"] = ", ".join(self.receivers)

            # 添加纯文本和 HTML 两个版本
            text_part = MIMEText(content, "plain", "utf-8")
            html_part = MIMEText(html_content, "html", "utf-8")
            msg.attach(text_part)
            msg.attach(html_part)

            # 自动识别 SMTP 配置
            domain = self.sender.split("@")[-1].lower()
            smtp_config = SMTP_CONFIGS.get(domain)

            if smtp_config:
                smtp_server: str = str(smtp_config["server"])
                smtp_port: int = int(smtp_config["port"])
                use_ssl: bool = bool(smtp_config["ssl"])
                logger.info(f"自动识别邮箱类型: {domain} -> {smtp_server}:{smtp_port}")
            else:
                smtp_server = f"smtp.{domain}"
                smtp_port = 465
                use_ssl = True
                logger.warning(f"未知邮箱类型 {domain}，尝试通用配置")

            # 发送邮件
            if use_ssl:
                server = smtplib.SMTP_SSL(smtp_server, smtp_port, timeout=30)
            else:
                server = smtplib.SMTP(smtp_server, smtp_port, timeout=30)
                server.starttls()

            server.login(self.sender, self.password)
            server.send_message(msg)
            server.quit()

            logger.info(f"邮件发送成功，收件人: {self.receivers}")
            return True

        except smtplib.SMTPAuthenticationError:
            logger.error("邮件发送失败：认证错误，请检查邮箱和授权码")
            return False
        except Exception as e:
            logger.error(f"发送邮件失败: {e}")
            return False

    def _markdown_to_html(self, markdown_text: str) -> str:
        """将 Markdown 转换为 HTML"""
        html_content = markdown2.markdown(
            markdown_text,
            extras=["tables", "fenced-code-blocks", "break-on-newline", "cuddled-lists"],
        )

        css_style = """
            body {
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
                line-height: 1.5;
                color: #24292e;
                font-size: 14px;
                padding: 15px;
                max-width: 900px;
                margin: 0 auto;
            }
            h1 { font-size: 20px; border-bottom: 1px solid #eaecef; padding-bottom: 0.3em; }
            h2 { font-size: 18px; border-bottom: 1px solid #eaecef; padding-bottom: 0.3em; }
            h3 { font-size: 16px; margin-top: 0.8em; margin-bottom: 0.4em; }
            table { border-collapse: collapse; width: 100%; margin: 12px 0; }
            th, td { border: 1px solid #dfe2e5; padding: 6px 10px; text-align: left; }
            th { background-color: #f6f8fa; font-weight: 600; }
        """

        return f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <style>{css_style}</style>
        </head>
        <body>{html_content}</body>
        </html>
        """
