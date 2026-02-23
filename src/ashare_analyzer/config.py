"""Pydantic Settings configuration management."""

import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Any

from pydantic import AfterValidator, BeforeValidator, Field, ValidationError, computed_field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from ashare_analyzer.exceptions import ConfigurationError
from ashare_analyzer.utils.stock_code import StockType, detect_stock_type

logger = logging.getLogger(__name__)


def _find_project_root() -> Path:
    """Find project root directory (contains pyproject.toml or .env).

    Priority check for PROJECT_ROOT environment variable for Docker scenarios.
    """
    # Priority: environment variable (for Docker)
    env_root = os.environ.get("PROJECT_ROOT")
    if env_root:
        path = Path(env_root)
        if path.exists():
            return path.resolve()

    # Auto-detect (development environment)
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").exists() or (parent / ".env").exists():
            return parent
    return current.parents[3]


_PROJECT_ROOT = _find_project_root()


def _parse_comma_list(value: str | None) -> list[str]:
    """Parse comma-separated string into list."""
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _parse_bool(value: str | bool | None) -> bool:
    """Parse boolean value from string or bool."""
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    return str(value).lower() in ("true", "1", "yes")


def _validate_temperature(v: float) -> float:
    """Validate LLM temperature is within valid range."""
    if not 0 <= v <= 2:
        raise ConfigurationError("Temperature must be between 0 and 2")
    return v


ValidTemperature = Annotated[float, AfterValidator(_validate_temperature)]

# Type alias for boolean fields from environment variables
EnvBool = Annotated[bool, BeforeValidator(_parse_bool)]

# Shared model configuration
_COMMON_CONFIG = SettingsConfigDict(
    env_file=_PROJECT_ROOT / ".env",
    env_file_encoding="utf-8",
    extra="ignore",
)


# ==========================================
# Nested configuration classes using BaseSettings
# ==========================================


class AIConfig(BaseSettings):
    """AI model configuration supporting multiple providers via litellm format."""

    model_config: SettingsConfigDict = _COMMON_CONFIG

    # Primary model configuration (litellm format: provider/model-name)
    llm_model: str | None = Field(default=None, validation_alias="LLM_MODEL")
    llm_api_key: str | None = Field(default=None, validation_alias="LLM_API_KEY")
    llm_base_url: str | None = Field(default=None, validation_alias="LLM_BASE_URL")

    # Fallback model configuration for failover
    llm_fallback_model: str | None = Field(default=None, validation_alias="LLM_FALLBACK_MODEL")
    llm_fallback_api_key: str | None = Field(default=None, validation_alias="LLM_FALLBACK_API_KEY")
    llm_fallback_base_url: str | None = Field(default=None, validation_alias="LLM_FALLBACK_BASE_URL")

    # Common generation parameters
    llm_temperature: ValidTemperature = Field(default=0.7, validation_alias="LLM_TEMPERATURE")
    llm_max_tokens: int = Field(default=8192, ge=1, validation_alias="LLM_MAX_TOKENS")
    llm_request_delay: float = Field(default=2.0, validation_alias="LLM_REQUEST_DELAY")
    llm_max_retries: int = Field(default=5, ge=0, le=10, validation_alias="LLM_MAX_RETRIES")
    llm_retry_delay: float = Field(default=5.0, validation_alias="LLM_RETRY_DELAY")


class SearchConfig(BaseSettings):
    """Search engine configuration."""

    model_config = _COMMON_CONFIG

    # Store raw values as str, return lists via computed_field
    bocha_api_keys_str: str = Field(default="", validation_alias="BOCHA_API_KEYS")
    tavily_api_keys_str: str = Field(default="", validation_alias="TAVILY_API_KEYS")
    brave_api_keys_str: str = Field(default="", validation_alias="BRAVE_API_KEYS")
    serpapi_keys_str: str = Field(default="", validation_alias="SERPAPI_API_KEYS")

    # SearXNG configuration
    searxng_base_url: str = Field(default="", validation_alias="SEARXNG_BASE_URL")
    searxng_username: str | None = Field(default=None, validation_alias="SEARXNG_USERNAME")
    searxng_password: str | None = Field(default=None, validation_alias="SEARXNG_PASSWORD")

    @computed_field
    @property
    def bocha_api_keys(self) -> list[str]:
        return _parse_comma_list(self.bocha_api_keys_str)

    @computed_field
    @property
    def tavily_api_keys(self) -> list[str]:
        return _parse_comma_list(self.tavily_api_keys_str)

    @computed_field
    @property
    def brave_api_keys(self) -> list[str]:
        return _parse_comma_list(self.brave_api_keys_str)

    @computed_field
    @property
    def serpapi_keys(self) -> list[str]:
        return _parse_comma_list(self.serpapi_keys_str)


class NewsFilterConfig(BaseSettings):
    """News filter configuration."""

    model_config = _COMMON_CONFIG

    news_filter_enabled: EnvBool = Field(default=True, validation_alias="NEWS_FILTER_ENABLED")
    news_filter_min_results: int = Field(default=3, ge=1, le=10, validation_alias="NEWS_FILTER_MIN_RESULTS")
    news_filter_model: str | None = Field(default=None, validation_alias="NEWS_FILTER_MODEL")


class NotificationChannelConfig(BaseSettings):
    """Notification channel configuration."""

    model_config = _COMMON_CONFIG

    # Email
    email_sender: str | None = Field(default=None, validation_alias="EMAIL_SENDER")
    email_sender_name: str = Field(default="ashare_analyzer股票分析助手", validation_alias="EMAIL_SENDER_NAME")
    email_password: str | None = Field(default=None, validation_alias="EMAIL_PASSWORD")
    email_receivers_str: str = Field(default="", validation_alias="EMAIL_RECEIVERS")

    # Telegram
    telegram_bot_token: str | None = Field(default=None, validation_alias="TELEGRAM_BOT_TOKEN")
    telegram_chat_id: str | None = Field(default=None, validation_alias="TELEGRAM_CHAT_ID")
    telegram_message_thread_id: str | None = Field(default=None, validation_alias="TELEGRAM_MESSAGE_THREAD_ID")

    # Discord
    discord_webhook_url: str | None = Field(default=None, validation_alias="DISCORD_WEBHOOK_URL")

    # Custom Webhook
    custom_webhook_urls_str: str = Field(default="", validation_alias="CUSTOM_WEBHOOK_URLS")
    custom_webhook_bearer_token: str | None = Field(default=None, validation_alias="CUSTOM_WEBHOOK_BEARER_TOKEN")

    @computed_field
    @property
    def email_receivers(self) -> list[str]:
        return _parse_comma_list(self.email_receivers_str)

    @computed_field
    @property
    def custom_webhook_urls(self) -> list[str]:
        return _parse_comma_list(self.custom_webhook_urls_str)


class NotificationMessageConfig(BaseSettings):
    """Notification message configuration."""

    model_config = _COMMON_CONFIG

    single_stock_notify: EnvBool = Field(default=False, validation_alias="SINGLE_STOCK_NOTIFY")


class LoggingConfig(BaseSettings):
    """Logging configuration."""

    model_config = _COMMON_CONFIG

    log_level: str = Field(default="INFO", validation_alias="LOG_LEVEL")

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        valid_levels = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")
        if v.upper() not in valid_levels:
            raise ConfigurationError(f"Log level must be one of {valid_levels}")
        return v.upper()


class SystemConfig(BaseSettings):
    """System configuration."""

    model_config = _COMMON_CONFIG

    max_workers: int = Field(default=3, ge=1, le=20, validation_alias="MAX_WORKERS")
    debug: EnvBool = Field(default=False, validation_alias="DEBUG")
    http_proxy: str | None = Field(default=None, validation_alias="HTTP_PROXY")
    https_proxy: str | None = Field(default=None, validation_alias="HTTPS_PROXY")


class ScheduleConfig(BaseSettings):
    """Scheduled task configuration."""

    model_config = _COMMON_CONFIG

    schedule_enabled: EnvBool = Field(default=False, validation_alias="SCHEDULE_ENABLED")
    schedule_time: str = Field(default="18:00", validation_alias="SCHEDULE_TIME")


class RealtimeQuoteConfig(BaseSettings):
    """Real-time quote configuration."""

    model_config = _COMMON_CONFIG

    realtime_source_priority: str = Field(
        default="tencent,akshare_sina,efinance,akshare_em",
        validation_alias="REALTIME_SOURCE_PRIORITY",
    )


class DataSourceConfig(BaseSettings):
    """Data source configuration."""

    model_config = _COMMON_CONFIG

    tushare_token: str | None = Field(default=None, validation_alias="TUSHARE_TOKEN")

    # Data source priority (lower value = higher priority)
    efinance_priority: int = Field(default=0, ge=0, le=10, validation_alias="EFINANCE_PRIORITY")
    akshare_priority: int = Field(default=1, ge=0, le=10, validation_alias="AKSHARE_PRIORITY")
    tushare_priority: int = Field(default=2, ge=0, le=10, validation_alias="TUSHARE_PRIORITY")
    pytdx_priority: int = Field(default=2, ge=0, le=10, validation_alias="PYTDX_PRIORITY")
    baostock_priority: int = Field(default=3, ge=0, le=10, validation_alias="BAOSTOCK_PRIORITY")
    yfinance_priority: int = Field(default=4, ge=0, le=10, validation_alias="YFINANCE_PRIORITY")


# ==========================================
# Main configuration class
# ==========================================


def _default_base_dir() -> str:
    """Get default base directory for application data.

    Priority:
    1. BASE_DIR environment variable
    2. ~/.ashare-analyzer (user home directory)

    This allows PyPI-installed usage without needing a project directory.
    """
    env_base = os.environ.get("BASE_DIR")
    if env_base:
        return env_base
    return str(Path.home() / ".ashare-analyzer")


class Config(BaseSettings):
    """Main system configuration class.

    Uses pydantic-settings to automatically load configuration from environment variables.
    Supports .env files and nested configuration models.
    Each nested configuration class loads environment variables independently.

    All runtime data (database, logs, reports) is stored under base_dir,
    which defaults to ~/.ashare-analyzer for PyPI-installed usage.
    """

    model_config = SettingsConfigDict(
        env_file=_PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
        env_parse_none_str="null",
    )

    # Base directory for all runtime data (data, logs, reports)
    base_dir: str = Field(default_factory=_default_base_dir, validation_alias="BASE_DIR")

    # Basic configuration
    stock_list_str: str = Field(default="", validation_alias="STOCK_LIST")

    # Nested configurations - each loads environment variables independently
    ai: AIConfig = Field(default_factory=AIConfig)
    search: SearchConfig = Field(default_factory=SearchConfig)
    news_filter: NewsFilterConfig = Field(default_factory=NewsFilterConfig)
    notification_channel: NotificationChannelConfig = Field(default_factory=NotificationChannelConfig)
    notification_message: NotificationMessageConfig = Field(default_factory=NotificationMessageConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    system: SystemConfig = Field(default_factory=SystemConfig)
    schedule: ScheduleConfig = Field(default_factory=ScheduleConfig)
    realtime_quote: RealtimeQuoteConfig = Field(default_factory=RealtimeQuoteConfig)
    datasource: DataSourceConfig = Field(default_factory=DataSourceConfig)

    @field_validator("stock_list_str", mode="before")
    @classmethod
    def parse_stock_list(cls, v: Any) -> Any:
        """Parse stock list from string and validate stock codes."""
        if isinstance(v, str):
            # Validate and filter invalid stock codes
            codes = [code.strip() for code in v.split(",") if code.strip()]
            valid_codes = []

            for code in codes:
                stock_type = detect_stock_type(code)
                if stock_type != StockType.UNKNOWN:
                    valid_codes.append(code)
                else:
                    logger.warning(f"无效的股票代码格式: {code}，已跳过")
            return ",".join(valid_codes)
        return ""

    @computed_field
    @property
    def stock_list(self) -> list[str]:
        return _parse_comma_list(self.stock_list_str)

    # ==========================================
    # Derived paths from base_dir
    # ==========================================

    @computed_field
    @property
    def data_dir(self) -> str:
        """Directory for database and data files."""
        return str(Path(self.base_dir) / "data")

    @computed_field
    @property
    def log_dir(self) -> str:
        """Directory for log files."""
        return str(Path(self.base_dir) / "logs")

    @computed_field
    @property
    def reports_dir(self) -> str:
        """Directory for report files."""
        return str(Path(self.base_dir) / "reports")

    @computed_field
    @property
    def database_path(self) -> str:
        """Path to SQLite database file."""
        return str(Path(self.base_dir) / "data" / "stock_analysis.db")

    # ==========================================
    # Methods
    # ==========================================

    def get_db_url(self) -> str:
        """Get SQLAlchemy database connection URL.

        Creates parent directories if they don't exist.
        """
        db_path = Path(self.database_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        return f"sqlite:///{db_path.absolute()}"

    def validate_config(self) -> list[str]:
        """Validate configuration completeness and return list of warnings."""
        warnings_list: list[str] = []

        if not self.stock_list:
            warnings_list.append("警告：未配置自选股列表 (STOCK_LIST)")

        if not self.ai.llm_api_key:
            warnings_list.append("警告：未配置大模型 API Key（LLM_API_KEY），AI 分析功能将不可用")

        if (
            not self.search.bocha_api_keys
            and not self.search.tavily_api_keys
            and not self.search.brave_api_keys
            and not self.search.serpapi_keys
            and not self.search.searxng_base_url
        ):
            warnings_list.append("提示：未配置任何搜索引擎，新闻搜索功能将不可用")

        # Check notification configuration
        has_notification = (
            (self.notification_channel.email_sender and self.notification_channel.email_password)
            or (self.notification_channel.telegram_bot_token and self.notification_channel.telegram_chat_id)
            or self.notification_channel.discord_webhook_url
            or self.notification_channel.custom_webhook_urls
        )
        if not has_notification:
            warnings_list.append("提示：未配置通知渠道，将不发送推送通知")

        return warnings_list

    def refresh_stock_list(self) -> None:
        """Hot reload STOCK_LIST from environment variable and update config."""
        # Clear cache and re-instantiate
        get_config.cache_clear()


@lru_cache
def get_config() -> Config:
    """Get cached configuration instance.

    Loads configuration from environment variables and .env file.
    Database configuration loading is handled separately in infrastructure layer.

    Returns:
        Config instance with all settings loaded.
    """
    return Config()


def get_config_safe() -> tuple[Config | None, list[str]]:
    """Safely load configuration, returning partial config even if incomplete.

    Returns:
        tuple: (Config object or None, list of error messages)
    """
    errors = []
    try:
        config = Config()
        return config, []
    except ValidationError as e:
        # Configuration validation failed
        errors.append(f"配置加载失败: {e}")
        return None, errors
    except ConfigurationError as e:
        errors.append(f"配置加载异常: {e}")
        return None, errors


def check_config_valid(config: Config | None) -> tuple[bool, list[str]]:
    """Check if configuration is valid (contains minimum required config).

    Returns:
        tuple: (is_valid, list of missing configuration items)
    """
    if config is None:
        return False, ["配置未加载"]

    missing = []

    # Check required AI configuration
    if not config.ai.llm_api_key:
        missing.append("LLM_API_KEY (AI 模型 API 密钥)")

    # Check stock list
    if not config.stock_list:
        missing.append("STOCK_LIST (股票代码列表)")

    # Check AI model
    if not config.ai.llm_model:
        missing.append("LLM_MODEL (AI 模型名称)")

    is_valid = len(missing) == 0
    return is_valid, missing


def get_project_root() -> Path:
    """Get project root directory path.

    Returns:
        Project root directory (contains pyproject.toml or .env)
    """
    return _PROJECT_ROOT
