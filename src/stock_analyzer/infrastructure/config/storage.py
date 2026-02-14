"""Infrastructure configuration storage implementation."""

import logging
from datetime import datetime
from typing import Any

from sqlalchemy import Column, DateTime, String, Text, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

from stock_analyzer.config.config import Config, get_project_root
from stock_analyzer.domain.exceptions import ConfigurationError

logger = logging.getLogger(__name__)

Base = declarative_base()


class AppConfigModel(Base):
    """Application configuration database model."""

    __tablename__ = "app_config"

    key = Column(String(100), primary_key=True)
    value = Column(Text)
    description = Column(String(255))
    updated_at = Column(DateTime)


class ConfigConverter:
    """Configuration converter utility.

    Converts Config objects to dictionaries suitable for environment variables.
    """

    def to_dict(self, config: Config) -> dict[str, Any]:
        """Convert Config object to environment variable dictionary.

        Args:
            config: Configuration object to convert.

        Returns:
            Dictionary with environment variable names as keys.
        """
        result: dict[str, Any] = {}

        if config.stock_list:
            result["STOCK_LIST"] = ",".join(config.stock_list)

        ai = config.ai
        if ai.llm_model:
            result["LLM_MODEL"] = ai.llm_model
        if ai.llm_api_key:
            result["LLM_API_KEY"] = ai.llm_api_key
        if ai.llm_base_url:
            result["LLM_BASE_URL"] = ai.llm_base_url
        if ai.llm_fallback_model:
            result["LLM_FALLBACK_MODEL"] = ai.llm_fallback_model
        if ai.llm_fallback_api_key:
            result["LLM_FALLBACK_API_KEY"] = ai.llm_fallback_api_key
        if ai.llm_fallback_base_url:
            result["LLM_FALLBACK_BASE_URL"] = ai.llm_fallback_base_url
        result["LLM_TEMPERATURE"] = ai.llm_temperature
        result["LLM_MAX_TOKENS"] = ai.llm_max_tokens
        result["LLM_REQUEST_DELAY"] = ai.llm_request_delay
        result["LLM_MAX_RETRIES"] = ai.llm_max_retries
        result["LLM_RETRY_DELAY"] = ai.llm_retry_delay

        search = config.search
        if search.bocha_api_keys_str:
            result["BOCHA_API_KEYS"] = search.bocha_api_keys_str
        if search.tavily_api_keys_str:
            result["TAVILY_API_KEYS"] = search.tavily_api_keys_str
        if search.brave_api_keys_str:
            result["BRAVE_API_KEYS"] = search.brave_api_keys_str
        if search.serpapi_keys_str:
            result["SERPAPI_API_KEYS"] = search.serpapi_keys_str
        if search.searxng_base_url:
            result["SEARXNG_BASE_URL"] = search.searxng_base_url

        notify = config.notification_channel
        if notify.telegram_bot_token:
            result["TELEGRAM_BOT_TOKEN"] = notify.telegram_bot_token
        if notify.telegram_chat_id:
            result["TELEGRAM_CHAT_ID"] = notify.telegram_chat_id
        if notify.email_sender:
            result["EMAIL_SENDER"] = notify.email_sender
        if notify.email_password:
            result["EMAIL_PASSWORD"] = notify.email_password
        if notify.email_receivers_str:
            result["EMAIL_RECEIVERS"] = notify.email_receivers_str
        if notify.discord_webhook_url:
            result["DISCORD_WEBHOOK_URL"] = notify.discord_webhook_url
        if notify.custom_webhook_urls_str:
            result["CUSTOM_WEBHOOK_URLS"] = notify.custom_webhook_urls_str
        if notify.custom_webhook_bearer_token:
            result["CUSTOM_WEBHOOK_BEARER_TOKEN"] = notify.custom_webhook_bearer_token

        db = config.database
        result["DATABASE_PATH"] = db.database_path

        log = config.logging
        result["LOG_DIR"] = log.log_dir
        result["LOG_LEVEL"] = log.log_level

        sys_cfg = config.system
        result["MAX_WORKERS"] = sys_cfg.max_workers
        result["DEBUG"] = sys_cfg.debug
        if sys_cfg.http_proxy:
            result["HTTP_PROXY"] = sys_cfg.http_proxy
        if sys_cfg.https_proxy:
            result["HTTPS_PROXY"] = sys_cfg.https_proxy

        return result


class ConfigStorage:
    """Configuration storage manager for .env files."""

    def __init__(self) -> None:
        self.project_root = get_project_root()
        self.env_file = self.project_root / ".env"
        self.converter = ConfigConverter()

    def save_to_env(self, config_dict: dict[str, Any]) -> None:
        """Save configuration dictionary to .env file.

        Args:
            config_dict: Configuration dictionary with env var names as keys.
        """
        self.env_file.parent.mkdir(parents=True, exist_ok=True)

        existing_lines: list[str] = []
        if self.env_file.exists():
            with open(self.env_file, encoding="utf-8") as f:
                existing_lines = f.readlines()

        new_config_lines: list[str] = []
        existing_keys: set[str] = set()

        for key, value in config_dict.items():
            if value is not None:
                if isinstance(value, list):
                    value = ",".join(str(v) for v in value)
                elif isinstance(value, bool):
                    value = "true" if value else "false"
                else:
                    value = str(value)

                if " " in value or "#" in value:
                    value = f'"{value}"'

                new_config_lines.append(f"{key}={value}\n")
                existing_keys.add(key)

        for line in existing_lines:
            line = line.rstrip("\n")
            if line.startswith("#") or not line.strip():
                new_config_lines.append(line + "\n")
            else:
                if "=" in line:
                    key = line.split("=", 1)[0].strip()
                    if key not in existing_keys:
                        new_config_lines.append(line + "\n")

        with open(self.env_file, "w", encoding="utf-8") as f:
            f.writelines(new_config_lines)

    def save_config_to_env(self, config: Config) -> None:
        """Save complete Config object to .env file.

        Args:
            config: Configuration object to save.
        """
        config_dict = self.converter.to_dict(config)
        self.save_to_env(config_dict)


class ConfigStorageImpl:
    """Database-backed configuration storage implementation.

    This class provides configuration storage using SQLAlchemy
    for database operations. It supports saving and loading configuration
    from a SQLite database.

    Configuration priority: Environment variables > .env file > Database
    """

    def __init__(self, db_url: str | None = None):
        """Initialize configuration storage.

        Args:
            db_url: Database URL. If None, uses default SQLite path.
        """
        self.project_root = get_project_root()
        self.env_file = self.project_root / ".env"
        self.converter = ConfigConverter()
        self._file_storage = ConfigStorage()

        self.db_session = None
        self._engine = None

        if db_url:
            self._init_db(db_url)
        else:
            db_path = self.project_root / "data" / "stock_analysis.db"
            db_path.parent.mkdir(parents=True, exist_ok=True)
            self._init_db(f"sqlite:///{db_path}")

    def _init_db(self, db_url: str) -> None:
        """Initialize database connection.

        Args:
            db_url: SQLAlchemy database URL.
        """
        self._engine = create_engine(db_url)
        Base.metadata.create_all(self._engine)
        Session = sessionmaker(bind=self._engine)
        self.db_session = Session()

    def save_to_env(self, config_dict: dict[str, Any]) -> None:
        """Save configuration to .env file.

        Args:
            config_dict: Configuration dictionary with env var names as keys.
        """
        self._file_storage.save_to_env(config_dict)

    def save_to_db(self, config_dict: dict[str, Any]) -> None:
        """Save configuration to database.

        Args:
            config_dict: Configuration dictionary.

        Raises:
            ConfigurationError: If database is not initialized.
        """
        if not self.db_session:
            raise ConfigurationError("Database not initialized, please provide db_url")

        for key, value in config_dict.items():
            if value is not None:
                if isinstance(value, list):
                    value = ",".join(str(v) for v in value)
                elif isinstance(value, bool):
                    value = "true" if value else "false"
                else:
                    value = str(value)

                config_item = self.db_session.query(AppConfigModel).filter_by(key=key).first()
                if config_item:
                    config_item.value = value
                    config_item.updated_at = datetime.now()
                else:
                    config_item = AppConfigModel(
                        key=key,
                        value=value,
                        updated_at=datetime.now(),
                    )
                    self.db_session.add(config_item)

        self.db_session.commit()

    def save_config_to_env(self, config: Config) -> None:
        """Save complete Config object to .env file.

        Args:
            config: Configuration object to save.
        """
        config_dict = self.converter.to_dict(config)
        self.save_to_env(config_dict)

    def save_config_to_db(self, config: Config) -> None:
        """Save complete Config object to database.

        Args:
            config: Configuration object to save.

        Raises:
            ConfigurationError: If database is not initialized.
        """
        if not self.db_session:
            raise ConfigurationError("Database not initialized, please provide db_url")

        config_dict = self.converter.to_dict(config)
        self.save_to_db(config_dict)

    def load_from_db(self) -> dict[str, str]:
        """Load configuration from database.

        Returns:
            Dictionary of configuration key-value pairs.

        Raises:
            ConfigurationError: If database is not initialized.
        """
        if not self.db_session:
            raise ConfigurationError("Database not initialized, please provide db_url")

        configs = self.db_session.query(AppConfigModel).all()
        result: dict[str, str] = {}
        for config in configs:
            key = str(config.key)
            value = str(config.value) if config.value else ""
            result[key] = value
        return result

    def close(self) -> None:
        """Close database connection and release resources."""
        if self.db_session:
            self.db_session.close()
            self.db_session = None
        if self._engine:
            self._engine.dispose()
            self._engine = None

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


def save_config_to_db_only(config_dict: dict[str, Any], db_url: str) -> None:
    """Convenience function: save configuration from dictionary to database.

    Args:
        config_dict: Configuration dictionary.
        db_url: Database URL.
    """
    storage = ConfigStorageImpl(db_url)
    try:
        storage.save_to_db(config_dict)
    finally:
        storage.close()
