"""Configuration storage module."""

from typing import Any

from stock_analyzer.config.config import Config, get_project_root


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

        # Basic configuration
        if config.stock_list:
            result["STOCK_LIST"] = ",".join(config.stock_list)

        # AI configuration
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

        # Search configuration
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

        # Notification configuration
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

        # Database configuration
        db = config.database
        result["DATABASE_PATH"] = db.database_path
        result["SAVE_CONTEXT_SNAPSHOT"] = db.save_context_snapshot

        # Logging configuration
        log = config.logging
        result["LOG_DIR"] = log.log_dir
        result["LOG_LEVEL"] = log.log_level

        # System configuration
        sys_cfg = config.system
        result["MAX_WORKERS"] = sys_cfg.max_workers
        result["DEBUG"] = sys_cfg.debug
        if sys_cfg.http_proxy:
            result["HTTP_PROXY"] = sys_cfg.http_proxy
        if sys_cfg.https_proxy:
            result["HTTPS_PROXY"] = sys_cfg.https_proxy

        return result


class ConfigStorage:
    """Configuration storage manager for .env files.

    This class handles saving configuration to .env files only.
    For database storage, use infrastructure.config.storage.ConfigStorageImpl.
    """

    def __init__(self) -> None:
        self.project_root = get_project_root()
        self.env_file = self.project_root / ".env"
        self.converter = ConfigConverter()

    def save_to_env(self, config_dict: dict[str, Any]) -> None:
        """Save configuration dictionary to .env file.

        Args:
            config_dict: Configuration dictionary with env var names as keys.
        """
        # Ensure directory exists
        self.env_file.parent.mkdir(parents=True, exist_ok=True)

        # Read existing content (preserve comments and non-config lines)
        existing_lines: list[str] = []
        if self.env_file.exists():
            with open(self.env_file, encoding="utf-8") as f:
                existing_lines = f.readlines()

        # Build new configuration content
        new_config_lines: list[str] = []
        existing_keys: set[str] = set()

        # First add new configuration
        for key, value in config_dict.items():
            if value is not None:
                # Handle list type
                if isinstance(value, list):
                    value = ",".join(str(v) for v in value)
                # Handle boolean
                elif isinstance(value, bool):
                    value = "true" if value else "false"
                else:
                    value = str(value)

                # Quote value if it contains special characters
                if " " in value or "#" in value:
                    value = f'"{value}"'

                new_config_lines.append(f"{key}={value}\n")
                existing_keys.add(key)

        # Preserve existing configurations not being overwritten
        for line in existing_lines:
            line = line.rstrip("\n")
            if line.startswith("#") or not line.strip():
                # Preserve comments and empty lines
                new_config_lines.append(line + "\n")
            else:
                # Check if it's a config line
                if "=" in line:
                    key = line.split("=", 1)[0].strip()
                    if key not in existing_keys:
                        new_config_lines.append(line + "\n")

        # Write to file
        with open(self.env_file, "w", encoding="utf-8") as f:
            f.writelines(new_config_lines)

    def save_config_to_env(self, config: Config) -> None:
        """Save complete Config object to .env file.

        Args:
            config: Configuration object to save.
        """
        config_dict = self.converter.to_dict(config)
        self.save_to_env(config_dict)


def load_merged_config() -> dict[str, str]:
    """Load merged configuration from .env file.

    Priority: Environment variables > .env file

    Returns:
        Merged configuration dictionary.
    """
    from dotenv import load_dotenv

    project_root = get_project_root()
    env_file = project_root / ".env"

    env_file_config: dict[str, str] = {}

    if env_file.exists():
        # Temporarily load .env file, then read
        load_dotenv(env_file, override=False)

        # Read all configuration keys
        config_keys = [
            "STOCK_LIST",
            "LLM_MODEL",
            "LLM_API_KEY",
            "LLM_BASE_URL",
            "LLM_FALLBACK_MODEL",
            "LLM_FALLBACK_API_KEY",
            "LLM_FALLBACK_BASE_URL",
            "LLM_TEMPERATURE",
            "LLM_MAX_TOKENS",
            "LLM_REQUEST_DELAY",
            "LLM_MAX_RETRIES",
            "LLM_RETRY_DELAY",
            "BOCHA_API_KEYS",
            "TAVILY_API_KEYS",
            "BRAVE_API_KEYS",
            "SERPAPI_API_KEYS",
            "SEARXNG_BASE_URL",
            "SEARXNG_USERNAME",
            "SEARXNG_PASSWORD",
            "TELEGRAM_BOT_TOKEN",
            "TELEGRAM_CHAT_ID",
            "TELEGRAM_MESSAGE_THREAD_ID",
            "EMAIL_SENDER",
            "EMAIL_PASSWORD",
            "EMAIL_RECEIVERS",
            "DISCORD_WEBHOOK_URL",
            "CUSTOM_WEBHOOK_URLS",
            "CUSTOM_WEBHOOK_BEARER_TOKEN",
            "DATABASE_PATH",
            "SAVE_CONTEXT_SNAPSHOT",
            "LOG_DIR",
            "LOG_LEVEL",
            "MAX_WORKERS",
            "DEBUG",
            "HTTP_PROXY",
            "HTTPS_PROXY",
            "TUSHARE_TOKEN",
        ]

        import os

        for key in config_keys:
            value = os.environ.get(key)
            if value:
                env_file_config[key] = value

    return env_file_config
