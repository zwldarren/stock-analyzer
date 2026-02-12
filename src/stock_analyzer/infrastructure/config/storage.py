"""Infrastructure configuration storage implementation."""

import logging
from datetime import datetime
from typing import Any

from sqlalchemy import Column, DateTime, String, Text, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

from stock_analyzer.config.config import Config, get_project_root
from stock_analyzer.config.storage import ConfigConverter
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

        # Initialize database connection
        self.db_session = None
        self._engine = None

        if db_url:
            self._init_db(db_url)
        else:
            # Use default database path
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
        # Delegate to file-based storage logic
        from stock_analyzer.config.storage import ConfigStorage

        file_storage = ConfigStorage()
        file_storage.save_to_env(config_dict)

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
                # Handle list type
                if isinstance(value, list):
                    value = ",".join(str(v) for v in value)
                # Handle boolean
                elif isinstance(value, bool):
                    value = "true" if value else "false"
                else:
                    value = str(value)

                # Query or create config item
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
