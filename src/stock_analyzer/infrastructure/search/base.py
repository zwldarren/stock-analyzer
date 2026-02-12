"""
Base classes and configurations for search engine providers.

Defines the fundamental interfaces and configuration classes for all search engine providers.
"""

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from itertools import cycle
from typing import ClassVar

from stock_analyzer.domain.models import SearchResponse

logger = logging.getLogger(__name__)


@dataclass
class ProviderConfig:
    """Base configuration class for providers."""

    enabled: bool = False
    priority: int = 100  # Lower value = higher priority


@dataclass
class ApiKeyProviderConfig(ProviderConfig):
    """Configuration for providers requiring API keys."""

    api_keys: list[str] | None = None

    def __post_init__(self):
        if self.api_keys:
            self.enabled = True


@dataclass
class SearxngProviderConfig(ProviderConfig):
    """Configuration for SearXNG provider."""

    base_url: str = ""
    username: str | None = None
    password: str | None = None

    def __post_init__(self):
        if self.base_url:
            self.enabled = True

    @property
    def use_basic_auth(self) -> bool:
        """Check if basic auth should be used."""
        return bool(self.username and self.password)


class BaseSearchProvider(ABC):
    """Base class for search engine providers."""

    # Class-level configuration type
    config_class: ClassVar[type[ProviderConfig]] = ProviderConfig

    def __init__(self, config: ProviderConfig, name: str):
        """
        Initialize the search engine provider.

        Args:
            config: Provider configuration object.
            name: Name of the search engine.
        """
        self._config = config
        self._name = name

    @property
    def name(self) -> str:
        """Get the provider name."""
        return self._name

    @property
    def is_available(self) -> bool:
        """Check if the provider is available."""
        return self._config.enabled

    @property
    def priority(self) -> int:
        """Get the provider priority."""
        return self._config.priority

    @abstractmethod
    def _do_search(self, query: str, max_results: int, days: int = 7) -> SearchResponse:
        """Execute search (implemented by subclasses)."""

    def search(self, query: str, max_results: int = 5, days: int = 7) -> SearchResponse:
        """
        Execute search using the template method pattern.

        Args:
            query: Search query string.
            max_results: Maximum number of results to return.
            days: Time range for search in days (default: 7).

        Returns:
            SearchResponse object containing search results.
        """
        if not self.is_available:
            return SearchResponse(
                query=query,
                results=[],
                provider=self._name,
                success=False,
                error_message=f"{self._name} not configured or unavailable",
            )

        start_time = time.time()
        try:
            response = self._do_search(query, max_results, days=days)
            response.search_time = time.time() - start_time

            if response.success:
                logger.info(
                    f"[{self._name}] 搜索 '{query}' 成功，"
                    f"返回 {len(response.results)} 条结果，"
                    f"耗时 {response.search_time:.2f}s"
                )
            else:
                logger.warning(f"[{self._name}] 搜索 '{query}' 失败: {response.error_message}")

            return response

        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"[{self._name}] 搜索 '{query}' 失败: {e}")
            return SearchResponse(
                query=query,
                results=[],
                provider=self._name,
                success=False,
                error_message=str(e),
                search_time=elapsed,
            )


class ApiKeySearchProvider(BaseSearchProvider):
    """Base class for search engine providers requiring API keys."""

    config_class = ApiKeyProviderConfig

    def __init__(self, config: ApiKeyProviderConfig, name: str):
        super().__init__(config, name)
        self._api_keys = config.api_keys or []
        self._key_cycle = cycle(self._api_keys) if self._api_keys else None
        self._key_usage: dict[str, int] = {key: 0 for key in self._api_keys}
        self._key_errors: dict[str, int] = {key: 0 for key in self._api_keys}

    @property
    def is_available(self) -> bool:
        """Check if any API key is available."""
        return bool(self._api_keys) and self._config.enabled

    def _get_next_key(self) -> str | None:
        """
        Get the next available API key using round-robin with error tracking.

        Strategy: Round-robin + skip keys with too many errors (>3).
        """
        if not self._key_cycle:
            return None

        # Try all keys at most once
        for _ in range(len(self._api_keys)):
            key = next(self._key_cycle)
            # Skip keys with too many errors (>3)
            if self._key_errors.get(key, 0) < 3:
                return key

        # All keys have errors, reset error counts and return first
        logger.warning(f"[{self._name}] 所有 API Key 都有错误记录，重置错误计数")
        self._key_errors = {key: 0 for key in self._api_keys}
        return self._api_keys[0] if self._api_keys else None

    def _record_success(self, key: str) -> None:
        """Record successful API key usage."""
        self._key_usage[key] = self._key_usage.get(key, 0) + 1
        # Decrease error count after success
        if key in self._key_errors and self._key_errors[key] > 0:
            self._key_errors[key] -= 1

    def _record_error(self, key: str) -> None:
        """Record API key error."""
        self._key_errors[key] = self._key_errors.get(key, 0) + 1
        logger.warning(f"[{self._name}] API Key {key[:8]}... 错误计数: {self._key_errors[key]}")

    def search(self, query: str, max_results: int = 5, days: int = 7) -> SearchResponse:
        """Execute search with API key error tracking."""
        api_key = self._get_next_key()
        if not api_key:
            return SearchResponse(
                query=query,
                results=[],
                provider=self._name,
                success=False,
                error_message=f"{self._name} API key not configured",
            )

        start_time = time.time()
        try:
            response = self._do_search_with_key(query, api_key, max_results, days=days)
            response.search_time = time.time() - start_time

            if response.success:
                self._record_success(api_key)
                logger.info(
                    f"[{self._name}] 搜索 '{query}' 成功，"
                    f"返回 {len(response.results)} 条结果，"
                    f"耗时 {response.search_time:.2f}s"
                )
            else:
                self._record_error(api_key)

            return response

        except Exception as e:
            self._record_error(api_key)
            elapsed = time.time() - start_time
            logger.error(f"[{self._name}] 搜索 '{query}' 失败: {e}")
            return SearchResponse(
                query=query,
                results=[],
                provider=self._name,
                success=False,
                error_message=str(e),
                search_time=elapsed,
            )

    @abstractmethod
    def _do_search_with_key(self, query: str, api_key: str, max_results: int, days: int = 7) -> SearchResponse:
        """Execute search with the specified API key (implemented by subclasses)."""

    def _do_search(self, query: str, max_results: int, days: int = 7) -> SearchResponse:
        """Default implementation (overridden by search method)."""
        api_key = self._get_next_key()
        if not api_key:
            return SearchResponse(
                query=query,
                results=[],
                provider=self._name,
                success=False,
                error_message=f"{self._name} API key not configured",
            )
        return self._do_search_with_key(query, api_key, max_results, days)
