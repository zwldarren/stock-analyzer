"""
Provider registry.

Manages registration and creation of all search engine providers.
"""

import logging
from collections.abc import Callable
from typing import Any, ClassVar

from stock_analyzer.search.base import (
    ApiKeyProviderConfig,
    BaseSearchProvider,
    ProviderConfig,
    SearxngProviderConfig,
)

logger = logging.getLogger(__name__)

# Provider factory function type
ProviderFactory = Callable[[Any], BaseSearchProvider]


class ProviderRegistry:
    """Registry for managing search engine provider registration and creation."""

    _providers: ClassVar[dict[str, tuple[type[ProviderConfig], ProviderFactory]]] = {}

    @classmethod
    def register(
        cls,
        name: str,
        config_class: type[ProviderConfig],
        factory: ProviderFactory,
    ) -> None:
        """
        Register a new provider.

        Args:
            name: Provider name (unique identifier).
            config_class: Configuration class type.
            factory: Factory function to create provider instances.
        """
        cls._providers[name] = (config_class, factory)
        logger.debug(f"已注册搜索引擎provider: {name}")

    @classmethod
    def get_config_class(cls, name: str) -> type[ProviderConfig] | None:
        """Get the configuration class for the specified provider."""
        entry = cls._providers.get(name)
        return entry[0] if entry else None

    @classmethod
    def create_provider(cls, name: str, config: ProviderConfig) -> BaseSearchProvider | None:
        """
        Create a provider instance.

        Args:
            name: Provider name.
            config: Provider configuration.

        Returns:
            Provider instance or None if not registered.
        """
        entry = cls._providers.get(name)
        if not entry:
            logger.warning(f"未找到provider: {name}")
            return None

        _, factory = entry
        try:
            return factory(config)
        except Exception as e:
            logger.error(f"创建provider {name} 失败: {e}")
            return None

    @classmethod
    def list_providers(cls) -> list[str]:
        """Get a list of all registered provider names."""
        return list(cls._providers.keys())

    @classmethod
    def is_registered(cls, name: str) -> bool:
        """Check if a provider is registered."""
        return name in cls._providers


def register_builtin_providers() -> None:
    """Register all built-in search engine providers."""
    # Delay import to avoid circular dependencies
    from stock_analyzer.search.impl import (
        BochaSearchProvider,
        BraveSearchProvider,
        SearxngSearchProvider,
        SerpAPISearchProvider,
        TavilySearchProvider,
    )

    # Priority 1: SearXNG (self-hosted, completely free)
    ProviderRegistry.register(
        "searxng",
        SearxngProviderConfig,
        lambda config: SearxngSearchProvider(config),
    )

    # Priority 2: Tavily (free tier available)
    ProviderRegistry.register(
        "tavily",
        ApiKeyProviderConfig,
        lambda config: TavilySearchProvider(config),
    )

    # Priority 3: Brave Search (free tier available)
    ProviderRegistry.register(
        "brave",
        ApiKeyProviderConfig,
        lambda config: BraveSearchProvider(config),
    )

    # Priority 4: SerpAPI (free tier available)
    ProviderRegistry.register(
        "serpapi",
        ApiKeyProviderConfig,
        lambda config: SerpAPISearchProvider(config),
    )

    # Priority 5: Bocha (paid only, Chinese optimized)
    ProviderRegistry.register(
        "bocha",
        ApiKeyProviderConfig,
        lambda config: BochaSearchProvider(config),
    )

    logger.info(f"已注册 {len(ProviderRegistry.list_providers())} 个搜索引擎provider")
