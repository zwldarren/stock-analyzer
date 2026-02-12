"""
Search service module.

Provides search service, providers, and configurations.
"""

from stock_analyzer.infrastructure.search.base import (
    ApiKeyProviderConfig,
    ApiKeySearchProvider,
    BaseSearchProvider,
    ProviderConfig,
    SearxngProviderConfig,
)
from stock_analyzer.infrastructure.search.impl import (
    BochaSearchProvider,
    BraveSearchProvider,
    SearxngSearchProvider,
    SerpAPISearchProvider,
    TavilySearchProvider,
)
from stock_analyzer.infrastructure.search.registry import (
    ProviderRegistry,
    register_builtin_providers,
)
from stock_analyzer.infrastructure.search.service import SearchService

register_builtin_providers()

__all__ = [
    "SearchService",
    "ProviderConfig",
    "ApiKeyProviderConfig",
    "SearxngProviderConfig",
    "BaseSearchProvider",
    "ApiKeySearchProvider",
    "TavilySearchProvider",
    "SerpAPISearchProvider",
    "BraveSearchProvider",
    "BochaSearchProvider",
    "SearxngSearchProvider",
    "ProviderRegistry",
    "register_builtin_providers",
]
