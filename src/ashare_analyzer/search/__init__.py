"""
Search service module.

Provides search service, providers, and configurations.
"""

from ashare_analyzer.search.base import (
    ApiKeyProviderConfig,
    ApiKeySearchProvider,
    BaseSearchProvider,
    ProviderConfig,
    ProviderRegistry,
    SearxngProviderConfig,
    register_builtin_providers,
)
from ashare_analyzer.search.impl import (
    BochaSearchProvider,
    BraveSearchProvider,
    SearxngSearchProvider,
    SerpAPISearchProvider,
    TavilySearchProvider,
)
from ashare_analyzer.search.service import SearchService

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
