"""
Caching utilities for data layer.

Provides thread-safe TTL caching using cachetools with additional features.
"""

import fnmatch
import logging
import threading
import time
from collections.abc import Callable
from typing import Any, TypeVar

from cachetools import TTLCache as BaseTTLCache

logger = logging.getLogger(__name__)

T = TypeVar("T")


class TTLCache:
    """
    Thread-safe TTL (Time-To-Live) cache with pattern-based invalidation.

    Wraps cachetools.TTLCache with additional features:
    - Thread-safe operations
    - Custom expiry times per entry
    - Pattern-based cache invalidation
    """

    def __init__(self, default_ttl: int = 600, max_size: int = 1000) -> None:
        """
        Initialize TTL cache.

        Args:
            default_ttl: Default time-to-live in seconds (default: 600 = 10 min)
            max_size: Maximum number of items in cache
        """
        self._cache: BaseTTLCache = BaseTTLCache(maxsize=max_size, ttl=default_ttl)
        self._cache_expiry: dict[str, float] = {}
        self._lock = threading.RLock()
        self._default_ttl = default_ttl

    def get(self, key: str) -> Any | None:
        """
        Get value from cache if not expired.

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found/expired
        """
        with self._lock:
            if key not in self._cache:
                return None

            # Check custom expiry if set
            if key in self._cache_expiry and time.time() > self._cache_expiry[key]:
                # Expired, remove from cache
                self._remove_entry(key)
                return None

            return self._cache.get(key)

    def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        """
        Set value in cache with TTL.

        Args:
            key: Cache key
            value: Value to cache
            ttl: Time-to-live in seconds (uses default if None)
        """
        ttl = ttl if ttl is not None else self._default_ttl

        with self._lock:
            self._cache[key] = value
            # Store custom expiry time for this entry
            self._cache_expiry[key] = time.time() + ttl

    def delete(self, key: str) -> None:
        """
        Delete value from cache.

        Args:
            key: Cache key
        """
        with self._lock:
            self._remove_entry(key)

    def clear(self) -> None:
        """Clear all cached values."""
        with self._lock:
            self._cache.clear()
            self._cache_expiry.clear()

    def invalidate(self, pattern: str | None = None) -> None:
        """
        Invalidate cache entries matching pattern.

        Args:
            pattern: Glob pattern to match keys (None = clear all)
        """
        with self._lock:
            if pattern is None:
                self.clear()
                logger.info("Cache cleared")
            else:
                keys_to_remove = [k for k in self._cache if fnmatch.fnmatch(k, pattern)]
                for key in keys_to_remove:
                    self._remove_entry(key)
                logger.info(f"Cache invalidated: {pattern} ({len(keys_to_remove)} entries)")

    def is_valid(self, key: str) -> bool:
        """
        Check if cache entry exists and is not expired.

        Args:
            key: Cache key

        Returns:
            True if valid, False otherwise
        """
        with self._lock:
            if key not in self._cache:
                return False

            if key in self._cache_expiry and time.time() > self._cache_expiry[key]:
                self._remove_entry(key)
                return False

            return True

    def _remove_entry(self, key: str) -> None:
        """Remove a single cache entry and its expiry record."""
        self._cache.pop(key, None)
        self._cache_expiry.pop(key, None)

    def cached(self, key_func: Callable[..., str] | None = None, ttl: int | None = None) -> Callable:
        """
        Decorator to cache function results.

        Args:
            key_func: Function to generate cache key from args
            ttl: Time-to-live in seconds

        Returns:
            Decorator function

        Example:
            @cache.cached(key_func=lambda code: f"stock:{code}", ttl=300)
            def get_stock_data(code: str):
                ...
        """

        def decorator(func: Callable[..., T]) -> Callable[..., T]:
            func_name = getattr(func, "__name__", str(func))

            def wrapper(*args: Any, **kwargs: Any) -> T:
                cache_key = key_func(*args, **kwargs) if key_func else f"{func_name}:{args}:{kwargs}"
                cached_result = self.get(cache_key)
                if cached_result is not None:
                    return cached_result

                result = func(*args, **kwargs)
                if result is not None:
                    self.set(cache_key, result, ttl)
                return result

            return wrapper

        return decorator
