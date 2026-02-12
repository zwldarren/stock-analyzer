"""
Search models - Entities for search functionality.

This module defines data structures for search results and responses.
"""

from dataclasses import dataclass
from typing import Any


@dataclass
class SearchResult:
    """
    Single search result item.

    Represents one search result with title, snippet, URL and metadata.

    Attributes:
        title: Result title
        snippet: Content snippet/summary
        url: Result URL
        source: Source website/domain
        published_date: Publication date (optional)
    """

    title: str
    snippet: str
    url: str
    source: str
    published_date: str | None = None

    def to_text(self) -> str:
        """Convert to formatted text."""
        date_str = f" ({self.published_date})" if self.published_date else ""
        return f"【{self.source}】{self.title}{date_str}\n{self.snippet}"


@dataclass
class SearchResponse:
    """
    Search response containing multiple results.

    Aggregates search results from a single query with metadata.

    Attributes:
        query: Search query string
        results: List of search results
        provider: Search provider name
        success: Whether search succeeded
        error_message: Error message if failed
        search_time: Search duration in seconds
    """

    query: str
    results: list[SearchResult]
    provider: str
    success: bool = True
    error_message: str | None = None
    search_time: float = 0.0

    def to_context(self, max_results: int = 5) -> str:
        """Convert search results to AI analysis context."""
        if not self.success or not self.results:
            return f"搜索 '{self.query}' 未找到相关结果。"

        lines = [f"【{self.query} 搜索结果】（来源：{self.provider}）"]
        for i, result in enumerate(self.results[:max_results], 1):
            lines.append(f"\n{i}. {result.to_text()}")

        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "query": self.query,
            "results": [
                {
                    "title": r.title,
                    "snippet": r.snippet,
                    "url": r.url,
                    "source": r.source,
                    "published_date": r.published_date,
                }
                for r in self.results
            ],
            "provider": self.provider,
            "success": self.success,
            "error_message": self.error_message,
            "search_time": self.search_time,
        }
