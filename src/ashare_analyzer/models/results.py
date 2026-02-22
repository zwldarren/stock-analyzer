"""
Result models for analysis and search.

Contains AnalysisResult, SearchResult, and SearchResponse.
"""

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class AnalysisResult:
    """
    AI Analysis Result - Decision-focused format.

    Encapsulates AI analysis results with clear trading decisions and reasoning.
    Designed for AI-driven decision making rather than human interpretation.
    """

    code: str
    name: str

    sentiment_score: int
    trend_prediction: str
    operation_advice: str
    decision_type: str = "hold"
    confidence_level: str = "中"

    final_action: str = ""
    position_ratio: float = 0.0
    decision_reasoning: str = ""

    dashboard: dict[str, Any] | None = None

    analysis_summary: str = ""
    risk_warning: str = ""

    market_snapshot: dict[str, Any] | None = None
    search_performed: bool = False
    success: bool = True
    error_message: str | None = None

    data_sources: str = ""
    raw_response: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert AnalysisResult to dictionary."""
        return {
            "code": self.code,
            "name": self.name,
            "sentiment_score": self.sentiment_score,
            "trend_prediction": self.trend_prediction,
            "operation_advice": self.operation_advice,
            "decision_type": self.decision_type,
            "confidence_level": self.confidence_level,
            "final_action": self.final_action,
            "position_ratio": self.position_ratio,
            "decision_reasoning": self.decision_reasoning,
            "dashboard": self.dashboard,
            "analysis_summary": self.analysis_summary,
            "risk_warning": self.risk_warning,
            "market_snapshot": self.market_snapshot,
            "search_performed": self.search_performed,
            "success": self.success,
            "error_message": self.error_message,
            "data_sources": self.data_sources,
            "raw_response": self.raw_response,
        }


@dataclass
class SearchResult:
    """
    Single search result item.

    Represents one search result with title, snippet, URL and metadata.
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
