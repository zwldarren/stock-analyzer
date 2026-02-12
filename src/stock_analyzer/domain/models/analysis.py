"""
Analysis models - Core analysis result entities and value objects.

This module defines all data structures related to stock analysis results.
"""

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class AnalysisResult:
    """
    AI Analysis Result - Decision-focused format.

    Encapsulates AI analysis results with clear trading decisions and reasoning.
    Designed for AI-driven decision making rather than human interpretation.

    Attributes:
        code: Stock code (e.g., "600519")
        name: Stock name (e.g., "贵州茅台")
        sentiment_score: Confidence score from 0-100
        trend_prediction: Trend prediction text
        operation_advice: Operation advice text
        decision_type: Decision type for statistics (buy/hold/sell)
        confidence_level: Confidence level (高/中/低)
        final_action: Final action recommendation (BUY/HOLD/SELL)
        position_ratio: Suggested position ratio 0.0-1.0
        decision_reasoning: Detailed reasoning for the decision
        dashboard: Complete decision dashboard data
        analysis_summary: Comprehensive analysis summary
        risk_warning: Risk warning text
        market_snapshot: Daily market snapshot for display
        search_performed: Whether web search was performed
        success: Whether analysis succeeded
        error_message: Error message if failed
        data_sources: Data source list for debugging
        raw_response: Raw AI response for debugging
    """

    code: str
    name: str

    # Core Decision Metrics
    sentiment_score: int
    trend_prediction: str
    operation_advice: str
    decision_type: str = "hold"
    confidence_level: str = "中"

    # Decision Details
    final_action: str = ""
    position_ratio: float = 0.0
    decision_reasoning: str = ""

    # Decision Dashboard
    dashboard: dict[str, Any] | None = None

    # Comprehensive Analysis
    analysis_summary: str = ""
    risk_warning: str = ""

    # Metadata
    market_snapshot: dict[str, Any] | None = None
    search_performed: bool = False
    success: bool = True
    error_message: str | None = None

    # Data Source Tracking (for debugging)
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
