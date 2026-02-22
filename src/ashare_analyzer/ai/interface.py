"""
AI Analyzer Interface

Defines the abstract interface for AI analysis service,
following the Dependency Inversion Principle.
"""

from abc import ABC, abstractmethod
from typing import Any

from ashare_analyzer.models import AnalysisResult


class IAIAnalyzer(ABC):
    """
    AI Analyzer Interface

    Responsible for calling AI models to analyze stocks,
    supporting multiple AI providers (Gemini, OpenAI, etc.).
    """

    @abstractmethod
    async def analyze(self, context: dict[str, Any]) -> AnalysisResult:
        """
        Analyze a single stock (async).

        Args:
            context: Stock analysis context data (technical, fundamental, etc.)

        Returns:
            AnalysisResult: Analysis result object
        """

    @abstractmethod
    def is_available(self) -> bool:
        """
        Check if analyzer is available.

        Returns:
            bool: Whether AI service is available (API Key configured correctly)
        """

    @abstractmethod
    async def batch_analyze(self, contexts: list[dict[str, Any]], delay_between: float = 2.0) -> list[AnalysisResult]:
        """
        Batch analyze multiple stocks (async).

        Args:
            contexts: List of context data
            delay_between: Delay between each analysis (seconds)

        Returns:
            list[AnalysisResult]: List of analysis results
        """
