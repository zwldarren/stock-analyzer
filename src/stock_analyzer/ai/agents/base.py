"""
Base Agent Module

Defines the base class for all analysis agents.
All agents must inherit from BaseAgent and implement the analyze method.
"""

from abc import ABC, abstractmethod
from typing import Any

from stock_analyzer.models import AgentSignal

__all__ = ["BaseAgent"]


class BaseAgent(ABC):
    """
    Base class for all analysis agents.

    All agents must inherit from this class and implement the analyze method.
    Agents are designed to be stateless and thread-safe.

    Example:
        class MyAgent(BaseAgent):
            def __init__(self):
                super().__init__("MyAgent")

            async def analyze(self, context: dict[str, Any]) -> AgentSignal:
                # Analysis logic here
                return AgentSignal(
                    agent_name=self.name,
                    signal=SignalType.BUY,
                    confidence=75,
                    reasoning="Strong technical setup"
                )
    """

    def __init__(self, name: str):
        """
        Initialize the agent.

        Args:
            name: Unique name for this agent
        """
        self.name = name

    @abstractmethod
    async def analyze(self, context: dict[str, Any]) -> AgentSignal:
        """
        Execute analysis and return a trading signal (async).

        Args:
            context: Complete analysis context containing:
                - code: Stock code
                - stock_name: Stock name
                - today: Current day's OHLCV data
                - raw_data: Historical price data
                - technical indicators (ma5, ma10, ma20, etc.)
                - chip: Chip distribution data
                - realtime: Real-time quote data
                - Any other data needed for analysis

        Returns:
            AgentSignal: Structured signal with confidence and reasoning

        Raises:
            Exception: If analysis fails (will be caught by coordinator)
        """

    def is_available(self) -> bool:
        """
        Check if the agent is available to run.

        Override this method if the agent requires external resources
        (API keys, data sources, etc.) that might be unavailable.

        Returns:
            True if the agent can execute, False otherwise
        """
        return True

    def get_signal_score(self, signal: AgentSignal) -> float:
        """
        Convert signal to a numeric score for aggregation.

        Args:
            signal: The agent's signal

        Returns:
            Numeric score between -100 and +100
        """
        return signal.get_signal_score()
