"""
Base Agent Module

Defines the base class and data structures for all analysis agents.
All agents must inherit from BaseAgent and implement the analyze method.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

from stock_analyzer.domain.exceptions import ValidationError

__all__ = ["BaseAgent", "AgentSignal", "SignalType"]


class SignalType(Enum):
    """交易信号类型枚举"""

    BUY = auto()
    SELL = auto()
    HOLD = auto()

    @classmethod
    def from_string(cls, value: str) -> "SignalType":
        """从字符串创建信号类型"""
        mapping = {
            "buy": cls.BUY,
            "sell": cls.SELL,
            "hold": cls.HOLD,
        }
        return mapping.get(value.lower(), cls.HOLD)

    def to_string(self) -> str:
        """转换为字符串"""
        mapping = {
            self.BUY: "buy",
            self.SELL: "sell",
            self.HOLD: "hold",
        }
        return mapping[self]

    def is_bullish(self) -> bool:
        """是否为看多信号"""
        return self == self.BUY

    def is_bearish(self) -> bool:
        """是否为看空信号"""
        return self == self.SELL

    def is_neutral(self) -> bool:
        """是否为中性信号"""
        return self == self.HOLD


@dataclass(frozen=True)
class AgentSignal:
    """
    Agent 分析信号值对象

    封装单个 Agent 的分析结果，包含信号类型、置信度和推理。
    """

    agent_name: str
    signal: SignalType
    confidence: int
    reasoning: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """验证信号值"""
        if not 0 <= self.confidence <= 100:
            raise ValidationError(f"置信度必须在 0-100 之间，当前值: {self.confidence}")

    def to_dict(self) -> dict[str, Any]:
        """转换为字典"""
        return {
            "agent_name": self.agent_name,
            "signal": self.signal.to_string(),
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AgentSignal":
        """从字典创建"""
        return cls(
            agent_name=data["agent_name"],
            signal=SignalType.from_string(data.get("signal", "neutral")),
            confidence=data.get("confidence", 0),
            reasoning=data.get("reasoning", ""),
            metadata=data.get("metadata", {}),
        )

    def get_signal_score(self) -> float:
        """将信号转换为数值分数 (-100 到 +100)"""
        signal_scores = {
            SignalType.BUY: 1.0,
            SignalType.HOLD: 0.0,
            SignalType.SELL: -1.0,
        }
        base_score = signal_scores.get(self.signal, 0.0)
        return base_score * self.confidence


class BaseAgent(ABC):
    """
    Base class for all analysis agents.

    All agents must inherit from this class and implement the analyze method.
    Agents are designed to be stateless and thread-safe.

    Example:
        class MyAgent(BaseAgent):
            def __init__(self):
                super().__init__("MyAgent")

            def analyze(self, context: dict[str, Any]) -> AgentSignal:
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
    def analyze(self, context: dict[str, Any]) -> AgentSignal:
        """
        Execute analysis and return a trading signal.

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
        pass

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
