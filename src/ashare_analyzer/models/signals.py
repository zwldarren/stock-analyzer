"""
Signal models for agent analysis.

Contains SignalType enumeration and AgentSignal value object.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any


class SignalType(Enum):
    """Trading signal type enumeration."""

    BUY = auto()
    SELL = auto()
    HOLD = auto()

    @classmethod
    def from_string(cls, value: str) -> "SignalType":
        """Create signal type from string."""
        mapping = {
            "buy": cls.BUY,
            "sell": cls.SELL,
            "hold": cls.HOLD,
        }
        return mapping.get(value.lower(), cls.HOLD)

    def to_string(self) -> str:
        """Convert to string."""
        mapping = {
            self.BUY: "buy",
            self.SELL: "sell",
            self.HOLD: "hold",
        }
        return mapping[self]

    def is_bullish(self) -> bool:
        """Check if signal is bullish."""
        return self == self.BUY

    def is_bearish(self) -> bool:
        """Check if signal is bearish."""
        return self == self.SELL

    def is_neutral(self) -> bool:
        """Check if signal is neutral."""
        return self == self.HOLD


@dataclass(frozen=True)
class AgentSignal:
    """
    Agent analysis signal value object.

    Encapsulates a single agent's analysis result with signal type, confidence, and reasoning.
    """

    agent_name: str
    signal: SignalType
    confidence: int
    reasoning: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate signal values."""
        if not 0 <= self.confidence <= 100:
            from ashare_analyzer.exceptions import ValidationError

            raise ValidationError(f"置信度必须在 0-100 之间，当前值: {self.confidence}")

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "agent_name": self.agent_name,
            "signal": self.signal.to_string(),
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AgentSignal":
        """Create from dictionary."""
        return cls(
            agent_name=data["agent_name"],
            signal=SignalType.from_string(data.get("signal", "hold")),
            confidence=data.get("confidence", 0),
            reasoning=data.get("reasoning", ""),
            metadata=data.get("metadata", {}),
        )

    def get_signal_score(self) -> float:
        """Convert signal to numeric score (-100 to +100)."""
        signal_scores = {
            SignalType.BUY: 1.0,
            SignalType.HOLD: 0.0,
            SignalType.SELL: -1.0,
        }
        base_score = signal_scores.get(self.signal, 0.0)
        return base_score * self.confidence
