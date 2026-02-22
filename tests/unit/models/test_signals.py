"""Tests for signal models."""

import pytest

from ashare_analyzer.exceptions import ValidationError
from ashare_analyzer.models import AgentSignal, SignalType


class TestSignalType:
    """Tests for SignalType enum."""

    def test_from_string_buy(self):
        """Test from_string returns BUY for 'buy'."""
        signal = SignalType.from_string("buy")
        assert signal == SignalType.BUY

    def test_from_string_sell_uppercase(self):
        """Test from_string is case-insensitive."""
        signal = SignalType.from_string("SELL")
        assert signal == SignalType.SELL

    def test_from_string_invalid_returns_hold(self):
        """Test from_string returns HOLD for invalid input."""
        signal = SignalType.from_string("invalid")
        assert signal == SignalType.HOLD

    def test_to_string(self):
        """Test to_string conversion."""
        assert SignalType.BUY.to_string() == "buy"
        assert SignalType.SELL.to_string() == "sell"
        assert SignalType.HOLD.to_string() == "hold"

    def test_is_bullish(self):
        """Test is_bullish method."""
        assert SignalType.BUY.is_bullish() is True
        assert SignalType.SELL.is_bullish() is False
        assert SignalType.HOLD.is_bullish() is False

    def test_is_bearish(self):
        """Test is_bearish method."""
        assert SignalType.SELL.is_bearish() is True
        assert SignalType.BUY.is_bearish() is False
        assert SignalType.HOLD.is_bearish() is False

    def test_is_neutral(self):
        """Test is_neutral method."""
        assert SignalType.HOLD.is_neutral() is True
        assert SignalType.BUY.is_neutral() is False
        assert SignalType.SELL.is_neutral() is False


class TestAgentSignal:
    """Tests for AgentSignal value object."""

    def test_create_signal(self):
        """Test creating a valid signal."""
        signal = AgentSignal(
            agent_name="TestAgent",
            signal=SignalType.BUY,
            confidence=75,
            reasoning="Test reasoning",
        )
        assert signal.agent_name == "TestAgent"
        assert signal.signal == SignalType.BUY
        assert signal.confidence == 75
        assert signal.reasoning == "Test reasoning"

    def test_confidence_validation_below_zero(self):
        """Test confidence must be >= 0."""
        with pytest.raises(ValidationError, match="置信度必须在 0-100"):
            AgentSignal(
                agent_name="TestAgent",
                signal=SignalType.BUY,
                confidence=-10,
                reasoning="Test",
            )

    def test_confidence_validation_above_100(self):
        """Test confidence must be <= 100."""
        with pytest.raises(ValidationError, match="置信度必须在 0-100"):
            AgentSignal(
                agent_name="TestAgent",
                signal=SignalType.BUY,
                confidence=150,
                reasoning="Test",
            )

    def test_to_dict(self, sample_agent_signal):
        """Test to_dict conversion."""
        result = sample_agent_signal.to_dict()
        assert result["agent_name"] == "TestAgent"
        assert result["signal"] == "buy"
        assert result["confidence"] == 75
        assert result["reasoning"] == "Test reasoning"

    def test_from_dict(self):
        """Test from_dict creation."""
        signal = AgentSignal.from_dict(
            {
                "agent_name": "TestAgent",
                "signal": "sell",
                "confidence": 80,
                "reasoning": "Test",
                "metadata": {"key": "value"},
            }
        )
        assert signal.signal == SignalType.SELL
        assert signal.confidence == 80
        assert signal.metadata == {"key": "value"}

    def test_get_signal_score_buy(self):
        """Test signal score for BUY."""
        signal = AgentSignal(
            agent_name="Test",
            signal=SignalType.BUY,
            confidence=80,
            reasoning="Test",
        )
        assert signal.get_signal_score() == 80.0

    def test_get_signal_score_sell(self):
        """Test signal score for SELL."""
        signal = AgentSignal(
            agent_name="Test",
            signal=SignalType.SELL,
            confidence=60,
            reasoning="Test",
        )
        assert signal.get_signal_score() == -60.0

    def test_get_signal_score_hold(self):
        """Test signal score for HOLD."""
        signal = AgentSignal(
            agent_name="Test",
            signal=SignalType.HOLD,
            confidence=50,
            reasoning="Test",
        )
        assert signal.get_signal_score() == 0.0
