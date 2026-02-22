"""Tests for BaseAgent."""

import pytest

from ashare_analyzer.ai.agents.base import BaseAgent
from ashare_analyzer.models import AgentSignal, SignalType


class ConcreteAgent(BaseAgent):
    """Concrete implementation for testing."""

    async def analyze(self, context: dict) -> AgentSignal:
        return AgentSignal(
            agent_name=self.name,
            signal=SignalType.HOLD,
            confidence=50,
            reasoning="Test analysis",
        )


class TestBaseAgent:
    """Tests for BaseAgent base class."""

    def test_init_sets_name(self):
        """Test initialization sets agent name."""
        agent = ConcreteAgent("TestAgent")
        assert agent.name == "TestAgent"

    def test_is_available_returns_true_by_default(self):
        """Test default is_available returns True."""
        agent = ConcreteAgent("TestAgent")
        assert agent.is_available() is True

    def test_get_signal_score_buy(self):
        """Test signal score calculation for BUY."""
        agent = ConcreteAgent("TestAgent")
        signal = AgentSignal(
            agent_name="TestAgent",
            signal=SignalType.BUY,
            confidence=80,
            reasoning="Test",
        )
        score = agent.get_signal_score(signal)
        assert score == 80.0

    def test_get_signal_score_sell(self):
        """Test signal score calculation for SELL."""
        agent = ConcreteAgent("TestAgent")
        signal = AgentSignal(
            agent_name="TestAgent",
            signal=SignalType.SELL,
            confidence=60,
            reasoning="Test",
        )
        score = agent.get_signal_score(signal)
        assert score == -60.0

    def test_get_signal_score_hold(self):
        """Test signal score calculation for HOLD."""
        agent = ConcreteAgent("TestAgent")
        signal = AgentSignal(
            agent_name="TestAgent",
            signal=SignalType.HOLD,
            confidence=50,
            reasoning="Test",
        )
        score = agent.get_signal_score(signal)
        assert score == 0.0

    @pytest.mark.asyncio
    async def test_analyze_returns_signal(self):
        """Test analyze method returns AgentSignal."""
        agent = ConcreteAgent("TestAgent")
        result = await agent.analyze({})
        assert isinstance(result, AgentSignal)
        assert result.agent_name == "TestAgent"
