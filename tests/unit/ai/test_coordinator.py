"""Tests for AgentCoordinator."""

import pytest

from ashare_analyzer.ai.agents.base import BaseAgent
from ashare_analyzer.ai.agents.coordinator import AgentAnalysisResult, AgentConsensus, AgentCoordinator
from ashare_analyzer.exceptions import ValidationError
from ashare_analyzer.models import AgentSignal, SignalType


class MockAgent(BaseAgent):
    """Mock agent for testing."""

    def __init__(self, name: str, signal: SignalType, confidence: int, should_fail: bool = False):
        super().__init__(name)
        self._signal = signal
        self._confidence = confidence
        self._should_fail = should_fail

    async def analyze(self, context: dict) -> AgentSignal:
        if self._should_fail:
            raise Exception("Agent failed")
        return AgentSignal(
            agent_name=self.name,
            signal=self._signal,
            confidence=self._confidence,
            reasoning=f"Test analysis from {self.name}",
        )


class TestAgentCoordinator:
    """Tests for AgentCoordinator."""

    def test_init_empty_agents(self):
        """Test coordinator initializes with empty agents dict."""
        coordinator = AgentCoordinator()
        assert coordinator.agents == {}
        assert coordinator.get_registered_agents() == []

    def test_register_agent(self):
        """Test registering an agent."""
        coordinator = AgentCoordinator()
        agent = MockAgent("TestAgent", SignalType.BUY, 75)
        coordinator.register_agent(agent)
        assert "TestAgent" in coordinator.agents
        assert coordinator.get_registered_agents() == ["TestAgent"]

    def test_register_duplicate_agent_raises_error(self):
        """Test registering duplicate agent raises ValidationError."""
        coordinator = AgentCoordinator()
        agent = MockAgent("TestAgent", SignalType.BUY, 75)
        coordinator.register_agent(agent)

        with pytest.raises(ValidationError, match="already registered"):
            coordinator.register_agent(agent)

    def test_unregister_agent(self):
        """Test unregistering an agent."""
        coordinator = AgentCoordinator()
        agent = MockAgent("TestAgent", SignalType.BUY, 75)
        coordinator.register_agent(agent)
        coordinator.unregister_agent("TestAgent")
        assert "TestAgent" not in coordinator.agents

    def test_unregister_nonexistent_agent_silently_succeeds(self):
        """Test unregistering non-existent agent doesn't raise."""
        coordinator = AgentCoordinator()
        coordinator.unregister_agent("NonExistent")  # Should not raise
        assert coordinator.get_registered_agents() == []

    @pytest.mark.asyncio
    async def test_analyze_with_single_agent(self):
        """Test analyze with single agent returns correct structure."""
        coordinator = AgentCoordinator()
        coordinator.register_agent(MockAgent("TestAgent", SignalType.BUY, 80))

        result = await coordinator.analyze({"code": "600519"})

        assert "agent_signals" in result
        assert "consensus" in result
        assert "consensus_level" in result
        assert "execution_summary" in result
        assert result["execution_summary"]["total_agents"] == 1
        assert result["execution_summary"]["successful_agents"] == 1

    @pytest.mark.asyncio
    async def test_analyze_with_multiple_agents(self):
        """Test analyze with multiple agents calculates consensus."""
        coordinator = AgentCoordinator()
        coordinator.register_agent(MockAgent("Agent1", SignalType.BUY, 80))
        coordinator.register_agent(MockAgent("Agent2", SignalType.BUY, 70))
        coordinator.register_agent(MockAgent("Agent3", SignalType.HOLD, 50))

        result = await coordinator.analyze({"code": "600519"})

        # 2 out of 3 agree on BUY
        assert result["consensus_level"] == 2 / 3

    @pytest.mark.asyncio
    async def test_analyze_handles_agent_failure(self):
        """Test analyze handles agent failure gracefully."""
        coordinator = AgentCoordinator()
        coordinator.register_agent(MockAgent("GoodAgent", SignalType.BUY, 80))
        coordinator.register_agent(MockAgent("BadAgent", SignalType.HOLD, 0, should_fail=True))

        result = await coordinator.analyze({"code": "600519"})

        assert result["execution_summary"]["successful_agents"] == 1
        assert result["agent_signals"]["BadAgent"]["success"] is False

    @pytest.mark.asyncio
    async def test_analyze_with_unavailable_agent(self):
        """Test analyze skips unavailable agents."""

        class UnavailableAgent(BaseAgent):
            def __init__(self):
                super().__init__("UnavailableAgent")

            def is_available(self) -> bool:
                return False

            async def analyze(self, context: dict) -> AgentSignal:
                return AgentSignal(
                    agent_name=self.name,
                    signal=SignalType.BUY,
                    confidence=80,
                    reasoning="Should not be called",
                )

        coordinator = AgentCoordinator()
        coordinator.register_agent(UnavailableAgent())
        coordinator.register_agent(MockAgent("AvailableAgent", SignalType.BUY, 80))

        result = await coordinator.analyze({"code": "600519"})

        assert result["agent_signals"]["UnavailableAgent"]["success"] is False


class TestAgentAnalysisResult:
    """Tests for AgentAnalysisResult dataclass."""

    def test_create_success_result(self):
        """Test creating successful result."""
        result = AgentAnalysisResult(
            agent_name="TestAgent",
            signal="buy",
            confidence=80,
            reasoning="Test reasoning",
        )
        assert result.success is True
        assert result.error is None

    def test_create_failure_result(self):
        """Test creating failure result."""
        result = AgentAnalysisResult(
            agent_name="TestAgent",
            signal="hold",
            confidence=0,
            reasoning="Failed",
            success=False,
            error="Connection error",
        )
        assert result.success is False
        assert result.error == "Connection error"


class TestAgentConsensus:
    """Tests for AgentConsensus dataclass."""

    def test_create_consensus(self):
        """Test creating consensus object."""
        consensus = AgentConsensus(
            agent_signals={"Agent1": {"signal": "buy"}},
            consensus_level=0.8,
            participating_agents=["Agent1"],
            risk_flags=["high_volatility"],
        )
        assert consensus.consensus_level == 0.8
        assert "high_volatility" in consensus.risk_flags
