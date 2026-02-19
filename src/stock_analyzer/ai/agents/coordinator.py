"""
Agent Coordinator Module

Manages multi-agent execution and aggregates signals for DecisionAgent.
The coordinator is now simplified - it only collects agent signals without
weighted voting, since the final decision is made by DecisionAgent.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any

from stock_analyzer.constants import SIGNAL_BUY, SIGNAL_HOLD, SIGNAL_SELL, normalize_signal
from stock_analyzer.exceptions import ValidationError
from stock_analyzer.utils import get_display

from .base import BaseAgent

logger = logging.getLogger(__name__)


@dataclass
class AgentAnalysisResult:
    """Result from a single agent analysis."""

    agent_name: str
    signal: str
    confidence: int
    reasoning: str
    metadata: dict[str, Any] = field(default_factory=dict)
    success: bool = True
    error: str | None = None


@dataclass
class AgentConsensus:
    """Aggregated agent signals for DecisionAgent."""

    agent_signals: dict[str, dict]  # Raw signals from each agent
    consensus_level: float  # 0-1, how much agents agree
    participating_agents: list[str] = field(default_factory=list)
    risk_flags: list[str] = field(default_factory=list)


class AgentCoordinator:
    """
    Simplified coordinator for multi-agent analysis.

    Responsibilities:
    1. Manage agent registration and parallel execution
    2. Collect agent signals (no weighted aggregation)
    3. Calculate consensus level
    4. Extract risk flags for DecisionAgent

    The final decision is made by DecisionAgent, not by weighted voting.

    Example:
        coordinator = AgentCoordinator()
        coordinator.register_agent(TechnicalAgent())
        coordinator.register_agent(FundamentalAgent())

        results = coordinator.analyze(context)
        # Returns raw signals for DecisionAgent to process
    """

    def __init__(self):
        """Initialize the coordinator."""
        self.agents: dict[str, BaseAgent] = {}
        self._logger = logging.getLogger(__name__)

    def register_agent(self, agent: BaseAgent) -> None:
        """Register an agent with the coordinator."""
        if agent.name in self.agents:
            raise ValidationError(f"Agent '{agent.name}' already registered")

        self.agents[agent.name] = agent
        self._logger.debug(f"注册Agent: {agent.name}")

    def unregister_agent(self, agent_name: str) -> None:
        """Unregister an agent."""
        if agent_name in self.agents:
            del self.agents[agent_name]
            self._logger.info(f"注销Agent: {agent_name}")

    async def analyze(self, context: dict[str, Any]) -> dict[str, Any]:
        """
        Execute multi-agent analysis and return aggregated results (async).

        Returns:
            Dictionary containing:
                - agent_signals: Dict of individual agent results
                - consensus: AgentConsensus object (simplified, no weights)
                - consensus_level: Float 0-1
                - execution_summary: Summary of execution
        """
        stock_code = context.get("code", "Unknown")
        self._logger.debug(f"[{stock_code}] AgentCoordinator开始多Agent分析")

        # 1. Execute all available agents
        agent_results = await self._execute_agents(context)

        # 2. Build consensus (simplified - no weighted scoring)
        consensus = self._build_consensus(agent_results)

        self._logger.debug(
            f"[{stock_code}] AgentCoordinator分析完成: "
            f"{len(consensus.participating_agents)}个Agent参与, "
            f"共识度{consensus.consensus_level:.2f}"
        )

        return {
            "agent_signals": {
                name: {
                    "signal": r.signal,
                    "confidence": r.confidence,
                    "reasoning": r.reasoning,
                    "metadata": r.metadata,
                    "success": r.success,
                }
                for name, r in agent_results.items()
            },
            "consensus": consensus,
            "consensus_level": consensus.consensus_level,
            "execution_summary": {
                "total_agents": len(self.agents),
                "executed_agents": len(agent_results),
                "successful_agents": sum(1 for r in agent_results.values() if r.success),
            },
        }

    async def _execute_agents(self, context: dict[str, Any]) -> dict[str, AgentAnalysisResult]:
        """Execute all available agents in parallel and collect results (async)."""
        results = {}
        stock_code = context.get("code", "Unknown")

        # Prepare agents to execute
        agents_to_execute = []
        for name, agent in self.agents.items():
            if not agent.is_available():
                self._logger.debug(f"[{stock_code}] Agent {name} 不可用，跳过")
                results[name] = AgentAnalysisResult(
                    agent_name=name,
                    signal=SIGNAL_HOLD,
                    confidence=0,
                    reasoning="Agent不可用",
                    success=False,
                    error="Agent not available",
                )
            else:
                agents_to_execute.append((name, agent))

        if not agents_to_execute:
            return results

        # Execute agents in parallel using asyncio
        display = get_display()

        async def execute_single_agent(name_agent_pair: tuple[str, BaseAgent]) -> tuple[str, AgentAnalysisResult]:
            name, agent = name_agent_pair
            display.start_agent(name)
            try:
                self._logger.debug(f"[{stock_code}] 执行Agent: {name}")
                signal = await agent.analyze(context)

                # 标准化信号类型
                normalized_signal = normalize_signal(signal.signal.to_string())

                result = AgentAnalysisResult(
                    agent_name=name,
                    signal=normalized_signal,
                    confidence=signal.confidence,
                    reasoning=signal.reasoning,
                    metadata=signal.metadata,
                    success=True,
                )

                display.complete_agent(name, normalized_signal, signal.confidence)
                self._logger.debug(
                    f"[{stock_code}] Agent {name} 完成: {normalized_signal} (置信度{signal.confidence}%)"
                )
                return name, result

            except Exception as e:
                self._logger.error(f"[{stock_code}] Agent {name} 执行失败: {e}")
                display.complete_agent(name, "hold", 0, error=str(e))
                result = AgentAnalysisResult(
                    agent_name=name,
                    signal=SIGNAL_HOLD,
                    confidence=0,
                    reasoning=f"执行失败: {str(e)}",
                    success=False,
                    error=str(e),
                )
                return name, result

        # Execute agents in parallel using asyncio.gather
        tasks = [execute_single_agent((name, agent)) for name, agent in agents_to_execute]
        completed = await asyncio.gather(*tasks)

        for name, result in completed:
            results[name] = result

        return results

    def _build_consensus(self, results: dict[str, AgentAnalysisResult]) -> AgentConsensus:
        """Build consensus from agent results (simplified, no weights)."""
        successful_results = {name: r for name, r in results.items() if r.success and r.confidence > 0}

        if not successful_results:
            return AgentConsensus(
                agent_signals={},
                consensus_level=0.0,
                participating_agents=[],
                risk_flags=[],
            )

        # Build agent signals dict
        agent_signals = {
            name: {
                "signal": r.signal,
                "confidence": r.confidence,
                "reasoning": r.reasoning,
                "metadata": r.metadata,
            }
            for name, r in successful_results.items()
        }

        # Calculate consensus level
        consensus_level = self._calculate_consensus_level(list(successful_results.values()))

        # Extract risk flags from RiskAgent and NewsSentimentAgent
        risk_flags = []
        risk_result = successful_results.get("RiskAgent")
        if risk_result:
            risk_flags.extend(risk_result.metadata.get("risk_factors", []))

        news_result = successful_results.get("NewsSentimentAgent")
        if news_result:
            risk_flags.extend(news_result.metadata.get("risk_factors", []))

        return AgentConsensus(
            agent_signals=agent_signals,
            consensus_level=consensus_level,
            participating_agents=list(successful_results.keys()),
            risk_flags=risk_flags,
        )

    def _calculate_consensus_level(self, results: list[AgentAnalysisResult]) -> float:
        """Calculate consensus level from results."""
        if not results:
            return 0.0

        # 标准化信号并计数
        buy_count = 0
        sell_count = 0
        hold_count = 0

        for r in results:
            normalized = normalize_signal(r.signal)
            if normalized == SIGNAL_BUY:
                buy_count += 1
            elif normalized == SIGNAL_SELL:
                sell_count += 1
            else:  # hold
                hold_count += 1

        total = len(results)
        max_consensus = max(buy_count, sell_count, hold_count)

        return max_consensus / total if total > 0 else 0.0

    def get_registered_agents(self) -> list[str]:
        """Get list of registered agent names."""
        return list(self.agents.keys())
