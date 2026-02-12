"""
Decision Agent Module

Final decision maker using LLM to generate clear trading actions.
"""

import logging
from typing import TYPE_CHECKING, Any

import json_repair

from stock_analyzer.ai.clients import get_llm_client
from stock_analyzer.ai.prompts import DECISION_SYSTEM_PROMPT, format_decision_prompt

from .base import AgentSignal, BaseAgent, SignalType

if TYPE_CHECKING:
    from stock_analyzer.ai.clients import LiteLLMClient

logger = logging.getLogger(__name__)


class DecisionAgent(BaseAgent):
    """
    Final decision agent using LLM to generate definitive trading actions.
    """

    def __init__(self):
        """Initialize the Decision Agent."""
        super().__init__("DecisionAgent")
        self._logger = logging.getLogger(__name__)
        self._llm_client: LiteLLMClient | None = None
        self._init_llm_client()

    def _init_llm_client(self) -> None:
        """Initialize LLM client for decision generation."""
        self._llm_client = get_llm_client()
        if self._llm_client:
            self._logger.info("DecisionAgent LLM client initialized successfully")
        else:
            self._logger.warning("No LLM API key configured, DecisionAgent will use rule-based fallback")

    def is_available(self) -> bool:
        """Check if decision agent is available (always true with fallback)."""
        return True

    def analyze(self, context: dict[str, Any]) -> AgentSignal:
        """
        Execute final decision analysis using LLM.

        Args:
            context: Analysis context containing:
                - code: Stock code
                - stock_name: Stock name
                - agent_signals: All agent signals from coordinator
                - consensus_data: Consensus data from coordinator
                - market_data: Additional market data

        Returns:
            AgentSignal with definitive trading action
        """
        stock_code = context.get("code", "")
        stock_name = context.get("stock_name", "")

        self._logger.info(f"[{stock_code}] DecisionAgent开始最终决策")

        try:
            # Extract inputs
            agent_signals = context.get("agent_signals", {})
            consensus_data = context.get("consensus_data", {})

            if not agent_signals:
                return AgentSignal(
                    agent_name=self.name,
                    signal=SignalType.HOLD,
                    confidence=0,
                    reasoning="无Agent信号输入，无法决策",
                    metadata={"error": "missing_agent_signals"},
                )

            # Use LLM for decision if available
            if self._llm_client and self._llm_client.is_available():
                decision = self._make_llm_decision(stock_code, stock_name, agent_signals, consensus_data)
            else:
                # Fallback to rule-based decision
                decision = self._make_rule_based_decision(agent_signals, consensus_data)

            self._logger.info(
                f"[{stock_code}] DecisionAgent决策完成: {decision['action']} "
                f"(置信度{decision['confidence']}%, 仓位{decision['position_ratio'] * 100:.0f}%)"
            )

            # Convert action to signal
            signal = self._action_to_signal(decision["action"])

            return AgentSignal(
                agent_name=self.name,
                signal=signal,
                confidence=decision["confidence"],
                reasoning=decision["reasoning"],
                metadata={
                    "action": decision["action"],
                    "position_ratio": decision["position_ratio"],
                    "key_factors": decision.get("key_factors", []),
                    "risk_assessment": decision.get("risk_assessment", {}),
                    "agent_signals_summary": self._summarize_signals(agent_signals),
                },
            )

        except Exception as e:
            self._logger.error(f"[{stock_code}] DecisionAgent决策失败: {e}")
            return AgentSignal(
                agent_name=self.name,
                signal=SignalType.HOLD,
                confidence=0,
                reasoning=f"决策失败: {str(e)}",
                metadata={"error": str(e)},
            )

    def _make_llm_decision(
        self,
        stock_code: str,
        stock_name: str,
        agent_signals: dict[str, Any],
        consensus_data: dict[str, Any],
    ) -> dict[str, Any]:
        """Use LLM to make final decision."""
        # Build prompt
        prompt = self._build_decision_prompt(stock_code, stock_name, agent_signals, consensus_data)

        self._logger.info(f"[{stock_code}] DecisionAgent调用LLM进行决策...")

        if not self._llm_client:
            self._logger.warning(f"[{stock_code}] LLM client not available, using rule-based fallback")
            return self._make_rule_based_decision(agent_signals, consensus_data)

        try:
            response = self._llm_client.generate(
                prompt=prompt,
                generation_config={"temperature": 0.2, "max_output_tokens": 2048},
                system_prompt=DECISION_SYSTEM_PROMPT,
            )

            # Parse JSON response
            result = json_repair.repair_json(response, return_objects=True)

            if result and isinstance(result, dict) and "action" in result:
                self._logger.debug(f"[{stock_code}] LLM决策成功: {result}")
                return result
            else:
                self._logger.warning(f"[{stock_code}] LLM返回格式无效，使用规则回退")
                return self._make_rule_based_decision(agent_signals, consensus_data)

        except Exception as e:
            self._logger.warning(f"[{stock_code}] LLM决策失败: {e}，使用规则回退")
            return self._make_rule_based_decision(agent_signals, consensus_data)

    def _make_rule_based_decision(
        self,
        agent_signals: dict[str, Any],
        consensus_data: dict[str, Any],
    ) -> dict[str, Any]:
        """Fallback rule-based decision when LLM unavailable."""
        weighted_score = consensus_data.get("weighted_score", 0)
        risk_flags = consensus_data.get("risk_flags", [])

        # Determine action based on score only
        if weighted_score >= 60:
            action = "BUY"
            reasoning = "强烈看多信号，建议买入"
        elif weighted_score >= 30:
            action = "BUY"
            reasoning = "看多信号，建议买入"
        elif weighted_score <= -60:
            action = "SELL"
            reasoning = "强烈看空信号，建议卖出"
        elif weighted_score <= -30:
            action = "SELL"
            reasoning = "看空信号，建议卖出"
        else:
            action = "HOLD"
            reasoning = "信号中性，建议观望"

        # Adjust confidence based on risk flags
        confidence = min(100, abs(int(weighted_score)))
        if risk_flags:
            confidence = max(30, confidence - len(risk_flags) * 10)
            reasoning += f"（注意：发现{len(risk_flags)}个风险点）"

        # Position ratio based on confidence
        if action == "BUY":
            position_ratio = min(0.5, confidence / 200)
        elif action == "SELL":
            position_ratio = 1.0  # Full exit
        else:
            position_ratio = 0.0

        return {
            "action": action,
            "confidence": confidence,
            "position_ratio": position_ratio,
            "reasoning": reasoning,
            "key_factors": [f"加权得分: {weighted_score:.1f}"],
            "risk_assessment": {
                "level": "high" if len(risk_flags) >= 2 else "medium" if risk_flags else "low",
                "concerns": risk_flags[:3] if risk_flags else [],
            },
        }

    def _build_decision_prompt(
        self,
        stock_code: str,
        stock_name: str,
        agent_signals: dict[str, Any],
        consensus_data: dict[str, Any],
    ) -> str:
        """Build decision prompt for LLM using template."""
        # Format agent signals for prompt
        signal_lines = []
        for agent_name, signal_data in agent_signals.items():
            signal = signal_data.get("signal", "unknown")
            confidence = signal_data.get("confidence", 0)
            reasoning = signal_data.get("reasoning", "")
            signal_lines.append(f"{agent_name}: {signal} (置信度{confidence}%) - {reasoning[:80]}...")

        agent_signals_str = "\n".join(signal_lines)

        # Get current price from context if available
        current_price = consensus_data.get("current_price", 0)

        return format_decision_prompt(
            stock_code=stock_code,
            stock_name=stock_name,
            current_price=current_price,
            agent_signals=agent_signals_str,
            weighted_score=consensus_data.get("weighted_score", 0),
            consensus_level=consensus_data.get("consensus_level", 0),
            risk_flags=consensus_data.get("risk_flags", []),
        )

    def _action_to_signal(self, action: str) -> SignalType:
        """Convert action to agent signal."""
        action_map = {
            "BUY": SignalType.BUY,
            "HOLD": SignalType.HOLD,
            "SELL": SignalType.SELL,
        }
        return action_map.get(action, SignalType.HOLD)

    def _summarize_signals(self, agent_signals: dict[str, Any]) -> dict[str, str]:
        """Create summary of agent signals for metadata."""
        return {
            name: f"{data.get('signal', 'unknown')}({data.get('confidence', 0)})"
            for name, data in agent_signals.items()
        }
