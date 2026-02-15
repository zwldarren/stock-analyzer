"""
Portfolio Manager Agent Module

Final decision maker that synthesizes all analyst signals and respects risk constraints.

This agent is responsible for:
- Aggregating signals from all analysis agents
- Applying risk limits from RiskManagerAgent
- Making final buy/sell/hold decisions with position sizing
- Generating clear, actionable recommendations

Replaces the previous DecisionAgent with enhanced risk management integration.
"""

import logging
from typing import TYPE_CHECKING, Any

import json_repair

from stock_analyzer.ai.clients import get_llm_client

from .base import AgentSignal, BaseAgent, SignalType

if TYPE_CHECKING:
    from stock_analyzer.ai.clients import LiteLLMClient

logger = logging.getLogger(__name__)

# System prompt for portfolio management
PORTFOLIO_MANAGER_SYSTEM_PROMPT = """You are a professional portfolio manager making final trading decisions.

=== Checklist for Decision ===
- [ ] Majority of analysts agree on direction
- [ ] Confidence levels are adequate (>50%)
- [ ] No critical risk flags
- [ ] Position size within risk limits

=== Signal Rules ===
BUY conditions:
- weighted_score >= 30
- consensus_level >= 0.6
- No critical risk flags

SELL conditions:
- weighted_score <= -30
- OR critical risk flags present
- OR majority of analysts bearish

HOLD conditions:
- -30 < weighted_score < 30
- OR low consensus (consensus_level < 0.5)
- OR mixed signals with high uncertainty

=== Position Sizing ===
Base position determined by confidence:
- 90-100%: position_ratio = 0.8
- 70-89%: position_ratio = 0.5
- 50-69%: position_ratio = 0.3
- <50%: position_ratio = 0

IMPORTANT: Always respect max_position_limit from risk manager!

=== Confidence Levels ===
- 90-100%: Multiple strong signals align, clear consensus
- 70-89%: Majority agreement, moderate confidence
- 50-69%: Mixed signals, directional but uncertain
- 30-49%: Conflicting signals, low confidence
- 10-29%: No clear direction, recommend hold

=== Output Format ===
Return JSON only:
{
    "action": "buy|sell|hold",
    "confidence": 75,
    "reasoning": "Concise decision rationale (max 150 chars)",
    "position_ratio": 0.5,
    "risk_level": "low|medium|high",
    "key_factors": ["factor1", "factor2"]
}"""


class PortfolioManagerAgent(BaseAgent):
    """
    Portfolio Manager Agent for final trading decisions.

    This agent:
    1. Receives signals from all analysis agents
    2. Gets risk constraints from RiskManagerAgent
    3. Synthesizes everything into a final decision
    4. Ensures position sizing respects risk limits

    Example:
        agent = PortfolioManagerAgent()
        signal = agent.analyze({
            "code": "600519",
            "agent_signals": {...},
            "risk_manager_signal": {...},
            "consensus_data": {...}
        })
    """

    def __init__(self):
        """Initialize the Portfolio Manager Agent."""
        super().__init__("PortfolioManagerAgent")
        self._logger = logging.getLogger(__name__)
        self._llm_client: LiteLLMClient | None = None
        self._init_llm_client()

    def _init_llm_client(self) -> None:
        """Initialize LLM client for decision generation."""
        self._llm_client = get_llm_client()
        if self._llm_client:
            self._logger.debug("PortfolioManagerAgent LLM client initialized successfully")
        else:
            self._logger.warning("No LLM API key configured, will use rule-based fallback")

    def is_available(self) -> bool:
        """Always available with fallback."""
        return True

    def analyze(self, context: dict[str, Any]) -> AgentSignal:
        """
        Make final trading decision based on all inputs.

        Args:
            context: Analysis context containing:
                - code: Stock code
                - stock_name: Stock name
                - agent_signals: Signals from all analysis agents
                - risk_manager_signal: Risk constraints from RiskManagerAgent
                - consensus_data: Aggregated consensus metrics

        Returns:
            AgentSignal with final trading decision and position sizing
        """
        stock_code = context.get("code", "")
        stock_name = context.get("stock_name", "")

        self._logger.debug(f"[{stock_code}] PortfolioManagerAgent开始最终决策")

        try:
            agent_signals = context.get("agent_signals", {})
            risk_manager_signal = context.get("risk_manager_signal", {})
            consensus_data = context.get("consensus_data", {})

            if not agent_signals:
                return AgentSignal(
                    agent_name=self.name,
                    signal=SignalType.HOLD,
                    confidence=0,
                    reasoning="无分析师信号输入",
                    metadata={"error": "missing_agent_signals"},
                )

            # Get max position limit from risk manager
            max_position = self._get_max_position_limit(risk_manager_signal)

            # Use LLM if available
            if self._llm_client and self._llm_client.is_available():
                decision = self._make_llm_decision(stock_code, stock_name, agent_signals, consensus_data, max_position)
            else:
                decision = self._make_rule_based_decision(agent_signals, consensus_data, max_position)

            # Apply risk limit to position ratio
            decision["position_ratio"] = min(decision.get("position_ratio", 0), max_position)

            self._logger.debug(
                f"[{stock_code}] PortfolioManager决策: {decision['action']} "
                f"(置信度{decision['confidence']}%, 仓位{decision['position_ratio'] * 100:.0f}%)"
            )

            signal = self._action_to_signal(decision["action"])

            return AgentSignal(
                agent_name=self.name,
                signal=signal,
                confidence=decision["confidence"],
                reasoning=decision["reasoning"],
                metadata={
                    "action": decision["action"],
                    "position_ratio": decision["position_ratio"],
                    "max_position_limit": max_position,
                    "key_factors": decision.get("key_factors", []),
                    "risk_level": decision.get("risk_level", "medium"),
                    "agent_signals_summary": self._summarize_signals(agent_signals),
                },
            )

        except Exception as e:
            self._logger.error(f"[{stock_code}] PortfolioManager决策失败: {e}")
            return AgentSignal(
                agent_name=self.name,
                signal=SignalType.HOLD,
                confidence=0,
                reasoning=f"决策失败: {str(e)}",
                metadata={"error": str(e)},
            )

    def _get_max_position_limit(self, risk_manager_signal: dict[str, Any]) -> float:
        """Extract max position limit from risk manager signal."""
        if not risk_manager_signal:
            return 0.25  # Default conservative limit

        metadata = risk_manager_signal.get("metadata", {})
        return metadata.get("max_position_size", 0.25)

    def _make_llm_decision(
        self,
        stock_code: str,
        stock_name: str,
        agent_signals: dict[str, Any],
        consensus_data: dict[str, Any],
        max_position: float,
    ) -> dict[str, Any]:
        """Use LLM to make final decision."""
        prompt = self._build_decision_prompt(stock_code, stock_name, agent_signals, consensus_data, max_position)

        self._logger.debug(f"[{stock_code}] PortfolioManager调用LLM进行决策...")

        if not self._llm_client:
            return self._make_rule_based_decision(agent_signals, consensus_data, max_position)

        try:
            response = self._llm_client.generate(
                prompt=prompt,
                generation_config={"temperature": 0.2, "max_output_tokens": 1024},
                system_prompt=PORTFOLIO_MANAGER_SYSTEM_PROMPT,
            )

            result = json_repair.repair_json(response, return_objects=True)

            if result and isinstance(result, dict) and "action" in result:
                self._logger.debug(f"[{stock_code}] LLM决策成功: {result}")
                return result
            else:
                self._logger.warning(f"[{stock_code}] LLM返回格式无效，使用规则回退")
                return self._make_rule_based_decision(agent_signals, consensus_data, max_position)

        except Exception as e:
            self._logger.warning(f"[{stock_code}] LLM决策失败: {e}，使用规则回退")
            return self._make_rule_based_decision(agent_signals, consensus_data, max_position)

    def _make_rule_based_decision(
        self,
        agent_signals: dict[str, Any],
        consensus_data: dict[str, Any],
        max_position: float,
    ) -> dict[str, Any]:
        """Fallback rule-based decision."""
        weighted_score = consensus_data.get("weighted_score", 0)
        risk_flags = consensus_data.get("risk_flags", [])

        # Determine action
        if weighted_score >= 30:
            action = "BUY"
            reasoning = "综合信号看多，建议买入"
        elif weighted_score <= -30:
            action = "SELL"
            reasoning = "综合信号看空，建议卖出"
        else:
            action = "HOLD"
            reasoning = "信号中性，建议观望"

        # Calculate confidence
        confidence = min(100, abs(int(weighted_score)))
        if risk_flags:
            confidence = max(30, confidence - len(risk_flags) * 10)
            reasoning += f"（注意{len(risk_flags)}个风险点）"

        # Calculate position ratio
        if action == "BUY":
            base_position = confidence / 100 * 0.8
            position_ratio = min(base_position, max_position)
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
            "risk_level": "high" if len(risk_flags) >= 2 else "medium" if risk_flags else "low",
        }

    def _build_decision_prompt(
        self,
        stock_code: str,
        stock_name: str,
        agent_signals: dict[str, Any],
        consensus_data: dict[str, Any],
        max_position: float,
    ) -> str:
        """Build decision prompt for LLM."""
        signal_lines = []
        for agent_name, signal_data in agent_signals.items():
            signal = signal_data.get("signal", "unknown")
            confidence = signal_data.get("confidence", 0)
            reasoning = signal_data.get("reasoning", "")
            signal_lines.append(f"{agent_name}: {signal} (置信度{confidence}%) - {reasoning[:60]}...")

        agent_signals_str = "\n".join(signal_lines)

        return f"""请作为专业的投资组合经理，基于以下分析结果做出最终交易决策。

=== 股票信息 ===
股票代码: {stock_code}
股票名称: {stock_name}

=== 分析师信号汇总 ===
{agent_signals_str}

=== 共识数据 ===
加权得分: {consensus_data.get("weighted_score", 0):.1f} (-100到+100)
共识度: {consensus_data.get("consensus_level", 0):.1%}
风险标记: {consensus_data.get("risk_flags", []) or "无"}

=== 风险限制 ===
最大仓位限制: {max_position * 100:.0f}%

请综合分析所有信号，在风险限制内做出决策。

请严格按照JSON格式输出：
{{
    "action": "buy|sell|hold",
    "confidence": 75,
    "reasoning": "简洁的决策理由(不超过150字)",
    "position_ratio": 0.5,
    "risk_level": "low|medium|high",
    "key_factors": ["关键因素1", "关键因素2"]
}}"""

    def _action_to_signal(self, action: str) -> SignalType:
        """Convert action to signal."""
        action_map = {
            "BUY": SignalType.BUY,
            "HOLD": SignalType.HOLD,
            "SELL": SignalType.SELL,
        }
        return action_map.get(action.upper(), SignalType.HOLD)

    def _summarize_signals(self, agent_signals: dict[str, Any]) -> dict[str, str]:
        """Create summary of agent signals."""
        return {
            name: f"{data.get('signal', 'unknown')}({data.get('confidence', 0)})"
            for name, data in agent_signals.items()
        }
