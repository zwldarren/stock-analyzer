"""
AI Analyzer Module - Multi-Agent Architecture

This module implements the AI analysis layer using a multi-agent system.
It coordinates specialized agents to analyze stocks and generate trading decisions.

Responsibilities:
1. Coordinate multiple specialized agents for parallel analysis
2. Aggregate agent signals to generate final trading decisions
3. Generate decision dashboard reports
4. Support multiple LLM providers with fallback
"""

import logging
from typing import Any

from stock_analyzer.config import get_config
from stock_analyzer.domain import get_stock_name_from_context
from stock_analyzer.domain.exceptions import AnalysisError
from stock_analyzer.domain.models import AnalysisResult
from stock_analyzer.domain.services import IAIAnalyzer

logger = logging.getLogger(__name__)


class AIAnalyzer(IAIAnalyzer):
    """
    AI Analyzer based on multi-agent architecture.

    This class coordinates multiple specialized agents to analyze stocks
    and generates trading decisions through the decision layer.

    Example:
        analyzer = AIAnalyzer()
        result = analyzer.analyze(context)
    """

    def __init__(self):
        """Initialize AI analyzer with multi-agent coordinator."""
        self._init_agent_coordinator()
        logger.info("AI分析器初始化成功 (多Agent模式)")

    def _init_agent_coordinator(self) -> None:
        """Initialize multi-agent coordinator with decision layer."""
        from stock_analyzer.agents import (
            AgentCoordinator,
            ChipAgent,
            DecisionAgent,
            FundamentalAgent,
            NewsSentimentAgent,
            RiskAgent,
            StyleAgent,
            TechnicalAgent,
            ValuationAgent,
        )

        self._agent_coordinator = AgentCoordinator()

        # Register analysis agents (provide signals)
        # Technical & A-share specific
        self._agent_coordinator.register_agent(TechnicalAgent())
        self._agent_coordinator.register_agent(ChipAgent())

        # Fundamental analysis
        self._agent_coordinator.register_agent(FundamentalAgent())
        self._agent_coordinator.register_agent(ValuationAgent())

        # News & sentiment
        self._agent_coordinator.register_agent(NewsSentimentAgent())

        # Investment style (merged Value/Growth/Momentum into single agent)
        self._agent_coordinator.register_agent(StyleAgent())

        # Risk management
        self._agent_coordinator.register_agent(RiskAgent())

        # Register decision agent (final decision maker)
        self._decision_agent = DecisionAgent()

        logger.info(f"Agent协调器初始化完成，已注册{len(self._agent_coordinator.agents)}个分析Agent + 1个决策Agent")

    def is_available(self) -> bool:
        """Check if analyzer is available."""
        return self._agent_coordinator is not None

    def analyze(self, context: dict[str, Any], news_context: str | None = None) -> AnalysisResult:
        """
        Analyze a single stock using multi-agent architecture.

        Analysis flow:
        1. Execute parallel multi-agent analysis (Technical/Fundamental/Chip)
        2. Aggregate agent signals to generate consensus data
        3. Decision layer (DecisionAgent) generates final trading decision
        4. Build decision dashboard
        5. Return structured result

        Args:
            context: Analysis context with stock data
            news_context: Pre-searched news content (optional, for backward compatibility)

        Returns:
            AnalysisResult with trading decision
        """
        code = context.get("code", "Unknown")
        name = context.get("stock_name", "")

        # Get stock name from context
        if not name or name.startswith("股票"):
            if "realtime" in context and context["realtime"].get("name"):
                name = context["realtime"]["name"]
            else:
                name = get_stock_name_from_context(code, context)

        logger.info(f"========== 多Agent分析 {name}({code}) ==========")

        # Ensure coordinator is initialized
        if self._agent_coordinator is None:
            logger.info("多Agent协调器未初始化，正在初始化...")
            self._init_agent_coordinator()

        if self._agent_coordinator is None:
            raise AnalysisError("Agent协调器初始化失败")

        try:
            # Step 1: Execute multi-agent analysis
            agent_results = self._agent_coordinator.analyze(context)

            # Step 2: Extract consensus data (simplified - no weighted scores)
            consensus = agent_results["consensus"]
            agent_signals = agent_results["agent_signals"]
            consensus_level = agent_results["consensus_level"]

            logger.info(
                f"[{code}] 分析Agent完成: {len(consensus.participating_agents)}个Agent参与, 共识度{consensus_level:.2f}"
            )

            # Step 3: Get current position status (can be extended to fetch from portfolio)
            current_position = context.get("current_position", "none")

            # Step 4: Decision layer makes final decision
            decision_context = {
                "code": code,
                "stock_name": name,
                "current_position": current_position,
                "agent_signals": agent_signals,
                "consensus_data": {
                    "consensus_level": consensus_level,
                    "participating_agents": consensus.participating_agents,
                    "risk_flags": consensus.risk_flags,
                },
                "market_data": context.get("today", {}),
            }

            final_signal = self._decision_agent.analyze(decision_context)

            logger.info(
                f"[{code}] 决策层完成: {final_signal.metadata.get('action', 'unknown')} "
                f"(置信度{final_signal.confidence}%)"
            )

            # Step 5: Build AnalysisResult from decision
            result = self._build_analysis_result_from_decision(
                code=code,
                name=name,
                final_signal=final_signal,
                agent_signals=agent_signals,
                consensus_level=consensus_level,
                context=context,
            )

            # Step 6: Populate debug fields (data_sources and raw_response)
            result.data_sources = self._collect_data_sources(agent_signals)
            result.raw_response = {
                "final_decision": {
                    "action": final_signal.metadata.get("action", final_signal.signal.to_string()),
                    "signal": final_signal.signal,
                    "confidence": final_signal.confidence,
                    "reasoning": final_signal.reasoning,
                    "metadata": final_signal.metadata,
                },
                "agent_signals": agent_signals,
                "consensus": {
                    "consensus_level": consensus_level,
                    "participating_agents": consensus.participating_agents,
                    "risk_flags": consensus.risk_flags,
                },
            }

            return result

        except Exception as e:
            logger.error(f"多Agent分析 {name}({code}) 失败: {e}")
            # Return error result
            return AnalysisResult(
                code=code,
                name=name,
                sentiment_score=50,
                trend_prediction="震荡",
                operation_advice="持有",
                decision_type="hold",
                confidence_level="低",
                analysis_summary=f"多Agent分析失败: {str(e)[:100]}",
                risk_warning="分析失败，请稍后重试",
                success=False,
                error_message=str(e),
            )

    def _build_analysis_result_from_decision(
        self,
        code: str,
        name: str,
        final_signal: Any,
        agent_signals: dict[str, Any],
        consensus_level: float,
        context: dict[str, Any],
    ) -> AnalysisResult:
        """Build AnalysisResult from decision agent output.

        Args:
            code: Stock code
            name: Stock name
            final_signal: Final signal from decision agent
            agent_signals: Signals from all analysis agents
            consensus_level: Consensus level among agents
            context: Analysis context

        Returns:
            AnalysisResult object
        """

        # Extract decision metadata
        metadata = final_signal.metadata
        action = metadata.get("action", "HOLD")
        position_ratio = metadata.get("position_ratio", 0.0)
        key_factors = metadata.get("key_factors", [])
        risk_assessment = metadata.get("risk_assessment", {})

        # Map action to Chinese operation advice
        action_map = {
            "BUY": "买入",
            "HOLD": "持有",
            "SELL": "卖出",
        }
        operation_advice = action_map.get(action, "观望")

        # Map internal signal to decision_type
        signal_to_decision = {
            "buy": "buy",
            "hold": "hold",
            "sell": "sell",
        }
        decision_type = signal_to_decision.get(final_signal.signal, "hold")

        # Determine trend prediction based on action
        if action == "BUY":
            trend_prediction = "看多" if final_signal.confidence >= 70 else "谨慎看多"
        elif action == "SELL":
            trend_prediction = "看空" if final_signal.confidence >= 70 else "谨慎看空"
        else:
            trend_prediction = "震荡"

        # Build new decision-focused dashboard
        dashboard = self._build_decision_dashboard(final_signal, agent_signals, consensus_level, context)

        # Build analysis summary with key factors
        analysis_summary = self._build_decision_summary(final_signal, agent_signals, consensus_level, key_factors)

        # Extract risk warnings
        risk_level = risk_assessment.get("level", "low")
        risk_concerns = risk_assessment.get("concerns", [])
        if risk_concerns:
            risk_warning = f"风险等级: {risk_level} | " + "; ".join(risk_concerns[:3])
        else:
            risk_warning = f"风险等级: {risk_level}"

        # Build decision reason
        decision_reason = final_signal.reasoning

        return AnalysisResult(
            code=code,
            name=name,
            sentiment_score=final_signal.confidence,
            trend_prediction=trend_prediction,
            operation_advice=operation_advice,
            decision_type=decision_type,
            confidence_level=self._score_to_confidence(final_signal.confidence),
            final_action=action,
            position_ratio=position_ratio,
            decision_reasoning=decision_reason,
            dashboard=dashboard,
            analysis_summary=analysis_summary,
            risk_warning=risk_warning,
            market_snapshot=self._build_market_snapshot(context),
            success=True,
        )

    def _get_consensus_advice(self, agent_signals: dict[str, Any]) -> str:
        """Generate consensus advice from agent signals."""
        buy_agents = [k for k, v in agent_signals.items() if v["signal"] == "buy"]
        sell_agents = [k for k, v in agent_signals.items() if v["signal"] == "sell"]

        if len(buy_agents) > len(sell_agents):
            return f"{len(buy_agents)}个Agent建议买入"
        elif len(sell_agents) > len(buy_agents):
            return f"{len(sell_agents)}个Agent建议卖出"
        return "Agent意见分歧，建议观望"

    def _get_no_position_advice(self, final_decision: Any) -> str:
        """Generate advice for users without existing position."""
        if final_decision.signal == "buy":
            if final_decision.confidence >= 70:
                return "可小仓位试探性买入"
            return "等待更明确信号"
        elif final_decision.signal == "sell":
            return "回避，不介入"
        return "继续观望"

    def _get_has_position_advice(self, final_decision: Any) -> str:
        """Generate advice for users with existing position."""
        if final_decision.signal == "sell":
            if final_decision.confidence >= 70:
                return "考虑减仓或止盈"
            return "提高警惕，设置止损"
        elif final_decision.signal == "buy":
            return "继续持有，关注加仓机会"
        return "继续持有，关注风险信号"

    def _build_decision_dashboard(
        self,
        final_signal: Any,
        agent_signals: dict[str, Any],
        consensus_level: float,
        context: dict[str, Any],
    ) -> dict[str, Any]:
        """Build decision-focused dashboard for visualization."""
        metadata = final_signal.metadata
        action = metadata.get("action", "HOLD")
        position_ratio = metadata.get("position_ratio", 0.0)
        key_factors = metadata.get("key_factors", [])
        risk_assessment = metadata.get("risk_assessment", {})

        return {
            "final_decision": {
                "action": action,
                "confidence": final_signal.confidence,
                "position_ratio": f"{position_ratio * 100:.0f}%",
                "reasoning": final_signal.reasoning,
            },
            "key_factors": key_factors,
            "risk_assessment": risk_assessment,
            "agent_consensus": {
                "signals": {k: v["signal"] for k, v in agent_signals.items()},
                "confidences": {k: v["confidence"] for k, v in agent_signals.items()},
                "reasonings": {k: v["reasoning"] for k, v in agent_signals.items()},
                "consensus_level": f"{consensus_level:.0%}",
                "participating_agents": list(agent_signals.items()),
            },
            "market_context": {
                "current_position": metadata.get("current_position", "none"),
                "price": context.get("today", {}).get("close"),
            },
        }

    def _build_decision_summary(
        self,
        final_signal: Any,
        agent_signals: dict[str, Any],
        consensus_level: float,
        key_factors: list[str],
    ) -> str:
        """Build decision-focused summary text."""
        metadata = final_signal.metadata
        action = metadata.get("action", "HOLD")

        parts = [
            f"最终决策: {action}",
            f"置信度: {final_signal.confidence}%",
            f"共识度: {consensus_level:.0%}",
        ]

        # Add key factors
        if key_factors:
            parts.append(f"关键因子: {', '.join(key_factors[:3])}")

        # Add agent breakdown
        for name, signal in agent_signals.items():
            parts.append(f"{name}: {signal['signal']}({signal['confidence']})")

        return " | ".join(parts)

    def _get_suggested_position(self, final_decision: Any) -> str:
        """Calculate suggested position size based on confidence."""
        if final_decision.signal == "buy":
            if final_decision.confidence >= 80:
                return "3-5成"
            elif final_decision.confidence >= 60:
                return "1-3成"
            return "轻仓试探"
        elif final_decision.signal == "sell":
            return "减仓或清仓"
        return "保持现有仓位"

    def _score_to_confidence(self, score: int) -> str:
        """Convert numeric score to confidence level text."""
        if score >= 80:
            return "高"
        elif score >= 60:
            return "中"
        return "低"

    def _build_market_snapshot(self, context: dict[str, Any]) -> dict[str, Any]:
        """Build market snapshot from analysis context."""
        today = context.get("today", {}) or {}
        realtime = context.get("realtime", {}) or {}
        yesterday = context.get("yesterday", {}) or {}

        prev_close = yesterday.get("close")
        close = today.get("close")
        high = today.get("high")
        low = today.get("low")

        # Calculate amplitude
        amplitude = None
        if prev_close not in (None, 0) and high is not None and low is not None:
            try:
                amplitude = (float(high) - float(low)) / float(prev_close) * 100
            except (TypeError, ValueError, ZeroDivisionError):
                amplitude = None

        snapshot = {
            "date": context.get("date", "未知"),
            "close": self._format_price(close),
            "open": self._format_price(today.get("open")),
            "high": self._format_price(high),
            "low": self._format_price(low),
            "prev_close": self._format_price(prev_close),
            "pct_chg": self._format_percent(today.get("pct_chg")),
            "amplitude": self._format_percent(amplitude),
            "volume": today.get("volume"),
            "amount": today.get("amount"),
        }

        if realtime:
            snapshot.update(
                {
                    "price": self._format_price(realtime.get("price")),
                    "change_amount": realtime.get("change_amount"),
                    "volume_ratio": realtime.get("volume_ratio"),
                    "turnover_rate": self._format_percent(realtime.get("turnover_rate")),
                    "source": realtime.get("source"),
                }
            )

        return snapshot

    def _collect_data_sources(self, agent_signals: dict[str, Any]) -> str:
        """Collect data source information from agent signals."""
        agent_names = list(agent_signals.keys())
        return ", ".join(agent_names) if agent_names else "multi-agent"

    @staticmethod
    def _format_price(value: float | None) -> str:
        """Format price value for display."""
        if value is None:
            return "N/A"
        try:
            return f"{float(value):.2f}"
        except (TypeError, ValueError):
            return "N/A"

    @staticmethod
    def _format_percent(value: float | None) -> str:
        """Format percentage value for display."""
        if value is None:
            return "N/A"
        try:
            return f"{float(value):.2f}%"
        except (TypeError, ValueError):
            return "N/A"

    def batch_analyze(
        self,
        contexts: list[dict[str, Any]],
        delay_between: float = 2.0,
    ) -> list[AnalysisResult]:
        """
        Batch analyze multiple stocks.

        Args:
            contexts: Context data list
            delay_between: Delay between analyses in seconds

        Returns:
            List of AnalysisResult
        """
        import time

        results = []

        for i, context in enumerate(contexts):
            if i > 0:
                logger.debug(f"等待 {delay_between} 秒后继续...")
                time.sleep(delay_between)

            result = self.analyze(context)
            results.append(result)

        return results

    def generate_market_review(self, prompt: str, generation_config: dict[str, Any]) -> str | None:
        """
        生成市场复盘报告

        Args:
            prompt: 提示词
            generation_config: 生成配置

        Returns:
            生成的复盘报告文本，失败返回 None
        """
        # For multi-agent mode, we need to use LLM for market review
        # This functionality requires LLM client
        config = get_config()

        try:
            from stock_analyzer.ai.clients import LiteLLMClient

            client = LiteLLMClient(
                model=config.ai.llm_model,
                api_key=config.ai.llm_api_key,
                base_url=config.ai.llm_base_url,
            )

            if not client.is_available():
                logger.warning("LLM 客户端不可用，无法生成市场复盘")
                return None

            return client.generate(prompt, generation_config)
        except Exception as e:
            logger.error(f"生成市场复盘报告失败: {e}")
            return None


# 便捷函数
def get_analyzer() -> AIAnalyzer:
    """获取 AI 分析器实例"""
    return AIAnalyzer()
