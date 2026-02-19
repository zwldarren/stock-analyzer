"""
AI Analyzer Module - Multi-Agent Architecture

This module implements the AI analysis layer using a multi-agent system.
It coordinates specialized agents to analyze stocks and generate trading decisions.

Responsibilities:
1. Coordinate multiple specialized agents for parallel analysis
2. Aggregate agent signals to generate final trading decisions
3. Generate decision dashboard reports
4. Support multiple LLM providers with fallback

Architecture:
    Analysis Agents (parallel) -> RiskManagerAgent -> PortfolioManagerAgent (final decision)
"""

import logging
from typing import Any

from stock_analyzer.ai.interface import IAIAnalyzer
from stock_analyzer.constants import normalize_signal
from stock_analyzer.data.stock_name_resolver import StockNameResolver
from stock_analyzer.exceptions import AnalysisError
from stock_analyzer.models import AnalysisResult
from stock_analyzer.utils import get_display

logger = logging.getLogger(__name__)


class AIAnalyzer(IAIAnalyzer):
    """
    AI Analyzer based on multi-agent architecture.

    This class coordinates multiple specialized agents to analyze stocks
    and generates trading decisions through the PortfolioManagerAgent.

    Execution Flow:
        1. RiskManagerAgent calculates position limits (runs first)
        2. Analysis agents run in parallel (Technical, Fundamental, etc.)
        3. PortfolioManagerAgent makes final decision respecting risk limits

    Example:
        analyzer = AIAnalyzer()
        result = analyzer.analyze(context)
    """

    def __init__(self):
        """Initialize AI analyzer with multi-agent coordinator."""
        self._init_agent_coordinator()
        logger.debug("AI分析器初始化成功 (多Agent模式)")

    def _init_agent_coordinator(self) -> None:
        """Initialize multi-agent coordinator with decision layer."""
        from stock_analyzer.ai.agents import (
            AgentCoordinator,
            ChipAgent,
            FundamentalAgent,
            NewsSentimentAgent,
            PortfolioManagerAgent,
            RiskManagerAgent,
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

        # Risk manager (calculates position limits, not a trading signal)
        self._risk_manager_agent = RiskManagerAgent()

        # Portfolio manager (final decision maker with risk constraints)
        self._portfolio_manager = PortfolioManagerAgent()

        agent_count = len(self._agent_coordinator.agents)
        logger.debug(f"Agent协调器初始化完成，已注册{agent_count}个分析Agent + RiskManager + PortfolioManager")

    def is_available(self) -> bool:
        """Check if analyzer is available."""
        return self._agent_coordinator is not None

    async def analyze(self, context: dict[str, Any]) -> AnalysisResult:
        """
        Analyze a single stock using multi-agent architecture (async).

        Analysis flow:
        1. RiskManagerAgent calculates position limits (runs first)
        2. Execute parallel multi-agent analysis (Technical/Fundamental/Chip)
        3. PortfolioManagerAgent makes final decision respecting risk limits
        4. Build decision dashboard
        5. Return structured result

        Args:
            context: Analysis context with stock data

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
                name = StockNameResolver.from_context(code, context)

        logger.debug(f"多Agent分析 {name}({code}) 开始")

        # Ensure coordinator is initialized
        if self._agent_coordinator is None:
            logger.debug("多Agent协调器未初始化，正在初始化...")
            self._init_agent_coordinator()

        if self._agent_coordinator is None:
            raise AnalysisError("Agent协调器初始化失败")

        try:
            display = get_display()

            # Step 1: RiskManagerAgent calculates position limits (runs first)
            display.start_agent("RiskManagerAgent")
            risk_manager_signal = await self._risk_manager_agent.analyze(context)
            display.complete_agent("RiskManagerAgent", "neutral", risk_manager_signal.confidence)
            max_pos = risk_manager_signal.metadata.get("max_position_size", 0.25) * 100
            logger.debug(f"[{code}] 风险管理完成: 仓位上限={max_pos:.0f}%")

            # Step 2: Execute multi-agent analysis (parallel)
            agent_results = await self._agent_coordinator.analyze(context)

            # Step 3: Extract consensus data
            consensus = agent_results["consensus"]
            agent_signals = agent_results["agent_signals"]
            consensus_level = agent_results["consensus_level"]

            logger.debug(
                f"[{code}] 分析Agent完成: {len(consensus.participating_agents)}个Agent参与, 共识度{consensus_level:.2f}"
            )

            # Step 4: Get current position status (can be extended to fetch from portfolio)
            current_position = context.get("current_position", "none")

            # Step 5: PortfolioManager makes final decision with risk constraints
            decision_context = {
                "code": code,
                "stock_name": name,
                "current_position": current_position,
                "agent_signals": agent_signals,
                "risk_manager_signal": {
                    "signal": risk_manager_signal.signal.to_string(),
                    "confidence": risk_manager_signal.confidence,
                    "reasoning": risk_manager_signal.reasoning,
                    "metadata": risk_manager_signal.metadata,
                },
                "consensus_data": {
                    "consensus_level": consensus_level,
                    "participating_agents": consensus.participating_agents,
                    "risk_flags": consensus.risk_flags,
                    "weighted_score": risk_manager_signal.metadata.get("risk_score", 50),
                },
                "market_data": context.get("today", {}),
            }

            display.start_agent("PortfolioManagerAgent")
            final_signal = await self._portfolio_manager.analyze(decision_context)
            action = final_signal.metadata.get("action", "HOLD")
            display.complete_agent("PortfolioManagerAgent", normalize_signal(action), final_signal.confidence)

            logger.debug(
                f"[{code}] 投资组合决策完成: {final_signal.metadata.get('action', 'unknown')} "
                f"(置信度{final_signal.confidence}%, 仓位{final_signal.metadata.get('position_ratio', 0) * 100:.0f}%)"
            )

            # Step 6: Build AnalysisResult from decision
            result = self._build_analysis_result_from_decision(
                code=code,
                name=name,
                final_signal=final_signal,
                agent_signals=agent_signals,
                consensus_level=consensus_level,
                context=context,
            )

            # Step 7: Populate debug fields (data_sources and raw_response)
            result.data_sources = self._collect_data_sources(agent_signals)
            result.raw_response = {
                "final_decision": {
                    "action": final_signal.metadata.get("action", final_signal.signal.to_string()),
                    "signal": final_signal.signal,
                    "confidence": final_signal.confidence,
                    "reasoning": final_signal.reasoning,
                    "metadata": final_signal.metadata,
                },
                "risk_management": {
                    "max_position_size": risk_manager_signal.metadata.get("max_position_size"),
                    "risk_score": risk_manager_signal.metadata.get("risk_score"),
                    "volatility_tier": risk_manager_signal.metadata.get("volatility_tier"),
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

    async def batch_analyze(
        self,
        contexts: list[dict[str, Any]],
        delay_between: float = 2.0,
    ) -> list[AnalysisResult]:
        """
        Batch analyze multiple stocks (async).

        Args:
            contexts: Context data list
            delay_between: Delay between analyses in seconds

        Returns:
            List of AnalysisResult
        """
        import asyncio

        results = []

        for i, context in enumerate(contexts):
            if i > 0:
                logger.debug(f"等待 {delay_between} 秒后继续...")
                await asyncio.sleep(delay_between)

            result = await self.analyze(context)
            results.append(result)

        return results
