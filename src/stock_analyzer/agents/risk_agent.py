"""
Risk Management Agent Module

Specialized agent for risk assessment and position sizing.

This agent focuses exclusively on:
- Volatility-based position sizing
- Risk factor aggregation
- Maximum drawdown assessment
- Risk-adjusted signal moderation

Inspired by ai-hedge-fund's Risk Manager Agent.
"""

import logging
import math
from typing import Any

from stock_analyzer.domain.constants import SIGNAL_BUY, SIGNAL_HOLD, SIGNAL_SELL, normalize_signal
from stock_analyzer.domain.exceptions import handle_errors

from .base import AgentSignal, BaseAgent, SignalType

logger = logging.getLogger(__name__)


class RiskAgent(BaseAgent):
    """
    Risk Management Agent for position sizing and risk assessment.

    This agent analyzes:
    - Volatility-based position limits
    - Risk factors from other agents
    - Maximum drawdown potential
    - Risk-adjusted confidence scores

    Key features:
    - Volatility-adjusted position sizing
    - Risk flag aggregation
    - Risk score calculation (0-100)

    Attributes:
        None - uses context data only

    Example:
        agent = RiskAgent()
        signal = agent.analyze({
            "code": "600519",
            "stock_name": "贵州茅台",
            "price_data": {...},
            "risk_factors": [...]
        })
    """

    def __init__(self):
        """Initialize the Risk Agent."""
        super().__init__("RiskAgent")
        self._logger = logging.getLogger(__name__)

    def is_available(self) -> bool:
        """Always available - only requires context data."""
        return True

    @handle_errors("风险分析失败", default_return=None)
    def analyze(self, context: dict[str, Any]) -> AgentSignal:
        """
        Execute risk analysis.

        Args:
            context: Analysis context containing:
                - code: Stock code
                - stock_name: Stock name
                - price_data: Historical price data (DataFrame or dict)
                - risk_factors: List of risk factors from other agents
                - agent_signals: Dict of signals from other agents

        Returns:
            AgentSignal with risk assessment and position sizing recommendation
        """
        stock_code = context.get("code", "")
        context.get("stock_name", "")

        self._logger.info(f"[{stock_code}] RiskAgent开始风险分析")

        # Get data from context
        price_data = context.get("price_data", {})
        risk_factors = context.get("risk_factors", [])
        agent_signals = context.get("agent_signals", {})

        # Calculate volatility metrics
        volatility_metrics = self._calculate_volatility(price_data)
        annualized_vol = volatility_metrics.get("annualized_volatility", 0.3)

        # Calculate position limit based on volatility
        position_limit = self._calculate_position_limit(annualized_vol)

        # Calculate risk score from various sources
        risk_score = self._calculate_risk_score(volatility_metrics, risk_factors, agent_signals)

        # Determine risk level
        risk_level = self._risk_score_to_level(risk_score)

        # Generate signal based on risk level
        signal, confidence = self._risk_to_signal(risk_score, agent_signals)

        # Build reasoning
        reasoning_parts = []
        reasoning_parts.append(f"风险等级: {risk_level}")
        reasoning_parts.append(f"年化波动率: {annualized_vol * 100:.1f}%")
        reasoning_parts.append(f"建议仓位上限: {position_limit * 100:.0f}%")

        if risk_factors:
            reasoning_parts.append(f"风险因素: {len(risk_factors)}项")

        reasoning = " / ".join(reasoning_parts)

        self._logger.info(
            f"[{stock_code}] RiskAgent分析完成: {signal} "
            f"(风险分{risk_score}/100, 波动率{annualized_vol * 100:.1f}%, 置信度{confidence}%)"
        )

        return AgentSignal(
            agent_name=self.name,
            signal=signal,
            confidence=confidence,
            reasoning=reasoning,
            metadata={
                "risk_score": risk_score,
                "risk_level": risk_level,
                "annualized_volatility": round(annualized_vol * 100, 1),
                "daily_volatility": round(volatility_metrics.get("daily_volatility", 0) * 100, 2),
                "position_limit": round(position_limit * 100, 0),
                "risk_factors": risk_factors,
                "volatility_percentile": volatility_metrics.get("volatility_percentile", 50),
            },
        )

    def _calculate_volatility(self, price_data: Any) -> dict[str, float]:
        """
        Calculate volatility metrics from price data.

        Returns:
            Dict with daily_volatility, annualized_volatility, volatility_percentile
        """
        try:
            # Try to extract close prices
            if hasattr(price_data, "close"):
                closes = price_data.close.values if hasattr(price_data.close, "values") else list(price_data.close)
            elif isinstance(price_data, dict) and "close" in price_data:
                closes = price_data["close"]
            elif isinstance(price_data, list):
                closes = price_data
            else:
                return {"daily_volatility": 0.02, "annualized_volatility": 0.30, "volatility_percentile": 50}

            if len(closes) < 10:
                return {"daily_volatility": 0.02, "annualized_volatility": 0.30, "volatility_percentile": 50}

            # Calculate daily returns
            returns = []
            for i in range(1, len(closes)):
                if closes[i - 1] > 0:
                    daily_return = (closes[i] - closes[i - 1]) / closes[i - 1]
                    returns.append(daily_return)

            if len(returns) < 5:
                return {"daily_volatility": 0.02, "annualized_volatility": 0.30, "volatility_percentile": 50}

            # Calculate daily volatility (standard deviation)
            mean_return = sum(returns) / len(returns)
            variance = sum((r - mean_return) ** 2 for r in returns) / len(returns)
            daily_vol = math.sqrt(variance)

            # Annualize (252 trading days)
            annualized_vol = daily_vol * math.sqrt(252)

            # Calculate percentile (simplified)
            # In practice, would compare to historical volatility
            if annualized_vol < 0.15:
                percentile = 20
            elif annualized_vol < 0.30:
                percentile = 50
            elif annualized_vol < 0.50:
                percentile = 80
            else:
                percentile = 95

            return {
                "daily_volatility": daily_vol,
                "annualized_volatility": annualized_vol,
                "volatility_percentile": percentile,
            }

        except Exception as e:
            self._logger.warning(f"波动率计算失败: {e}")
            return {"daily_volatility": 0.02, "annualized_volatility": 0.30, "volatility_percentile": 50}

    def _calculate_position_limit(self, annualized_vol: float) -> float:
        """
        Calculate position size limit based on volatility.

        Volatility-adjusted limits (inspired by ai-hedge-fund):
        - Low volatility (<15%): Up to 25% allocation
        - Medium volatility (15-30%): 12.5-20% allocation
        - High volatility (30-50%): 5-15% allocation
        - Very high volatility (>50%): Max 10% allocation
        """
        if annualized_vol < 0.15:
            # Low volatility
            return 0.25
        elif annualized_vol < 0.30:
            # Medium volatility - linear interpolation
            vol_multiplier = 1.0 - (annualized_vol - 0.15) * 0.5
            return max(0.125, vol_multiplier * 0.20)
        elif annualized_vol < 0.50:
            # High volatility
            vol_multiplier = 0.75 - (annualized_vol - 0.30) * 0.5
            return max(0.05, vol_multiplier * 0.20)
        else:
            # Very high volatility
            return 0.10

    def _calculate_risk_score(
        self,
        volatility_metrics: dict[str, float],
        risk_factors: list[str],
        agent_signals: dict[str, Any],
    ) -> int:
        """
        Calculate overall risk score (0-100, higher = more risky).

        Components:
        - Volatility risk (0-40 points)
        - Risk factor count (0-30 points)
        - Signal divergence (0-30 points)
        """
        score = 0

        # Volatility risk (0-40 points)
        annualized_vol = volatility_metrics.get("annualized_volatility", 0.3)
        if annualized_vol > 0.50:
            score += 40
        elif annualized_vol > 0.30:
            score += 30
        elif annualized_vol > 0.15:
            score += 20
        else:
            score += 10

        # Risk factor count (0-30 points)
        risk_factor_count = len(risk_factors)
        if risk_factor_count >= 5:
            score += 30
        elif risk_factor_count >= 3:
            score += 20
        elif risk_factor_count >= 1:
            score += 10

        # Signal divergence (0-30 points)
        if agent_signals:
            # 标准化信号并计数
            buy_count = 0
            sell_count = 0
            hold_count = 0

            for s in agent_signals.values():
                normalized = normalize_signal(s.get("signal", SIGNAL_HOLD))
                if normalized == SIGNAL_BUY:
                    buy_count += 1
                elif normalized == SIGNAL_SELL:
                    sell_count += 1
                else:
                    hold_count += 1

            total = len(agent_signals)
            if total > 0:
                # High divergence = high risk
                max_consensus = max(buy_count, sell_count, hold_count)
                divergence = 1 - (max_consensus / total)
                score += int(divergence * 30)

        return min(score, 100)

    def _risk_score_to_level(self, score: int) -> str:
        """Convert risk score to risk level."""
        if score >= 70:
            return "高风险"
        elif score >= 50:
            return "中高风险"
        elif score >= 30:
            return "中等风险"
        else:
            return "低风险"

    def _risk_to_signal(self, risk_score: int, agent_signals: dict[str, Any]) -> tuple[SignalType, int]:
        """
        Convert risk assessment to signal and confidence.

        Risk Agent doesn't generate directional signals directly.
        Instead, it provides risk-adjusted confidence for other signals.

        Logic:
        - High risk (score >= 70): Reduce position or hold
        - Medium risk: Moderate confidence
        - Low risk: Full confidence
        """
        # Get consensus from other agents
        if not agent_signals:
            return (SignalType.HOLD, 50)

        # 标准化信号并计数
        buy_count = 0
        sell_count = 0

        for s in agent_signals.values():
            normalized = normalize_signal(s.get("signal", SIGNAL_HOLD))
            if normalized == SIGNAL_BUY:
                buy_count += 1
            elif normalized == SIGNAL_SELL:
                sell_count += 1

        # Base signal on majority
        if buy_count > sell_count:
            base_signal = SignalType.BUY
        elif sell_count > buy_count:
            base_signal = SignalType.SELL
        else:
            base_signal = SignalType.HOLD

        # Adjust confidence based on risk
        if risk_score >= 70:
            # High risk - reduce confidence significantly
            confidence = max(20, 50 - (risk_score - 70))
            # May override to hold if very risky
            if risk_score >= 85:
                return (SignalType.HOLD, confidence)
        elif risk_score >= 50:
            # Medium risk - moderate confidence
            confidence = max(40, 70 - (risk_score - 50) // 2)
        else:
            # Low risk - high confidence
            confidence = min(95, 80 + (50 - risk_score) // 2)

        return (base_signal, confidence)
