"""
Risk Manager Agent Module

Independent risk management agent focused on position sizing limits.

This agent is responsible for:
- Calculating maximum position size based on volatility
- Providing risk constraints for portfolio decisions
- NOT generating trading signals (that's PortfolioManager's job)

Inspired by ai-hedge-fund's Risk Manager architecture.
"""

import logging
import math
from typing import Any

from ashare_analyzer.exceptions import handle_errors
from ashare_analyzer.models import AgentSignal, SignalType

from .base import BaseAgent

logger = logging.getLogger(__name__)


class RiskManagerAgent(BaseAgent):
    """
    Risk Manager Agent for position sizing and risk constraints.

    This agent DOES NOT generate trading signals. Instead, it calculates
    risk constraints that the PortfolioManager must respect.

    Key Outputs (in metadata):
    - max_position_size: Maximum allowed position (0-1)
    - volatility_tier: Volatility classification
    - risk_score: Overall risk assessment (0-100)
    - correlation_adjustment: Portfolio-level correlation adjustment (placeholder)

    Example:
        agent = RiskManagerAgent()
        signal = agent.analyze({
            "code": "600519",
            "price_data": {"close": [...]},
            "existing_positions": [...]  # Optional, for correlation
        })
        max_position = signal.metadata.get("max_position_size", 0.25)
    """

    # Volatility thresholds for position sizing
    VOL_THRESHOLDS = {
        "low": {"max_vol": 0.15, "base_limit": 0.25},
        "medium": {"max_vol": 0.30, "base_limit": 0.20},
        "high": {"max_vol": 0.50, "base_limit": 0.15},
        "very_high": {"max_vol": float("inf"), "base_limit": 0.10},
    }

    def __init__(self):
        """Initialize the Risk Manager Agent."""
        super().__init__("RiskManagerAgent")
        self._logger = logging.getLogger(__name__)

    def is_available(self) -> bool:
        """Always available - only requires context data."""
        return True

    @handle_errors("风险管理计算失败", default_return=None)
    async def analyze(self, context: dict[str, Any]) -> AgentSignal:
        """
        Calculate risk constraints and position limits (async).

        Args:
            context: Analysis context containing:
                - code: Stock code
                - price_data: Historical price data
                - existing_positions: List of existing positions (for correlation)

        Returns:
            AgentSignal with signal=HOLD (risk manager doesn't trade)
            and metadata containing risk constraints
        """
        stock_code = context.get("code", "")
        self._logger.debug(f"[{stock_code}] RiskManagerAgent计算风险限制")

        # Get price data
        price_data = context.get("price_data", {})

        # Calculate volatility metrics
        volatility_metrics = self._calculate_volatility(price_data)
        annualized_vol = volatility_metrics.get("annualized_volatility", 0.3)

        # Classify volatility tier
        vol_tier = self._classify_volatility_tier(annualized_vol)

        # Calculate base position limit
        base_limit = self._calculate_base_position_limit(annualized_vol)

        # Get correlation adjustment (placeholder for future implementation)
        correlation_adj = self._get_correlation_adjustment(context)

        # Apply correlation adjustment
        final_limit = base_limit * correlation_adj

        # Calculate overall risk score
        risk_score = self._calculate_risk_score(
            volatility_metrics,
            context.get("risk_factors", []),
            context.get("agent_signals", {}),
        )

        self._logger.debug(
            f"[{stock_code}] RiskManager: 波动率档次={vol_tier}, 仓位上限={final_limit * 100:.0f}%, 风险分={risk_score}"
        )

        # Build reasoning (for documentation purposes)
        reasoning = f"波动率{annualized_vol * 100:.1f}%({vol_tier}), 建议仓位上限{final_limit * 100:.0f}%"

        return AgentSignal(
            agent_name=self.name,
            signal=SignalType.HOLD,  # Risk manager never generates trade signals
            confidence=100,  # Confidence in risk calculation
            reasoning=reasoning,
            metadata={
                # Core outputs for PortfolioManager
                "max_position_size": round(final_limit, 3),
                "base_position_limit": round(base_limit, 3),
                "correlation_adjustment": round(correlation_adj, 3),
                # Risk metrics
                "risk_score": risk_score,
                "volatility_tier": vol_tier,
                # Volatility details
                "annualized_volatility": round(annualized_vol * 100, 1),
                "daily_volatility": round(volatility_metrics.get("daily_volatility", 0) * 100, 2),
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
            # Extract close prices
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

            # Calculate daily volatility
            mean_return = sum(returns) / len(returns)
            variance = sum((r - mean_return) ** 2 for r in returns) / len(returns)
            daily_vol = math.sqrt(variance)

            # Annualize (252 trading days)
            annualized_vol = daily_vol * math.sqrt(252)

            # Calculate percentile
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

    def _classify_volatility_tier(self, annualized_vol: float) -> str:
        """Classify volatility into tiers."""
        if annualized_vol < 0.15:
            return "low"
        elif annualized_vol < 0.30:
            return "medium"
        elif annualized_vol < 0.50:
            return "high"
        else:
            return "very_high"

    def _calculate_base_position_limit(self, annualized_vol: float) -> float:
        """
        Calculate base position size limit based on volatility.

        Volatility-adjusted limits (inspired by ai-hedge-fund):
        - Low volatility (<15%): Up to 25% allocation
        - Medium volatility (15-30%): 12.5-20% allocation
        - High volatility (30-50%): 5-15% allocation
        - Very high volatility (>50%): Max 10% allocation
        """
        if annualized_vol < 0.15:
            # Low volatility - allow higher allocation
            return 0.25
        elif annualized_vol < 0.30:
            # Medium volatility - linear interpolation
            vol_multiplier = 1.0 - (annualized_vol - 0.15) * 0.5
            return max(0.125, vol_multiplier * 0.20)
        elif annualized_vol < 0.50:
            # High volatility - reduce allocation
            vol_multiplier = 0.75 - (annualized_vol - 0.30) * 0.5
            return max(0.05, vol_multiplier * 0.20)
        else:
            # Very high volatility - minimum allocation
            return 0.10

    def _get_correlation_adjustment(self, context: dict[str, Any]) -> float:
        """
        Calculate correlation-based adjustment for position sizing.

        This is a placeholder for future portfolio-level correlation analysis.
        When implemented, it would:
        - Analyze correlation with existing positions
        - Reduce position limit for highly correlated assets
        - Increase limit for diversifying assets

        Returns:
            Adjustment multiplier (0.7 - 1.1)
        """
        # Placeholder implementation
        # TODO: Implement portfolio-level correlation analysis
        existing_positions = context.get("existing_positions", [])

        if not existing_positions:
            return 1.0  # No adjustment for single position

        # Placeholder: Assume moderate correlation adjustment for multiple positions
        # In production, would calculate actual correlations
        if len(existing_positions) >= 5:
            return 0.85  # Reduce for concentrated portfolio
        elif len(existing_positions) >= 3:
            return 0.95
        else:
            return 1.0

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
            buy_count = sum(1 for s in agent_signals.values() if s.get("signal", "").lower() == "buy")
            sell_count = sum(1 for s in agent_signals.values() if s.get("signal", "").lower() == "sell")
            hold_count = sum(1 for s in agent_signals.values() if s.get("signal", "").lower() == "hold")

            total = len(agent_signals)
            if total > 0:
                max_consensus = max(buy_count, sell_count, hold_count)
                divergence = 1 - (max_consensus / total)
                score += int(divergence * 30)

        return min(score, 100)
