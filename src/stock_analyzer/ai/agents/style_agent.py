"""
Style Agent Module

Merged investment style agent combining Value, Growth, and Momentum analysis.

This agent provides a unified investment style analysis with:
- Value investing (Buffett/Graham principles): 30%
- Growth investing (disruptive innovation): 35%
- Momentum investing (trend following): 35%
"""

import logging
import math
from typing import Any

from stock_analyzer.exceptions import handle_errors
from stock_analyzer.models import AgentSignal, SignalType

from .base import BaseAgent
from .utils import FinancialScorer

logger = logging.getLogger(__name__)


class StyleAgent(BaseAgent):
    """
    Unified Investment Style Agent.

    Combines three investment styles into one comprehensive analysis:
    1. Value (30%): Margin of safety, strong fundamentals, moat indicators
    2. Growth (35%): Revenue/EPS growth, innovation, market opportunity
    3. Momentum (35%): Price trends, volume confirmation, relative strength

    This consolidation reduces agent count from 9 to 7 while maintaining
    full analytical coverage of investment styles.

    Attributes:
        None - uses context data only

    Example:
        agent = StyleAgent()
        signal = agent.analyze({
            "code": "600519",
            "stock_name": "贵州茅台",
            "financial_data": {...},
            "valuation_data": {...},
            "growth_data": {...},
            "price_data": {...},
            "technical_data": {...},
            "market_data": {...},
        })
    """

    # Weight distribution across three styles
    STYLE_WEIGHTS = {
        "value": 0.30,
        "growth": 0.35,
        "momentum": 0.35,
    }

    # Value sub-weights
    VALUE_WEIGHTS = {
        "valuation": 0.30,
        "fundamentals": 0.25,
        "moat": 0.20,
        "management": 0.15,
        "safety": 0.10,
    }

    # Growth sub-weights
    GROWTH_WEIGHTS = {
        "growth_metrics": 0.40,
        "innovation": 0.25,
        "market": 0.20,
        "valuation": 0.15,
    }

    # Momentum sub-weights
    MOMENTUM_WEIGHTS = {
        "price_momentum": 0.35,
        "trend": 0.25,
        "volume": 0.20,
        "relative_strength": 0.20,
    }

    def __init__(self):
        """Initialize the Style Agent."""
        super().__init__("StyleAgent")
        self._logger = logging.getLogger(__name__)

    def is_available(self) -> bool:
        """Always available - only requires context data."""
        return True

    @handle_errors("投资风格分析失败", default_return=None)
    async def analyze(self, context: dict[str, Any]) -> AgentSignal:
        """
        Execute unified investment style analysis (async).

        Args:
            context: Analysis context containing:
                - code: Stock code
                - stock_name: Stock name
                - current_price: Current price
                - financial_data: Financial metrics
                - valuation_data: Valuation inputs
                - growth_data: Growth metrics
                - price_data: Price history
                - technical_data: Technical indicators
                - market_data: Market comparison data

        Returns:
            AgentSignal with comprehensive style analysis
        """
        stock_code = context.get("code", "")
        self._logger.debug(f"[{stock_code}] StyleAgent开始投资风格分析")

        # Get data from context
        current_price = context.get("current_price", 0)
        financial_data = context.get("financial_data", {})
        valuation_data = context.get("valuation_data", {})
        growth_data = context.get("growth_data", {})
        price_data = context.get("price_data", {})
        technical_data = context.get("technical_data", {})
        market_data = context.get("market_data", {})

        # Calculate scores for each style
        value_score = self._analyze_value_style(current_price, financial_data, valuation_data)
        growth_score = self._analyze_growth_style(current_price, financial_data, growth_data)
        momentum_score = self._analyze_momentum_style(price_data, technical_data, market_data)

        # Calculate weighted total score
        total_score = (
            value_score * self.STYLE_WEIGHTS["value"]
            + growth_score * self.STYLE_WEIGHTS["growth"]
            + momentum_score * self.STYLE_WEIGHTS["momentum"]
        )

        # Generate signal based on score
        signal, confidence = self._score_to_signal(total_score)

        # Build comprehensive reasoning
        reasoning = self._build_reasoning(
            value_score,
            growth_score,
            momentum_score,
            financial_data,
            valuation_data,
            growth_data,
            price_data,
            technical_data,
        )

        self._logger.debug(
            f"[{stock_code}] StyleAgent分析完成: {signal} "
            f"(价值{value_score:.0f}/成长{growth_score:.0f}/动量{momentum_score:.0f}, "
            f"总分{total_score:.1f}/100, 置信度{confidence}%)"
        )

        return AgentSignal(
            agent_name=self.name,
            signal=signal,
            confidence=confidence,
            reasoning=reasoning,
            metadata={
                "total_score": round(total_score, 1),
                "value_score": round(value_score, 1),
                "growth_score": round(growth_score, 1),
                "momentum_score": round(momentum_score, 1),
                "value_breakdown": self._get_value_breakdown(valuation_data, financial_data),
                "growth_breakdown": self._get_growth_breakdown(growth_data),
                "momentum_breakdown": self._get_momentum_breakdown(price_data, technical_data),
            },
        )

    def _analyze_value_style(self, current_price: float, financial_data: dict, valuation_data: dict) -> float:
        """Analyze value investing characteristics (0-100 scale)."""
        if not financial_data:
            return 50.0  # Neutral if no data

        # Calculate sub-scores (0-10 each)
        valuation = self._score_valuation(current_price, valuation_data)
        fundamentals = self._score_fundamentals(financial_data)
        moat = self._score_moat(financial_data)
        management = self._score_management(financial_data)
        safety = FinancialScorer.analyze_safety(financial_data)

        # Weighted average, scaled to 0-100
        score = (
            valuation * self.VALUE_WEIGHTS["valuation"]
            + fundamentals * self.VALUE_WEIGHTS["fundamentals"]
            + moat * self.VALUE_WEIGHTS["moat"]
            + management * self.VALUE_WEIGHTS["management"]
            + safety * self.VALUE_WEIGHTS["safety"]
        ) * 10

        return min(score, 100)

    def _analyze_growth_style(self, current_price: float, financial_data: dict, growth_data: dict) -> float:
        """Analyze growth investing characteristics (0-100 scale)."""
        if not growth_data:
            return 50.0  # Neutral if no data

        # Check for detailed growth metrics
        has_detailed = any(
            growth_data.get(k) is not None and growth_data.get(k) != 0 for k in ["revenue_cagr", "eps_cagr", "fcf_cagr"]
        )

        if not has_detailed:
            # Use simplified analysis based on price momentum
            return self._analyze_basic_growth(growth_data)

        # Calculate sub-scores (0-10 each)
        growth_metrics = self._score_growth_metrics(growth_data)
        innovation = self._score_innovation(growth_data)
        market = self._score_market_opportunity(growth_data)
        valuation = self._score_growth_valuation(current_price, financial_data, growth_data)

        # Weighted average, scaled to 0-100
        score = (
            growth_metrics * self.GROWTH_WEIGHTS["growth_metrics"]
            + innovation * self.GROWTH_WEIGHTS["innovation"]
            + market * self.GROWTH_WEIGHTS["market"]
            + valuation * self.GROWTH_WEIGHTS["valuation"]
        ) * 10

        return min(score, 100)

    def _analyze_momentum_style(self, price_data: dict, technical_data: dict, market_data: dict) -> float:
        """Analyze momentum investing characteristics (0-100 scale)."""
        if not price_data:
            return 50.0  # Neutral if no data

        # Calculate sub-scores (0-10 each)
        price_momentum = self._score_price_momentum(price_data)
        trend = self._score_trend_strength(technical_data)
        volume = self._score_volume(technical_data)
        rs = self._score_relative_strength(market_data)

        # Weighted average, scaled to 0-100
        score = (
            price_momentum * self.MOMENTUM_WEIGHTS["price_momentum"]
            + trend * self.MOMENTUM_WEIGHTS["trend"]
            + volume * self.MOMENTUM_WEIGHTS["volume"]
            + rs * self.MOMENTUM_WEIGHTS["relative_strength"]
        ) * 10

        return min(score, 100)

    # Value Analysis Methods
    def _score_valuation(self, current_price: float, data: dict) -> int:
        """Score valuation metrics (0-10)."""
        score = 0
        if current_price <= 0:
            return 5

        # P/E scoring
        eps = data.get("eps", 0)
        if eps > 0:
            pe = current_price / eps
            if pe < 15:
                score += 3
            elif pe < 20:
                score += 2
            elif pe < 25:
                score += 1

        # P/B scoring
        bvps = data.get("book_value_per_share", 0)
        if bvps > 0:
            pb = current_price / bvps
            if pb < 2:
                score += 2
            elif pb < 3:
                score += 1

        # Margin of safety
        graham = self._calculate_graham_number(data)
        if graham > 0 and current_price > 0:
            mos = (graham - current_price) / current_price * 100
            if mos > 30:
                score += 3
            elif mos > 15:
                score += 2

        return min(score, 10)

    def _score_fundamentals(self, data: dict) -> int:
        """Score fundamental metrics (0-10)."""
        score = 0
        roe = data.get("roe", 0)
        if roe > 15:
            score += 3
        elif roe > 12:
            score += 2
        elif roe > 8:
            score += 1

        net_margin = data.get("net_margin", 0)
        if net_margin > 20:
            score += 2
        elif net_margin > 15:
            score += 1

        op_margin = data.get("operating_margin", 0)
        if op_margin > 15:
            score += 2
        elif op_margin > 10:
            score += 1

        roa = data.get("roa", 0)
        if roa > 8:
            score += 1

        return min(score, 10)

    def _score_moat(self, data: dict) -> int:
        """Score competitive moat (0-10)."""
        score = 0
        roe_consistency = data.get("roe_consistency", 0)
        if roe_consistency > 80:
            score += 3
        elif roe_consistency > 60:
            score += 2

        gross_margin = data.get("gross_margin", 0)
        if gross_margin > 40:
            score += 2
        elif gross_margin > 30:
            score += 1

        asset_turnover = data.get("asset_turnover", 0)
        if asset_turnover > 1.0:
            score += 1

        stability = data.get("performance_stability", 0)
        if stability > 70:
            score += 1

        return min(score, 10)

    def _score_management(self, data: dict) -> int:
        """Score management quality (0-10)."""
        score = 0
        if data.get("share_buybacks", False):
            score += 3
        if data.get("dividend_record", False):
            score += 2

        payout_ratio = data.get("payout_ratio", 0)
        if 0 < payout_ratio < 50:
            score += 2

        roic = data.get("roic", 0)
        if roic > 12:
            score += 3
        elif roic > 8:
            score += 2

        return min(score, 10)

    # Growth Analysis Methods
    def _score_growth_metrics(self, data: dict) -> int:
        """Score growth metrics (0-10)."""
        score = 0
        revenue_cagr = data.get("revenue_cagr", 0)
        if revenue_cagr > 30:
            score += 4
        elif revenue_cagr > 20:
            score += 3
        elif revenue_cagr > 15:
            score += 2
        elif revenue_cagr > 10:
            score += 1

        eps_cagr = data.get("eps_cagr", 0)
        if eps_cagr > 30:
            score += 3
        elif eps_cagr > 20:
            score += 2
        elif eps_cagr > 15:
            score += 1

        fcf_cagr = data.get("fcf_cagr", 0)
        if fcf_cagr > 20:
            score += 2
        elif fcf_cagr > 10:
            score += 1

        return min(score, 10)

    def _score_innovation(self, data: dict) -> int:
        """Score innovation metrics (0-10)."""
        score = 0
        rd_intensity = data.get("rd_intensity", 0)
        if rd_intensity > 15:
            score += 4
        elif rd_intensity > 10:
            score += 3
        elif rd_intensity > 5:
            score += 2

        gm_expansion = data.get("gross_margin_expansion", 0)
        if gm_expansion > 5:
            score += 2
        elif gm_expansion > 2:
            score += 1

        if data.get("operating_leverage", False):
            score += 2

        return min(score, 10)

    def _score_market_opportunity(self, data: dict) -> int:
        """Score market opportunity (0-10)."""
        score = 0
        tam = data.get("total_addressable_market", 0)
        if tam > 100:
            score += 3
        elif tam > 50:
            score += 2
        elif tam > 10:
            score += 1

        market_share = data.get("market_share", 0)
        if 0 < market_share < 10:
            score += 2

        if data.get("scalability", False):
            score += 3

        return min(score, 10)

    def _score_growth_valuation(self, price: float, fin_data: dict, growth_data: dict) -> int:
        """Score growth-adjusted valuation (0-10)."""
        score = 0
        if price <= 0:
            return 5

        eps = fin_data.get("eps", 0)
        growth_rate = growth_data.get("eps_cagr", 0)
        if eps > 0 and growth_rate > 0:
            peg = (price / eps) / growth_rate
            if peg < 1.0:
                score += 4
            elif peg < 1.5:
                score += 3
            elif peg < 2.0:
                score += 2

        revenue_cagr = growth_data.get("revenue_cagr", 0)
        if revenue_cagr > 25:
            score += 1

        return min(score, 10)

    def _analyze_basic_growth(self, growth_data: dict) -> float:
        """Simplified growth analysis when detailed data unavailable."""
        score = 50
        revenue_cagr = growth_data.get("revenue_cagr", 0)
        price_momentum = growth_data.get("price_momentum_1m", 0)
        forward_pe = growth_data.get("forward_pe", 0)

        if revenue_cagr > 30:
            score += 20
        elif revenue_cagr > 20:
            score += 15
        elif revenue_cagr > 10:
            score += 10
        elif revenue_cagr < -20:
            score -= 20
        elif revenue_cagr < -10:
            score -= 10

        if price_momentum > 10:
            score += 10
        elif price_momentum < -10:
            score -= 10

        if 0 < forward_pe < 20:
            score += 5
        elif forward_pe > 50:
            score -= 10

        return max(0, min(100, score))

    # Momentum Analysis Methods
    def _score_price_momentum(self, price_data: dict) -> int:
        """Score price momentum (0-10)."""
        score = 0
        ret_1m = self._get_return(price_data, 20)
        ret_3m = self._get_return(price_data, 60)
        ret_6m = self._get_return(price_data, 120)

        if ret_1m > 15:
            score += 3
        elif ret_1m > 8:
            score += 2
        elif ret_1m > 0:
            score += 1

        if ret_3m > 30:
            score += 3
        elif ret_3m > 15:
            score += 2
        elif ret_3m > 5:
            score += 1

        if ret_6m > 50:
            score += 2
        elif ret_6m > 25:
            score += 1

        if ret_1m > 0 and ret_3m > 0 and ret_6m > 0:
            score += 1

        return min(score, 10)

    def _score_trend_strength(self, technical_data: dict) -> int:
        """Score trend strength (0-10)."""
        score = 0
        adx = technical_data.get("adx", 0)
        if adx > 30:
            score += 3
        elif adx > 25:
            score += 2
        elif adx > 20:
            score += 1

        ma20 = technical_data.get("ma20", 0)
        ma60 = technical_data.get("ma60", 0)
        ma120 = technical_data.get("ma120", 0)
        current = technical_data.get("current_price", 0)

        if all([ma20, ma60, ma120, current]):
            if current > ma20 > ma60 > ma120:
                score += 3
            elif current > ma20 > ma60:
                score += 2
            elif current > ma20:
                score += 1

        return min(score, 10)

    def _score_volume(self, technical_data: dict) -> int:
        """Score volume patterns (0-10)."""
        score = 0
        vol_momentum = technical_data.get("volume_momentum", 0)
        if vol_momentum > 150:
            score += 3
        elif vol_momentum > 120:
            score += 2
        elif vol_momentum > 100:
            score += 1

        if technical_data.get("obv_trend") == "up":
            score += 2

        up_ratio = technical_data.get("up_volume_ratio", 0.5)
        if up_ratio > 0.6:
            score += 2
        elif up_ratio > 0.55:
            score += 1

        if technical_data.get("volume_spike", False):
            score += 2

        return min(score, 10)

    def _score_relative_strength(self, market_data: dict) -> int:
        """Score relative strength (0-10)."""
        score = 0
        rs_ratio = market_data.get("relative_strength_ratio", 1.0)
        if rs_ratio > 1.2:
            score += 3
        elif rs_ratio > 1.1:
            score += 2
        elif rs_ratio > 1.0:
            score += 1

        if market_data.get("outperforming_in_down_market", False):
            score += 3

        if market_data.get("rs_trend") == "improving":
            score += 2

        rs_rank = market_data.get("relative_strength_rank", 50)
        if rs_rank > 80:
            score += 2
        elif rs_rank > 60:
            score += 1

        return min(score, 10)

    # Helper Methods
    def _calculate_graham_number(self, data: dict) -> float:
        """Calculate Graham Number."""
        eps = data.get("eps", 0)
        bvps = data.get("book_value_per_share", 0)
        if eps <= 0 or bvps <= 0:
            return 0
        return math.sqrt(22.5 * eps * bvps)

    def _get_return(self, price_data: dict, days: int) -> float:
        """Calculate return over specified days."""
        try:
            if isinstance(price_data, dict) and "close" in price_data:
                closes = price_data["close"]
            else:
                return 0

            if len(closes) < days + 1:
                return 0

            current = closes[-1]
            past = closes[-(days + 1)]

            if past <= 0:
                return 0

            return (current - past) / past * 100
        except Exception:
            logger.debug(f"Error calculating change in {days} days")
            return 0

    def _score_to_signal(self, score: float) -> tuple[SignalType, int]:
        """Convert total score to signal and confidence."""
        if score >= 65:
            confidence = min(100, int(60 + (score - 65) * 1.1))
            return (SignalType.BUY, confidence)
        elif score <= 30:
            confidence = min(100, int(60 + (30 - score) * 1.1))
            return (SignalType.SELL, confidence)
        else:
            distance = abs(score - 50)
            confidence = max(30, int(40 + distance))
            return (SignalType.HOLD, confidence)

    def _build_reasoning(
        self,
        value_score: float,
        growth_score: float,
        momentum_score: float,
        financial_data: dict,
        valuation_data: dict,
        growth_data: dict,
        price_data: dict,
        technical_data: dict,
    ) -> str:
        """Build comprehensive reasoning string."""
        parts = []

        # Value component
        if value_score >= 70:
            parts.append(f"价值属性强({value_score:.0f})")
        elif value_score <= 40:
            parts.append(f"价值属性弱({value_score:.0f})")

        # Growth component
        if growth_score >= 70:
            parts.append(f"成长属性强({growth_score:.0f})")
        elif growth_score <= 40:
            parts.append(f"成长属性弱({growth_score:.0f})")

        # Momentum component
        if momentum_score >= 70:
            parts.append(f"动量属性强({momentum_score:.0f})")
        elif momentum_score <= 40:
            parts.append(f"动量属性弱({momentum_score:.0f})")

        if not parts:
            parts.append(f"风格均衡(价值{value_score:.0f}/成长{growth_score:.0f}/动量{momentum_score:.0f})")

        return " / ".join(parts)

    def _get_value_breakdown(self, valuation_data: dict, financial_data: dict) -> dict:
        """Get value analysis breakdown."""
        return {
            "pe_ratio": valuation_data.get("pe_ratio"),
            "pb_ratio": valuation_data.get("pb_ratio"),
            "roe": financial_data.get("roe"),
            "net_margin": financial_data.get("net_margin"),
        }

    def _get_growth_breakdown(self, growth_data: dict) -> dict:
        """Get growth analysis breakdown."""
        return {
            "revenue_cagr": growth_data.get("revenue_cagr"),
            "price_momentum_1m": growth_data.get("price_momentum_1m"),
            "forward_pe": growth_data.get("forward_pe"),
        }

    def _get_momentum_breakdown(self, price_data: dict, technical_data: dict) -> dict:
        """Get momentum analysis breakdown."""
        return {
            "returns_1m": self._get_return(price_data, 20),
            "returns_3m": self._get_return(price_data, 60),
            "adx": technical_data.get("adx"),
            "volume_momentum": technical_data.get("volume_momentum"),
        }
