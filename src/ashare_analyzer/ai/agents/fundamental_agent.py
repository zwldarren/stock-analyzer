"""
Fundamental Analysis Agent Module

Specialized agent for analyzing company fundamentals and financial metrics.

This agent focuses exclusively on:
- Financial statement analysis (ROE, ROA, margins)
- Balance sheet health (debt ratios, current ratio)
- Earnings consistency and growth
- Profitability metrics

Inspired by ai-hedge-fund's Fundamentals Analyst.
"""

import logging
from typing import Any

from ashare_analyzer.exceptions import handle_errors
from ashare_analyzer.models import AgentSignal, SignalType

from .base import BaseAgent
from .utils import FinancialScorer

logger = logging.getLogger(__name__)


class FundamentalAgent(BaseAgent):
    """
    Fundamental Analysis Agent for financial metrics assessment.

    This agent analyzes:
    - Profitability metrics (ROE, ROA, net margin)
    - Financial health (debt-to-equity, current ratio)
    - Earnings stability and growth
    - Balance sheet strength

    All analysis is quantitative without LLM involvement.

    Attributes:
        None - uses context data only

    Example:
        agent = FundamentalAgent()
        signal = agent.analyze({
            "code": "600519",
            "stock_name": "贵州茅台",
            "financial_data": {...}
        })
    """

    def __init__(self):
        """Initialize the Fundamental Agent."""
        super().__init__("FundamentalAgent")
        self._logger = logging.getLogger(__name__)

    def is_available(self) -> bool:
        """Always available - only requires context data."""
        return True

    @handle_errors("基本面分析失败", default_return=None)
    async def analyze(self, context: dict[str, Any]) -> AgentSignal:
        """
        Execute fundamental analysis (async).

        Args:
            context: Analysis context containing:
                - code: Stock code
                - stock_name: Stock name
                - financial_data: Financial metrics dict with keys:
                    - roe: Return on equity
                    - roa: Return on assets
                    - net_margin: Net profit margin
                    - gross_margin: Gross profit margin
                    - debt_to_equity: Debt to equity ratio
                    - current_ratio: Current ratio
                    - revenue_growth: Revenue growth rate
                    - earnings_growth: Earnings growth rate
                    - eps_consistency: EPS consistency score

        Returns:
            AgentSignal with fundamental-based trading signal
        """
        stock_code = context.get("code", "")

        self._logger.debug(f"[{stock_code}] FundamentalAgent开始基本面分析")

        # Get financial data from context
        financial_data = context.get("financial_data", {})

        if not financial_data:
            return AgentSignal(
                agent_name=self.name,
                signal=SignalType.HOLD,
                confidence=0,
                reasoning="无财务数据可用",
                metadata={"error": "no_financial_data"},
            )

        # 检查是否有基本财务数据（ROE等）还是只有PE/PB
        has_basic_metrics = any(
            financial_data.get(k) is not None and financial_data.get(k) != 0
            for k in ["roe", "roa", "net_margin", "gross_margin", "debt_to_equity"]
        )

        # 如果只有PE/PB等基础数据，进行简化分析
        if not has_basic_metrics:
            return self._analyze_with_basic_data(stock_code, financial_data)

        # Calculate scores for each dimension
        profitability_score = self._analyze_profitability(financial_data)
        health_score = self._analyze_financial_health(financial_data)
        growth_score = self._analyze_growth(financial_data)
        consistency_score = self._analyze_consistency(financial_data)

        # Calculate weighted total score (max 100)
        total_score = profitability_score * 0.35 + health_score * 0.30 + growth_score * 0.20 + consistency_score * 0.15

        # Generate signal based on score thresholds
        signal, confidence = self._score_to_signal(total_score)

        # Build reasoning
        reasoning_parts = []
        if profitability_score >= 8:
            reasoning_parts.append(f"盈利能力优秀({profitability_score}/10)")
        elif profitability_score >= 5:
            reasoning_parts.append(f"盈利能力良好({profitability_score}/10)")
        else:
            reasoning_parts.append(f"盈利能力较弱({profitability_score}/10)")

        if health_score >= 8:
            reasoning_parts.append(f"财务健康优秀({health_score}/10)")
        elif health_score >= 5:
            reasoning_parts.append(f"财务健康良好({health_score}/10)")
        else:
            reasoning_parts.append(f"财务健康需关注({health_score}/10)")

        reasoning = " / ".join(reasoning_parts)

        self._logger.debug(
            f"[{stock_code}] FundamentalAgent分析完成: {signal} (总分{total_score:.1f}/100, 置信度{confidence}%)"
        )

        return AgentSignal(
            agent_name=self.name,
            signal=signal,
            confidence=confidence,
            reasoning=reasoning,
            metadata={
                "total_score": round(total_score, 1),
                "profitability_score": profitability_score,
                "health_score": health_score,
                "growth_score": growth_score,
                "consistency_score": consistency_score,
                "roe": financial_data.get("roe"),
                "net_margin": financial_data.get("net_margin"),
                "debt_to_equity": financial_data.get("debt_to_equity"),
                "current_ratio": financial_data.get("current_ratio"),
                "revenue_growth": financial_data.get("revenue_growth"),
            },
        )

    def _analyze_profitability(self, data: dict[str, Any]) -> int:
        """
        Analyze profitability metrics (0-10 scale).

        Scoring:
        - ROE > 15%: +3, > 10%: +2, > 5%: +1
        - Net Margin > 20%: +3, > 15%: +2, > 10%: +1
        - Gross Margin > 40%: +2, > 30%: +1
        - ROA > 8%: +2, > 5%: +1
        """
        score = 0

        # ROE scoring
        roe = data.get("roe", 0)
        if roe > 15:
            score += 3
        elif roe > 10:
            score += 2
        elif roe > 5:
            score += 1

        # Net margin scoring
        net_margin = data.get("net_margin", 0)
        if net_margin > 20:
            score += 3
        elif net_margin > 15:
            score += 2
        elif net_margin > 10:
            score += 1

        # Gross margin scoring
        gross_margin = data.get("gross_margin", 0)
        if gross_margin > 40:
            score += 2
        elif gross_margin > 30:
            score += 1

        # ROA scoring
        roa = data.get("roa", 0)
        if roa > 8:
            score += 2
        elif roa > 5:
            score += 1

        return min(score, 10)

    def _analyze_financial_health(self, data: dict[str, Any]) -> int:
        """Analyze financial health metrics (0-10 scale)."""
        return FinancialScorer.analyze_financial_health(data)

    def _analyze_growth(self, data: dict[str, Any]) -> int:
        """
        Analyze growth metrics (0-10 scale).

        Scoring:
        - Revenue Growth > 20%: +3, > 10%: +2, > 5%: +1
        - Earnings Growth > 20%: +3, > 10%: +2, > 5%: +1
        - Book Value Growth > 15%: +2, > 10%: +1
        """
        score = 0

        # Revenue growth scoring
        revenue_growth = data.get("revenue_growth", 0)
        if revenue_growth > 20:
            score += 3
        elif revenue_growth > 10:
            score += 2
        elif revenue_growth > 5:
            score += 1

        # Earnings growth scoring
        earnings_growth = data.get("earnings_growth", 0)
        if earnings_growth > 20:
            score += 3
        elif earnings_growth > 10:
            score += 2
        elif earnings_growth > 5:
            score += 1

        # Book value growth scoring
        bv_growth = data.get("book_value_growth", 0)
        if bv_growth > 15:
            score += 2
        elif bv_growth > 10:
            score += 1

        return min(score, 10)

    def _analyze_consistency(self, data: dict[str, Any]) -> int:
        """
        Analyze earnings consistency (0-10 scale).

        Scoring:
        - EPS consistency > 80%: +4, > 60%: +2
        - Revenue consistency > 80%: +3, > 60%: +1
        - Dividend track record: +3
        """
        score = 0

        # EPS consistency scoring
        eps_consistency = data.get("eps_consistency", 0)
        if eps_consistency > 80:
            score += 4
        elif eps_consistency > 60:
            score += 2

        # Revenue consistency scoring
        revenue_consistency = data.get("revenue_consistency", 0)
        if revenue_consistency > 80:
            score += 3
        elif revenue_consistency > 60:
            score += 1

        # Dividend track record
        dividend_record = data.get("dividend_record", False)
        if dividend_record:
            score += 3

        return min(score, 10)

    def _score_to_signal(self, score: float) -> tuple[SignalType, int]:
        """
        Convert fundamental score to signal and confidence.

        Signal thresholds:
        - BUY: score >= 70 (strong fundamentals)
        - SELL: score <= 30 (weak fundamentals)
        - HOLD: 30 < score < 70 (mixed fundamentals)

        Confidence calculation:
        - Based on distance from neutral zone (50)
        """
        if score >= 70:
            # Strong buy zone
            confidence = min(100, int(50 + (score - 70) * 1.5))
            return (SignalType.BUY, confidence)
        elif score <= 30:
            # Strong sell zone
            confidence = min(100, int(50 + (30 - score) * 1.5))
            return (SignalType.SELL, confidence)
        else:
            # Hold zone - confidence based on certainty
            distance_from_neutral = abs(score - 50)
            confidence = max(30, int(50 + distance_from_neutral))
            return (SignalType.HOLD, confidence)

    def _analyze_with_basic_data(self, stock_code: str, financial_data: dict[str, Any]) -> AgentSignal:
        """
        当只有基础数据（PE/PB等）时进行简化分析。

        基于PE/PB、波动率等指标给出基本面评估。
        """
        pe_ratio = financial_data.get("pe_ratio", 0)
        pb_ratio = financial_data.get("pb_ratio", 0)
        volatility = financial_data.get("volatility", 0)
        price_momentum = financial_data.get("price_momentum_20d", 0)

        # 简化评分（0-100）
        score = 50  # 中性起点

        # PE评分
        if pe_ratio > 0:
            if pe_ratio < 15:
                score += 15
            elif pe_ratio < 25:
                score += 5
            elif pe_ratio > 50:
                score -= 15
            elif pe_ratio > 30:
                score -= 5

        # PB评分
        if pb_ratio > 0:
            if pb_ratio < 1.5:
                score += 10
            elif pb_ratio > 5:
                score -= 10

        # 波动率评分（低波动加分）
        if volatility > 0:
            if volatility < 20:
                score += 5
            elif volatility > 50:
                score -= 5

        # 价格动量评分
        if price_momentum > 10:
            score += 5
        elif price_momentum < -10:
            score -= 5

        # 限制分数范围
        score = max(0, min(100, score))

        # 生成信号
        signal, confidence = self._score_to_signal(score)

        # 构建理由
        reasoning_parts = []
        if pe_ratio > 0:
            if pe_ratio < 15:
                reasoning_parts.append(f"PE{pe_ratio:.1f}偏低")
            elif pe_ratio > 30:
                reasoning_parts.append(f"PE{pe_ratio:.1f}偏高")
            else:
                reasoning_parts.append(f"PE{pe_ratio:.1f}合理")

        if pb_ratio > 0:
            if pb_ratio < 1.5:
                reasoning_parts.append(f"PB{pb_ratio:.1f}偏低")
            elif pb_ratio > 3:
                reasoning_parts.append(f"PB{pb_ratio:.1f}偏高")

        if volatility > 0:
            reasoning_parts.append(f"波动率{volatility:.1f}%")

        reasoning = " / ".join(reasoning_parts) if reasoning_parts else "基于有限数据的基础分析"

        self._logger.debug(
            f"[{stock_code}] FundamentalAgent简化分析完成: {signal} (基础评分{score:.1f}/100, 置信度{confidence}%)"
        )

        return AgentSignal(
            agent_name=self.name,
            signal=signal,
            confidence=confidence,
            reasoning=reasoning,
            metadata={
                "analysis_type": "basic",
                "score": round(score, 1),
                "pe_ratio": pe_ratio,
                "pb_ratio": pb_ratio,
                "volatility": volatility,
                "price_momentum_20d": price_momentum,
            },
        )
