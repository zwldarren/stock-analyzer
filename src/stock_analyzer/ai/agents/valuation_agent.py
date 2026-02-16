"""
Valuation Agent Module

Specialized agent for company valuation analysis.

This agent focuses exclusively on:
- Multiple valuation methods (DCF, Graham Number, Relative Valuation)
- Margin of safety calculation
- Overvalued/undervalued assessment
- Industry-relative valuation comparison

All calculations use ONLY actual data, no estimations or defaults.
"""

import logging
import math
from typing import Any

from stock_analyzer.exceptions import handle_errors

from .base import AgentSignal, BaseAgent, SignalType

logger = logging.getLogger(__name__)


class ValuationAgent(BaseAgent):
    """
    Valuation Agent for determining fair value and margin of safety.

    This agent uses multiple valuation methods with ACTUAL DATA ONLY:
    1. DCF (Discounted Cash Flow) - requires FCF, shares, growth_rate, discount_rate
    2. Owner Earnings (Buffett Method) - requires net_income, shares, etc.
    3. Graham Number - requires EPS and Book Value per Share
    4. Relative Valuation (P/E, P/B comparison) - requires industry multiples
    5. PE-based Valuation - requires industry PE
    6. PB-based Valuation - requires industry PB

    NO default values or estimations are used. If data is missing, that method is skipped.

    Attributes:
        None - uses context data only

    Example:
        agent = ValuationAgent()
        signal = agent.analyze({
            "code": "600519",
            "stock_name": "贵州茅台",
            "current_price": 1500.0,
            "valuation_data": {...}
        })
    """

    def __init__(self):
        """Initialize the Valuation Agent."""
        super().__init__("ValuationAgent")
        self._logger = logging.getLogger(__name__)

    def is_available(self) -> bool:
        """Always available - only requires context data."""
        return True

    @handle_errors("估值分析失败", default_return=None)
    def analyze(self, context: dict[str, Any]) -> AgentSignal:
        """
        Execute valuation analysis using ONLY actual data.

        Args:
            context: Analysis context containing:
                - code: Stock code
                - stock_name: Stock name
                - current_price: Current stock price
                - valuation_data: Valuation inputs dict with keys:
                    - eps: Earnings per share
                    - book_value_per_share: Book value per share
                    - free_cash_flow: Free cash flow
                    - shares_outstanding: Shares outstanding
                    - growth_rate: Expected growth rate
                    - discount_rate: Discount rate
                    - terminal_multiple: Terminal multiple
                    - industry_pe: Industry average P/E
                    - industry_pb: Industry average P/B

        Returns:
            AgentSignal with valuation-based trading signal
        """
        stock_code = context.get("code", "")
        current_price = context.get("current_price", 0)

        self._logger.debug(f"[{stock_code}] ValuationAgent开始估值分析")

        # Get valuation data from context
        valuation_data = context.get("valuation_data", {})

        # 记录原始估值数据用于调试
        pe_ratio = valuation_data.get("pe_ratio", 0)
        pb_ratio = valuation_data.get("pb_ratio", 0)
        eps = valuation_data.get("eps", 0)
        bvps = valuation_data.get("book_value_per_share", 0)
        self._logger.debug(f"[{stock_code}] 估值数据: PE={pe_ratio}, PB={pb_ratio}, EPS={eps}, BVPS={bvps}")

        if not valuation_data or current_price <= 0:
            return AgentSignal(
                agent_name=self.name,
                signal=SignalType.HOLD,
                confidence=0,
                reasoning="无估值数据或当前价格",
                metadata={"error": "no_valuation_data"},
            )

        # Calculate valuations using different methods (ONLY with actual data)
        valuations = {}
        available_methods = []

        # 1. DCF Valuation - requires actual FCF, shares, growth_rate, discount_rate
        dcf_value = self._calculate_dcf(valuation_data)
        if dcf_value > 0:
            valuations["dcf"] = dcf_value
            available_methods.append("dcf")

        # 2. Owner Earnings (Buffett Method) - requires actual data
        oe_value = self._calculate_owner_earnings(valuation_data)
        if oe_value > 0:
            valuations["owner_earnings"] = oe_value
            available_methods.append("owner_earnings")

        # 3. Graham Number - requires actual EPS and BVPS
        graham_value = self._calculate_graham_number(valuation_data)
        if graham_value > 0:
            valuations["graham_number"] = graham_value
            available_methods.append("graham_number")

        # 4. Relative Valuation - requires actual industry multiples
        relative_value = self._calculate_relative_valuation(valuation_data)
        if relative_value > 0:
            valuations["relative"] = relative_value
            available_methods.append("relative")

        # 5. PE-based Valuation - requires actual industry PE
        pe_value = self._calculate_pe_valuation(valuation_data)
        if pe_value > 0:
            valuations["pe_based"] = pe_value
            available_methods.append("pe_based")

        # 6. PB-based Valuation - requires actual industry PB
        pb_value = self._calculate_pb_valuation(valuation_data)
        if pb_value > 0:
            valuations["pb_based"] = pb_value
            available_methods.append("pb_based")

        if not valuations:
            # 没有可用的估值方法，返回中性信号
            self._logger.debug(f"[{stock_code}] 无足够数据进行任何估值计算")
            return AgentSignal(
                agent_name=self.name,
                signal=SignalType.HOLD,
                confidence=0,
                reasoning="数据不足，无法计算估值",
                metadata={
                    "error": "insufficient_data_for_valuation",
                    "available_data": list(valuation_data.keys()),
                },
            )

        # Calculate weighted average fair value using equal weights for available methods
        method_count = len(valuations)
        equal_weight = 1.0 / method_count

        weighted_value = sum(value * equal_weight for value in valuations.values())
        fair_value = weighted_value

        # Calculate margin of safety
        margin_of_safety = (fair_value - current_price) / current_price if current_price > 0 else 0

        # Generate signal based on margin of safety
        signal, confidence = self._margin_to_signal(margin_of_safety)

        # Build reasoning
        reasoning_parts = []
        reasoning_parts.append(f"公允价值¥{fair_value:.2f}")
        reasoning_parts.append(f"当前价格¥{current_price:.2f}")
        reasoning_parts.append(f"安全边际{margin_of_safety * 100:+.1f}%")
        reasoning_parts.append(f"基于{method_count}种方法: {', '.join(available_methods)}")

        if "graham_number" in valuations:
            reasoning_parts.append(f"Graham数¥{valuations['graham_number']:.2f}")

        reasoning = " / ".join(reasoning_parts)

        self._logger.debug(
            f"[{stock_code}] ValuationAgent分析完成: {signal} "
            f"(公允价值¥{fair_value:.2f}, 安全边际{margin_of_safety * 100:+.1f}%, "
            f"置信度{confidence}%, 方法:{method_count}种)"
        )

        metadata = {
            "fair_value": round(fair_value, 2),
            "current_price": current_price,
            "margin_of_safety": round(margin_of_safety * 100, 1),
            "valuations": {k: round(v, 2) for k, v in valuations.items()},
            "available_methods": available_methods,
            "method_count": method_count,
            "eps": valuation_data.get("eps") if valuation_data.get("eps", 0) > 0 else None,
            "book_value_per_share": valuation_data.get("book_value_per_share")
            if valuation_data.get("book_value_per_share", 0) > 0
            else None,
            "pe_ratio": current_price / valuation_data.get("eps", 1) if valuation_data.get("eps", 0) > 0 else None,
            "pb_ratio": current_price / valuation_data.get("book_value_per_share", 1)
            if valuation_data.get("book_value_per_share", 0) > 0
            else None,
        }

        industry_pe = valuation_data.get("industry_pe")
        industry_pb = valuation_data.get("industry_pb")
        industry_name = valuation_data.get("industry_name")
        pb_deviation = valuation_data.get("pb_deviation_from_industry")

        if industry_name:
            metadata["industry_name"] = industry_name
        if industry_pe and industry_pe > 0:
            metadata["industry_pe"] = round(industry_pe, 2)
        if industry_pb and industry_pb > 0:
            metadata["industry_pb"] = round(industry_pb, 2)
        if pb_deviation is not None:
            metadata["pb_deviation_from_industry"] = pb_deviation

        return AgentSignal(
            agent_name=self.name,
            signal=signal,
            confidence=confidence,
            reasoning=reasoning,
            metadata=metadata,
        )

    def _calculate_dcf(self, data: dict[str, Any]) -> float:
        """
        Calculate DCF (Discounted Cash Flow) valuation.

        Three-stage model using ONLY actual data:
        - Stage 1 (Years 1-5): High growth
        - Stage 2 (Years 6-10): Transition
        - Stage 3: Terminal value

        Returns 0 if any required data is missing.
        """
        fcf = data.get("free_cash_flow", 0)
        shares = data.get("shares_outstanding", 0)
        growth = data.get("growth_rate")
        discount = data.get("discount_rate")
        terminal_multiple = data.get("terminal_multiple")

        # All required data must be present and valid
        if fcf <= 0 or shares <= 0 or growth is None or discount is None or terminal_multiple is None:
            return 0

        if growth <= 0 or discount <= 0 or terminal_multiple <= 0:
            return 0

        # Cap growth rate at reasonable levels
        growth = min(growth, 0.20)  # Max 20%

        # Stage 1: Years 1-5
        stage1_value = 0
        current_fcf = fcf
        for year in range(1, 6):
            current_fcf *= 1 + growth
            stage1_value += current_fcf / ((1 + discount) ** year)

        # Stage 2: Years 6-10 (transition)
        stage2_value = 0
        terminal_growth = 0.03  # 3% terminal growth
        for year in range(6, 11):
            # Linear fade of growth to terminal rate
            fade_factor = (year - 5) / 5
            year_growth = growth * (1 - fade_factor) + terminal_growth * fade_factor
            current_fcf *= 1 + year_growth
            stage2_value += current_fcf / ((1 + discount) ** year)

        # Terminal value
        terminal_fcf = current_fcf * (1 + terminal_growth)
        terminal_value = terminal_fcf * terminal_multiple / ((1 + discount) ** 10)

        total_value = stage1_value + stage2_value + terminal_value
        value_per_share = total_value / shares

        return value_per_share

    def _calculate_owner_earnings(self, data: dict[str, Any]) -> float:
        """
        Calculate Owner Earnings valuation (Buffett Method).

        Owner Earnings = Net Income + Depreciation - Maintenance CapEx - WC Changes

        Returns 0 if required data is missing.
        """
        net_income = data.get("net_income", 0)
        depreciation = data.get("depreciation")
        maintenance_capex = data.get("maintenance_capex")
        wc_changes = data.get("working_capital_changes")
        shares = data.get("shares_outstanding", 0)
        growth = data.get("growth_rate")

        # 必须所有数据都存在才计算
        if (
            net_income <= 0
            or shares <= 0
            or depreciation is None
            or maintenance_capex is None
            or wc_changes is None
            or growth is None
        ):
            return 0

        # Calculate owner earnings
        owner_earnings = net_income + depreciation - maintenance_capex - wc_changes

        if owner_earnings <= 0:
            return 0

        # 使用实际的 required_return 如果提供，否则不计算
        required_return = data.get("required_return")
        if required_return is None or required_return <= 0:
            return 0

        margin_of_safety = data.get("margin_of_safety")
        if margin_of_safety is None:
            return 0

        # Simple 5-year DCF of owner earnings
        total_value = 0
        current_oe = owner_earnings
        for year in range(1, 6):
            current_oe *= 1 + growth
            total_value += current_oe / ((1 + required_return) ** year)

        # Terminal value (conservative)
        terminal_multiple = data.get("owner_earnings_terminal_multiple", 10)
        terminal_value = current_oe * terminal_multiple / ((1 + required_return) ** 5)

        total_value += terminal_value
        value_per_share = (total_value / shares) * (1 - margin_of_safety)

        return value_per_share

    def _calculate_graham_number(self, data: dict[str, Any]) -> float:
        """
        Calculate Graham Number.

        Formula: sqrt(22.5 * EPS * Book Value per Share)

        Returns 0 if EPS or BVPS is missing or invalid.
        """
        eps = data.get("eps", 0)
        bvps = data.get("book_value_per_share", 0)

        # 必须同时有EPS和BVPS才计算
        if eps <= 0 or bvps <= 0:
            return 0

        # Graham Number formula
        graham_number = math.sqrt(22.5 * eps * bvps)

        return graham_number

    def _calculate_relative_valuation(self, data: dict[str, Any]) -> float:
        """
        Calculate relative valuation based on industry multiples.

        Uses P/E and P/B ratios compared to industry averages.
        Returns 0 if industry multiples are not available.
        """
        eps = data.get("eps", 0)
        bvps = data.get("book_value_per_share", 0)
        industry_pe = data.get("industry_pe", 0)
        industry_pb = data.get("industry_pb", 0)
        industry_name = data.get("industry_name", "")

        # 必须至少有一个有效的行业倍数
        if industry_pe <= 0 and industry_pb <= 0:
            return 0

        # P/E based valuation (only if both EPS and industry PE available)
        pe_value = 0
        if eps > 0 and industry_pe > 0:
            pe_value = eps * industry_pe
            self._logger.debug(f"相对估值(PE): EPS={eps} * 行业PE={industry_pe} = {pe_value}")

        # P/B based valuation (only if both BVPS and industry PB available)
        pb_value = 0
        if bvps > 0 and industry_pb > 0:
            pb_value = bvps * industry_pb
            self._logger.debug(f"相对估值(PB): BVPS={bvps} * 行业PB={industry_pb} = {pb_value}")

        # Average if both available
        if pe_value > 0 and pb_value > 0:
            avg_value = (pe_value + pb_value) / 2
            self._logger.debug(
                f"行业相对估值({industry_name}): PE估值={pe_value:.2f}, PB估值={pb_value:.2f}, 平均={avg_value:.2f}"
            )
            return avg_value
        elif pe_value > 0:
            self._logger.debug(f"行业相对估值(仅PE): {pe_value:.2f}")
            return pe_value
        elif pb_value > 0:
            self._logger.debug(f"行业相对估值(仅PB): {pb_value:.2f}")
            return pb_value

        return 0

    def _calculate_pe_valuation(self, data: dict[str, Any]) -> float:
        """
        Calculate valuation based on P/E ratio.

        Uses industry average P/E ONLY - no defaults.
        Returns 0 if industry PE is not available.
        """
        eps = data.get("eps", 0)
        industry_pe = data.get("industry_pe", 0)
        industry_name = data.get("industry_name", "")

        # 必须有EPS和有效的行业PE
        if eps <= 0 or industry_pe <= 0:
            return 0

        value = eps * industry_pe
        self._logger.info(f"PE估值: EPS={eps:.2f} * 行业PE({industry_name})={industry_pe:.2f} = {value:.2f}")
        return value

    def _calculate_pb_valuation(self, data: dict[str, Any]) -> float:
        """
        Calculate valuation based on P/B ratio.

        Uses industry average P/B ONLY - no defaults.
        Returns 0 if industry PB is not available.
        """
        bvps = data.get("book_value_per_share", 0)
        industry_pb = data.get("industry_pb", 0)
        industry_name = data.get("industry_name", "")
        pb_deviation = data.get("pb_deviation_from_industry")

        # 必须有BVPS和有效的行业PB
        if bvps <= 0 or industry_pb <= 0:
            return 0

        value = bvps * industry_pb

        # 记录偏离度信息
        if pb_deviation is not None:
            self._logger.debug(
                f"PB估值: BVPS={bvps:.2f} * 行业PB({industry_name})={industry_pb:.2f} = {value:.2f} "
                f"(当前PB偏离行业{pb_deviation:+.1f}%)"
            )
        else:
            self._logger.debug(f"PB估值: BVPS={bvps:.2f} * 行业PB({industry_name})={industry_pb:.2f} = {value:.2f}")

        return value

    def _margin_to_signal(self, margin: float) -> tuple[SignalType, int]:
        """
        Convert margin of safety to signal and confidence.

        Signal thresholds:
        - BUY: margin >= 15% (undervalued)
        - SELL: margin <= -15% (overvalued)
        - HOLD: -15% < margin < 15% (fairly valued)

        Confidence based on magnitude of mispricing.
        """
        if margin >= 0.15:
            # Undervalued - buy zone
            confidence = min(100, int(50 + margin * 200))  # 15% -> 80%, 25% -> 100%
            return (SignalType.BUY, confidence)
        elif margin <= -0.15:
            # Overvalued - sell zone
            confidence = min(100, int(50 + abs(margin) * 200))
            return (SignalType.SELL, confidence)
        else:
            # Fairly valued - hold zone
            confidence = max(30, int(50 - abs(margin) * 100))
            return (SignalType.HOLD, confidence)
