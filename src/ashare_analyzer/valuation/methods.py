"""Valuation methods for A-share stock analysis."""

import math
from dataclasses import dataclass

from ashare_analyzer.constants import (
    A_SHARE_MARKET_PREMIUM,
    A_SHARE_RISK_FREE_RATE,
    A_SHARE_RISK_PREMIUM,
    DEFAULT_PB,
    DEFAULT_PS,
)


@dataclass
class ValuationResult:
    """
    Result of a valuation calculation.

    Attributes:
        value: The calculated fair value (0 if calculation was skipped)
        method: Name of the valuation method used
        skip_reason: Reason why calculation was skipped (None if successful)
    """

    value: float
    method: str
    skip_reason: str | None = None


def calculate_adjusted_graham(eps: float, bvps: float, roe: float = 10.0) -> ValuationResult:
    """
    Calculate fair value using adjusted Graham formula for A-share market.

    Formula: sqrt(A_share_factor × EPS × BVPS)
    - A_share_factor = 22.5 × A_SHARE_MARKET_PREMIUM × quality_adjustment
    - Quality adjustment: ROE > 15% → 1.2, ROE 10-15% → 1.0, ROE < 10% → 0.8

    Args:
        eps: Earnings per share (must be positive)
        bvps: Book value per share (must be positive)
        roe: Return on equity in percent (default 10.0)

    Returns:
        ValuationResult with fair value or skip_reason if inputs are invalid
    """
    method_name = "adjusted_graham"

    # Validate inputs
    if eps <= 0:
        return ValuationResult(value=0.0, method=method_name, skip_reason=f"EPS must be positive, got {eps}")

    if bvps <= 0:
        return ValuationResult(value=0.0, method=method_name, skip_reason=f"BVPS must be positive, got {bvps}")

    # Calculate quality adjustment based on ROE
    if roe > 15:
        quality_adjustment = 1.2
    elif roe >= 10:
        quality_adjustment = 1.0
    else:
        quality_adjustment = 0.8

    # Calculate A-share factor
    # Graham's original factor is 22.5 (for P/E × P/B = 22.5 for fair value)
    # Adjusted for A-share market premium and quality
    a_share_factor = 22.5 * A_SHARE_MARKET_PREMIUM * quality_adjustment

    # Calculate fair value using Graham formula: sqrt(factor × EPS × BVPS)
    fair_value = math.sqrt(a_share_factor * eps * bvps)

    return ValuationResult(value=fair_value, method=method_name)


def calculate_peg_valuation(eps: float, pe_ratio: float, growth_rate: float) -> ValuationResult:
    """
    Calculate fair value using PEG (Price/Earnings-to-Growth) method.

    Fair PEG determination:
    - Growth > 30% → Fair PEG = 0.8 (high growth stocks are risky)
    - Growth 15-30% → Fair PEG = 1.0 (reasonable growth)
    - Growth 10-15% → Fair PEG = 1.2 (lower growth needs discount)

    Formula:
    - Fair PE = Fair PEG × Growth Rate
    - Fair Value = EPS × Fair PE

    Args:
        eps: Earnings per share (must be positive)
        pe_ratio: Current P/E ratio (used for context, not in calculation)
        growth_rate: Expected growth rate in percent (must be positive)

    Returns:
        ValuationResult with fair value or skip_reason if inputs are invalid
    """
    method_name = "peg"

    # Validate inputs
    if eps <= 0:
        return ValuationResult(value=0.0, method=method_name, skip_reason=f"EPS must be positive, got {eps}")

    if growth_rate <= 0:
        return ValuationResult(
            value=0.0, method=method_name, skip_reason=f"Growth rate must be positive, got {growth_rate}"
        )

    # Determine fair PEG based on growth rate
    if growth_rate > 30:
        fair_peg = 0.8
    elif growth_rate >= 15:
        fair_peg = 1.0
    else:  # growth_rate < 15 (low growth, uses higher fair_peg)
        fair_peg = 1.2

    # Calculate fair PE and fair value
    fair_pe = fair_peg * growth_rate
    fair_value = eps * fair_pe

    return ValuationResult(value=fair_value, method=method_name)


def calculate_pb_percentile(
    bvps: float,
    current_pb: float,
    historical_pb_median: float | None = None,
    industry_pb: float | None = None,
) -> ValuationResult:
    """
    Calculate fair value using P/B (Price-to-Book) percentile method.

    Reference PB priority: historical_pb_median > industry_pb > DEFAULT_PB

    Formula: Fair Value = BVPS × Reference PB

    Args:
        bvps: Book value per share (must be positive)
        current_pb: Current P/B ratio (for context, not used in calculation)
        historical_pb_median: Historical median P/B for the stock
        industry_pb: Industry average P/B

    Returns:
        ValuationResult with fair value or skip_reason if inputs are invalid
    """
    method_name = "pb_percentile"

    # Validate inputs
    if bvps <= 0:
        return ValuationResult(value=0.0, method=method_name, skip_reason=f"BVPS must be positive, got {bvps}")

    # Determine reference PB with priority: historical > industry > default
    if historical_pb_median is not None:
        reference_pb = historical_pb_median
    elif industry_pb is not None:
        reference_pb = industry_pb
    else:
        reference_pb = DEFAULT_PB

    # Calculate fair value
    fair_value = bvps * reference_pb

    return ValuationResult(value=fair_value, method=method_name)


def calculate_ps_valuation(
    current_price: float,
    current_ps: float,
    industry_ps: float | None = None,
    historical_ps_median: float | None = None,
) -> ValuationResult:
    """
    Calculate fair value using P/S (Price-to-Sales) method.

    Reference PS priority: industry_ps > historical_ps_median > DEFAULT_PS

    Formula: Fair Value = Current Price × (Reference PS / Current PS)

    Args:
        current_price: Current stock price
        current_ps: Current P/S ratio (must be positive)
        industry_ps: Industry average P/S
        historical_ps_median: Historical median P/S for the stock

    Returns:
        ValuationResult with fair value or skip_reason if inputs are invalid
    """
    method_name = "ps"

    # Validate inputs
    if current_ps <= 0:
        return ValuationResult(
            value=0.0, method=method_name, skip_reason=f"Current PS must be positive, got {current_ps}"
        )

    # Determine reference PS with priority: industry > historical > default
    if industry_ps is not None:
        reference_ps = industry_ps
    elif historical_ps_median is not None:
        reference_ps = historical_ps_median
    else:
        reference_ps = DEFAULT_PS

    # Calculate fair value
    fair_value = current_price * (reference_ps / current_ps)

    return ValuationResult(value=fair_value, method=method_name)


def calculate_dividend_discount(dividend: float, required_return: float | None = None) -> ValuationResult:
    """
    Calculate fair value using dividend discount model.

    Default required_return = A_SHARE_RISK_FREE_RATE + A_SHARE_RISK_PREMIUM

    Formula: Fair Value = Dividend / Required Return

    Args:
        dividend: Annual dividend per share (must be positive)
        required_return: Required rate of return (default: risk-free + risk premium)

    Returns:
        ValuationResult with fair value or skip_reason if inputs are invalid
    """
    method_name = "dividend_discount"

    # Validate inputs
    if dividend <= 0:
        return ValuationResult(value=0.0, method=method_name, skip_reason=f"Dividend must be positive, got {dividend}")

    # Use default required return if not provided
    if required_return is None:
        required_return = A_SHARE_RISK_FREE_RATE + A_SHARE_RISK_PREMIUM

    # Calculate fair value
    fair_value = dividend / required_return

    return ValuationResult(value=fair_value, method=method_name)
