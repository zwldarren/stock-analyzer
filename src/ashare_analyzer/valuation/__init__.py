"""Valuation module for A-share stock analysis."""

from .classifier import StockType, classify_stock
from .methods import (
    ValuationResult,
    calculate_adjusted_graham,
    calculate_dividend_discount,
    calculate_pb_percentile,
    calculate_peg_valuation,
    calculate_ps_valuation,
)
from .registry import METHOD_WEIGHTS, MethodSelection, get_methods_for_stock_type, select_valuation_methods

__all__ = [
    "METHOD_WEIGHTS",
    "MethodSelection",
    "StockType",
    "ValuationResult",
    "calculate_adjusted_graham",
    "calculate_dividend_discount",
    "calculate_pb_percentile",
    "calculate_peg_valuation",
    "calculate_ps_valuation",
    "classify_stock",
    "get_methods_for_stock_type",
    "select_valuation_methods",
]
