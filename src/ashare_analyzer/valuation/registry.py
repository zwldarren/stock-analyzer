"""Method registry for valuation method selection based on stock type."""

from dataclasses import dataclass

from ashare_analyzer.valuation.classifier import StockType


@dataclass
class MethodSelection:
    """Selection of valuation methods with weights.

    Attributes:
        methods: List of valuation method names to use
        weights: Dictionary mapping method names to their weights
    """

    methods: list[str]
    weights: dict[str, float]


METHOD_WEIGHTS: dict[StockType, dict[str, float]] = {
    StockType.VALUE: {
        "pb_percentile": 0.30,
        "dividend_discount": 0.30,
        "adjusted_graham": 0.40,
    },
    StockType.GROWTH: {
        "peg": 0.50,
        "ps": 0.30,
        "adjusted_graham": 0.20,
    },
    StockType.CYCLICAL: {
        "pb_percentile": 0.50,
        "adjusted_graham": 0.30,
        "dividend_discount": 0.20,
    },
    StockType.FINANCIAL: {
        "pb_percentile": 0.40,
        "dividend_discount": 0.60,
    },
    StockType.LOSS_MAKING: {
        "ps": 0.80,
        "adjusted_graham": 0.20,
    },
    StockType.DEFAULT: {
        "pb_percentile": 0.40,
        "adjusted_graham": 0.60,
    },
}


def get_methods_for_stock_type(stock_type: StockType) -> list[str]:
    """
    Get list of valuation method names for a stock type.

    Args:
        stock_type: The stock type classification

    Returns:
        List of valuation method names to use for this stock type
    """
    weights = METHOD_WEIGHTS.get(stock_type, METHOD_WEIGHTS[StockType.DEFAULT])
    return list(weights.keys())


def select_valuation_methods(stock_type: StockType) -> MethodSelection:
    """
    Select valuation methods and weights for a stock type.

    Args:
        stock_type: The stock type classification

    Returns:
        MethodSelection containing methods list and weights dictionary
    """
    weights = METHOD_WEIGHTS.get(stock_type, METHOD_WEIGHTS[StockType.DEFAULT])
    methods = list(weights.keys())
    return MethodSelection(methods=methods, weights=weights.copy())
