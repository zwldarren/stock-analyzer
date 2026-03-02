"""Tests for valuation method registry."""

import pytest

from ashare_analyzer.valuation.classifier import StockType
from ashare_analyzer.valuation.registry import (
    METHOD_WEIGHTS,
    MethodSelection,
    get_methods_for_stock_type,
    select_valuation_methods,
)


class TestMethodSelection:
    """Tests for MethodSelection dataclass."""

    def test_method_selection_creation(self) -> None:
        """Test MethodSelection can be created with valid data."""
        selection = MethodSelection(
            methods=["pb_percentile", "dividend_discount"],
            weights={"pb_percentile": 0.6, "dividend_discount": 0.4},
        )
        assert selection.methods == ["pb_percentile", "dividend_discount"]
        assert selection.weights == {"pb_percentile": 0.6, "dividend_discount": 0.4}


class TestMethodWeights:
    """Tests for METHOD_WEIGHTS dictionary."""

    def test_method_weights_has_all_stock_types(self) -> None:
        """Test METHOD_WEIGHTS contains all StockType values."""
        expected_types = {
            StockType.VALUE,
            StockType.GROWTH,
            StockType.CYCLICAL,
            StockType.FINANCIAL,
            StockType.LOSS_MAKING,
            StockType.DEFAULT,
        }
        assert set(METHOD_WEIGHTS.keys()) == expected_types

    def test_value_weights(self) -> None:
        """Test VALUE stock type weights."""
        weights = METHOD_WEIGHTS[StockType.VALUE]
        assert weights == {
            "pb_percentile": 0.30,
            "dividend_discount": 0.30,
            "adjusted_graham": 0.40,
        }
        assert sum(weights.values()) == pytest.approx(1.0)

    def test_growth_weights(self) -> None:
        """Test GROWTH stock type weights."""
        weights = METHOD_WEIGHTS[StockType.GROWTH]
        assert weights == {
            "peg": 0.50,
            "ps": 0.30,
            "adjusted_graham": 0.20,
        }
        assert sum(weights.values()) == pytest.approx(1.0)

    def test_cyclical_weights(self) -> None:
        """Test CYCLICAL stock type weights."""
        weights = METHOD_WEIGHTS[StockType.CYCLICAL]
        assert weights == {
            "pb_percentile": 0.50,
            "adjusted_graham": 0.30,
            "dividend_discount": 0.20,
        }
        assert sum(weights.values()) == pytest.approx(1.0)

    def test_financial_weights(self) -> None:
        """Test FINANCIAL stock type weights."""
        weights = METHOD_WEIGHTS[StockType.FINANCIAL]
        assert weights == {
            "pb_percentile": 0.40,
            "dividend_discount": 0.60,
        }
        assert sum(weights.values()) == pytest.approx(1.0)

    def test_loss_making_weights(self) -> None:
        """Test LOSS_MAKING stock type weights."""
        weights = METHOD_WEIGHTS[StockType.LOSS_MAKING]
        assert weights == {
            "ps": 0.80,
            "adjusted_graham": 0.20,
        }
        assert sum(weights.values()) == pytest.approx(1.0)

    def test_default_weights(self) -> None:
        """Test DEFAULT stock type weights."""
        weights = METHOD_WEIGHTS[StockType.DEFAULT]
        assert weights == {
            "pb_percentile": 0.40,
            "adjusted_graham": 0.60,
        }
        assert sum(weights.values()) == pytest.approx(1.0)

    def test_all_weights_sum_to_one(self) -> None:
        """Test all stock type weights sum to approximately 1.0."""
        for stock_type, weights in METHOD_WEIGHTS.items():
            total = sum(weights.values())
            assert total == pytest.approx(1.0), f"{stock_type} weights sum to {total}, not 1.0"


class TestGetMethodsForStockType:
    """Tests for get_methods_for_stock_type function."""

    def test_get_methods_for_value(self) -> None:
        """Test getting methods for VALUE stock type."""
        methods = get_methods_for_stock_type(StockType.VALUE)
        assert methods == ["pb_percentile", "dividend_discount", "adjusted_graham"]

    def test_get_methods_for_growth(self) -> None:
        """Test getting methods for GROWTH stock type."""
        methods = get_methods_for_stock_type(StockType.GROWTH)
        assert methods == ["peg", "ps", "adjusted_graham"]

    def test_get_methods_for_cyclical(self) -> None:
        """Test getting methods for CYCLICAL stock type."""
        methods = get_methods_for_stock_type(StockType.CYCLICAL)
        assert methods == ["pb_percentile", "adjusted_graham", "dividend_discount"]

    def test_get_methods_for_financial(self) -> None:
        """Test getting methods for FINANCIAL stock type."""
        methods = get_methods_for_stock_type(StockType.FINANCIAL)
        assert methods == ["pb_percentile", "dividend_discount"]

    def test_get_methods_for_loss_making(self) -> None:
        """Test getting methods for LOSS_MAKING stock type."""
        methods = get_methods_for_stock_type(StockType.LOSS_MAKING)
        assert methods == ["ps", "adjusted_graham"]

    def test_get_methods_for_default(self) -> None:
        """Test getting methods for DEFAULT stock type."""
        methods = get_methods_for_stock_type(StockType.DEFAULT)
        assert methods == ["pb_percentile", "adjusted_graham"]


class TestSelectValuationMethods:
    """Tests for select_valuation_methods function."""

    def test_select_for_value(self) -> None:
        """Test selecting methods for VALUE stock type."""
        selection = select_valuation_methods(StockType.VALUE)
        assert isinstance(selection, MethodSelection)
        assert selection.methods == ["pb_percentile", "dividend_discount", "adjusted_graham"]
        assert selection.weights == {
            "pb_percentile": 0.30,
            "dividend_discount": 0.30,
            "adjusted_graham": 0.40,
        }

    def test_select_for_growth(self) -> None:
        """Test selecting methods for GROWTH stock type."""
        selection = select_valuation_methods(StockType.GROWTH)
        assert isinstance(selection, MethodSelection)
        assert selection.methods == ["peg", "ps", "adjusted_graham"]
        assert selection.weights == {
            "peg": 0.50,
            "ps": 0.30,
            "adjusted_graham": 0.20,
        }

    def test_select_for_cyclical(self) -> None:
        """Test selecting methods for CYCLICAL stock type."""
        selection = select_valuation_methods(StockType.CYCLICAL)
        assert isinstance(selection, MethodSelection)
        assert selection.methods == ["pb_percentile", "adjusted_graham", "dividend_discount"]
        assert selection.weights == {
            "pb_percentile": 0.50,
            "adjusted_graham": 0.30,
            "dividend_discount": 0.20,
        }

    def test_select_for_financial(self) -> None:
        """Test selecting methods for FINANCIAL stock type."""
        selection = select_valuation_methods(StockType.FINANCIAL)
        assert isinstance(selection, MethodSelection)
        assert selection.methods == ["pb_percentile", "dividend_discount"]
        assert selection.weights == {
            "pb_percentile": 0.40,
            "dividend_discount": 0.60,
        }

    def test_select_for_loss_making(self) -> None:
        """Test selecting methods for LOSS_MAKING stock type."""
        selection = select_valuation_methods(StockType.LOSS_MAKING)
        assert isinstance(selection, MethodSelection)
        assert selection.methods == ["ps", "adjusted_graham"]
        assert selection.weights == {
            "ps": 0.80,
            "adjusted_graham": 0.20,
        }

    def test_select_for_default(self) -> None:
        """Test selecting methods for DEFAULT stock type."""
        selection = select_valuation_methods(StockType.DEFAULT)
        assert isinstance(selection, MethodSelection)
        assert selection.methods == ["pb_percentile", "adjusted_graham"]
        assert selection.weights == {
            "pb_percentile": 0.40,
            "adjusted_graham": 0.60,
        }

    def test_methods_match_weights_keys(self) -> None:
        """Test that methods list matches weights dictionary keys."""
        for stock_type in StockType:
            selection = select_valuation_methods(stock_type)
            assert set(selection.methods) == set(selection.weights.keys()), (
                f"{stock_type}: methods {selection.methods} don't match weights keys {list(selection.weights.keys())}"
            )
