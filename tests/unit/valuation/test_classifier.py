"""Tests for stock classifier."""

from ashare_analyzer.valuation.classifier import StockType, classify_stock


class TestClassifyStock:
    """Tests for stock classification."""

    def test_classify_value_stock(self):
        """Test classification of value stock."""
        data = {
            "pe_ratio": 12.0,  # < 15
            "roe": 12.0,  # > 10%
            "dividend_yield": 3.0,  # > 2%
            "industry": "银行",
        }
        result = classify_stock(data)
        assert result == StockType.VALUE

    def test_classify_growth_stock_by_revenue(self):
        """Test classification of growth stock by revenue growth."""
        data = {
            "pe_ratio": 30.0,
            "revenue_growth": 25.0,  # > 20%
            "industry": "计算机",
        }
        result = classify_stock(data)
        assert result == StockType.GROWTH

    def test_classify_growth_stock_by_eps(self):
        """Test classification of growth stock by EPS growth."""
        data = {
            "pe_ratio": 30.0,
            "eps_growth": 18.0,  # > 15%
            "industry": "医药生物",
        }
        result = classify_stock(data)
        assert result == StockType.GROWTH

    def test_classify_cyclical_stock(self):
        """Test classification of cyclical stock."""
        data = {
            "pe_ratio": 20.0,
            "industry": "有色金属",
        }
        result = classify_stock(data)
        assert result == StockType.CYCLICAL

    def test_classify_financial_stock_bank(self):
        """Test classification of financial stock (bank)."""
        data = {
            "pe_ratio": 8.0,
            "industry": "银行",
        }
        result = classify_stock(data)
        assert result == StockType.FINANCIAL

    def test_classify_financial_stock_insurance(self):
        """Test classification of financial stock (insurance)."""
        data = {
            "pe_ratio": 15.0,
            "industry": "保险",
        }
        result = classify_stock(data)
        assert result == StockType.FINANCIAL

    def test_classify_loss_making_stock_negative_pe(self):
        """Test classification of loss-making stock (negative PE)."""
        data = {
            "pe_ratio": -5.0,  # Negative = loss
            "industry": "计算机",
        }
        result = classify_stock(data)
        assert result == StockType.LOSS_MAKING

    def test_classify_loss_making_stock_zero_pe(self):
        """Test classification of loss-making stock (zero PE)."""
        data = {
            "pe_ratio": 0.0,  # Zero = no earnings
            "industry": "电子",
        }
        result = classify_stock(data)
        assert result == StockType.LOSS_MAKING

    def test_classify_default_stock(self):
        """Test classification defaults to DEFAULT for unclassified stocks."""
        data = {
            "pe_ratio": 25.0,
            "industry": "食品饮料",
        }
        result = classify_stock(data)
        assert result == StockType.DEFAULT

    def test_classify_with_missing_data(self):
        """Test classification with minimal data returns DEFAULT."""
        data = {}
        result = classify_stock(data)
        assert result == StockType.DEFAULT

    def test_classify_missing_pe_ratio_returns_default(self):
        """Test classification with missing PE ratio returns DEFAULT."""
        data = {
            "industry": "计算机",
            "roe": 15.0,
        }
        result = classify_stock(data)
        assert result == StockType.DEFAULT

    def test_classify_cyclical_with_value_characteristics(self):
        """Test cyclical stock with value characteristics is classified as VALUE."""
        # A cyclical stock meeting value criteria should use value methods
        data = {
            "pe_ratio": 12.0,  # < 15
            "roe": 15.0,  # > 10%
            "dividend_yield": 3.0,  # > 2%
            "industry": "有色金属",  # Cyclical industry
        }
        result = classify_stock(data)
        assert result == StockType.VALUE


class TestStockType:
    """Tests for StockType enum."""

    def test_stock_type_values(self):
        """Test StockType enum has all expected values."""
        assert StockType.VALUE.value == "value"
        assert StockType.GROWTH.value == "growth"
        assert StockType.CYCLICAL.value == "cyclical"
        assert StockType.FINANCIAL.value == "financial"
        assert StockType.LOSS_MAKING.value == "loss_making"
        assert StockType.DEFAULT.value == "default"
