"""Tests for ValuationAgent with multi-factor adaptive approach."""

import pytest

from ashare_analyzer.ai.agents.valuation_agent import ValuationAgent
from ashare_analyzer.models import SignalType
from ashare_analyzer.valuation import StockType


class TestValuationAgent:
    """Tests for ValuationAgent."""

    def test_init(self):
        """Test agent initialization."""
        agent = ValuationAgent()
        assert agent.name == "ValuationAgent"

    def test_is_available_returns_true(self):
        """Test is_available always returns True."""
        agent = ValuationAgent()
        assert agent.is_available() is True

    @pytest.mark.asyncio
    async def test_analyze_undervalued_returns_buy(self):
        """Test undervalued stock returns BUY signal."""
        agent = ValuationAgent()

        # Create context with value stock characteristics
        # PE < 15, ROE > 10, Dividend > 2% -> VALUE type
        # For VALUE: pb_percentile(0.30), dividend_discount(0.30), adjusted_graham(0.40)
        context = {
            "code": "600519",
            "stock_name": "Test Stock",
            "current_price": 100.0,
            "valuation_data": {
                "eps": 10.0,
                "book_value_per_share": 80.0,
                "pe_ratio": 10.0,  # PE < 15
                "pb_ratio": 1.25,
                "roe": 15.0,  # ROE > 10%
                "dividend_yield": 3.0,  # Dividend > 2%
                "industry_name": "白酒",
                "industry_pb": 5.0,
            },
        }

        result = await agent.analyze(context)

        assert result.signal == SignalType.BUY
        assert result.confidence > 0
        assert "公允价值" in result.reasoning
        assert "股票类型" in result.reasoning

    @pytest.mark.asyncio
    async def test_analyze_overvalued_returns_sell(self):
        """Test overvalued stock returns SELL signal."""
        agent = ValuationAgent()

        # Create context where fair value is lower than current price
        context = {
            "code": "600519",
            "stock_name": "Test Stock",
            "current_price": 500.0,
            "valuation_data": {
                "eps": 10.0,
                "book_value_per_share": 80.0,
                "pe_ratio": 50.0,  # High PE
                "pb_ratio": 6.25,
                "roe": 15.0,
                "dividend_yield": 1.0,
                "industry_name": "白酒",
                "industry_pb": 3.0,
            },
        }

        result = await agent.analyze(context)

        assert result.signal == SignalType.SELL
        assert result.confidence > 0
        assert "安全边际" in result.reasoning

    @pytest.mark.asyncio
    async def test_analyze_fairly_valued_returns_hold(self):
        """Test fairly valued stock returns HOLD signal."""
        agent = ValuationAgent()

        # Create context where fair value is close to current price
        # We need margin between -5% and +5% for HOLD
        # Adjusted Graham with ROE 15%: sqrt(22.5 * 2.0 * 10 * 80) ≈ 189.7
        # PB percentile with industry_pb 2.0: 80 * 2.0 = 160
        # Weighted average for DEFAULT: 0.4 * 160 + 0.6 * 189.7 ≈ 177.8
        # For margin ~0%, set price to 178
        context = {
            "code": "600519",
            "stock_name": "Test Stock",
            "current_price": 178.0,
            "valuation_data": {
                "eps": 10.0,
                "book_value_per_share": 80.0,
                "pe_ratio": 17.8,
                "pb_ratio": 2.225,
                "roe": 15.0,
                "dividend_yield": 1.0,
                "industry_name": "白酒",
                "industry_pb": 2.0,
            },
        }

        result = await agent.analyze(context)

        # Margin should be within HOLD range (-5% to +5%)
        margin = result.metadata.get("margin_of_safety", 0)
        assert -6 <= margin <= 6, f"Expected margin within [-5%, +5%], got {margin}%"
        assert result.signal == SignalType.HOLD
        assert result.confidence >= 10

    @pytest.mark.asyncio
    async def test_analyze_no_valuation_data_returns_hold(self):
        """Test analysis with no valuation data returns HOLD."""
        agent = ValuationAgent()

        context = {
            "code": "600519",
            "stock_name": "Test Stock",
            "current_price": 1000.0,
            "valuation_data": {},
        }

        result = await agent.analyze(context)

        assert result.signal == SignalType.HOLD
        assert result.confidence == 0
        assert "error" in result.metadata

    @pytest.mark.asyncio
    async def test_analyze_zero_price_returns_hold(self):
        """Test analysis with zero price returns HOLD."""
        agent = ValuationAgent()

        context = {
            "code": "600519",
            "stock_name": "Test Stock",
            "current_price": 0,
            "valuation_data": {
                "eps": 50.0,
                "book_value_per_share": 150.0,
                "pe_ratio": 20.0,
            },
        }

        result = await agent.analyze(context)

        assert result.signal == SignalType.HOLD
        assert result.confidence == 0


class TestValuationAgentStockTypeClassification:
    """Tests for stock type classification integration."""

    @pytest.mark.asyncio
    async def test_cyclical_stock_uses_pb_percentile_method(self):
        """Test cyclical stock uses pb_percentile as primary method."""
        agent = ValuationAgent()

        # 洛阳钼业-like: cyclical industry (有色金属)
        context = {
            "code": "603993",
            "stock_name": "洛阳钼业",
            "current_price": 6.0,
            "valuation_data": {
                "eps": 0.3,
                "book_value_per_share": 2.5,
                "pe_ratio": 20.0,
                "pb_ratio": 2.4,
                "roe": 10.0,
                "dividend_yield": 2.0,
                "industry_name": "有色金属",  # Cyclical industry
            },
        }

        result = await agent.analyze(context)

        # Should classify as CYCLICAL
        assert result.metadata.get("stock_type") == StockType.CYCLICAL.value

        # Should use pb_percentile method (primary for cyclical)
        valuations = result.metadata.get("valuations", {})
        assert "pb_percentile" in valuations

        # Should not have 100% confidence
        assert result.confidence < 100

    @pytest.mark.asyncio
    async def test_cyclical_stock_does_not_get_extreme_sell_confidence(self):
        """Test that cyclical stock (洛阳钼业-like) doesn't get 100% SELL confidence.

        This is the key test case for the A-share valuation redesign.
        Cyclical stocks at low PB should not be rated as extreme SELL.
        """
        agent = ValuationAgent()

        # 洛阳钼业-like: cyclical stock with PB at low percentile
        # Current PB = 2.4, Industry PB = 3.0
        # This should not produce 100% SELL confidence
        context = {
            "code": "603993",
            "stock_name": "洛阳钼业",
            "current_price": 6.0,
            "valuation_data": {
                "eps": 0.3,
                "book_value_per_share": 2.5,
                "pe_ratio": 20.0,
                "pb_ratio": 2.4,
                "roe": 10.0,
                "dividend_yield": 2.0,
                "industry_name": "有色金属",  # Cyclical industry
                "industry_pb": 3.0,
            },
        }

        result = await agent.analyze(context)

        # Key assertion: confidence should NOT be 100
        assert result.confidence < 100, f"Cyclical stock should not have 100% confidence, got {result.confidence}"

        # Verify stock type is CYCLICAL
        assert result.metadata.get("stock_type") == StockType.CYCLICAL.value

        # Verify reasoning includes stock type
        assert "周期型" in result.reasoning or "股票类型" in result.reasoning

    @pytest.mark.asyncio
    async def test_financial_stock_uses_dividend_discount(self):
        """Test financial stock uses dividend_discount as primary method."""
        agent = ValuationAgent()

        # Financial stock that doesn't meet VALUE criteria (PE not < 15, or ROE not > 10%)
        # Using a securities company with higher PE
        context = {
            "code": "601211",
            "stock_name": "国泰君安",
            "current_price": 15.0,
            "valuation_data": {
                "eps": 0.5,
                "book_value_per_share": 10.0,
                "pe_ratio": 30.0,  # High PE - not VALUE
                "pb_ratio": 1.5,
                "roe": 5.0,  # Low ROE - not VALUE
                "dividend_yield": 2.0,
                "industry_name": "证券",  # Financial industry
                "industry_pb": 1.6,
            },
        }

        result = await agent.analyze(context)

        # Should classify as FINANCIAL (not VALUE because PE >= 15)
        assert result.metadata.get("stock_type") == StockType.FINANCIAL.value

        # Should use dividend_discount method (primary for financial)
        valuations = result.metadata.get("valuations", {})
        assert "dividend_discount" in valuations

    @pytest.mark.asyncio
    async def test_growth_stock_uses_peg_method(self):
        """Test growth stock uses PEG as primary method."""
        agent = ValuationAgent()

        # Growth stock with high revenue growth
        context = {
            "code": "300750",
            "stock_name": "宁德时代",
            "current_price": 200.0,
            "valuation_data": {
                "eps": 10.0,
                "book_value_per_share": 50.0,
                "pe_ratio": 20.0,
                "pb_ratio": 4.0,
                "roe": 20.0,
                "dividend_yield": 0.5,
                "revenue_growth": 50.0,  # High growth > 20%
                "industry_name": "电池",
            },
        }

        result = await agent.analyze(context)

        # Should classify as GROWTH
        assert result.metadata.get("stock_type") == StockType.GROWTH.value

        # Should use PEG method (primary for growth)
        valuations = result.metadata.get("valuations", {})
        assert "peg" in valuations

    @pytest.mark.asyncio
    async def test_value_stock_classification(self):
        """Test value stock with low PE, high ROE, dividend payer."""
        agent = ValuationAgent()

        context = {
            "code": "000651",
            "stock_name": "格力电器",
            "current_price": 40.0,
            "valuation_data": {
                "eps": 4.0,
                "book_value_per_share": 20.0,
                "pe_ratio": 10.0,  # PE < 15
                "pb_ratio": 2.0,
                "roe": 20.0,  # ROE > 10%
                "dividend_yield": 5.0,  # Dividend > 2%
                "industry_name": "家电",
            },
        }

        result = await agent.analyze(context)

        # Should classify as VALUE
        assert result.metadata.get("stock_type") == StockType.VALUE.value


class TestValuationAgentConfidenceDecay:
    """Tests for confidence decay when data is missing."""

    @pytest.mark.asyncio
    async def test_confidence_decay_no_historical_data(self):
        """Test confidence decay when historical PB/PE data is missing."""
        agent = ValuationAgent()

        # Context without historical data
        context = {
            "code": "600519",
            "stock_name": "Test Stock",
            "current_price": 100.0,
            "valuation_data": {
                "eps": 10.0,
                "book_value_per_share": 80.0,
                "pe_ratio": 10.0,
                "pb_ratio": 1.25,
                "roe": 15.0,
                "dividend_yield": 3.0,
                "industry_name": "白酒",
                "industry_pb": 5.0,
                # No historical_pb_median
            },
        }

        result = await agent.analyze(context)

        # Confidence decay should be applied
        assert result.metadata.get("confidence_decay", 1.0) < 1.0

    @pytest.mark.asyncio
    async def test_confidence_decay_no_industry_data(self):
        """Test confidence decay when industry data is missing."""
        agent = ValuationAgent()

        # Context without industry data
        context = {
            "code": "600519",
            "stock_name": "Test Stock",
            "current_price": 100.0,
            "valuation_data": {
                "eps": 10.0,
                "book_value_per_share": 80.0,
                "pe_ratio": 10.0,
                "pb_ratio": 1.25,
                "roe": 15.0,
                "dividend_yield": 3.0,
                "industry_name": "白酒",
                # No industry_pb
            },
        }

        result = await agent.analyze(context)

        # Confidence decay should be applied
        assert result.metadata.get("confidence_decay", 1.0) < 1.0

    @pytest.mark.asyncio
    async def test_confidence_decay_single_method(self):
        """Test confidence decay when only one valuation method is available."""
        agent = ValuationAgent()

        # Context with minimal data - only one method available
        context = {
            "code": "600519",
            "stock_name": "Test Stock",
            "current_price": 100.0,
            "valuation_data": {
                "book_value_per_share": 80.0,
                "pb_ratio": 1.25,
                "industry_name": "有色金属",  # Cyclical - uses pb_percentile
                # Minimal data for pb_percentile only
            },
        }

        result = await agent.analyze(context)

        # If only one method, decay should be significant (x0.6)
        if len(result.metadata.get("valuations", {})) == 1:
            assert result.metadata.get("confidence_decay", 1.0) < 0.7

    @pytest.mark.asyncio
    async def test_higher_confidence_with_full_data(self):
        """Test higher confidence when full data is available."""
        agent = ValuationAgent()

        # Context with full data
        context_full = {
            "code": "600519",
            "stock_name": "Test Stock",
            "current_price": 100.0,
            "valuation_data": {
                "eps": 10.0,
                "book_value_per_share": 80.0,
                "pe_ratio": 10.0,
                "pb_ratio": 1.25,
                "roe": 15.0,
                "dividend_yield": 3.0,
                "industry_name": "白酒",
                "industry_pb": 5.0,
                "historical_pb_median": 4.5,  # Historical data available
            },
        }

        # Context with missing data
        context_partial = {
            "code": "600519",
            "stock_name": "Test Stock",
            "current_price": 100.0,
            "valuation_data": {
                "eps": 10.0,
                "book_value_per_share": 80.0,
                "pe_ratio": 10.0,
                "pb_ratio": 1.25,
                "roe": 15.0,
                "dividend_yield": 3.0,
                "industry_name": "白酒",
                # No historical or industry data
            },
        }

        result_full = await agent.analyze(context_full)
        result_partial = await agent.analyze(context_partial)

        # Full data should have higher confidence (less decay)
        decay_full = result_full.metadata.get("confidence_decay", 1.0)
        decay_partial = result_partial.metadata.get("confidence_decay", 1.0)
        assert decay_full >= decay_partial


class TestValuationAgentMetadata:
    """Tests for metadata content."""

    @pytest.mark.asyncio
    async def test_metadata_includes_stock_type(self):
        """Test metadata includes stock_type field."""
        agent = ValuationAgent()

        context = {
            "code": "600519",
            "stock_name": "Test Stock",
            "current_price": 100.0,
            "valuation_data": {
                "eps": 10.0,
                "book_value_per_share": 80.0,
                "pe_ratio": 10.0,
                "pb_ratio": 1.25,
                "roe": 15.0,
                "dividend_yield": 3.0,
                "industry_name": "白酒",
            },
        }

        result = await agent.analyze(context)

        assert "stock_type" in result.metadata
        assert result.metadata["stock_type"] in [t.value for t in StockType]

    @pytest.mark.asyncio
    async def test_metadata_includes_valuations(self):
        """Test metadata includes valuations dict."""
        agent = ValuationAgent()

        context = {
            "code": "600519",
            "stock_name": "Test Stock",
            "current_price": 100.0,
            "valuation_data": {
                "eps": 10.0,
                "book_value_per_share": 80.0,
                "pe_ratio": 10.0,
                "pb_ratio": 1.25,
                "roe": 15.0,
                "dividend_yield": 3.0,
                "industry_name": "白酒",
                "industry_pb": 5.0,
            },
        }

        result = await agent.analyze(context)

        assert "valuations" in result.metadata
        assert isinstance(result.metadata["valuations"], dict)
        assert len(result.metadata["valuations"]) > 0

    @pytest.mark.asyncio
    async def test_metadata_includes_method_weights(self):
        """Test metadata includes method_weights dict."""
        agent = ValuationAgent()

        context = {
            "code": "600519",
            "stock_name": "Test Stock",
            "current_price": 100.0,
            "valuation_data": {
                "eps": 10.0,
                "book_value_per_share": 80.0,
                "pe_ratio": 10.0,
                "pb_ratio": 1.25,
                "roe": 15.0,
                "dividend_yield": 3.0,
                "industry_name": "白酒",
                "industry_pb": 5.0,
            },
        }

        result = await agent.analyze(context)

        assert "method_weights" in result.metadata
        assert isinstance(result.metadata["method_weights"], dict)

    @pytest.mark.asyncio
    async def test_metadata_includes_confidence_decay(self):
        """Test metadata includes confidence_decay field."""
        agent = ValuationAgent()

        context = {
            "code": "600519",
            "stock_name": "Test Stock",
            "current_price": 100.0,
            "valuation_data": {
                "eps": 10.0,
                "book_value_per_share": 80.0,
                "pe_ratio": 10.0,
                "pb_ratio": 1.25,
                "roe": 15.0,
                "dividend_yield": 3.0,
                "industry_name": "白酒",
                "industry_pb": 5.0,
            },
        }

        result = await agent.analyze(context)

        assert "confidence_decay" in result.metadata
        assert 0 < result.metadata["confidence_decay"] <= 1.0


class TestValuationAgentMarginToSignal:
    """Tests for margin to signal conversion with new thresholds."""

    def test_margin_to_signal_strong_buy(self):
        """Test margin >= 20% returns BUY with high confidence."""
        agent = ValuationAgent()

        signal, confidence = agent._margin_to_signal(0.20)

        assert signal == SignalType.BUY
        assert confidence == 70  # 70% base for strong buy

    def test_margin_to_signal_strong_buy_with_excess(self):
        """Test margin > 20% returns BUY with higher confidence."""
        agent = ValuationAgent()

        signal, confidence = agent._margin_to_signal(0.30)

        assert signal == SignalType.BUY
        # 70 + (30-20) * 0.5 = 75
        assert confidence == 75

    def test_margin_to_signal_moderate_buy(self):
        """Test margin 10-20% returns BUY with moderate confidence."""
        agent = ValuationAgent()

        signal, confidence = agent._margin_to_signal(0.15)

        assert signal == SignalType.BUY
        # 60 + (15-10) * 100 = 65
        assert confidence == 65

    def test_margin_to_signal_hold(self):
        """Test margin -5% to +5% returns HOLD."""
        agent = ValuationAgent()

        signal, confidence = agent._margin_to_signal(0.0)

        assert signal == SignalType.HOLD
        assert confidence == 50

    def test_margin_to_signal_hold_slight_negative(self):
        """Test small negative margin returns HOLD."""
        agent = ValuationAgent()

        signal, confidence = agent._margin_to_signal(-0.03)

        assert signal == SignalType.HOLD
        assert confidence == 50

    def test_margin_to_signal_moderate_sell(self):
        """Test margin -10% to -20% returns SELL with moderate confidence."""
        agent = ValuationAgent()

        signal, confidence = agent._margin_to_signal(-0.15)

        assert signal == SignalType.SELL
        # 60 + (15-10) = 65
        assert confidence == 65

    def test_margin_to_signal_strong_sell(self):
        """Test margin <= -20% returns SELL with high confidence."""
        agent = ValuationAgent()

        signal, confidence = agent._margin_to_signal(-0.20)

        assert signal == SignalType.SELL
        assert confidence == 70  # 70% base for strong sell

    def test_margin_to_signal_strong_sell_with_excess(self):
        """Test margin < -20% returns SELL with higher confidence."""
        agent = ValuationAgent()

        signal, confidence = agent._margin_to_signal(-0.30)

        assert signal == SignalType.SELL
        # 70 + (30-20) * 0.5 = 75
        assert confidence == 75


class TestValuationAgentCalculateMethods:
    """Tests for individual valuation calculation methods."""

    def test_calculate_single_method_adjusted_graham(self):
        """Test adjusted Graham calculation."""
        agent = ValuationAgent()

        valuation_data = {
            "eps": 10.0,
            "book_value_per_share": 80.0,
            "roe": 15.0,
        }

        result = agent._calculate_single_method("adjusted_graham", valuation_data, 100.0)

        assert result.value > 0
        assert result.method == "adjusted_graham"
        assert result.skip_reason is None

    def test_calculate_single_method_peg(self):
        """Test PEG calculation."""
        agent = ValuationAgent()

        valuation_data = {
            "eps": 10.0,
            "pe_ratio": 20.0,
            "revenue_growth": 20.0,
        }

        result = agent._calculate_single_method("peg", valuation_data, 100.0)

        assert result.value > 0
        assert result.method == "peg"
        assert result.skip_reason is None

    def test_calculate_single_method_pb_percentile(self):
        """Test PB percentile calculation."""
        agent = ValuationAgent()

        valuation_data = {
            "book_value_per_share": 80.0,
            "pb_ratio": 1.25,
            "industry_pb": 3.0,
        }

        result = agent._calculate_single_method("pb_percentile", valuation_data, 100.0)

        assert result.value > 0
        assert result.method == "pb_percentile"
        assert result.skip_reason is None

    def test_calculate_single_method_ps(self):
        """Test PS calculation."""
        agent = ValuationAgent()

        valuation_data = {
            "ps_ratio": 2.0,
            "industry_ps": 3.0,
        }

        result = agent._calculate_single_method("ps", valuation_data, 100.0)

        assert result.value > 0
        assert result.method == "ps"
        assert result.skip_reason is None

    def test_calculate_single_method_dividend_discount(self):
        """Test dividend discount calculation."""
        agent = ValuationAgent()

        valuation_data = {
            "dividend_yield": 3.0,
        }

        result = agent._calculate_single_method("dividend_discount", valuation_data, 100.0)

        assert result.value > 0
        assert result.method == "dividend_discount"
        assert result.skip_reason is None

    def test_calculate_single_method_missing_data(self):
        """Test method returns skip_reason when data is missing."""
        agent = ValuationAgent()

        valuation_data = {
            "eps": 0,  # Invalid EPS
        }

        result = agent._calculate_single_method("adjusted_graham", valuation_data, 100.0)

        assert result.value == 0
        assert result.skip_reason is not None
