"""Tests for ValuationAgent."""

import math

import pytest

from stock_analyzer.ai.agents.valuation_agent import ValuationAgent
from stock_analyzer.models import SignalType


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

        # Fair value calculation with multiple methods:
        # Graham = sqrt(22.5 * 50 * 150) = 410.79
        # Relative = 50 * 30 = 1500 (only PE, no PB)
        # PE-based = 50 * 30 = 1500
        # Average = (410.79 + 1500 + 1500) / 3 = 1136.93
        # For BUY: margin >= 0.15, so price <= 1136.93 / 1.15 = 988.6
        # Use price = 800 -> margin = (1136.93 - 800) / 800 = 0.42 > 0.15 -> BUY
        context = {
            "code": "600519",
            "stock_name": "Test Stock",
            "current_price": 800.0,
            "valuation_data": {
                "eps": 50.0,
                "book_value_per_share": 150.0,
                "industry_pe": 30.0,
            },
        }

        result = await agent.analyze(context)

        assert result.signal == SignalType.BUY
        assert result.confidence > 0
        assert "公允价值" in result.reasoning

    @pytest.mark.asyncio
    async def test_analyze_overvalued_returns_sell(self):
        """Test overvalued stock returns SELL signal."""
        agent = ValuationAgent()

        # Fair value = 1136.93 (same calculation as above)
        # For SELL: margin <= -0.15, so price >= 1136.93 / 0.85 = 1337.6
        # Use price = 1400 -> margin = (1136.93 - 1400) / 1400 = -0.188 < -0.15 -> SELL
        context = {
            "code": "600519",
            "stock_name": "Test Stock",
            "current_price": 1400.0,
            "valuation_data": {
                "eps": 50.0,
                "book_value_per_share": 150.0,
                "industry_pe": 30.0,
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

        # Fair value = 1136.93 (same calculation as above)
        # For HOLD: -0.15 < margin < 0.15
        # Use price = 1100 -> margin = (1136.93 - 1100) / 1100 = 0.034 -> HOLD
        context = {
            "code": "600519",
            "stock_name": "Test Stock",
            "current_price": 1100.0,
            "valuation_data": {
                "eps": 50.0,
                "book_value_per_share": 150.0,
                "industry_pe": 30.0,
            },
        }

        result = await agent.analyze(context)

        assert result.signal == SignalType.HOLD
        assert result.confidence >= 30

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
                "industry_pe": 30.0,
            },
        }

        result = await agent.analyze(context)

        assert result.signal == SignalType.HOLD
        assert result.confidence == 0


class TestValuationAgentCalculations:
    """Tests for ValuationAgent calculation methods."""

    def test_calculate_graham_number(self):
        """Test Graham Number calculation: sqrt(22.5 * eps * bvps)."""
        agent = ValuationAgent()

        # Graham Number = sqrt(22.5 * 50 * 150) = sqrt(168750) = 410.82
        data = {
            "eps": 50.0,
            "book_value_per_share": 150.0,
        }

        result = agent._calculate_graham_number(data)

        expected = math.sqrt(22.5 * 50.0 * 150.0)
        assert abs(result - expected) < 0.01

    def test_calculate_graham_number_missing_eps_returns_zero(self):
        """Test Graham Number returns zero when EPS is missing."""
        agent = ValuationAgent()

        data = {
            "book_value_per_share": 150.0,
        }

        result = agent._calculate_graham_number(data)

        assert result == 0

    def test_calculate_graham_number_missing_bvps_returns_zero(self):
        """Test Graham Number returns zero when BVPS is missing."""
        agent = ValuationAgent()

        data = {
            "eps": 50.0,
        }

        result = agent._calculate_graham_number(data)

        assert result == 0

    def test_calculate_graham_number_negative_values_returns_zero(self):
        """Test Graham Number returns zero for negative values."""
        agent = ValuationAgent()

        data = {
            "eps": -10.0,
            "book_value_per_share": 150.0,
        }

        result = agent._calculate_graham_number(data)

        assert result == 0

    def test_calculate_pe_valuation(self):
        """Test PE-based valuation: eps * industry_pe."""
        agent = ValuationAgent()

        # PE valuation = 50 * 30 = 1500
        data = {
            "eps": 50.0,
            "industry_pe": 30.0,
        }

        result = agent._calculate_pe_valuation(data)

        assert result == 1500.0

    def test_calculate_pe_valuation_missing_eps_returns_zero(self):
        """Test PE valuation returns zero when EPS is missing."""
        agent = ValuationAgent()

        data = {
            "industry_pe": 30.0,
        }

        result = agent._calculate_pe_valuation(data)

        assert result == 0

    def test_calculate_pe_valuation_missing_industry_pe_returns_zero(self):
        """Test PE valuation returns zero when industry PE is missing."""
        agent = ValuationAgent()

        data = {
            "eps": 50.0,
        }

        result = agent._calculate_pe_valuation(data)

        assert result == 0

    def test_calculate_pb_valuation(self):
        """Test PB-based valuation: bvps * industry_pb."""
        agent = ValuationAgent()

        # PB valuation = 150 * 8 = 1200
        data = {
            "book_value_per_share": 150.0,
            "industry_pb": 8.0,
        }

        result = agent._calculate_pb_valuation(data)

        assert result == 1200.0

    def test_calculate_pb_valuation_missing_bvps_returns_zero(self):
        """Test PB valuation returns zero when BVPS is missing."""
        agent = ValuationAgent()

        data = {
            "industry_pb": 8.0,
        }

        result = agent._calculate_pb_valuation(data)

        assert result == 0

    def test_calculate_pb_valuation_missing_industry_pb_returns_zero(self):
        """Test PB valuation returns zero when industry PB is missing."""
        agent = ValuationAgent()

        data = {
            "book_value_per_share": 150.0,
        }

        result = agent._calculate_pb_valuation(data)

        assert result == 0

    def test_calculate_relative_valuation_with_both(self):
        """Test relative valuation with both PE and PB data."""
        agent = ValuationAgent()

        # PE valuation = 50 * 30 = 1500
        # PB valuation = 150 * 8 = 1200
        # Average = (1500 + 1200) / 2 = 1350
        data = {
            "eps": 50.0,
            "book_value_per_share": 150.0,
            "industry_pe": 30.0,
            "industry_pb": 8.0,
        }

        result = agent._calculate_relative_valuation(data)

        assert result == 1350.0

    def test_calculate_relative_valuation_with_pe_only(self):
        """Test relative valuation with only PE data."""
        agent = ValuationAgent()

        # PE valuation = 50 * 30 = 1500
        data = {
            "eps": 50.0,
            "industry_pe": 30.0,
        }

        result = agent._calculate_relative_valuation(data)

        assert result == 1500.0

    def test_calculate_relative_valuation_with_pb_only(self):
        """Test relative valuation with only PB data."""
        agent = ValuationAgent()

        # PB valuation = 150 * 8 = 1200
        data = {
            "book_value_per_share": 150.0,
            "industry_pb": 8.0,
        }

        result = agent._calculate_relative_valuation(data)

        assert result == 1200.0

    def test_calculate_relative_valuation_no_data_returns_zero(self):
        """Test relative valuation returns zero with no data."""
        agent = ValuationAgent()

        data = {}

        result = agent._calculate_relative_valuation(data)

        assert result == 0

    def test_margin_to_signal_buy_threshold(self):
        """Test margin >= 0.15 returns BUY signal."""
        agent = ValuationAgent()

        # margin = 0.15 -> BUY
        signal, confidence = agent._margin_to_signal(0.15)

        assert signal == SignalType.BUY
        assert confidence >= 50  # 50 + 0.15 * 200 = 80

    def test_margin_to_signal_buy_large_margin(self):
        """Test large positive margin returns BUY with high confidence."""
        agent = ValuationAgent()

        # margin = 0.25 -> BUY, confidence capped at 100
        signal, confidence = agent._margin_to_signal(0.25)

        assert signal == SignalType.BUY
        assert confidence == 100  # 50 + 0.25 * 200 = 100

    def test_margin_to_signal_sell_threshold(self):
        """Test margin <= -0.15 returns SELL signal."""
        agent = ValuationAgent()

        # margin = -0.15 -> SELL
        signal, confidence = agent._margin_to_signal(-0.15)

        assert signal == SignalType.SELL
        assert confidence >= 50  # 50 + 0.15 * 200 = 80

    def test_margin_to_signal_sell_large_margin(self):
        """Test large negative margin returns SELL with high confidence."""
        agent = ValuationAgent()

        # margin = -0.25 -> SELL, confidence capped at 100
        signal, confidence = agent._margin_to_signal(-0.25)

        assert signal == SignalType.SELL
        assert confidence == 100  # 50 + 0.25 * 200 = 100

    def test_margin_to_signal_hold_range(self):
        """Test margin in hold range returns HOLD signal."""
        agent = ValuationAgent()

        # margin = 0.05 -> HOLD
        signal, confidence = agent._margin_to_signal(0.05)

        assert signal == SignalType.HOLD
        assert confidence >= 30  # max(30, 50 - 5) = 45

    def test_margin_to_signal_hold_zero(self):
        """Test zero margin returns HOLD signal."""
        agent = ValuationAgent()

        # margin = 0 -> HOLD
        signal, confidence = agent._margin_to_signal(0.0)

        assert signal == SignalType.HOLD
        assert confidence == 50  # 50 - 0 = 50

    def test_margin_to_signal_hold_negative_small(self):
        """Test small negative margin returns HOLD signal."""
        agent = ValuationAgent()

        # margin = -0.10 -> HOLD
        signal, confidence = agent._margin_to_signal(-0.10)

        assert signal == SignalType.HOLD
        assert confidence >= 30  # max(30, 50 - 10) = 40
