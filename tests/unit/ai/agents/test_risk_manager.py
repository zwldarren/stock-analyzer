"""Tests for RiskManagerAgent."""

import pytest

from ashare_analyzer.ai.agents.risk_manager import RiskManagerAgent
from ashare_analyzer.models import SignalType


class TestRiskManagerAgent:
    """Tests for RiskManagerAgent."""

    def test_init(self):
        """Test agent initialization."""
        agent = RiskManagerAgent()
        assert agent.name == "RiskManagerAgent"

    def test_is_available_returns_true(self):
        """Test is_available always returns True."""
        agent = RiskManagerAgent()
        assert agent.is_available() is True

    @pytest.mark.asyncio
    async def test_analyze_returns_hold_signal(self):
        """Test that risk manager never generates trade signals."""
        agent = RiskManagerAgent()

        context = {
            "code": "600519",
            "price_data": {"close": [100.0] * 20},
        }

        result = await agent.analyze(context)

        # Risk manager always returns HOLD
        assert result.signal == SignalType.HOLD
        assert result.confidence == 100  # Confidence in calculation

    @pytest.mark.asyncio
    async def test_analyze_low_volatility_higher_position(self):
        """Test low volatility (< 15%) allows 25% position limit."""
        agent = RiskManagerAgent()

        # Create price data with low volatility (steady upward trend)
        # Volatility will be ~10% annualized
        base_price = 100.0
        prices = [base_price + i * 0.05 for i in range(50)]  # Very steady trend

        context = {
            "code": "600519",
            "price_data": {"close": prices},
        }

        result = await agent.analyze(context)

        assert result.metadata["volatility_tier"] == "low"
        assert result.metadata["max_position_size"] == 0.25
        assert "25%" in result.reasoning or "25.0%" in result.reasoning

    @pytest.mark.asyncio
    async def test_analyze_high_volatility_lower_position(self):
        """Test high volatility (> 30%) reduces position limit."""
        agent = RiskManagerAgent()

        # Create price data with high volatility
        # Daily moves of ~4% -> ~63% annualized volatility
        import random

        random.seed(42)  # For reproducibility
        base_price = 100.0
        prices = [base_price]
        for _ in range(50):
            # Use larger swings to ensure high volatility
            change = random.uniform(-0.04, 0.04)
            prices.append(prices[-1] * (1 + change))

        context = {
            "code": "600519",
            "price_data": {"close": prices},
        }

        result = await agent.analyze(context)

        # High volatility should result in reduced position limit
        assert result.metadata["volatility_tier"] in ("high", "very_high")
        assert result.metadata["max_position_size"] < 0.20  # Less than medium limit

    @pytest.mark.asyncio
    async def test_analyze_calculates_risk_score(self):
        """Test risk score calculation."""
        agent = RiskManagerAgent()

        context = {
            "code": "600519",
            "price_data": {"close": [100.0 + i * 0.5 for i in range(50)]},
            "risk_factors": ["factor1", "factor2", "factor3"],
            "agent_signals": {
                "technical": {"signal": "buy"},
                "fundamental": {"signal": "buy"},
                "sentiment": {"signal": "sell"},
            },
        }

        result = await agent.analyze(context)

        # Risk score should be calculated
        assert "risk_score" in result.metadata
        assert 0 <= result.metadata["risk_score"] <= 100

    @pytest.mark.asyncio
    async def test_analyze_with_multiple_positions_reduces_limit(self):
        """Test that multiple existing positions reduce position limit via correlation adjustment."""
        agent = RiskManagerAgent()

        # Low volatility prices
        prices = [100.0 + i * 0.05 for i in range(50)]

        # No existing positions - should get full limit
        context_single = {
            "code": "600519",
            "price_data": {"close": prices},
        }

        result_single = await agent.analyze(context_single)

        # With 5+ existing positions - correlation adjustment should reduce limit
        context_multiple = {
            "code": "600519",
            "price_data": {"close": prices},
            "existing_positions": ["600000", "600001", "600002", "600003", "600004"],
        }

        result_multiple = await agent.analyze(context_multiple)

        # Position limit should be reduced with multiple positions
        assert result_multiple.metadata["correlation_adjustment"] < result_single.metadata["correlation_adjustment"]
        assert result_multiple.metadata["max_position_size"] < result_single.metadata["max_position_size"]


class TestRiskManagerCalculations:
    """Tests for RiskManagerAgent calculation methods."""

    def test_classify_volatility_tier_low(self):
        """Test volatility < 15% is classified as 'low'."""
        agent = RiskManagerAgent()

        # Test boundary cases for low tier
        assert agent._classify_volatility_tier(0.10) == "low"
        assert agent._classify_volatility_tier(0.14) == "low"
        assert agent._classify_volatility_tier(0.001) == "low"

    def test_classify_volatility_tier_medium(self):
        """Test volatility 15-30% is classified as 'medium'."""
        agent = RiskManagerAgent()

        assert agent._classify_volatility_tier(0.15) == "medium"
        assert agent._classify_volatility_tier(0.20) == "medium"
        assert agent._classify_volatility_tier(0.29) == "medium"

    def test_classify_volatility_tier_high(self):
        """Test volatility 30-50% is classified as 'high'."""
        agent = RiskManagerAgent()

        assert agent._classify_volatility_tier(0.30) == "high"
        assert agent._classify_volatility_tier(0.40) == "high"
        assert agent._classify_volatility_tier(0.49) == "high"

    def test_classify_volatility_tier_very_high(self):
        """Test volatility > 50% is classified as 'very_high'."""
        agent = RiskManagerAgent()

        assert agent._classify_volatility_tier(0.50) == "very_high"
        assert agent._classify_volatility_tier(0.60) == "very_high"
        assert agent._classify_volatility_tier(1.0) == "very_high"

    def test_calculate_base_position_limit_low_vol(self):
        """Test base position limit for low volatility."""
        agent = RiskManagerAgent()

        # Low volatility should return 25% limit
        assert agent._calculate_base_position_limit(0.10) == 0.25
        assert agent._calculate_base_position_limit(0.14) == 0.25

    def test_calculate_base_position_limit_high_vol(self):
        """Test base position limit for high/very high volatility."""
        agent = RiskManagerAgent()

        # Very high volatility should return 10% limit
        assert agent._calculate_base_position_limit(0.55) == 0.10
        assert agent._calculate_base_position_limit(0.80) == 0.10

        # High volatility should return reduced limit
        limit_high = agent._calculate_base_position_limit(0.35)
        assert 0.05 <= limit_high <= 0.15

    def test_get_correlation_adjustment_no_positions(self):
        """Test correlation adjustment with no existing positions."""
        agent = RiskManagerAgent()

        # No positions should return 1.0 (no adjustment)
        context = {"existing_positions": []}
        assert agent._get_correlation_adjustment(context) == 1.0

        # Missing key should also return 1.0
        context = {}
        assert agent._get_correlation_adjustment(context) == 1.0

    def test_get_correlation_adjustment_many_positions(self):
        """Test correlation adjustment with many positions."""
        agent = RiskManagerAgent()

        # 1-2 positions: no adjustment
        context = {"existing_positions": ["600000"]}
        assert agent._get_correlation_adjustment(context) == 1.0

        context = {"existing_positions": ["600000", "600001"]}
        assert agent._get_correlation_adjustment(context) == 1.0

        # 3-4 positions: 0.95 adjustment
        context = {"existing_positions": ["600000", "600001", "600002"]}
        assert agent._get_correlation_adjustment(context) == 0.95

        context = {"existing_positions": ["600000", "600001", "600002", "600003"]}
        assert agent._get_correlation_adjustment(context) == 0.95

        # 5+ positions: 0.85 adjustment
        context = {"existing_positions": ["600000", "600001", "600002", "600003", "600004"]}
        assert agent._get_correlation_adjustment(context) == 0.85

    def test_calculate_volatility_from_list(self):
        """Test volatility calculation from price list."""
        agent = RiskManagerAgent()

        # Create price series with known volatility
        # Daily volatility ~2% -> annual ~32%
        import random

        random.seed(42)
        base_price = 100.0
        prices = [base_price]
        for _ in range(50):
            change = random.gauss(0, 0.02)  # 2% daily vol
            prices.append(prices[-1] * (1 + change))

        result = agent._calculate_volatility({"close": prices})

        assert "daily_volatility" in result
        assert "annualized_volatility" in result
        assert "volatility_percentile" in result

        # Check reasonable values
        assert 0 < result["daily_volatility"] < 0.10
        assert 0 < result["annualized_volatility"] < 1.0

    def test_calculate_volatility_insufficient_data(self):
        """Test volatility calculation with insufficient data returns defaults."""
        agent = RiskManagerAgent()

        # Empty price data
        result = agent._calculate_volatility({"close": []})
        assert result["annualized_volatility"] == 0.30  # Default
        assert result["daily_volatility"] == 0.02

        # Only a few data points
        result = agent._calculate_volatility({"close": [100.0, 101.0, 102.0]})
        assert result["annualized_volatility"] == 0.30

    def test_calculate_volatility_with_dict_close(self):
        """Test volatility calculation with dict containing close prices."""
        agent = RiskManagerAgent()

        # Dict with close prices
        prices = [100.0 + i * 0.5 for i in range(50)]
        result = agent._calculate_volatility({"close": prices})

        assert result["annualized_volatility"] < 0.50  # Low volatility

    def test_calculate_volatility_with_dataframe_like_object(self):
        """Test volatility calculation with DataFrame-like object."""

        agent = RiskManagerAgent()

        # Mock DataFrame-like object
        class MockDataFrame:
            def __init__(self, closes):
                self.close = type("MockClose", (), {"values": closes})()

        prices = [100.0 + i * 0.5 for i in range(50)]
        mock_df = MockDataFrame(prices)

        result = agent._calculate_volatility(mock_df)
        assert "annualized_volatility" in result

    def test_calculate_volatility_with_list_directly(self):
        """Test volatility calculation when price_data is a list directly."""
        agent = RiskManagerAgent()

        prices = [100.0 + i * 0.5 for i in range(50)]
        result = agent._calculate_volatility(prices)

        assert "annualized_volatility" in result
        assert result["annualized_volatility"] < 0.50

    def test_calculate_volatility_with_invalid_data(self):
        """Test volatility calculation with invalid data returns defaults."""
        agent = RiskManagerAgent()

        # Invalid data type
        result = agent._calculate_volatility("invalid")
        assert result["annualized_volatility"] == 0.30

    def test_calculate_risk_score_volatility_component(self):
        """Test risk score volatility component."""
        agent = RiskManagerAgent()

        # Low volatility: 10 points
        score_low = agent._calculate_risk_score({"annualized_volatility": 0.10}, [], {})
        assert 10 <= score_low <= 40  # At least 10 for volatility

        # Very high volatility: 40 points
        score_vhigh = agent._calculate_risk_score({"annualized_volatility": 0.60}, [], {})
        assert score_vhigh >= 40  # At least 40 for volatility

    def test_calculate_risk_score_risk_factors(self):
        """Test risk score risk factor component."""
        agent = RiskManagerAgent()

        base_vol = {"annualized_volatility": 0.10}  # Low vol = 10 points

        # No risk factors: 0 additional points
        score_none = agent._calculate_risk_score(base_vol, [], {})
        assert score_none == 10  # Only volatility points

        # 1-2 risk factors: 10 additional points
        score_few = agent._calculate_risk_score(base_vol, ["factor1"], {})
        assert score_few == 20  # 10 + 10

        # 3-4 risk factors: 20 additional points
        score_medium = agent._calculate_risk_score(base_vol, ["f1", "f2", "f3"], {})
        assert score_medium == 30  # 10 + 20

        # 5+ risk factors: 30 additional points
        score_many = agent._calculate_risk_score(base_vol, ["f1", "f2", "f3", "f4", "f5"], {})
        assert score_many == 40  # 10 + 30

    def test_calculate_risk_score_signal_divergence(self):
        """Test risk score signal divergence component."""
        agent = RiskManagerAgent()

        base_vol = {"annualized_volatility": 0.10}  # Low vol = 10 points

        # Full consensus (all buy): 0 divergence points
        signals_consensus = {
            "agent1": {"signal": "buy"},
            "agent2": {"signal": "buy"},
            "agent3": {"signal": "buy"},
        }
        score_consensus = agent._calculate_risk_score(base_vol, [], signals_consensus)
        assert score_consensus == 10  # Only volatility points

        # Full divergence (1 buy, 1 sell, 1 hold): max divergence
        signals_diverged = {
            "agent1": {"signal": "buy"},
            "agent2": {"signal": "sell"},
            "agent3": {"signal": "hold"},
        }
        score_diverged = agent._calculate_risk_score(base_vol, [], signals_diverged)
        assert score_diverged > 10  # Divergence adds points
