"""Tests for valuation methods."""

import math

from ashare_analyzer.valuation.methods import (
    ValuationResult,
    calculate_adjusted_graham,
    calculate_dividend_discount,
    calculate_pb_percentile,
    calculate_peg_valuation,
    calculate_ps_valuation,
)


class TestValuationResult:
    """Tests for ValuationResult dataclass."""

    def test_valuation_result_creation(self):
        """Test basic ValuationResult creation."""
        result = ValuationResult(value=100.0, method="graham")
        assert result.value == 100.0
        assert result.method == "graham"
        assert result.skip_reason is None

    def test_valuation_result_with_skip_reason(self):
        """Test ValuationResult with skip reason."""
        result = ValuationResult(value=0.0, method="graham", skip_reason="EPS is negative")
        assert result.value == 0.0
        assert result.method == "graham"
        assert result.skip_reason == "EPS is negative"


class TestCalculateAdjustedGraham:
    """Tests for adjusted Graham valuation method."""

    def test_basic_calculation(self):
        """Test basic Graham calculation with valid inputs."""
        # EPS=5, BVPS=20, ROE=12%
        # A_share_factor = 22.5 * 2.0 * 1.0 = 45.0
        # Value = sqrt(45.0 * 5 * 20) = sqrt(4500) = 67.08
        result = calculate_adjusted_graham(eps=5.0, bvps=20.0, roe=12.0)

        assert result.method == "adjusted_graham"
        assert result.skip_reason is None
        assert math.isclose(result.value, math.sqrt(45.0 * 5.0 * 20.0), rel_tol=1e-6)

    def test_high_roe_quality_adjustment(self):
        """Test quality adjustment for ROE > 15%."""
        # ROE > 15% -> quality_adjustment = 1.2
        # A_share_factor = 22.5 * 2.0 * 1.2 = 54.0
        result = calculate_adjusted_graham(eps=5.0, bvps=20.0, roe=18.0)

        expected = math.sqrt(54.0 * 5.0 * 20.0)
        assert math.isclose(result.value, expected, rel_tol=1e-6)

    def test_medium_roe_quality_adjustment(self):
        """Test quality adjustment for ROE 10-15%."""
        # ROE 10-15% -> quality_adjustment = 1.0
        # A_share_factor = 22.5 * 2.0 * 1.0 = 45.0
        result = calculate_adjusted_graham(eps=5.0, bvps=20.0, roe=12.0)

        expected = math.sqrt(45.0 * 5.0 * 20.0)
        assert math.isclose(result.value, expected, rel_tol=1e-6)

    def test_low_roe_quality_adjustment(self):
        """Test quality adjustment for ROE < 10%."""
        # ROE < 10% -> quality_adjustment = 0.8
        # A_share_factor = 22.5 * 2.0 * 0.8 = 36.0
        result = calculate_adjusted_graham(eps=5.0, bvps=20.0, roe=8.0)

        expected = math.sqrt(36.0 * 5.0 * 20.0)
        assert math.isclose(result.value, expected, rel_tol=1e-6)

    def test_boundary_roe_10_percent(self):
        """Test ROE exactly at 10% boundary (should be medium adjustment)."""
        # ROE = 10% is in [10, 15], so quality_adjustment = 1.0
        result = calculate_adjusted_graham(eps=5.0, bvps=20.0, roe=10.0)

        expected = math.sqrt(45.0 * 5.0 * 20.0)  # 1.0 adjustment
        assert math.isclose(result.value, expected, rel_tol=1e-6)

    def test_boundary_roe_15_percent(self):
        """Test ROE exactly at 15% boundary (should be medium adjustment)."""
        # ROE = 15% is in [10, 15], so quality_adjustment = 1.0
        result = calculate_adjusted_graham(eps=5.0, bvps=20.0, roe=15.0)

        expected = math.sqrt(45.0 * 5.0 * 20.0)  # 1.0 adjustment
        assert math.isclose(result.value, expected, rel_tol=1e-6)

    def test_negative_eps_returns_zero(self):
        """Test that negative EPS returns value=0 with skip_reason."""
        result = calculate_adjusted_graham(eps=-1.0, bvps=20.0, roe=12.0)

        assert result.value == 0.0
        assert result.method == "adjusted_graham"
        assert result.skip_reason is not None
        assert "EPS" in result.skip_reason or "eps" in result.skip_reason

    def test_zero_eps_returns_zero(self):
        """Test that zero EPS returns value=0 with skip_reason."""
        result = calculate_adjusted_graham(eps=0.0, bvps=20.0, roe=12.0)

        assert result.value == 0.0
        assert result.skip_reason is not None

    def test_negative_bvps_returns_zero(self):
        """Test that negative BVPS returns value=0 with skip_reason."""
        result = calculate_adjusted_graham(eps=5.0, bvps=-10.0, roe=12.0)

        assert result.value == 0.0
        assert result.skip_reason is not None
        assert "BVPS" in result.skip_reason or "bvps" in result.skip_reason

    def test_zero_bvps_returns_zero(self):
        """Test that zero BVPS returns value=0 with skip_reason."""
        result = calculate_adjusted_graham(eps=5.0, bvps=0.0, roe=12.0)

        assert result.value == 0.0
        assert result.skip_reason is not None

    def test_default_roe_value(self):
        """Test that default ROE=10.0 is used when not provided."""
        # Default roe=10.0 is in [10, 15], so quality_adjustment = 1.0
        result = calculate_adjusted_graham(eps=5.0, bvps=20.0)

        expected = math.sqrt(45.0 * 5.0 * 20.0)
        assert math.isclose(result.value, expected, rel_tol=1e-6)


class TestCalculatePegValuation:
    """Tests for PEG valuation method."""

    def test_basic_calculation(self):
        """Test basic PEG calculation with valid inputs."""
        # EPS=5, PE=20, Growth=20%
        # Fair PEG for 15-30% growth = 1.0
        # Fair PE = 1.0 * 20 = 20
        # Fair Value = 5 * 20 = 100
        result = calculate_peg_valuation(eps=5.0, pe_ratio=20.0, growth_rate=20.0)

        assert result.method == "peg"
        assert result.skip_reason is None
        assert math.isclose(result.value, 100.0, rel_tol=1e-6)

    def test_high_growth_fair_peg(self):
        """Test fair PEG for growth > 30%."""
        # Growth > 30% -> Fair PEG = 0.8
        # Fair PE = 0.8 * 40 = 32
        # Fair Value = 5 * 32 = 160
        result = calculate_peg_valuation(eps=5.0, pe_ratio=25.0, growth_rate=40.0)

        assert math.isclose(result.value, 5.0 * 0.8 * 40.0, rel_tol=1e-6)

    def test_medium_growth_fair_peg(self):
        """Test fair PEG for growth 15-30%."""
        # Growth 15-30% -> Fair PEG = 1.0
        result = calculate_peg_valuation(eps=5.0, pe_ratio=20.0, growth_rate=20.0)

        assert math.isclose(result.value, 5.0 * 1.0 * 20.0, rel_tol=1e-6)

    def test_low_growth_fair_peg(self):
        """Test fair PEG for growth 10-15%."""
        # Growth 10-15% -> Fair PEG = 1.2
        result = calculate_peg_valuation(eps=5.0, pe_ratio=15.0, growth_rate=12.0)

        assert math.isclose(result.value, 5.0 * 1.2 * 12.0, rel_tol=1e-6)

    def test_boundary_growth_10_percent(self):
        """Test growth exactly at 10% boundary."""
        # Growth = 10% is in [10, 15], so Fair PEG = 1.2
        result = calculate_peg_valuation(eps=5.0, pe_ratio=15.0, growth_rate=10.0)

        assert math.isclose(result.value, 5.0 * 1.2 * 10.0, rel_tol=1e-6)

    def test_boundary_growth_15_percent(self):
        """Test growth exactly at 15% boundary."""
        # Growth = 15% is in [15, 30], so Fair PEG = 1.0
        result = calculate_peg_valuation(eps=5.0, pe_ratio=20.0, growth_rate=15.0)

        assert math.isclose(result.value, 5.0 * 1.0 * 15.0, rel_tol=1e-6)

    def test_boundary_growth_30_percent(self):
        """Test growth exactly at 30% boundary."""
        # Growth = 30% is in [15, 30], so Fair PEG = 1.0
        result = calculate_peg_valuation(eps=5.0, pe_ratio=25.0, growth_rate=30.0)

        assert math.isclose(result.value, 5.0 * 1.0 * 30.0, rel_tol=1e-6)

    def test_negative_eps_returns_zero(self):
        """Test that negative EPS returns value=0 with skip_reason."""
        result = calculate_peg_valuation(eps=-1.0, pe_ratio=20.0, growth_rate=20.0)

        assert result.value == 0.0
        assert result.method == "peg"
        assert result.skip_reason is not None

    def test_zero_eps_returns_zero(self):
        """Test that zero EPS returns value=0 with skip_reason."""
        result = calculate_peg_valuation(eps=0.0, pe_ratio=20.0, growth_rate=20.0)

        assert result.value == 0.0
        assert result.skip_reason is not None

    def test_negative_growth_returns_zero(self):
        """Test that negative growth rate returns value=0 with skip_reason."""
        result = calculate_peg_valuation(eps=5.0, pe_ratio=20.0, growth_rate=-5.0)

        assert result.value == 0.0
        assert result.skip_reason is not None

    def test_zero_growth_returns_zero(self):
        """Test that zero growth rate returns value=0 with skip_reason."""
        result = calculate_peg_valuation(eps=5.0, pe_ratio=20.0, growth_rate=0.0)

        assert result.value == 0.0
        assert result.skip_reason is not None


class TestCalculatePbPercentile:
    """Tests for PB percentile valuation method."""

    def test_basic_calculation_with_historical_pb(self):
        """Test PB calculation with historical PB median."""
        # BVPS=20, Historical PB Median=1.5
        # Fair Value = 20 * 1.5 = 30
        result = calculate_pb_percentile(bvps=20.0, current_pb=2.0, historical_pb_median=1.5)

        assert result.method == "pb_percentile"
        assert result.skip_reason is None
        assert math.isclose(result.value, 30.0, rel_tol=1e-6)

    def test_calculation_with_industry_pb(self):
        """Test PB calculation with industry PB (no historical)."""
        # BVPS=20, Industry PB=1.8
        # Fair Value = 20 * 1.8 = 36
        result = calculate_pb_percentile(bvps=20.0, current_pb=2.0, industry_pb=1.8)

        assert result.method == "pb_percentile"
        assert math.isclose(result.value, 36.0, rel_tol=1e-6)

    def test_calculation_with_default_pb(self):
        """Test PB calculation with default PB (no historical or industry)."""
        # BVPS=20, Default PB=2.0
        # Fair Value = 20 * 2.0 = 40
        result = calculate_pb_percentile(bvps=20.0, current_pb=2.5)

        assert result.method == "pb_percentile"
        assert math.isclose(result.value, 40.0, rel_tol=1e-6)

    def test_historical_pb_priority_over_industry(self):
        """Test that historical PB has priority over industry PB."""
        # Historical PB should be used even if industry PB is provided
        result = calculate_pb_percentile(bvps=20.0, current_pb=2.0, historical_pb_median=1.5, industry_pb=2.5)

        # Should use historical (1.5), not industry (2.5)
        assert math.isclose(result.value, 30.0, rel_tol=1e-6)

    def test_industry_pb_priority_over_default(self):
        """Test that industry PB has priority over default PB."""
        result = calculate_pb_percentile(bvps=20.0, current_pb=2.0, industry_pb=1.8)

        # Should use industry (1.8), not default (2.0)
        assert math.isclose(result.value, 36.0, rel_tol=1e-6)

    def test_negative_bvps_returns_zero(self):
        """Test that negative BVPS returns value=0 with skip_reason."""
        result = calculate_pb_percentile(bvps=-10.0, current_pb=2.0)

        assert result.value == 0.0
        assert result.method == "pb_percentile"
        assert result.skip_reason is not None

    def test_zero_bvps_returns_zero(self):
        """Test that zero BVPS returns value=0 with skip_reason."""
        result = calculate_pb_percentile(bvps=0.0, current_pb=2.0)

        assert result.value == 0.0
        assert result.skip_reason is not None


class TestCalculatePsValuation:
    """Tests for PS valuation method."""

    def test_basic_calculation_with_industry_ps(self):
        """Test PS calculation with industry PS."""
        # Current Price=50, Current PS=5.0, Industry PS=3.0
        # Fair Value = 50 * (3.0 / 5.0) = 30
        result = calculate_ps_valuation(current_price=50.0, current_ps=5.0, industry_ps=3.0)

        assert result.method == "ps"
        assert result.skip_reason is None
        assert math.isclose(result.value, 30.0, rel_tol=1e-6)

    def test_calculation_with_historical_ps(self):
        """Test PS calculation with historical PS (no industry)."""
        # Current Price=50, Current PS=5.0, Historical PS=4.0
        # Fair Value = 50 * (4.0 / 5.0) = 40
        result = calculate_ps_valuation(current_price=50.0, current_ps=5.0, historical_ps_median=4.0)

        assert math.isclose(result.value, 40.0, rel_tol=1e-6)

    def test_calculation_with_default_ps(self):
        """Test PS calculation with default PS (no industry or historical)."""
        # Current Price=50, Current PS=5.0, Default PS=2.0
        # Fair Value = 50 * (2.0 / 5.0) = 20
        result = calculate_ps_valuation(current_price=50.0, current_ps=5.0)

        assert math.isclose(result.value, 20.0, rel_tol=1e-6)

    def test_industry_ps_priority_over_historical(self):
        """Test that industry PS has priority over historical PS."""
        # Industry PS should be used even if historical PS is provided
        result = calculate_ps_valuation(current_price=50.0, current_ps=5.0, industry_ps=3.0, historical_ps_median=4.0)

        # Should use industry (3.0), not historical (4.0)
        assert math.isclose(result.value, 30.0, rel_tol=1e-6)

    def test_historical_ps_priority_over_default(self):
        """Test that historical PS has priority over default PS."""
        result = calculate_ps_valuation(current_price=50.0, current_ps=5.0, historical_ps_median=4.0)

        # Should use historical (4.0), not default (2.0)
        assert math.isclose(result.value, 40.0, rel_tol=1e-6)

    def test_zero_current_ps_returns_zero(self):
        """Test that zero current PS returns value=0 with skip_reason."""
        result = calculate_ps_valuation(current_price=50.0, current_ps=0.0)

        assert result.value == 0.0
        assert result.method == "ps"
        assert result.skip_reason is not None

    def test_negative_current_ps_returns_zero(self):
        """Test that negative current PS returns value=0 with skip_reason."""
        result = calculate_ps_valuation(current_price=50.0, current_ps=-1.0)

        assert result.value == 0.0
        assert result.skip_reason is not None


class TestCalculateDividendDiscount:
    """Tests for dividend discount valuation method."""

    def test_basic_calculation(self):
        """Test dividend discount calculation with valid inputs."""
        # Dividend=2.0, Required Return=0.08 (default)
        # Fair Value = 2.0 / 0.08 = 25.0
        result = calculate_dividend_discount(dividend=2.0)

        assert result.method == "dividend_discount"
        assert result.skip_reason is None
        assert math.isclose(result.value, 25.0, rel_tol=1e-6)

    def test_calculation_with_custom_required_return(self):
        """Test dividend discount with custom required return."""
        # Dividend=2.0, Required Return=0.10
        # Fair Value = 2.0 / 0.10 = 20.0
        result = calculate_dividend_discount(dividend=2.0, required_return=0.10)

        assert math.isclose(result.value, 20.0, rel_tol=1e-6)

    def test_default_required_return_calculation(self):
        """Test that default required return is calculated correctly."""
        # Default = A_SHARE_RISK_FREE_RATE + A_SHARE_RISK_PREMIUM = 0.03 + 0.05 = 0.08
        result = calculate_dividend_discount(dividend=1.6)

        # Fair Value = 1.6 / 0.08 = 20.0
        assert math.isclose(result.value, 20.0, rel_tol=1e-6)

    def test_negative_dividend_returns_zero(self):
        """Test that negative dividend returns value=0 with skip_reason."""
        result = calculate_dividend_discount(dividend=-1.0)

        assert result.value == 0.0
        assert result.method == "dividend_discount"
        assert result.skip_reason is not None

    def test_zero_dividend_returns_zero(self):
        """Test that zero dividend returns value=0 with skip_reason."""
        result = calculate_dividend_discount(dividend=0.0)

        assert result.value == 0.0
        assert result.skip_reason is not None
