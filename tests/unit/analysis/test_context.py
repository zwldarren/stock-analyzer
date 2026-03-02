"""
Tests for analysis context builders.

Focuses on EPS/BVPS range validation for A-share stocks.
"""

import pytest

from ashare_analyzer.analysis.context import build_valuation_context
from ashare_analyzer.models.quotes import UnifiedRealtimeQuote


class TestBuildValuationContext:
    """Tests for build_valuation_context function."""

    @pytest.mark.asyncio
    async def test_eps_calculation_within_new_range(self):
        """Test EPS calculation with PE ratios in the new extended range (2-200)."""
        # Mock data service that returns nothing (no industry data)
        mock_data_service = type(
            "MockDataService",
            (),
            {
                "_try_realtime_by_source": None,
            },
        )()

        # Test with PE = 2 (lower bound)
        mock_quote = UnifiedRealtimeQuote(
            code="000001",
            pe_ratio=2.0,
            pb_ratio=1.0,
        )

        result, alt_quote = await build_valuation_context(
            realtime_quote=mock_quote,
            daily_data=None,
            current_price=10.0,
            data_service=mock_data_service,
            stock_code="000001",
        )

        # EPS should be calculated: 10.0 / 2.0 = 5.0
        assert "eps" in result
        assert result["eps"] == 5.0

    @pytest.mark.asyncio
    async def test_eps_calculation_upper_bound(self):
        """Test EPS calculation at upper PE bound (200)."""
        mock_data_service = type("MockDataService", (), {"_try_realtime_by_source": None})()

        mock_quote = UnifiedRealtimeQuote(
            code="000001",
            pe_ratio=200.0,
            pb_ratio=1.0,
        )

        result, alt_quote = await build_valuation_context(
            realtime_quote=mock_quote,
            daily_data=None,
            current_price=100.0,
            data_service=mock_data_service,
            stock_code="000001",
        )

        # EPS should be calculated: 100.0 / 200.0 = 0.5
        assert "eps" in result
        assert result["eps"] == 0.5

    @pytest.mark.asyncio
    async def test_eps_calculation_below_range(self):
        """Test that EPS is NOT calculated when PE is below valid range (< 2)."""
        mock_data_service = type("MockDataService", (), {"_try_realtime_by_source": None})()

        mock_quote = UnifiedRealtimeQuote(
            code="000001",
            pe_ratio=1.5,  # Below range
            pb_ratio=1.0,
        )

        result, alt_quote = await build_valuation_context(
            realtime_quote=mock_quote,
            daily_data=None,
            current_price=10.0,
            data_service=mock_data_service,
            stock_code="000001",
        )

        # EPS should NOT be calculated
        assert "eps" not in result

    @pytest.mark.asyncio
    async def test_eps_calculation_above_range(self):
        """Test that EPS is NOT calculated when PE is above valid range (> 200)."""
        mock_data_service = type("MockDataService", (), {"_try_realtime_by_source": None})()

        mock_quote = UnifiedRealtimeQuote(
            code="000001",
            pe_ratio=250.0,  # Above range
            pb_ratio=1.0,
        )

        result, alt_quote = await build_valuation_context(
            realtime_quote=mock_quote,
            daily_data=None,
            current_price=100.0,
            data_service=mock_data_service,
            stock_code="000001",
        )

        # EPS should NOT be calculated
        assert "eps" not in result

    @pytest.mark.asyncio
    async def test_bvps_calculation_within_new_range(self):
        """Test BVPS calculation with PB ratios in the new extended range (0.3-30)."""
        mock_data_service = type("MockDataService", (), {"_try_realtime_by_source": None})()

        # Test with PB = 0.3 (lower bound)
        mock_quote = UnifiedRealtimeQuote(
            code="000001",
            pe_ratio=10.0,
            pb_ratio=0.3,
        )

        result, alt_quote = await build_valuation_context(
            realtime_quote=mock_quote,
            daily_data=None,
            current_price=30.0,
            data_service=mock_data_service,
            stock_code="000001",
        )

        # BVPS should be calculated: 30.0 / 0.3 = 100.0
        assert "book_value_per_share" in result
        assert result["book_value_per_share"] == 100.0

    @pytest.mark.asyncio
    async def test_bvps_calculation_upper_bound(self):
        """Test BVPS calculation at upper PB bound (30)."""
        mock_data_service = type("MockDataService", (), {"_try_realtime_by_source": None})()

        mock_quote = UnifiedRealtimeQuote(
            code="000001",
            pe_ratio=10.0,
            pb_ratio=30.0,
        )

        result, alt_quote = await build_valuation_context(
            realtime_quote=mock_quote,
            daily_data=None,
            current_price=300.0,
            data_service=mock_data_service,
            stock_code="000001",
        )

        # BVPS should be calculated: 300.0 / 30.0 = 10.0
        assert "book_value_per_share" in result
        assert result["book_value_per_share"] == 10.0

    @pytest.mark.asyncio
    async def test_bvps_calculation_below_range(self):
        """Test that BVPS is NOT calculated when PB is below valid range (< 0.3)."""
        mock_data_service = type("MockDataService", (), {"_try_realtime_by_source": None})()

        mock_quote = UnifiedRealtimeQuote(
            code="000001",
            pe_ratio=10.0,
            pb_ratio=0.2,  # Below range
        )

        result, alt_quote = await build_valuation_context(
            realtime_quote=mock_quote,
            daily_data=None,
            current_price=30.0,
            data_service=mock_data_service,
            stock_code="000001",
        )

        # BVPS should NOT be calculated
        assert "book_value_per_share" not in result

    @pytest.mark.asyncio
    async def test_bvps_calculation_above_range(self):
        """Test that BVPS is NOT calculated when PB is above valid range (> 30)."""
        mock_data_service = type("MockDataService", (), {"_try_realtime_by_source": None})()

        mock_quote = UnifiedRealtimeQuote(
            code="000001",
            pe_ratio=10.0,
            pb_ratio=35.0,  # Above range
        )

        result, alt_quote = await build_valuation_context(
            realtime_quote=mock_quote,
            daily_data=None,
            current_price=300.0,
            data_service=mock_data_service,
            stock_code="000001",
        )

        # BVPS should NOT be calculated
        assert "book_value_per_share" not in result

    @pytest.mark.asyncio
    async def test_old_range_still_works(self):
        """Test that stocks in the old range (PE 5-100, PB 0.5-20) still work."""
        mock_data_service = type("MockDataService", (), {"_try_realtime_by_source": None})()

        mock_quote = UnifiedRealtimeQuote(
            code="000001",
            pe_ratio=50.0,  # In old range
            pb_ratio=10.0,  # In old range
        )

        result, alt_quote = await build_valuation_context(
            realtime_quote=mock_quote,
            daily_data=None,
            current_price=100.0,
            data_service=mock_data_service,
            stock_code="000001",
        )

        # Both EPS and BVPS should be calculated
        assert "eps" in result
        assert result["eps"] == 2.0  # 100.0 / 50.0
        assert "book_value_per_share" in result
        assert result["book_value_per_share"] == 10.0  # 100.0 / 10.0

    @pytest.mark.asyncio
    async def test_bank_stock_low_pe(self):
        """Test A-share bank stock with low PE (< 5) now works."""
        mock_data_service = type("MockDataService", (), {"_try_realtime_by_source": None})()

        # Typical bank stock: PE = 4, PB = 0.5
        mock_quote = UnifiedRealtimeQuote(
            code="600000",
            pe_ratio=4.0,  # Below old range, within new range
            pb_ratio=0.5,
        )

        result, alt_quote = await build_valuation_context(
            realtime_quote=mock_quote,
            daily_data=None,
            current_price=20.0,
            data_service=mock_data_service,
            stock_code="600000",
        )

        # EPS should now be calculated: 20.0 / 4.0 = 5.0
        assert "eps" in result
        assert result["eps"] == 5.0

    @pytest.mark.asyncio
    async def test_growth_stock_high_pe(self):
        """Test growth stock with high PE (> 100) now works."""
        mock_data_service = type("MockDataService", (), {"_try_realtime_by_source": None})()

        # Typical growth stock: PE = 150, PB = 25
        mock_quote = UnifiedRealtimeQuote(
            code="300750",
            pe_ratio=150.0,  # Above old range, within new range
            pb_ratio=25.0,  # Above old range, within new range
        )

        result, alt_quote = await build_valuation_context(
            realtime_quote=mock_quote,
            daily_data=None,
            current_price=300.0,
            data_service=mock_data_service,
            stock_code="300750",
        )

        # Both should be calculated
        assert "eps" in result
        assert result["eps"] == 2.0  # 300.0 / 150.0
        assert "book_value_per_share" in result
        assert result["book_value_per_share"] == 12.0  # 300.0 / 25.0

    @pytest.mark.asyncio
    async def test_zero_price_returns_no_valuation(self):
        """Test that zero current price returns no EPS/BVPS."""
        mock_data_service = type("MockDataService", (), {"_try_realtime_by_source": None})()

        mock_quote = UnifiedRealtimeQuote(
            code="000001",
            pe_ratio=10.0,
            pb_ratio=1.0,
        )

        result, alt_quote = await build_valuation_context(
            realtime_quote=mock_quote,
            daily_data=None,
            current_price=0.0,
            data_service=mock_data_service,
            stock_code="000001",
        )

        # Neither EPS nor BVPS should be calculated
        assert "eps" not in result
        assert "book_value_per_share" not in result

    @pytest.mark.asyncio
    async def test_zero_pe_returns_no_eps(self):
        """Test that zero or negative PE returns no EPS."""
        mock_data_service = type("MockDataService", (), {"_try_realtime_by_source": None})()

        mock_quote = UnifiedRealtimeQuote(
            code="000001",
            pe_ratio=0.0,
            pb_ratio=1.0,
        )

        result, alt_quote = await build_valuation_context(
            realtime_quote=mock_quote,
            daily_data=None,
            current_price=10.0,
            data_service=mock_data_service,
            stock_code="000001",
        )

        assert "eps" not in result
