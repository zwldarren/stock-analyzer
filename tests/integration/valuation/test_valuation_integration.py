"""Integration tests for valuation system."""

import pytest

from ashare_analyzer.ai.agents.valuation_agent import ValuationAgent


class TestValuationIntegration:
    """Integration tests for complete valuation flow."""

    @pytest.mark.asyncio
    async def test_cyclical_stock_valuation(self):
        """Test cyclical stock (洛阳钼业-like) gets reasonable valuation."""
        agent = ValuationAgent()

        context = {
            "code": "603993",
            "stock_name": "洛阳钼业",
            "current_price": 23.46,
            "valuation_data": {
                "pe_ratio": 26.0,
                "pb_ratio": 2.7,
                "eps": 0.90,
                "book_value_per_share": 8.69,
                "industry_name": "有色金属",
                "roe": 12.0,
            },
        }

        result = await agent.analyze(context)

        # Key assertion: NOT 100% confidence SELL
        assert result.confidence < 100, f"Confidence should decay, got {result.confidence}%"
        assert result.metadata.get("stock_type") == "cyclical"
        print(f"Signal: {result.signal}, Confidence: {result.confidence}%")
        print(f"Reasoning: {result.reasoning}")

    @pytest.mark.asyncio
    async def test_growth_stock_valuation(self):
        """Test growth stock gets PEG-based valuation."""
        agent = ValuationAgent()

        context = {
            "code": "test",
            "stock_name": "Growth Stock",
            "current_price": 100.0,
            "valuation_data": {
                "pe_ratio": 40.0,
                "eps": 2.5,
                "revenue_growth": 25.0,  # PEG uses revenue_growth
            },
        }

        result = await agent.analyze(context)

        assert result.metadata.get("stock_type") == "growth"
        # Growth stocks should use PEG method
        valuations = result.metadata.get("valuations", {})
        assert "peg" in valuations, f"Expected peg in valuations, got {valuations}"

    @pytest.mark.asyncio
    async def test_financial_stock_valuation(self):
        """Test financial stock (bank) gets PB + Dividend valuation.

        Note: If a financial stock meets value criteria (PE < 15, ROE > 10%, dividend > 2%),
        it will be classified as VALUE first. This test uses a bank that doesn't meet
        value criteria to test FINANCIAL classification.
        """
        agent = ValuationAgent()

        context = {
            "code": "601398",
            "stock_name": "某银行",
            "current_price": 5.0,
            "valuation_data": {
                "pe_ratio": 18.0,  # Higher PE (doesn't meet value PE < 15)
                "pb_ratio": 0.8,
                "eps": 0.28,
                "book_value_per_share": 6.25,
                "industry_name": "银行",
                "roe": 8.0,  # Lower ROE (doesn't meet value ROE > 10%)
                "dividend_yield": 1.5,  # Lower dividend (doesn't meet value > 2%)
            },
        }

        result = await agent.analyze(context)

        # This bank doesn't meet VALUE criteria, so should be classified as FINANCIAL
        assert result.metadata.get("stock_type") == "financial"

    @pytest.mark.asyncio
    async def test_value_stock_valuation(self):
        """Test value stock gets Graham + PB valuation."""
        agent = ValuationAgent()

        context = {
            "code": "test-value",
            "stock_name": "Value Stock",
            "current_price": 20.0,
            "valuation_data": {
                "pe_ratio": 12.0,
                "pb_ratio": 1.5,
                "eps": 1.67,
                "book_value_per_share": 13.33,
                "roe": 12.0,
                "dividend_yield": 3.0,
            },
        }

        result = await agent.analyze(context)

        assert result.metadata.get("stock_type") == "value"
        # Value stocks should use Graham and PB methods
        valuations = result.metadata.get("valuations", {})
        assert "adjusted_graham" in valuations or "pb_percentile" in valuations

    @pytest.mark.asyncio
    async def test_loss_making_stock_valuation(self):
        """Test loss-making stock handling."""
        agent = ValuationAgent()

        context = {
            "code": "test-loss",
            "stock_name": "Loss Making Stock",
            "current_price": 10.0,
            "valuation_data": {
                "pe_ratio": -10.0,  # Negative PE (loss)
                "pb_ratio": 5.0,
                "eps": -1.0,  # Negative EPS
                "book_value_per_share": 2.0,
                "ps_ratio": 0.5,  # P/S ratio
            },
        }

        result = await agent.analyze(context)

        # Should classify as loss_making
        assert result.metadata.get("stock_type") == "loss_making"
