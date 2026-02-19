"""Tests for TechnicalAgent."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from stock_analyzer.ai.agents.technical_agent import TechnicalAgent
from stock_analyzer.models import SignalType


class TestTechnicalAgent:
    """Tests for TechnicalAgent."""

    def test_init(self):
        """Test agent initialization."""
        with patch("stock_analyzer.ai.agents.technical_agent.get_llm_client", return_value=None):
            agent = TechnicalAgent()
            assert agent.name == "TechnicalAgent"

    def test_is_available_returns_true(self):
        """Test is_available always returns True."""
        with patch("stock_analyzer.ai.agents.technical_agent.get_llm_client", return_value=None):
            agent = TechnicalAgent()
            assert agent.is_available() is True

    @pytest.mark.asyncio
    async def test_analyze_bullish_trend_returns_buy(self, sample_analysis_context):
        """Test bullish trend analysis returns BUY signal."""
        with patch("stock_analyzer.ai.agents.technical_agent.get_llm_client", return_value=None):
            agent = TechnicalAgent()

            # Create bullish context
            context = sample_analysis_context.copy()
            context["today"] = context["today"].copy()
            context["today"]["ma5"] = 1795.0  # close > ma5 > ma10 > ma20
            context["today"]["ma10"] = 1780.0
            context["today"]["ma20"] = 1770.0
            context["today"]["close"] = 1800.0
            context["ma_status"] = "多头排列"

            result = await agent.analyze(context)

            assert result.signal == SignalType.BUY
            assert result.confidence > 0
            assert "多头排列" in result.reasoning

    @pytest.mark.asyncio
    async def test_analyze_bearish_trend_returns_sell(self, sample_analysis_context):
        """Test bearish trend analysis returns SELL signal."""
        with patch("stock_analyzer.ai.agents.technical_agent.get_llm_client", return_value=None):
            agent = TechnicalAgent()

            # Create bearish context
            context = sample_analysis_context.copy()
            context["today"] = context["today"].copy()
            context["today"]["close"] = 1760.0
            context["today"]["ma5"] = 1770.0
            context["today"]["ma10"] = 1780.0
            context["today"]["ma20"] = 1790.0
            context["ma_status"] = "空头排列"

            result = await agent.analyze(context)

            assert result.signal == SignalType.SELL
            assert "空头" in result.reasoning

    @pytest.mark.asyncio
    async def test_analyze_missing_price_data_returns_hold(self):
        """Test analysis with missing price data returns HOLD."""
        with patch("stock_analyzer.ai.agents.technical_agent.get_llm_client", return_value=None):
            agent = TechnicalAgent()

            result = await agent.analyze(
                {
                    "code": "600519",
                    "today": {"close": 0, "ma5": 0},  # Invalid data
                    "ma_status": "unknown",
                }
            )

            assert result.signal == SignalType.HOLD
            assert result.confidence == 0

    @pytest.mark.asyncio
    async def test_analyze_with_llm_success(self, sample_analysis_context, mock_acompletion):
        """Test LLM-based analysis returns parsed result."""
        # Create a mock LLM client
        mock_llm_client = MagicMock()
        mock_llm_client.is_available.return_value = True
        mock_llm_client.generate_with_tool = AsyncMock(
            return_value={
                "signal": "buy",
                "confidence": 85,
                "reasoning": "Strong bullish trend",
                "trend_assessment": "bullish",
                "trend_strength": 80,
            }
        )

        with patch("stock_analyzer.ai.agents.technical_agent.get_llm_client", return_value=mock_llm_client):
            agent = TechnicalAgent()

            result = await agent.analyze(sample_analysis_context)

            assert result.signal == SignalType.BUY
            assert result.confidence == 85
            assert "trend_assessment" in result.metadata

    @pytest.mark.asyncio
    async def test_analyze_exception_returns_hold(self):
        """Test exception during analysis returns HOLD."""
        with patch("stock_analyzer.ai.agents.technical_agent.get_llm_client", return_value=None):
            agent = TechnicalAgent()

            # Context that will cause exception
            result = await agent.analyze({"code": "600519", "today": None})

            assert result.signal == SignalType.HOLD
            assert result.confidence == 0
            assert "error" in result.metadata


class TestTechnicalAgentHelpers:
    """Tests for TechnicalAgent helper methods."""

    def test_analyze_volume_significant(self):
        """Test volume analysis for significant volume."""
        with patch("stock_analyzer.ai.agents.technical_agent.get_llm_client", return_value=None):
            agent = TechnicalAgent()
            result = agent._analyze_volume(2.5)
            assert result == "显著放量"

    def test_analyze_volume_normal(self):
        """Test volume analysis for normal volume."""
        with patch("stock_analyzer.ai.agents.technical_agent.get_llm_client", return_value=None):
            agent = TechnicalAgent()
            result = agent._analyze_volume(1.0)
            assert result == "量能正常"

    def test_analyze_volume_shrinking(self):
        """Test volume analysis for shrinking volume."""
        with patch("stock_analyzer.ai.agents.technical_agent.get_llm_client", return_value=None):
            agent = TechnicalAgent()
            result = agent._analyze_volume(0.3)
            assert result == "明显缩量"

    def test_interpret_rsi_overbought(self):
        """Test RSI interpretation for overbought."""
        with patch("stock_analyzer.ai.agents.technical_agent.get_llm_client", return_value=None):
            agent = TechnicalAgent()
            result = agent._interpret_rsi(75)
            assert result == "超买"

    def test_interpret_rsi_oversold(self):
        """Test RSI interpretation for oversold."""
        with patch("stock_analyzer.ai.agents.technical_agent.get_llm_client", return_value=None):
            agent = TechnicalAgent()
            result = agent._interpret_rsi(15)
            assert result == "严重超卖"

    def test_interpret_adx_strong_trend(self):
        """Test ADX interpretation for strong trend."""
        with patch("stock_analyzer.ai.agents.technical_agent.get_llm_client", return_value=None):
            agent = TechnicalAgent()
            result = agent._interpret_adx(45)
            assert result == "很强趋势"

    def test_interpret_adx_no_trend(self):
        """Test ADX interpretation for no trend."""
        with patch("stock_analyzer.ai.agents.technical_agent.get_llm_client", return_value=None):
            agent = TechnicalAgent()
            result = agent._interpret_adx(15)
            assert result == "无趋势"
