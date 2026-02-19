"""Tests for AIAnalyzer."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from stock_analyzer.ai.analyzer import AIAnalyzer
from stock_analyzer.models import AgentSignal, AnalysisResult, SignalType


class TestAIAnalyzer:
    """Tests for AIAnalyzer."""

    def test_init(self):
        """Test AIAnalyzer initializes correctly."""
        with patch("stock_analyzer.ai.analyzer.get_display"):
            analyzer = AIAnalyzer()

            assert analyzer._agent_coordinator is not None
            assert analyzer._risk_manager_agent is not None
            assert analyzer._portfolio_manager is not None

    def test_is_available_returns_true(self):
        """Test is_available returns True when coordinator is initialized."""
        analyzer = AIAnalyzer.__new__(AIAnalyzer)
        analyzer._agent_coordinator = MagicMock()

        assert analyzer.is_available() is True

    def test_is_available_returns_false_when_none(self):
        """Test is_available returns False when coordinator is None."""
        analyzer = AIAnalyzer.__new__(AIAnalyzer)
        analyzer._agent_coordinator = None

        assert analyzer.is_available() is False

    @pytest.mark.asyncio
    async def test_analyze_returns_analysis_result(self, sample_analysis_context):
        """Test analyze returns AnalysisResult object."""
        # Create mock agents
        mock_coordinator = MagicMock()
        mock_coordinator.analyze = AsyncMock(
            return_value={
                "consensus": MagicMock(
                    participating_agents=["TechnicalAgent"],
                    risk_flags=[],
                ),
                "agent_signals": {
                    "TechnicalAgent": {
                        "signal": "buy",
                        "confidence": 75,
                        "reasoning": "Test reasoning",
                    }
                },
                "consensus_level": 0.8,
            }
        )

        mock_risk_signal = AgentSignal(
            agent_name="RiskManagerAgent",
            signal=SignalType.HOLD,
            confidence=50,
            reasoning="Risk assessment",
            metadata={"max_position_size": 0.25, "risk_score": 50},
        )

        mock_portfolio_signal = AgentSignal(
            agent_name="PortfolioManagerAgent",
            signal=SignalType.BUY,
            confidence=75,
            reasoning="Strong buy signal",
            metadata={
                "action": "BUY",
                "position_ratio": 0.2,
                "key_factors": ["趋势向上"],
                "risk_assessment": {"level": "low", "concerns": []},
            },
        )

        mock_risk_manager = MagicMock()
        mock_risk_manager.analyze = AsyncMock(return_value=mock_risk_signal)

        mock_portfolio_manager = MagicMock()
        mock_portfolio_manager.analyze = AsyncMock(return_value=mock_portfolio_signal)

        with patch("stock_analyzer.ai.analyzer.get_display"):
            analyzer = AIAnalyzer.__new__(AIAnalyzer)
            analyzer._agent_coordinator = mock_coordinator
            analyzer._risk_manager_agent = mock_risk_manager
            analyzer._portfolio_manager = mock_portfolio_manager

            result = await analyzer.analyze(sample_analysis_context)

        assert isinstance(result, AnalysisResult)
        assert result.code == "600519"
        assert result.name == "贵州茅台"
        assert result.success is True
        assert result.decision_type in ["buy", "hold", "sell"]

    @pytest.mark.asyncio
    async def test_analyze_builds_decision_dashboard(self, sample_analysis_context):
        """Test analyze builds decision dashboard with expected keys."""
        mock_coordinator = MagicMock()
        mock_coordinator.analyze = AsyncMock(
            return_value={
                "consensus": MagicMock(
                    participating_agents=["TechnicalAgent", "FundamentalAgent"],
                    risk_flags=[],
                ),
                "agent_signals": {
                    "TechnicalAgent": {
                        "signal": "buy",
                        "confidence": 80,
                        "reasoning": "Test",
                    },
                    "FundamentalAgent": {
                        "signal": "buy",
                        "confidence": 70,
                        "reasoning": "Test",
                    },
                },
                "consensus_level": 0.9,
            }
        )

        mock_risk_signal = AgentSignal(
            agent_name="RiskManagerAgent",
            signal=SignalType.HOLD,
            confidence=50,
            reasoning="Risk assessment",
            metadata={"max_position_size": 0.3, "risk_score": 40},
        )

        mock_portfolio_signal = AgentSignal(
            agent_name="PortfolioManagerAgent",
            signal=SignalType.BUY,
            confidence=85,
            reasoning="Strong fundamentals and technicals",
            metadata={
                "action": "BUY",
                "position_ratio": 0.25,
                "key_factors": ["估值合理", "趋势向上"],
                "risk_assessment": {"level": "medium", "concerns": ["市场波动"]},
            },
        )

        mock_risk_manager = MagicMock()
        mock_risk_manager.analyze = AsyncMock(return_value=mock_risk_signal)

        mock_portfolio_manager = MagicMock()
        mock_portfolio_manager.analyze = AsyncMock(return_value=mock_portfolio_signal)

        with patch("stock_analyzer.ai.analyzer.get_display"):
            analyzer = AIAnalyzer.__new__(AIAnalyzer)
            analyzer._agent_coordinator = mock_coordinator
            analyzer._risk_manager_agent = mock_risk_manager
            analyzer._portfolio_manager = mock_portfolio_manager

            result = await analyzer.analyze(sample_analysis_context)

        assert result.dashboard is not None
        assert "final_decision" in result.dashboard
        assert "key_factors" in result.dashboard
        assert "risk_assessment" in result.dashboard
        assert "agent_consensus" in result.dashboard

        # Check final_decision structure
        assert "action" in result.dashboard["final_decision"]
        assert "confidence" in result.dashboard["final_decision"]
        assert "position_ratio" in result.dashboard["final_decision"]

        # Check agent_consensus structure
        assert "signals" in result.dashboard["agent_consensus"]
        assert "confidences" in result.dashboard["agent_consensus"]
        assert "consensus_level" in result.dashboard["agent_consensus"]

    @pytest.mark.asyncio
    async def test_analyze_handles_exception_gracefully(self, sample_analysis_context):
        """Test analyze returns error result on failure."""
        mock_coordinator = MagicMock()
        mock_coordinator.analyze = AsyncMock(side_effect=Exception("API Error"))

        mock_risk_signal = AgentSignal(
            agent_name="RiskManagerAgent",
            signal=SignalType.HOLD,
            confidence=50,
            reasoning="Risk",
            metadata={"max_position_size": 0.25},
        )

        mock_risk_manager = MagicMock()
        mock_risk_manager.analyze = AsyncMock(return_value=mock_risk_signal)

        with patch("stock_analyzer.ai.analyzer.get_display"):
            analyzer = AIAnalyzer.__new__(AIAnalyzer)
            analyzer._agent_coordinator = mock_coordinator
            analyzer._risk_manager_agent = mock_risk_manager

            result = await analyzer.analyze(sample_analysis_context)

        assert isinstance(result, AnalysisResult)
        assert result.success is False
        assert result.error_message == "API Error"
        assert result.decision_type == "hold"
        assert result.confidence_level == "低"

    @pytest.mark.asyncio
    async def test_analyze_handles_risk_manager_exception(self, sample_analysis_context):
        """Test analyze handles risk manager failure."""
        mock_risk_manager = MagicMock()
        mock_risk_manager.analyze = AsyncMock(side_effect=Exception("Risk manager failed"))

        with patch("stock_analyzer.ai.analyzer.get_display"):
            analyzer = AIAnalyzer.__new__(AIAnalyzer)
            analyzer._agent_coordinator = MagicMock()
            analyzer._risk_manager_agent = mock_risk_manager

            result = await analyzer.analyze(sample_analysis_context)

        assert result.success is False
        assert "Risk manager failed" in result.error_message

    @pytest.mark.asyncio
    async def test_batch_analyze_processes_multiple(self, sample_analysis_context):
        """Test batch_analyze processes multiple stocks."""
        contexts = [
            {**sample_analysis_context, "code": "600519"},
            {**sample_analysis_context, "code": "000858"},
            {**sample_analysis_context, "code": "000001"},
        ]

        # Mock single analyze method
        mock_results = [
            AnalysisResult(
                code="600519",
                name="贵州茅台",
                sentiment_score=80,
                trend_prediction="看多",
                operation_advice="买入",
                success=True,
            ),
            AnalysisResult(
                code="000858",
                name="五粮液",
                sentiment_score=70,
                trend_prediction="震荡",
                operation_advice="持有",
                success=True,
            ),
            AnalysisResult(
                code="000001",
                name="平安银行",
                sentiment_score=50,
                trend_prediction="看空",
                operation_advice="卖出",
                success=True,
            ),
        ]

        with patch("stock_analyzer.ai.analyzer.get_display"):
            analyzer = AIAnalyzer.__new__(AIAnalyzer)
            analyzer._agent_coordinator = MagicMock()

            # Mock analyze to return results in order
            analyze_mock = AsyncMock(side_effect=mock_results)
            analyzer.analyze = analyze_mock

            results = await analyzer.batch_analyze(contexts, delay_between=0)

        assert len(results) == 3
        assert results[0].code == "600519"
        assert results[1].code == "000858"
        assert results[2].code == "000001"

    @pytest.mark.asyncio
    async def test_analyze_uses_realtime_name_when_available(self):
        """Test analyze uses realtime name from context."""
        context = {
            "code": "600519",
            "stock_name": "",  # Empty name
            "realtime": {"name": "贵州茅台"},
            "today": {},
        }

        mock_coordinator = MagicMock()
        mock_coordinator.analyze = AsyncMock(
            return_value={
                "consensus": MagicMock(participating_agents=[], risk_flags=[]),
                "agent_signals": {},
                "consensus_level": 0.5,
            }
        )

        mock_risk_signal = AgentSignal(
            agent_name="RiskManagerAgent",
            signal=SignalType.HOLD,
            confidence=50,
            reasoning="Test",
            metadata={"max_position_size": 0.25},
        )

        mock_portfolio_signal = AgentSignal(
            agent_name="PortfolioManagerAgent",
            signal=SignalType.HOLD,
            confidence=50,
            reasoning="Test",
            metadata={"action": "HOLD", "position_ratio": 0.0, "key_factors": [], "risk_assessment": {}},
        )

        mock_risk_manager = MagicMock()
        mock_risk_manager.analyze = AsyncMock(return_value=mock_risk_signal)

        mock_portfolio_manager = MagicMock()
        mock_portfolio_manager.analyze = AsyncMock(return_value=mock_portfolio_signal)

        with patch("stock_analyzer.ai.analyzer.get_display"):
            analyzer = AIAnalyzer.__new__(AIAnalyzer)
            analyzer._agent_coordinator = mock_coordinator
            analyzer._risk_manager_agent = mock_risk_manager
            analyzer._portfolio_manager = mock_portfolio_manager

            result = await analyzer.analyze(context)

        assert result.name == "贵州茅台"


class TestAIAnalyzerHelpers:
    """Tests for AIAnalyzer helper methods."""

    def test_score_to_confidence_high(self):
        """Test score >= 80 returns '高'."""
        analyzer = AIAnalyzer.__new__(AIAnalyzer)
        analyzer._agent_coordinator = MagicMock()

        assert analyzer._score_to_confidence(80) == "高"
        assert analyzer._score_to_confidence(85) == "高"
        assert analyzer._score_to_confidence(100) == "高"

    def test_score_to_confidence_medium(self):
        """Test score 60-79 returns '中'."""
        analyzer = AIAnalyzer.__new__(AIAnalyzer)
        analyzer._agent_coordinator = MagicMock()

        assert analyzer._score_to_confidence(60) == "中"
        assert analyzer._score_to_confidence(70) == "中"
        assert analyzer._score_to_confidence(79) == "中"

    def test_score_to_confidence_low(self):
        """Test score < 60 returns '低'."""
        analyzer = AIAnalyzer.__new__(AIAnalyzer)
        analyzer._agent_coordinator = MagicMock()

        assert analyzer._score_to_confidence(59) == "低"
        assert analyzer._score_to_confidence(50) == "低"
        assert analyzer._score_to_confidence(0) == "低"

    def test_format_price_valid(self):
        """Test formatting valid price."""
        assert AIAnalyzer._format_price(1800.5) == "1800.50"
        assert AIAnalyzer._format_price(100.0) == "100.00"
        assert AIAnalyzer._format_price(12.345) == "12.35"

    def test_format_price_none(self):
        """Test formatting None price returns N/A."""
        assert AIAnalyzer._format_price(None) == "N/A"

    def test_format_price_zero(self):
        """Test formatting zero price."""
        assert AIAnalyzer._format_price(0) == "0.00"

    def test_format_percent_valid(self):
        """Test formatting valid percentage."""
        assert AIAnalyzer._format_percent(1.5) == "1.50%"
        assert AIAnalyzer._format_percent(10.0) == "10.00%"
        assert AIAnalyzer._format_percent(-2.5) == "-2.50%"

    def test_format_percent_none(self):
        """Test formatting None percentage returns N/A."""
        assert AIAnalyzer._format_percent(None) == "N/A"

    def test_format_percent_zero(self):
        """Test formatting zero percentage."""
        assert AIAnalyzer._format_percent(0) == "0.00%"

    def test_collect_data_sources(self):
        """Test collecting data sources from agent signals."""
        analyzer = AIAnalyzer.__new__(AIAnalyzer)
        analyzer._agent_coordinator = MagicMock()

        agent_signals = {
            "TechnicalAgent": {"signal": "buy"},
            "FundamentalAgent": {"signal": "hold"},
        }

        result = analyzer._collect_data_sources(agent_signals)

        assert "TechnicalAgent" in result
        assert "FundamentalAgent" in result

    def test_collect_data_sources_empty(self):
        """Test collecting data sources from empty signals."""
        analyzer = AIAnalyzer.__new__(AIAnalyzer)
        analyzer._agent_coordinator = MagicMock()

        result = analyzer._collect_data_sources({})

        assert result == "multi-agent"

    def test_build_market_snapshot(self):
        """Test building market snapshot from context."""
        analyzer = AIAnalyzer.__new__(AIAnalyzer)
        analyzer._agent_coordinator = MagicMock()

        context = {
            "date": "2024-01-15",
            "today": {
                "close": 1800.0,
                "open": 1790.0,
                "high": 1810.0,
                "low": 1785.0,
                "pct_chg": 1.5,
                "volume": 1000000,
                "amount": 1800000000,
            },
            "yesterday": {
                "close": 1773.0,
            },
            "realtime": {
                "price": 1805.0,
                "change_amount": 5.0,
                "volume_ratio": 1.2,
                "turnover_rate": 0.8,
                "source": "sina",
            },
        }

        snapshot = analyzer._build_market_snapshot(context)

        assert snapshot["date"] == "2024-01-15"
        assert snapshot["close"] == "1800.00"
        assert snapshot["open"] == "1790.00"
        assert snapshot["high"] == "1810.00"
        assert snapshot["low"] == "1785.00"
        assert snapshot["pct_chg"] == "1.50%"
        assert snapshot["amplitude"] is not None
        assert snapshot["price"] == "1805.00"
        assert snapshot["source"] == "sina"

    def test_build_market_snapshot_minimal(self):
        """Test building market snapshot with minimal data."""
        analyzer = AIAnalyzer.__new__(AIAnalyzer)
        analyzer._agent_coordinator = MagicMock()

        context = {
            "date": "2024-01-15",
        }

        snapshot = analyzer._build_market_snapshot(context)

        assert snapshot["date"] == "2024-01-15"
        assert snapshot["close"] == "N/A"
        assert snapshot["amplitude"] == "N/A"

    def test_build_market_snapshot_with_none_values(self):
        """Test building market snapshot handles None values gracefully."""
        analyzer = AIAnalyzer.__new__(AIAnalyzer)
        analyzer._agent_coordinator = MagicMock()

        context = {
            "date": "2024-01-15",
            "today": {
                "close": None,
                "high": None,
                "low": None,
                "pct_chg": None,
            },
            "yesterday": {"close": None},
        }

        snapshot = analyzer._build_market_snapshot(context)

        assert snapshot["close"] == "N/A"
        assert snapshot["amplitude"] == "N/A"


class TestAIAnalyzerDecisionDashboard:
    """Tests for decision dashboard building."""

    def test_build_decision_dashboard_structure(self):
        """Test _build_decision_dashboard returns correct structure."""
        analyzer = AIAnalyzer.__new__(AIAnalyzer)
        analyzer._agent_coordinator = MagicMock()

        # Create a mock final signal
        mock_signal = MagicMock()
        mock_signal.confidence = 80
        mock_signal.reasoning = "Strong buy signal"
        mock_signal.metadata = {
            "action": "BUY",
            "position_ratio": 0.25,
            "key_factors": ["趋势向上", "估值合理"],
            "risk_assessment": {"level": "low", "concerns": []},
            "current_position": "none",
        }

        agent_signals = {
            "TechnicalAgent": {"signal": "buy", "confidence": 85, "reasoning": "MACD金叉"},
            "FundamentalAgent": {"signal": "buy", "confidence": 75, "reasoning": "业绩增长"},
        }

        context = {"today": {"close": 1800.0}}

        dashboard = analyzer._build_decision_dashboard(mock_signal, agent_signals, 0.8, context)

        assert dashboard["final_decision"]["action"] == "BUY"
        assert dashboard["final_decision"]["confidence"] == 80
        assert dashboard["final_decision"]["position_ratio"] == "25%"
        assert len(dashboard["key_factors"]) == 2
        assert dashboard["agent_consensus"]["consensus_level"] == "80%"

    def test_build_decision_summary(self):
        """Test _build_decision_summary returns formatted text."""
        analyzer = AIAnalyzer.__new__(AIAnalyzer)
        analyzer._agent_coordinator = MagicMock()

        mock_signal = MagicMock()
        mock_signal.confidence = 75
        mock_signal.metadata = {"action": "BUY"}

        agent_signals = {
            "TechnicalAgent": {"signal": "buy", "confidence": 80},
            "FundamentalAgent": {"signal": "hold", "confidence": 50},
        }

        key_factors = ["趋势向上", "估值合理"]

        summary = analyzer._build_decision_summary(mock_signal, agent_signals, 0.75, key_factors)

        assert "最终决策: BUY" in summary
        assert "置信度: 75%" in summary
        assert "共识度: 75%" in summary
        assert "关键因子" in summary
        assert "TechnicalAgent" in summary

    def test_build_analysis_result_from_decision_buy(self):
        """Test building AnalysisResult for BUY decision."""
        analyzer = AIAnalyzer.__new__(AIAnalyzer)
        analyzer._agent_coordinator = MagicMock()

        mock_signal = MagicMock()
        mock_signal.signal = "buy"  # String representation for dict lookup
        mock_signal.confidence = 85
        mock_signal.reasoning = "Strong buy signal"
        mock_signal.metadata = {
            "action": "BUY",
            "position_ratio": 0.25,
            "key_factors": ["趋势向上"],
            "risk_assessment": {"level": "low", "concerns": []},
        }

        agent_signals = {
            "TechnicalAgent": {"signal": "buy", "confidence": 80, "reasoning": "Test"},
        }

        context = {"today": {"close": 1800.0}, "date": "2024-01-15"}

        result = analyzer._build_analysis_result_from_decision(
            code="600519",
            name="贵州茅台",
            final_signal=mock_signal,
            agent_signals=agent_signals,
            consensus_level=0.8,
            context=context,
        )

        assert result.code == "600519"
        assert result.name == "贵州茅台"
        assert result.sentiment_score == 85
        assert result.trend_prediction == "看多"  # High confidence buy (85 >= 70)
        assert result.operation_advice == "买入"
        assert result.decision_type == "buy"
        assert result.confidence_level == "高"  # 85 >= 80
        assert result.final_action == "BUY"
        assert result.position_ratio == 0.25
        assert result.success is True

    def test_build_analysis_result_from_decision_sell(self):
        """Test building AnalysisResult for SELL decision."""
        analyzer = AIAnalyzer.__new__(AIAnalyzer)
        analyzer._agent_coordinator = MagicMock()

        mock_signal = MagicMock()
        mock_signal.signal = "sell"  # String representation for dict lookup
        mock_signal.confidence = 75
        mock_signal.reasoning = "Sell signal"
        mock_signal.metadata = {
            "action": "SELL",
            "position_ratio": 1.0,  # Full exit
            "key_factors": ["趋势向下"],
            "risk_assessment": {"level": "high", "concerns": ["市场风险"]},
        }

        agent_signals = {
            "TechnicalAgent": {"signal": "sell", "confidence": 70, "reasoning": "Test"},
        }

        context = {"today": {"close": 1700.0}, "date": "2024-01-15"}

        result = analyzer._build_analysis_result_from_decision(
            code="600519",
            name="贵州茅台",
            final_signal=mock_signal,
            agent_signals=agent_signals,
            consensus_level=0.7,
            context=context,
        )

        assert result.trend_prediction == "看空"  # 75 >= 70
        assert result.operation_advice == "卖出"
        assert result.decision_type == "sell"
        assert result.confidence_level == "中"  # 60 <= 75 < 80

    def test_build_analysis_result_from_decision_hold(self):
        """Test building AnalysisResult for HOLD decision."""
        analyzer = AIAnalyzer.__new__(AIAnalyzer)
        analyzer._agent_coordinator = MagicMock()

        mock_signal = MagicMock()
        mock_signal.signal = SignalType.HOLD
        mock_signal.confidence = 50
        mock_signal.reasoning = "Hold signal"
        mock_signal.metadata = {
            "action": "HOLD",
            "position_ratio": 0.0,
            "key_factors": [],
            "risk_assessment": {"level": "medium", "concerns": []},
        }

        agent_signals = {}

        context = {"today": {"close": 1800.0}, "date": "2024-01-15"}

        result = analyzer._build_analysis_result_from_decision(
            code="600519",
            name="贵州茅台",
            final_signal=mock_signal,
            agent_signals=agent_signals,
            consensus_level=0.5,
            context=context,
        )

        assert result.trend_prediction == "震荡"
        assert result.operation_advice == "持有"
        assert result.decision_type == "hold"
        assert result.confidence_level == "低"  # 50 < 60
