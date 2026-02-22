"""Shared fixtures for ashare_analyzer tests."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.fixture
def mock_llm_response():
    """Create a mock LLM response with tool call."""

    def _create_response(result: dict) -> MagicMock:
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.tool_calls = [MagicMock()]
        mock_response.choices[0].message.tool_calls[0].function.arguments = json.dumps(result)
        return mock_response

    return _create_response


@pytest.fixture
def mock_acompletion(mock_llm_response):
    """Patch acompletion for LLM tests."""
    with patch("ashare_analyzer.ai.clients.acompletion", new_callable=AsyncMock) as mock:
        yield mock


@pytest.fixture
def sample_stock_data():
    """Sample OHLCV data for testing."""
    return {
        "close": 1800.0,
        "open": 1790.0,
        "high": 1810.0,
        "low": 1785.0,
        "volume": 1000000,
        "ma5": 1790.0,
        "ma10": 1780.0,
        "ma20": 1770.0,
        "pct_chg": 1.5,
        "volume_ratio": 1.2,
        "rsi_14": 55.0,
        "macd": 5.0,
        "macd_signal": 4.0,
        "macd_hist": 1.0,
        "adx": 25.0,
    }


@pytest.fixture
def sample_analysis_context(sample_stock_data):
    """Sample analysis context for testing agents."""
    return {
        "code": "600519",
        "stock_name": "贵州茅台",
        "today": sample_stock_data,
        "ma_status": "多头排列",
        "current_price": 1800.0,
        "valuation_data": {
            "eps": 50.0,
            "book_value_per_share": 150.0,
            "pe_ratio": 36.0,
            "pb_ratio": 12.0,
            "industry_pe": 30.0,
            "industry_pb": 8.0,
        },
    }


@pytest.fixture
def sample_agent_signal():
    """Sample agent signal for testing."""
    from ashare_analyzer.models import AgentSignal, SignalType

    return AgentSignal(
        agent_name="TestAgent",
        signal=SignalType.BUY,
        confidence=75,
        reasoning="Test reasoning",
        metadata={"test_key": "test_value"},
    )
