"""Tests for LiteLLMClient."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestLiteLLMClientGenerateWithTool:
    """Tests for generate_with_tool method."""

    @pytest.mark.asyncio
    async def test_generate_with_tool_returns_parsed_arguments(self):
        """Test that generate_with_tool extracts and parses tool call arguments."""
        from ashare_analyzer.ai.clients import LiteLLMClient

        client = LiteLLMClient(
            model="openai/gpt-4",
            api_key="test-key",
        )

        tool = {
            "type": "function",
            "function": {
                "name": "test_tool",
                "parameters": {
                    "type": "object",
                    "properties": {"result": {"type": "string"}},
                },
            },
        }

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.tool_calls = [MagicMock()]
        mock_response.choices[0].message.tool_calls[0].function.arguments = json.dumps({"result": "success"})

        with patch("ashare_analyzer.ai.clients.acompletion", new_callable=AsyncMock, return_value=mock_response):
            result = await client.generate_with_tool(
                prompt="test prompt",
                tool=tool,
                generation_config={"temperature": 0.2},
            )

        assert result == {"result": "success"}

    @pytest.mark.asyncio
    async def test_generate_with_tool_returns_none_on_no_tool_calls(self):
        """Test that generate_with_tool returns None when no tool calls."""
        from ashare_analyzer.ai.clients import LiteLLMClient

        client = LiteLLMClient(
            model="openai/gpt-4",
            api_key="test-key",
        )

        tool = {
            "type": "function",
            "function": {
                "name": "test_tool",
                "parameters": {"type": "object"},
            },
        }

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.tool_calls = None

        with patch("ashare_analyzer.ai.clients.acompletion", new_callable=AsyncMock, return_value=mock_response):
            result = await client.generate_with_tool(
                prompt="test prompt",
                tool=tool,
                generation_config={"temperature": 0.2},
            )

        assert result is None

    @pytest.mark.asyncio
    async def test_generate_with_tool_returns_none_when_unavailable(self):
        """Test that generate_with_tool returns None when client is unavailable."""
        from ashare_analyzer.ai.clients import LiteLLMClient

        client = LiteLLMClient(
            model="openai/gpt-4",
            api_key=None,
        )

        result = await client.generate_with_tool(
            prompt="test prompt",
            tool={},
            generation_config={},
        )

        assert result is None


class TestGetFilterLLMClient:
    """Tests for get_filter_llm_client function."""

    def test_returns_none_when_no_api_key(self):
        """Test that get_filter_llm_client returns None when no API key configured."""
        from ashare_analyzer.ai.clients import get_filter_llm_client

        mock_config = MagicMock()
        mock_config.ai.llm_api_key = None

        with patch("ashare_analyzer.ai.clients.get_config", return_value=mock_config):
            result = get_filter_llm_client()

        assert result is None

    def test_uses_filter_model_when_configured(self):
        """Test that get_filter_llm_client uses filter-specific model when configured."""
        from ashare_analyzer.ai.clients import get_filter_llm_client

        mock_config = MagicMock()
        mock_config.ai.llm_api_key = "test-key"
        mock_config.ai.llm_model = "deepseek/deepseek-reasoner"
        mock_config.ai.llm_base_url = None
        mock_config.news_filter.news_filter_model = "openai/gpt-4o-mini"

        with patch("ashare_analyzer.ai.clients.get_config", return_value=mock_config):
            result = get_filter_llm_client()

        assert result is not None
        assert result.model == "openai/gpt-4o-mini"

    def test_falls_back_to_main_model_when_no_filter_model(self):
        """Test that get_filter_llm_client uses main model when filter model not configured."""
        from ashare_analyzer.ai.clients import get_filter_llm_client

        mock_config = MagicMock()
        mock_config.ai.llm_api_key = "test-key"
        mock_config.ai.llm_model = "deepseek/deepseek-reasoner"
        mock_config.ai.llm_base_url = None
        mock_config.news_filter.news_filter_model = None

        with patch("ashare_analyzer.ai.clients.get_config", return_value=mock_config):
            result = get_filter_llm_client()

        assert result is not None
        assert result.model == "deepseek/deepseek-reasoner"

    def test_returns_none_on_exception(self):
        """Test that get_filter_llm_client returns None on exception."""
        from ashare_analyzer.ai.clients import get_filter_llm_client

        with patch("ashare_analyzer.ai.clients.get_config", side_effect=Exception("Config error")):
            result = get_filter_llm_client()

        assert result is None
