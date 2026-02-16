"""Tests for LiteLLMClient."""

import json
from unittest.mock import MagicMock, patch


class TestLiteLLMClientGenerateWithTool:
    """Tests for generate_with_tool method."""

    def test_generate_with_tool_returns_parsed_arguments(self):
        """Test that generate_with_tool extracts and parses tool call arguments."""
        from stock_analyzer.ai.clients import LiteLLMClient

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

        with patch("stock_analyzer.ai.clients.completion", return_value=mock_response):
            result = client.generate_with_tool(
                prompt="test prompt",
                tool=tool,
                generation_config={"temperature": 0.2},
            )

        assert result == {"result": "success"}

    def test_generate_with_tool_returns_none_on_no_tool_calls(self):
        """Test that generate_with_tool returns None when no tool calls."""
        from stock_analyzer.ai.clients import LiteLLMClient

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

        with patch("stock_analyzer.ai.clients.completion", return_value=mock_response):
            result = client.generate_with_tool(
                prompt="test prompt",
                tool=tool,
                generation_config={"temperature": 0.2},
            )

        assert result is None

    def test_generate_with_tool_returns_none_when_unavailable(self):
        """Test that generate_with_tool returns None when client is unavailable."""
        from stock_analyzer.ai.clients import LiteLLMClient

        client = LiteLLMClient(
            model="openai/gpt-4",
            api_key=None,  # No API key = unavailable
        )

        result = client.generate_with_tool(
            prompt="test prompt",
            tool={},
            generation_config={},
        )

        assert result is None
