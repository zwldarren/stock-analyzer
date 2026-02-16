"""
AI Tools Module

Defines Function Call schemas for LLM structured output.
"""

from typing import Any

# Unified function schema for agent analysis signals
ANALYZE_SIGNAL_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "analyze_signal",
        "description": "Analyze stock data and generate trading signal with reasoning",
        "parameters": {
            "type": "object",
            "properties": {
                "signal": {
                    "type": "string",
                    "enum": ["buy", "sell", "hold"],
                    "description": "Trading signal type",
                },
                "confidence": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": 100,
                    "description": "Confidence level (0-100)",
                },
                "reasoning": {
                    "type": "string",
                    "description": "Analysis reasoning (max 200 characters)",
                },
                "metadata": {
                    "type": "object",
                    "description": "Agent-specific analysis metadata",
                    "properties": {
                        # Technical Agent fields
                        "trend_assessment": {"type": "string"},
                        "trend_strength": {"type": "integer"},
                        "key_levels": {
                            "type": "object",
                            "properties": {
                                "support": {"type": "number"},
                                "resistance": {"type": "number"},
                                "ideal_entry": {"type": "number"},
                                "stop_loss": {"type": "number"},
                            },
                        },
                        "technical_factors": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "risk_factors": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "recommendation": {"type": "string"},
                        # Chip Agent fields
                        "control_assessment": {"type": "string"},
                        "phase": {"type": "string"},
                        "risk_level": {"type": "string"},
                        "key_factors": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        # News Sentiment fields
                        "sentiment": {"type": "string"},
                        "sentiment_score": {"type": "integer"},
                        "bullish_articles": {"type": "integer"},
                        "bearish_articles": {"type": "integer"},
                        "neutral_articles": {"type": "integer"},
                        "positive_catalysts": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "key_headlines": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "irrelevant_results": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        # Portfolio Manager fields
                        "action": {"type": "string"},
                        "position_ratio": {"type": "number"},
                    },
                },
            },
            "required": ["signal", "confidence", "reasoning"],
        },
    },
}


__all__ = ["ANALYZE_SIGNAL_TOOL"]
