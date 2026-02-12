"""股票分析 AI 模块 - 多Agent架构"""

from stock_analyzer.ai.analyzer import AIAnalyzer
from stock_analyzer.ai.clients import LiteLLMClient, get_llm_client
from stock_analyzer.ai.prompts import (
    DECISION_SYSTEM_PROMPT,
    DECISION_USER_PROMPT_TEMPLATE,
    NEWS_SENTIMENT_SYSTEM_PROMPT,
    NEWS_SENTIMENT_USER_PROMPT_TEMPLATE,
    format_decision_prompt,
)
from stock_analyzer.domain.models import AnalysisResult

__all__ = [
    # Analyzer
    "AIAnalyzer",
    "AnalysisResult",
    # LLM Clients
    "LiteLLMClient",
    "get_llm_client",
    # Prompts
    "NEWS_SENTIMENT_SYSTEM_PROMPT",
    "NEWS_SENTIMENT_USER_PROMPT_TEMPLATE",
    "DECISION_SYSTEM_PROMPT",
    "DECISION_USER_PROMPT_TEMPLATE",
    "format_decision_prompt",
]
