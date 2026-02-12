"""
News Sentiment Agent Module

Specialized agent for analyzing news sentiment and market sentiment using LLM.

This agent focuses exclusively on:
- News article sentiment classification
- Market sentiment analysis
- Positive/negative catalyst identification
- Risk factor detection from news

Inspired by ai-hedge-fund's News Sentiment Analyst.
"""

import logging
from typing import TYPE_CHECKING, Any

import json_repair

from stock_analyzer.ai.clients import get_llm_client
from stock_analyzer.ai.prompts import (
    NEWS_SENTIMENT_SYSTEM_PROMPT,
    NEWS_SENTIMENT_USER_PROMPT_TEMPLATE,
)
from stock_analyzer.infrastructure.search import SearchService

from .base import AgentSignal, BaseAgent, SignalType

if TYPE_CHECKING:
    from stock_analyzer.ai.clients import LiteLLMClient

logger = logging.getLogger(__name__)


class NewsSentimentAgent(BaseAgent):
    """
    News Sentiment Agent for analyzing market news and sentiment.

    This agent specializes in:
    - News sentiment classification using LLM
    - Bullish/bearish article counting
    - Risk and catalyst identification from news
    - Sentiment score aggregation

    Attributes:
        search_service: Search service for news retrieval
        llm_client: LLM client for sentiment analysis

    Example:
        agent = NewsSentimentAgent(search_service)
        signal = agent.analyze({
            "code": "600519",
            "stock_name": "贵州茅台"
        })
    """

    def __init__(self, search_service: SearchService | None = None):
        """
        Initialize the News Sentiment Agent.

        Args:
            search_service: SearchService instance. If None, will attempt to
                          get from global configuration.
        """
        super().__init__("NewsSentimentAgent")
        self._search_service = search_service
        self._logger = logging.getLogger(__name__)
        self._llm_client: LiteLLMClient | None = None
        self._init_llm_client()

    def _init_llm_client(self) -> None:
        """Initialize LLM client for sentiment analysis."""
        self._llm_client = get_llm_client()
        if self._llm_client:
            self._logger.info("NewsSentimentAgent LLM client initialized successfully")
        else:
            self._logger.warning("No LLM API key configured, NewsSentimentAgent will be unavailable")

    def _get_search_service(self) -> SearchService:
        """Get or initialize search service."""
        if self._search_service is None:
            from stock_analyzer.core.dependencies import get_search_service

            self._search_service = get_search_service()
        return self._search_service

    def is_available(self) -> bool:
        """Check if both search service and LLM are available."""
        try:
            has_search = self._get_search_service().is_available
            has_llm = self._llm_client is not None and self._llm_client.is_available()
            return has_search and has_llm
        except Exception as e:
            logger.debug(f"Error checking availability: {e}")
            return False

    def analyze(self, context: dict[str, Any]) -> AgentSignal:
        """
        Execute news sentiment analysis.

        Args:
            context: Analysis context containing:
                - code: Stock code
                - stock_name: Stock name

        Returns:
            AgentSignal with sentiment-based trading signal
        """
        stock_code = context.get("code", "")
        stock_name = context.get("stock_name", "")

        self._logger.info(f"[{stock_code}] NewsSentimentAgent开始新闻情绪分析")

        try:
            search_service = self._get_search_service()

            # Search for latest news
            news_results = search_service.search_stock_news(
                stock_code=stock_code,
                stock_name=stock_name,
                max_results=10,
            )

            if not news_results or not news_results.results:
                return AgentSignal(
                    agent_name=self.name,
                    signal=SignalType.HOLD,
                    confidence=0,
                    reasoning="未找到相关新闻",
                    metadata={"error": "no_news_found"},
                )

            # Use LLM to analyze sentiment
            llm_analysis = self._analyze_sentiment_with_llm(stock_code, stock_name, news_results.results)

            if not llm_analysis:
                return AgentSignal(
                    agent_name=self.name,
                    signal=SignalType.HOLD,
                    confidence=0,
                    reasoning="LLM情感分析失败",
                    metadata={"error": "llm_analysis_failed"},
                )

            # Extract signal from LLM analysis
            signal_str = llm_analysis.get("signal", "neutral")
            signal = SignalType.from_string(signal_str)
            confidence = llm_analysis.get("confidence", 50)
            reasoning = llm_analysis.get("reasoning", "无详细理由")

            self._logger.info(
                f"[{stock_code}] NewsSentimentAgent分析完成: {signal_str} "
                f"(置信度{confidence}%, 情绪{llm_analysis.get('sentiment', 'neutral')})"
            )

            return AgentSignal(
                agent_name=self.name,
                signal=signal,
                confidence=confidence,
                reasoning=reasoning,
                metadata={
                    "sentiment": llm_analysis.get("sentiment", "neutral"),
                    "sentiment_score": llm_analysis.get("sentiment_score", 50),
                    "bullish_articles": llm_analysis.get("bullish_articles", 0),
                    "bearish_articles": llm_analysis.get("bearish_articles", 0),
                    "neutral_articles": llm_analysis.get("neutral_articles", 0),
                    "risk_factors": llm_analysis.get("risk_factors", []),
                    "positive_catalysts": llm_analysis.get("positive_catalysts", []),
                    "key_headlines": llm_analysis.get("key_headlines", []),
                    "irrelevant_results": llm_analysis.get("irrelevant_results", []),
                    "total_articles": len(news_results.results),
                },
            )

        except Exception as e:
            self._logger.error(f"[{stock_code}] NewsSentimentAgent分析失败: {e}")
            return AgentSignal(
                agent_name=self.name,
                signal=SignalType.HOLD,
                confidence=0,
                reasoning=f"新闻情绪分析失败: {str(e)}",
                metadata={"error": str(e)},
            )

    def _analyze_sentiment_with_llm(
        self, stock_code: str, stock_name: str, news_items: list[Any]
    ) -> dict[str, Any] | None:
        """
        Use LLM to analyze news sentiment and generate signal.

        Args:
            stock_code: Stock code
            stock_name: Stock name
            news_items: List of news articles

        Returns:
            Parsed LLM analysis result with sentiment and signal
        """
        if not self._llm_client or not self._llm_client.is_available():
            self._logger.warning("LLM client not available, cannot perform sentiment analysis")
            return None

        try:
            # Format news items for LLM
            news_context = self._format_news_items(news_items)

            # Build prompt using template
            prompt = NEWS_SENTIMENT_USER_PROMPT_TEMPLATE.format(
                stock_code=stock_code,
                stock_name=stock_name,
                news_context=news_context,
            )

            # Call LLM
            self._logger.info(f"[{stock_code}] NewsSentimentAgent调用LLM分析新闻情绪...")
            response = self._llm_client.generate(
                prompt=prompt,
                generation_config={"temperature": 0.3, "max_output_tokens": 2048},
                system_prompt=NEWS_SENTIMENT_SYSTEM_PROMPT,
            )

            # Parse JSON response
            result = json_repair.repair_json(response, return_objects=True)

            if result and isinstance(result, dict) and "signal" in result:
                self._logger.debug(f"[{stock_code}] LLM情绪分析成功: {result}")
                return result
            else:
                self._logger.warning(f"[{stock_code}] LLM返回格式无效")
                return None

        except Exception as e:
            self._logger.error(f"[{stock_code}] LLM情绪分析失败: {e}")
            return None

    def _format_news_items(self, news_items: list[Any]) -> str:
        """
        Format news items as text context for LLM analysis.

        Args:
            news_items: List of news articles

        Returns:
            Formatted text with all news items
        """
        lines = []

        for i, item in enumerate(news_items, 1):
            lines.append(f"\n{i}. {item.title}")
            if hasattr(item, "snippet") and item.snippet:
                lines.append(f"   {item.snippet[:200]}...")
            if hasattr(item, "source") and item.source:
                lines.append(f"   来源: {item.source}")

        return "\n".join(lines) if lines else "未获取到新闻数据"
