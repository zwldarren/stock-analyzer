"""
Multi-Agent Analysis System

This module implements a multi-agent architecture for stock analysis,
inspired by ai-hedge-fund but adapted for A-share market.

Agents:
    - NewsSentimentAgent: News sentiment analysis using LLM
    - FundamentalAgent: Financial metrics and fundamentals analysis
    - ValuationAgent: Multi-method valuation (DCF, Graham, Relative)
    - RiskAgent: Risk assessment and position sizing
    - TechnicalAgent: Technical analysis (trend, MA, patterns)
    - ChipAgent: Chip distribution analysis (A-share specific)
    - StyleAgent: Unified investment style analysis (Value/Growth/Momentum)
    - DecisionAgent: Final decision aggregation

Usage:
    from stock_analyzer.agents import AgentCoordinator, NewsSentimentAgent

    coordinator = AgentCoordinator()
    coordinator.register_agent(NewsSentimentAgent(search_service))
    results = coordinator.analyze(context)
"""

from .base import AgentSignal, BaseAgent
from .chip_agent import ChipAgent
from .coordinator import AgentCoordinator
from .decision_agent import DecisionAgent
from .fundamental_agent import FundamentalAgent
from .news_sentiment_agent import NewsSentimentAgent
from .risk_agent import RiskAgent
from .style_agent import StyleAgent
from .technical_agent import TechnicalAgent
from .valuation_agent import ValuationAgent

__all__ = [
    # Base classes
    "BaseAgent",
    "AgentSignal",
    # Core agents
    "AgentCoordinator",
    "DecisionAgent",
    "NewsSentimentAgent",
    "FundamentalAgent",
    "ValuationAgent",
    "RiskAgent",
    "TechnicalAgent",
    "ChipAgent",
    # Style agent (merged Value/Growth/Momentum)
    "StyleAgent",
]
