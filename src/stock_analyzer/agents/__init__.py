"""
Multi-Agent Analysis System

This module implements a multi-agent architecture for stock analysis,
inspired by ai-hedge-fund but adapted for A-share market.

Agents:
    - NewsSentimentAgent: News sentiment analysis using LLM
    - FundamentalAgent: Financial metrics and fundamentals analysis
    - ValuationAgent: Multi-method valuation (DCF, Graham, Relative)
    - RiskManagerAgent: Position sizing and risk constraints (not a trading signal)
    - TechnicalAgent: Technical analysis (trend, MA, patterns, RSI, MACD, etc.)
    - ChipAgent: Chip distribution analysis (A-share specific)
    - StyleAgent: Unified investment style analysis (Value/Growth/Momentum)
    - PortfolioManagerAgent: Final decision with risk constraints

Execution Flow:
    1. RiskManagerAgent calculates position limits (first)
    2. Analysis agents run in parallel (Technical, Fundamental, Valuation, etc.)
    3. PortfolioManagerAgent makes final decision respecting risk limits

Usage:
    from stock_analyzer.agents import AgentCoordinator, TechnicalAgent

    coordinator = AgentCoordinator()
    coordinator.register_agent(TechnicalAgent())
    results = coordinator.analyze(context)
"""

from .base import AgentSignal, BaseAgent
from .chip_agent import ChipAgent
from .coordinator import AgentCoordinator
from .fundamental_agent import FundamentalAgent
from .news_sentiment_agent import NewsSentimentAgent
from .portfolio_manager import PortfolioManagerAgent
from .risk_manager import RiskManagerAgent
from .style_agent import StyleAgent
from .technical_agent import TechnicalAgent
from .valuation_agent import ValuationAgent

__all__ = [
    # Base classes
    "BaseAgent",
    "AgentSignal",
    # Core agents
    "AgentCoordinator",
    "PortfolioManagerAgent",
    "NewsSentimentAgent",
    "FundamentalAgent",
    "ValuationAgent",
    "RiskManagerAgent",
    "TechnicalAgent",
    "ChipAgent",
    "StyleAgent",
]
