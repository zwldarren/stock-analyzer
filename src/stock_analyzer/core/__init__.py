"""
Core module - Application orchestration layer.

This module serves as the application layer that coordinates between
infrastructure (data sources) and domain (agents) layers.

Responsibilities:
1. Coordinate data preparation from infrastructure layer
2. Delegate AI analysis to domain agents layer
3. Manage application workflow and scheduling

Architecture:
- Core layer (this module): Data preparation, workflow coordination
- Domain layer (agents/): AI analysis and decision making
- Infrastructure layer: Data sources, notifications, persistence
"""

from stock_analyzer.core.analyzer import analyze_stock, batch_analyze
from stock_analyzer.core.dependencies import (
    get_ai_analyzer,
    get_data_service,
    get_db,
    get_notification_service,
    get_search_service,
)
from stock_analyzer.core.scheduler import run_with_schedule
from stock_analyzer.domain.models import AnalysisResult

__all__ = [
    # Analysis functions
    "analyze_stock",
    "batch_analyze",
    # Scheduler
    "run_with_schedule",
    # Dependency getters
    "get_data_service",
    "get_ai_analyzer",
    "get_db",
    "get_search_service",
    "get_notification_service",
    # Models (re-exported from domain)
    "AnalysisResult",
]
