"""
Industry classification module for A-share market.

Provides industry categorization and normalization utilities
to handle variations in industry names from different data sources.

Main Components:
    - CYCLICAL_INDUSTRIES: Set of cyclical industry names
    - FINANCIAL_INDUSTRIES: Set of financial industry names
    - is_cyclical_industry(): Check if an industry is cyclical
    - is_financial_industry(): Check if an industry is financial
    - get_industry_category(): Get the category of an industry
"""

from ashare_analyzer.industry.classifier import (
    get_all_cyclical_industries,
    get_all_financial_industries,
    get_industry_category,
    is_cyclical_industry,
    is_financial_industry,
)
from ashare_analyzer.industry.constants import (
    CYCLICAL_INDUSTRIES,
    FINANCIAL_INDUSTRIES,
    INDUSTRY_ALIASES,
    INDUSTRY_KEYWORDS,
    IndustryCategory,
)

__all__ = [
    # Constants
    "CYCLICAL_INDUSTRIES",
    "FINANCIAL_INDUSTRIES",
    "INDUSTRY_ALIASES",
    "INDUSTRY_KEYWORDS",
    "IndustryCategory",
    # Functions
    "is_cyclical_industry",
    "is_financial_industry",
    "get_industry_category",
    "get_all_cyclical_industries",
    "get_all_financial_industries",
]
