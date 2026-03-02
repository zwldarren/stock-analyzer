"""
Industry classifier for stock type determination.

Provides functions to classify industries into categories (cyclical, financial, etc.)
using multiple matching strategies: exact match, alias resolution, and keyword matching.
"""

import logging

from ashare_analyzer.industry.constants import (
    CYCLICAL_INDUSTRIES,
    FINANCIAL_INDUSTRIES,
    IndustryCategory,
)
from ashare_analyzer.industry.normalizer import classify_industry

logger = logging.getLogger(__name__)


def is_cyclical_industry(industry: str) -> bool:
    """
    Check if an industry is cyclical.

    Uses multi-strategy matching:
    1. Exact match after normalization
    2. Alias resolution
    3. Keyword partial matching

    Cyclical industries are those whose performance is highly correlated
    with economic cycles. They typically have:
    - High earnings volatility
    - Sensitivity to GDP growth
    - Capital-intensive operations

    Examples include: 有色金属, 煤炭, 钢铁, 化工, 汽车, 房地产, etc.

    Args:
        industry: Industry name from data source

    Returns:
        True if industry is cyclical, False otherwise

    Examples:
        >>> is_cyclical_industry("有色金属")
        True
        >>> is_cyclical_industry("券商")  # Financial, not cyclical
        False
        >>> is_cyclical_industry("锂电池材料")  # Matches via keyword
        True
        >>> is_cyclical_industry("计算机")
        False
    """
    if not industry:
        return False

    classified = classify_industry(industry)
    if classified is None:
        return False

    return classified in CYCLICAL_INDUSTRIES


def is_financial_industry(industry: str) -> bool:
    """
    Check if an industry is financial.

    Uses multi-strategy matching:
    1. Exact match after normalization
    2. Alias resolution
    3. Keyword partial matching

    Financial industries have special valuation considerations:
    - Balance-sheet driven analysis
    - Use PB/ROE valuation methods
    - Subject to regulatory constraints

    Examples include: 银行, 保险, 证券, 券商, 信托, etc.

    Args:
        industry: Industry name from data source

    Returns:
        True if industry is financial, False otherwise

    Examples:
        >>> is_financial_industry("银行")
        True
        >>> is_financial_industry("券商")  # Alias for 证券
        True
        >>> is_financial_industry("证券公司")  # Normalized to 证券
        True
        >>> is_financial_industry("有色金属")
        False
    """
    if not industry:
        return False

    classified = classify_industry(industry)
    if classified is None:
        return False

    return classified in FINANCIAL_INDUSTRIES


def get_industry_category(industry: str) -> IndustryCategory:
    """
    Get the category of an industry.

    Args:
        industry: Industry name from data source

    Returns:
        IndustryCategory enum value

    Examples:
        >>> get_industry_category("有色金属")
        <IndustryCategory.CYCLICAL: 'cyclical'>
        >>> get_industry_category("银行")
        <IndustryCategory.FINANCIAL: 'financial'>
        >>> get_industry_category("计算机")
        <IndustryCategory.OTHER: 'other'>
        >>> get_industry_category("")
        <IndustryCategory.OTHER: 'other'>
    """
    if not industry:
        return IndustryCategory.OTHER

    classified = classify_industry(industry)
    if classified is None:
        return IndustryCategory.OTHER

    if classified in CYCLICAL_INDUSTRIES:
        return IndustryCategory.CYCLICAL
    elif classified in FINANCIAL_INDUSTRIES:
        return IndustryCategory.FINANCIAL

    return IndustryCategory.OTHER


def get_all_cyclical_industries() -> set[str]:
    """
    Get all cyclical industry names (canonical names only).

    Returns:
        Set of canonical cyclical industry names

    Examples:
        >>> industries = get_all_cyclical_industries()
        >>> "有色金属" in industries
        True
        >>> "煤炭" in industries
        True
    """
    return CYCLICAL_INDUSTRIES.copy()


def get_all_financial_industries() -> set[str]:
    """
    Get all financial industry names (canonical names only).

    Returns:
        Set of canonical financial industry names

    Examples:
        >>> industries = get_all_financial_industries()
        >>> "银行" in industries
        True
        >>> "证券" in industries
        True
    """
    return FINANCIAL_INDUSTRIES.copy()
