"""
Industry name normalization utilities.

Handles variations in industry names from different data sources
(akshare, tushare, efinance) through normalization, alias resolution,
and keyword matching.
"""

import logging
import re

from ashare_analyzer.industry.constants import (
    COMMON_INDUSTRY_SUFFIXES,
    CYCLICAL_INDUSTRIES,
    FINANCIAL_INDUSTRIES,
    INDUSTRY_ALIASES,
    INDUSTRY_KEYWORDS,
)

logger = logging.getLogger(__name__)


def normalize_industry_name(industry: str) -> str:
    """
    Normalize industry name by removing common suffixes and whitespace.

    Args:
        industry: Raw industry name from data source

    Returns:
        Normalized industry name

    Examples:
        >>> normalize_industry_name("有色金属行业")
        '有色金属'
        >>> normalize_industry_name("  证券板块  ")
        '证券'
        >>> normalize_industry_name("银行一级行业")
        '银行'
    """
    if not industry:
        return ""

    # Strip whitespace
    normalized = industry.strip()

    # Remove common suffixes (longest match first to handle overlapping suffixes)
    # Sort by length descending so longer suffixes are checked first
    # Use explicit type to satisfy type checker
    suffixes_by_length: list[str] = list(COMMON_INDUSTRY_SUFFIXES)
    suffixes_by_length.sort(key=len, reverse=True)
    for suffix in suffixes_by_length:
        if normalized.endswith(suffix) and len(normalized) > len(suffix):
            normalized = normalized[: -len(suffix)]
            break  # Only remove one suffix (the longest match)

    return normalized


def resolve_alias(industry: str) -> str:
    """
    Resolve industry alias to canonical name.

    Args:
        industry: Industry name (may be an alias)

    Returns:
        Canonical industry name, or original if no alias found

    Examples:
        >>> resolve_alias("券商")
        '证券'
        >>> resolve_alias("煤炭开采")
        '煤炭'
        >>> resolve_alias("有色金属")
        '有色金属'
    """
    if not industry:
        return ""

    # Normalize first
    normalized = normalize_industry_name(industry)

    # Check direct alias
    if normalized in INDUSTRY_ALIASES:
        return INDUSTRY_ALIASES[normalized]

    # Check original (in case it's already canonical)
    if industry in INDUSTRY_ALIASES:
        return INDUSTRY_ALIASES[industry]

    return normalized


def match_by_keyword(industry: str, industry_list: set[str]) -> str | None:
    """
    Match industry using keyword partial matching.

    This is used as a fallback when exact matching fails.
    It checks if any keyword from INDUSTRY_KEYWORDS appears in the industry name.

    To avoid false positives, keywords are matched with word boundaries where
    appropriate. For example, "汽车" should match "汽车整车" but not "汽车服务"
    (if "服务" is excluded).

    Args:
        industry: Industry name to match
        industry_list: Set of canonical industry names to match against

    Returns:
        Matched canonical industry name, or None if no match

    Examples:
        >>> match_by_keyword("锂电池材料", CYCLICAL_INDUSTRIES)
        '稀有金属'
        >>> match_by_keyword("稀土永磁", CYCLICAL_INDUSTRIES)
        '稀有金属'
        >>> match_by_keyword("计算机", CYCLICAL_INDUSTRIES) is None
        True
    """
    if not industry:
        return None

    normalized = normalize_industry_name(industry)

    # First try exact match
    if normalized in industry_list:
        return normalized

    # Try alias resolution
    resolved = resolve_alias(normalized)
    if resolved in industry_list:
        return resolved

    # Try keyword matching
    for canonical_name, keywords in INDUSTRY_KEYWORDS.items():
        if canonical_name not in industry_list:
            continue

        for keyword in keywords:
            # Use word boundary matching for single-character keywords
            # to avoid false positives like "汽车服务" matching "汽车"
            if len(keyword) == 1:
                # Single char: use boundary check
                pattern = rf"(?:^|[^服务]){re.escape(keyword)}(?:$|[^务服])"
            else:
                # Multi-char: check if keyword is contained
                # But avoid matching "服务" suffix
                pattern = rf"{re.escape(keyword)}(?!服务)"

            if re.search(pattern, normalized):
                logger.debug(f"Industry keyword match: '{industry}' -> '{canonical_name}' via keyword '{keyword}'")
                return canonical_name

    return None


def classify_industry(industry: str) -> str | None:
    """
    Classify industry into canonical name with fallback strategies.

    Matching strategy (in order):
    1. Exact match after normalization
    2. Alias resolution
    3. Keyword partial matching

    Args:
        industry: Raw industry name from data source

    Returns:
        Canonical industry name if classified, None otherwise

    Examples:
        >>> classify_industry("券商")
        '证券'
        >>> classify_industry("锂电池材料")
        '稀有金属'
        >>> classify_industry("计算机") is None
        True
    """
    if not industry:
        return None

    # Step 1: Normalize
    normalized = normalize_industry_name(industry)

    # Step 2: Try exact match in cyclical industries
    if normalized in CYCLICAL_INDUSTRIES:
        return normalized

    # Step 3: Try exact match in financial industries
    if normalized in FINANCIAL_INDUSTRIES:
        return normalized

    # Step 4: Try alias resolution
    resolved = resolve_alias(normalized)
    if resolved != normalized and (resolved in CYCLICAL_INDUSTRIES or resolved in FINANCIAL_INDUSTRIES):
        return resolved

    # Step 5: Try keyword matching as fallback
    all_industries = CYCLICAL_INDUSTRIES | FINANCIAL_INDUSTRIES
    keyword_match = match_by_keyword(normalized, all_industries)
    if keyword_match:
        return keyword_match

    # No match found
    logger.debug(f"Industry not classified: '{industry}' (normalized: '{normalized}')")
    return None
