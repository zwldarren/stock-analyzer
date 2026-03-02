"""Stock classifier for determining valuation method selection."""

from enum import Enum

from ashare_analyzer.industry import is_cyclical_industry, is_financial_industry


class StockType(Enum):
    """Stock type classification for valuation method selection."""

    VALUE = "value"  # Low PE, high ROE, dividend payer
    GROWTH = "growth"  # High revenue/EPS growth
    CYCLICAL = "cyclical"  # Cyclical industry (metals, coal, etc.)
    FINANCIAL = "financial"  # Banks, insurance, securities
    LOSS_MAKING = "loss_making"  # Negative earnings
    DEFAULT = "default"  # Default classification


def classify_stock(data: dict) -> StockType:
    """
    Classify a stock into a type for valuation method selection.

    Classification priority (first match wins):
    1. Loss-making: PE <= 0 (only if PE is explicitly provided)
    2. Value: PE < 15, ROE > 10%, Dividend > 2%
    3. Financial: Industry in financial list (银行, 保险, 证券, 多元金融)
    4. Cyclical: Industry in cyclical list (有色金属, 煤炭, etc.)
    5. Growth: Revenue growth > 20% or EPS growth > 15%
    6. Default: Everything else

    Note: Value is checked before Financial because a bank meeting value criteria
    should use value-based valuation methods for better accuracy.

    Args:
        data: Stock data dict with keys:
            - pe_ratio: P/E ratio (negative for loss-making, None if unknown)
            - roe: Return on equity (%)
            - dividend_yield: Dividend yield (%)
            - revenue_growth: Revenue growth rate (%)
            - eps_growth: EPS growth rate (%)
            - industry: Industry name

    Returns:
        StockType classification
    """
    pe_ratio = data.get("pe_ratio")
    industry = data.get("industry", "")
    roe = data.get("roe", 0)
    dividend_yield = data.get("dividend_yield", 0)

    # 1. Loss-making: negative or zero PE (only if explicitly provided)
    if pe_ratio is not None and pe_ratio <= 0:
        return StockType.LOSS_MAKING

    # 2. Value stock: low PE, high ROE, dividend payer
    # Check value first - even financial stocks can be value stocks
    if pe_ratio is not None and pe_ratio < 15 and roe > 10 and dividend_yield > 2:
        return StockType.VALUE

    # 3. Financial industry (banks, insurance, securities)
    if industry and is_financial_industry(industry):
        return StockType.FINANCIAL

    # 4. Cyclical industry
    if industry and is_cyclical_industry(industry):
        return StockType.CYCLICAL

    # 5. Growth stock: high revenue or EPS growth
    revenue_growth = data.get("revenue_growth", 0)
    eps_growth = data.get("eps_growth", 0)

    if revenue_growth > 20 or eps_growth > 15:
        return StockType.GROWTH

    # 6. Default
    return StockType.DEFAULT
