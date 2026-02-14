"""
Financial Scoring Utilities

Shared financial analysis logic across multiple agents to eliminate duplication.
"""

from typing import Any


class FinancialScorer:
    """
    Financial scoring utility class.

    Provides shared scoring methods for financial health, profitability,
    and safety metrics used across multiple agents.
    """

    @staticmethod
    def analyze_financial_health(data: dict[str, Any]) -> int:
        """
        Analyze financial health metrics (0-10 scale).

        Scoring:
        - Debt-to-Equity < 0.3: +3, < 0.5: +2, < 1.0: +1
        - Current Ratio > 2.0: +3, > 1.5: +2, > 1.0: +1
        - Interest Coverage > 5: +2, > 3: +1
        - Free Cash Flow positive: +2
        """
        # Reuse shared balance sheet strength scoring
        score = FinancialScorer._score_balance_sheet_strength(data)

        # Free cash flow scoring
        fcf = data.get("free_cash_flow", 0)
        if fcf > 0:
            score += 2

        return min(score, 10)

    @staticmethod
    def analyze_safety(data: dict[str, Any]) -> int:
        """
        Analyze balance sheet safety (0-10 scale).

        Scoring:
        - Debt-to-Equity < 0.3: +3, < 0.5: +2, < 1.0: +1
        - Current Ratio > 2.0: +3, > 1.5: +2, > 1.0: +1
        - Interest Coverage > 5: +2, > 3: +1
        - Net cash position: +2
        """
        # Reuse financial health scoring for common metrics
        score = FinancialScorer._score_balance_sheet_strength(data)

        # Net cash position (specific to safety analysis)
        if data.get("net_cash_position", False):
            score += 2

        return min(score, 10)

    @staticmethod
    def _score_balance_sheet_strength(data: dict[str, Any]) -> int:
        """
        Score balance sheet strength metrics (shared logic).

        Returns:
            Base score (0-8) from debt-to-equity, current ratio, and interest coverage
        """
        score = 0

        # Debt-to-equity scoring (lower is better)
        dte = data.get("debt_to_equity", 0)
        if dte < 0.3:
            score += 3
        elif dte < 0.5:
            score += 2
        elif dte < 1.0:
            score += 1

        # Current ratio scoring (higher is better)
        current_ratio = data.get("current_ratio", 0)
        if current_ratio > 2.0:
            score += 3
        elif current_ratio > 1.5:
            score += 2
        elif current_ratio > 1.0:
            score += 1

        # Interest coverage scoring
        interest_coverage = data.get("interest_coverage", 0)
        if interest_coverage > 5:
            score += 2
        elif interest_coverage > 3:
            score += 1

        return score

    @staticmethod
    def analyze_profitability(data: dict[str, Any]) -> int:
        """
        Analyze profitability metrics (0-10 scale).

        Scoring:
        - ROE > 15%: +3, > 10%: +2, > 5%: +1
        - Net Margin > 20%: +3, > 15%: +2, > 10%: +1
        - Gross Margin > 40%: +2, > 30%: +1
        - ROA > 8%: +2, > 5%: +1
        """
        score = 0

        # ROE scoring
        roe = data.get("roe", 0)
        if roe > 15:
            score += 3
        elif roe > 10:
            score += 2
        elif roe > 5:
            score += 1

        # Net margin scoring
        net_margin = data.get("net_margin", 0)
        if net_margin > 20:
            score += 3
        elif net_margin > 15:
            score += 2
        elif net_margin > 10:
            score += 1

        # Gross margin scoring
        gross_margin = data.get("gross_margin", 0)
        if gross_margin > 40:
            score += 2
        elif gross_margin > 30:
            score += 1

        # ROA scoring
        roa = data.get("roa", 0)
        if roa > 8:
            score += 2
        elif roa > 5:
            score += 1

        return min(score, 10)
