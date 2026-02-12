"""
Typed analysis context for type safety.

Replaces dict[str, Any] with a structured dataclass for better type safety
and IDE support. This addresses the review recommendation to "Type the context".
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class PriceData:
    """Price and volume data for momentum analysis."""

    close: list[float] = field(default_factory=list)
    volume: list[float] = field(default_factory=list)


@dataclass
class TechnicalIndicators:
    """Technical analysis indicators."""

    current_price: float = 0.0
    ma20: float = 0.0
    ma60: float = 0.0
    ma120: float = 0.0
    ma_alignment: str = "neutral"
    volume_momentum: float = 100.0
    price_volume_correlation: float = 0.3
    adx: float = 20.0
    obv_trend: str = "neutral"
    up_volume_ratio: float = 0.5
    volume_spike: bool = False


@dataclass
class ValuationData:
    """Valuation metrics."""

    pe_ratio: float | None = None
    pb_ratio: float | None = None
    eps: float | None = None
    book_value_per_share: float | None = None
    industry_pe: float = 0.0
    industry_pb: float = 0.0
    industry_name: str = ""
    pb_deviation_from_industry: float = 0.0


@dataclass
class FinancialData:
    """Financial metrics."""

    pe_ratio: float | None = None
    pb_ratio: float | None = None
    volatility: float = 0.0
    price_momentum_20d: float = 0.0
    roe: float = 0.0
    net_margin: float = 0.0
    gross_margin: float = 0.0
    debt_to_equity: float = 0.0


@dataclass
class GrowthData:
    """Growth metrics."""

    revenue_cagr: float = 0.0
    eps_cagr: float = 0.0
    fcf_cagr: float = 0.0
    price_momentum_1m: float = 0.0
    forward_pe: float | None = None
    rd_intensity: float = 0.0


@dataclass
class MarketData:
    """Market context data."""

    relative_strength_ratio: float = 1.0
    rs_trend: str = "neutral"
    relative_strength_rank: float = 50.0


@dataclass
class ChipData:
    """Chip distribution data."""

    profit_ratio: float = 0.0
    avg_cost: float = 0.0
    concentration_90: float = 0.0
    concentration_70: float = 0.0


@dataclass
class RealtimeQuote:
    """Real-time quote data."""

    name: str = ""
    price: float | None = None
    change_amount: float | None = None
    volume_ratio: float | None = None
    turnover_rate: float | None = None
    pe_ratio: float | None = None
    pb_ratio: float | None = None
    source: str = ""


@dataclass
class DailyData:
    """Daily OHLCV data."""

    date: str = ""
    open: float | None = None
    high: float | None = None
    low: float | None = None
    close: float | None = None
    volume: float | None = None
    amount: float | None = None
    pct_chg: float | None = None
    ma5: float | None = None
    ma10: float | None = None
    ma20: float | None = None
    volume_ratio: float | None = None


@dataclass
class AnalysisContext:
    """
    Complete analysis context with type safety.

    This dataclass replaces the dict[str, Any] context for better
    type safety, IDE autocomplete, and documentation.

    Usage:
        context = AnalysisContext(
            code="600519",
            stock_name="贵州茅台",
            current_price=1800.0,
            technical_data=TechnicalIndicators(...),
        )

        # Convert to dict for backward compatibility
        context_dict = context.to_dict()
    """

    # Basic info
    code: str = ""
    stock_name: str = ""
    date: str = ""
    current_price: float = 0.0
    industry: str = ""

    # Raw data
    raw_data: list[dict] = field(default_factory=list)

    # Today's data
    today: DailyData = field(default_factory=DailyData)
    yesterday: dict[str, Any] = field(default_factory=dict)

    # Status
    ma_status: str = ""
    price_change_ratio: float = 0.0
    volume_change_ratio: float = 0.0

    # Realtime quote
    realtime: RealtimeQuote = field(default_factory=RealtimeQuote)

    # Analysis data
    price_data: PriceData = field(default_factory=PriceData)
    technical_data: TechnicalIndicators = field(default_factory=TechnicalIndicators)
    valuation_data: ValuationData = field(default_factory=ValuationData)
    financial_data: FinancialData = field(default_factory=FinancialData)
    growth_data: GrowthData = field(default_factory=GrowthData)
    market_data: MarketData = field(default_factory=MarketData)
    chip: ChipData | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for backward compatibility."""
        result: dict[str, Any] = {
            "code": self.code,
            "stock_name": self.stock_name,
            "date": self.date,
            "current_price": self.current_price,
            "industry": self.industry,
            "raw_data": self.raw_data,
            "today": self._daily_to_dict(self.today),
            "yesterday": self.yesterday,
            "ma_status": self.ma_status,
            "price_change_ratio": self.price_change_ratio,
            "volume_change_ratio": self.volume_change_ratio,
            "realtime": self._realtime_to_dict(self.realtime),
            "price_data": self._price_data_to_dict(self.price_data),
            "technical_data": self._technical_to_dict(self.technical_data),
            "valuation_data": self._valuation_to_dict(self.valuation_data),
            "financial_data": self._financial_to_dict(self.financial_data),
            "growth_data": self._growth_to_dict(self.growth_data),
            "market_data": self._market_to_dict(self.market_data),
        }

        if self.chip:
            result["chip"] = self._chip_to_dict(self.chip)

        return result

    @staticmethod
    def _daily_to_dict(d: DailyData) -> dict[str, Any]:
        return {
            "date": d.date,
            "open": d.open,
            "high": d.high,
            "low": d.low,
            "close": d.close,
            "volume": d.volume,
            "amount": d.amount,
            "pct_chg": d.pct_chg,
            "ma5": d.ma5,
            "ma10": d.ma10,
            "ma20": d.ma20,
            "volume_ratio": d.volume_ratio,
        }

    @staticmethod
    def _realtime_to_dict(r: RealtimeQuote) -> dict[str, Any]:
        return {
            "name": r.name,
            "price": r.price,
            "change_amount": r.change_amount,
            "volume_ratio": r.volume_ratio,
            "turnover_rate": r.turnover_rate,
            "pe_ratio": r.pe_ratio,
            "pb_ratio": r.pb_ratio,
            "source": r.source,
        }

    @staticmethod
    def _price_data_to_dict(p: PriceData) -> dict[str, Any]:
        return {"close": p.close, "volume": p.volume}

    @staticmethod
    def _technical_to_dict(t: TechnicalIndicators) -> dict[str, Any]:
        return {
            "current_price": t.current_price,
            "ma20": t.ma20,
            "ma60": t.ma60,
            "ma120": t.ma120,
            "ma_alignment": t.ma_alignment,
            "volume_momentum": t.volume_momentum,
            "price_volume_correlation": t.price_volume_correlation,
            "adx": t.adx,
            "obv_trend": t.obv_trend,
            "up_volume_ratio": t.up_volume_ratio,
            "volume_spike": t.volume_spike,
        }

    @staticmethod
    def _valuation_to_dict(v: ValuationData) -> dict[str, Any]:
        return {
            "pe_ratio": v.pe_ratio,
            "pb_ratio": v.pb_ratio,
            "eps": v.eps,
            "book_value_per_share": v.book_value_per_share,
            "industry_pe": v.industry_pe,
            "industry_pb": v.industry_pb,
            "industry_name": v.industry_name,
            "pb_deviation_from_industry": v.pb_deviation_from_industry,
        }

    @staticmethod
    def _financial_to_dict(f: FinancialData) -> dict[str, Any]:
        return {
            "pe_ratio": f.pe_ratio,
            "pb_ratio": f.pb_ratio,
            "volatility": f.volatility,
            "price_momentum_20d": f.price_momentum_20d,
            "roe": f.roe,
            "net_margin": f.net_margin,
            "gross_margin": f.gross_margin,
            "debt_to_equity": f.debt_to_equity,
        }

    @staticmethod
    def _growth_to_dict(g: GrowthData) -> dict[str, Any]:
        return {
            "revenue_cagr": g.revenue_cagr,
            "eps_cagr": g.eps_cagr,
            "fcf_cagr": g.fcf_cagr,
            "price_momentum_1m": g.price_momentum_1m,
            "forward_pe": g.forward_pe,
            "rd_intensity": g.rd_intensity,
        }

    @staticmethod
    def _market_to_dict(m: MarketData) -> dict[str, Any]:
        return {
            "relative_strength_ratio": m.relative_strength_ratio,
            "rs_trend": m.rs_trend,
            "relative_strength_rank": m.relative_strength_rank,
        }

    @staticmethod
    def _chip_to_dict(c: ChipData) -> dict[str, Any]:
        return {
            "profit_ratio": c.profit_ratio,
            "avg_cost": c.avg_cost,
            "concentration_90": c.concentration_90,
            "concentration_70": c.concentration_70,
        }


# Type alias for backward compatibility
ContextDict = dict[str, Any]
