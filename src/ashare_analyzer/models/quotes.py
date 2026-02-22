"""
Quote models for realtime stock data.

Contains RealtimeSource enum and UnifiedRealtimeQuote dataclass.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any


class RealtimeSource(Enum):
    """实时行情数据源"""

    EFINANCE = "efinance"
    AKSHARE_EM = "akshare_em"
    AKSHARE_SINA = "akshare_sina"
    AKSHARE_QQ = "akshare_qq"
    TUSHARE = "tushare"
    TENCENT = "tencent"
    SINA = "sina"
    FALLBACK = "fallback"


@dataclass
class UnifiedRealtimeQuote:
    """
    统一实时行情数据结构

    设计原则：
    - 各数据源返回的字段可能不同，缺失字段用 None 表示
    - 主流程使用 getattr(quote, field, None) 获取，保证兼容性
    - source 字段标记数据来源，便于调试
    """

    code: str
    name: str = ""
    source: RealtimeSource = RealtimeSource.FALLBACK

    price: float | None = None
    change_pct: float | None = None
    change_amount: float | None = None

    volume: int | None = None
    amount: float | None = None
    volume_ratio: float | None = None
    turnover_rate: float | None = None
    amplitude: float | None = None

    open_price: float | None = None
    high: float | None = None
    low: float | None = None
    pre_close: float | None = None

    pe_ratio: float | None = None
    pb_ratio: float | None = None
    total_mv: float | None = None
    circ_mv: float | None = None

    change_60d: float | None = None
    high_52w: float | None = None
    low_52w: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """转换为字典（过滤 None 值）"""
        result: dict[str, Any] = {
            "code": self.code,
            "name": self.name,
            "source": self.source.value,
        }
        fields = [
            ("price", self.price),
            ("change_pct", self.change_pct),
            ("change_amount", self.change_amount),
            ("volume", self.volume),
            ("amount", self.amount),
            ("volume_ratio", self.volume_ratio),
            ("turnover_rate", self.turnover_rate),
            ("amplitude", self.amplitude),
            ("open_price", self.open_price),
            ("high", self.high),
            ("low", self.low),
            ("pre_close", self.pre_close),
            ("pe_ratio", self.pe_ratio),
            ("pb_ratio", self.pb_ratio),
            ("total_mv", self.total_mv),
            ("circ_mv", self.circ_mv),
            ("change_60d", self.change_60d),
            ("high_52w", self.high_52w),
            ("low_52w", self.low_52w),
        ]
        for key, value in fields:
            if value is not None:
                result[key] = value
        return result

    def has_basic_data(self) -> bool:
        """检查是否有基本的价格数据"""
        return self.price is not None and self.price > 0

    def has_volume_data(self) -> bool:
        """检查是否有量价数据"""
        return self.volume_ratio is not None or self.turnover_rate is not None
