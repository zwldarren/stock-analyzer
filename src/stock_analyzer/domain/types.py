"""
Domain Types - Shared data structures for domain layer.

These types are used by domain interfaces and infrastructure implementations.
Placed in domain layer to avoid circular dependencies following DDD principles.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any


class RealtimeSource(Enum):
    """实时行情数据源"""

    EFINANCE = "efinance"  # 东方财富（efinance库）
    AKSHARE_EM = "akshare_em"  # 东方财富（akshare库）
    AKSHARE_SINA = "akshare_sina"  # 新浪财经
    AKSHARE_QQ = "akshare_qq"  # 腾讯财经
    TUSHARE = "tushare"  # Tushare Pro
    TENCENT = "tencent"  # 腾讯直连
    SINA = "sina"  # 新浪直连
    FALLBACK = "fallback"  # 降级兜底


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

    # === 核心价格数据（几乎所有源都有）===
    price: float | None = None  # 最新价
    change_pct: float | None = None  # 涨跌幅(%)
    change_amount: float | None = None  # 涨跌额

    # === 量价指标（部分源可能缺失）===
    volume: int | None = None  # 成交量（手）
    amount: float | None = None  # 成交额（元）
    volume_ratio: float | None = None  # 量比
    turnover_rate: float | None = None  # 换手率(%)
    amplitude: float | None = None  # 振幅(%)

    # === 价格区间 ===
    open_price: float | None = None  # 开盘价
    high: float | None = None  # 最高价
    low: float | None = None  # 最低价
    pre_close: float | None = None  # 昨收价

    # === 估值指标（仅东财等全量接口有）===
    pe_ratio: float | None = None  # 市盈率(动态)
    pb_ratio: float | None = None  # 市净率
    total_mv: float | None = None  # 总市值(元)
    circ_mv: float | None = None  # 流通市值(元)

    # === 其他指标 ===
    change_60d: float | None = None  # 60日涨跌幅(%)
    high_52w: float | None = None  # 52周最高
    low_52w: float | None = None  # 52周最低

    def to_dict(self) -> dict[str, Any]:
        """转换为字典（过滤 None 值）"""
        result: dict[str, Any] = {
            "code": self.code,
            "name": self.name,
            "source": self.source.value,
        }
        # 只添加非 None 的字段
        if self.price is not None:
            result["price"] = self.price
        if self.change_pct is not None:
            result["change_pct"] = self.change_pct
        if self.change_amount is not None:
            result["change_amount"] = self.change_amount
        if self.volume is not None:
            result["volume"] = self.volume
        if self.amount is not None:
            result["amount"] = self.amount
        if self.volume_ratio is not None:
            result["volume_ratio"] = self.volume_ratio
        if self.turnover_rate is not None:
            result["turnover_rate"] = self.turnover_rate
        if self.amplitude is not None:
            result["amplitude"] = self.amplitude
        if self.open_price is not None:
            result["open_price"] = self.open_price
        if self.high is not None:
            result["high"] = self.high
        if self.low is not None:
            result["low"] = self.low
        if self.pre_close is not None:
            result["pre_close"] = self.pre_close
        if self.pe_ratio is not None:
            result["pe_ratio"] = self.pe_ratio
        if self.pb_ratio is not None:
            result["pb_ratio"] = self.pb_ratio
        if self.total_mv is not None:
            result["total_mv"] = self.total_mv
        if self.circ_mv is not None:
            result["circ_mv"] = self.circ_mv
        if self.change_60d is not None:
            result["change_60d"] = self.change_60d
        if self.high_52w is not None:
            result["high_52w"] = self.high_52w
        if self.low_52w is not None:
            result["low_52w"] = self.low_52w
        return result

    def has_basic_data(self) -> bool:
        """检查是否有基本的价格数据"""
        return self.price is not None and self.price > 0

    def has_volume_data(self) -> bool:
        """检查是否有量价数据"""
        return self.volume_ratio is not None or self.turnover_rate is not None


@dataclass
class ChipDistribution:
    """
    筹码分布数据

    反映持仓成本分布和获利情况
    """

    code: str
    date: str = ""
    source: str = "akshare"

    # 获利情况
    profit_ratio: float = 0.0  # 获利比例(0-1)
    avg_cost: float = 0.0  # 平均成本

    # 筹码集中度
    cost_90_low: float = 0.0  # 90%筹码成本下限
    cost_90_high: float = 0.0  # 90%筹码成本上限
    concentration_90: float = 0.0  # 90%筹码集中度（越小越集中）

    cost_70_low: float = 0.0  # 70%筹码成本下限
    cost_70_high: float = 0.0  # 70%筹码成本上限
    concentration_70: float = 0.0  # 70%筹码集中度

    def to_dict(self) -> dict[str, Any]:
        """转换为字典"""
        return {
            "code": self.code,
            "date": self.date,
            "source": self.source,
            "profit_ratio": self.profit_ratio,
            "avg_cost": self.avg_cost,
            "cost_90_low": self.cost_90_low,
            "cost_90_high": self.cost_90_high,
            "concentration_90": self.concentration_90,
            "concentration_70": self.concentration_70,
        }

    def get_chip_status(self, current_price: float) -> str:
        """
        获取筹码状态描述

        Args:
            current_price: 当前股价

        Returns:
            筹码状态描述
        """
        status_parts = []

        # 获利比例分析
        if self.profit_ratio >= 0.9:
            status_parts.append("获利盘极高(>90%)")
        elif self.profit_ratio >= 0.7:
            status_parts.append("获利盘较高(70-90%)")
        elif self.profit_ratio >= 0.5:
            status_parts.append("获利盘中等(50-70%)")
        elif self.profit_ratio >= 0.3:
            status_parts.append("套牢盘较多(>30%)")
        else:
            status_parts.append("套牢盘极重(>70%)")

        # 筹码集中度分析 (90%集中度 < 10% 表示集中)
        if self.concentration_90 < 0.08:
            status_parts.append("筹码高度集中")
        elif self.concentration_90 < 0.15:
            status_parts.append("筹码较集中")
        elif self.concentration_90 < 0.25:
            status_parts.append("筹码分散度中等")
        else:
            status_parts.append("筹码较分散")

        # 成本与现价关系
        if current_price > 0 and self.avg_cost > 0:
            cost_diff = (current_price - self.avg_cost) / self.avg_cost * 100
            if cost_diff > 20:
                status_parts.append(f"现价高于平均成本{cost_diff:.1f}%")
            elif cost_diff > 5:
                status_parts.append(f"现价略高于成本{cost_diff:.1f}%")
            elif cost_diff > -5:
                status_parts.append("现价接近平均成本")
            else:
                status_parts.append(f"现价低于平均成本{abs(cost_diff):.1f}%")

        return "，".join(status_parts)
