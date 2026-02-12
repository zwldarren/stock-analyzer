"""
===================================
实时行情统一类型定义 & 熔断机制
===================================

设计目标：
1. 统一各数据源的实时行情返回结构
2. 实现熔断/冷却机制，避免连续失败时反复请求
3. 支持多数据源故障切换

使用方式：
- 所有 Fetcher 的 get_realtime_quote() 统一返回 UnifiedRealtimeQuote
- CircuitBreaker 管理各数据源的熔断状态
"""

import logging
import time
from typing import Any

logger = logging.getLogger(__name__)


# ============================================
# 通用类型转换工具函数
# ============================================
# 设计说明：
# 各数据源返回的原始数据类型不一致（str/float/int/NaN），
# 使用这些函数统一转换，避免在各 Fetcher 中重复定义。


def safe_float(val: Any, default: float | None = None) -> float | None:
    """
    安全转换为浮点数

    处理场景：
    - None / 空字符串 → default
    - pandas NaN / numpy NaN → default
    - 数值字符串 → float
    - 已是数值 → float

    Args:
        val: 待转换的值
        default: 转换失败时的默认值

    Returns:
        转换后的浮点数，或默认值
    """
    try:
        if val is None:
            return default

        # 处理字符串
        if isinstance(val, str):
            val = val.strip()
            if val == "" or val == "-" or val == "--":
                return default

        # 处理 pandas/numpy NaN
        # 使用 math.isnan 而不是 pd.isna，避免强制依赖 pandas
        import math

        try:
            if math.isnan(float(val)):
                return default
        except (ValueError, TypeError):
            pass

        return float(val)
    except (ValueError, TypeError):
        return default


def safe_int(val: Any, default: int | None = None) -> int | None:
    """
    安全转换为整数

    先转换为 float，再取整，处理 "123.0" 这类情况

    Args:
        val: 待转换的值
        default: 转换失败时的默认值

    Returns:
        转换后的整数，或默认值
    """
    f_val = safe_float(val, default=None)
    if f_val is not None:
        return int(f_val)
    return default


# ============================================
# Circuit Breaker - 熔断器
# ============================================


class CircuitBreaker:
    """
    熔断器 - 管理数据源的熔断/冷却状态

    策略：
    - 连续失败 N 次后进入熔断状态
    - 熔断期间跳过该数据源
    - 冷却时间后自动恢复半开状态
    - 半开状态下单次成功则完全恢复，失败则继续熔断

    状态机：
    CLOSED（正常） --失败N次--> OPEN（熔断）--冷却时间到--> HALF_OPEN（半开）
    HALF_OPEN --成功--> CLOSED
    HALF_OPEN --失败--> OPEN
    """

    # 状态常量
    CLOSED = "closed"  # 正常状态
    OPEN = "open"  # 熔断状态（不可用）
    HALF_OPEN = "half_open"  # 半开状态（试探性请求）

    def __init__(
        self,
        failure_threshold: int = 3,  # 连续失败次数阈值
        cooldown_seconds: float = 300.0,  # 冷却时间（秒），默认5分钟
        half_open_max_calls: int = 1,  # 半开状态最大尝试次数
    ):
        self.failure_threshold = failure_threshold
        self.cooldown_seconds = cooldown_seconds
        self.half_open_max_calls = half_open_max_calls

        # 各数据源状态 {source_name: {state, failures, last_failure_time, half_open_calls}}
        self._states: dict[str, dict[str, Any]] = {}

    def _get_state(self, source: str) -> dict[str, Any]:
        """获取或初始化数据源状态"""
        if source not in self._states:
            self._states[source] = {
                "state": self.CLOSED,
                "failures": 0,
                "last_failure_time": 0.0,
                "half_open_calls": 0,
            }
        return self._states[source]

    def is_available(self, source: str) -> bool:
        """
        检查数据源是否可用

        返回 True 表示可以尝试请求
        返回 False 表示应跳过该数据源
        """
        state = self._get_state(source)
        current_time = time.time()

        if state["state"] == self.CLOSED:
            return True

        if state["state"] == self.OPEN:
            # 检查冷却时间
            time_since_failure = current_time - state["last_failure_time"]
            if time_since_failure >= self.cooldown_seconds:
                # 冷却完成，进入半开状态
                state["state"] = self.HALF_OPEN
                state["half_open_calls"] = 0
                logger.info(f"[熔断器] {source} 冷却完成，进入半开状态")
                return True
            else:
                remaining = self.cooldown_seconds - time_since_failure
                logger.debug(f"[熔断器] {source} 处于熔断状态，剩余冷却时间: {remaining:.0f}s")
                return False

        if state["state"] == self.HALF_OPEN:
            # 半开状态下限制请求次数
            return state["half_open_calls"] < self.half_open_max_calls

        return True

    def record_success(self, source: str) -> None:
        """记录成功请求"""
        state = self._get_state(source)

        if state["state"] == self.HALF_OPEN:
            # 半开状态下成功，完全恢复
            logger.info(f"[熔断器] {source} 半开状态请求成功，恢复正常")

        # 重置状态
        state["state"] = self.CLOSED
        state["failures"] = 0
        state["half_open_calls"] = 0

    def record_failure(self, source: str, error: str | None = None) -> None:
        """记录失败请求"""
        state = self._get_state(source)
        current_time = time.time()

        state["failures"] += 1
        state["last_failure_time"] = current_time

        if state["state"] == self.HALF_OPEN:
            # 半开状态下失败，继续熔断
            state["state"] = self.OPEN
            state["half_open_calls"] = 0
            logger.warning(f"[熔断器] {source} 半开状态请求失败，继续熔断 {self.cooldown_seconds}s")
        elif state["failures"] >= self.failure_threshold:
            # 达到阈值，进入熔断
            state["state"] = self.OPEN
            logger.warning(
                f"[熔断器] {source} 连续失败 {state['failures']} 次，进入熔断状态 (冷却 {self.cooldown_seconds}s)"
            )
            if error:
                logger.warning(f"[熔断器] 最后错误: {error}")

    def get_status(self) -> dict[str, str]:
        """获取所有数据源状态"""
        return {source: info["state"] for source, info in self._states.items()}

    def reset(self, source: str | None = None) -> None:
        """重置熔断器状态"""
        if source:
            if source in self._states:
                del self._states[source]
        else:
            self._states.clear()


# 全局熔断器实例（实时行情专用）
_realtime_circuit_breaker = CircuitBreaker(
    failure_threshold=3,  # 连续失败3次熔断
    cooldown_seconds=300.0,  # 冷却5分钟
    half_open_max_calls=1,
)

# 筹码接口熔断器（更保守的策略，因为该接口更不稳定）
_chip_circuit_breaker = CircuitBreaker(
    failure_threshold=2,  # 连续失败2次熔断
    cooldown_seconds=600.0,  # 冷却10分钟
    half_open_max_calls=1,
)


def get_realtime_circuit_breaker() -> CircuitBreaker:
    """获取实时行情熔断器"""
    return _realtime_circuit_breaker


def get_chip_circuit_breaker() -> CircuitBreaker:
    """获取筹码接口熔断器"""
    return _chip_circuit_breaker
