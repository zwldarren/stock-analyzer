"""股票代码处理工具"""

import re
from enum import Enum


class StockType(Enum):
    """股票类型"""

    A_SHARE = "a_share"  # A股 (6位数字)
    HK = "hk"  # 港股 (5位数字)
    US = "us"  # 美股 (字母)
    ETF = "etf"  # ETF
    UNKNOWN = "unknown"


def detect_stock_type(code: str) -> StockType:
    """检测股票类型"""
    if not code:
        return StockType.UNKNOWN

    code = code.strip().upper()

    # 美股: 1-5个字母，可能包含 . 后跟字母（如 BRK.B）
    if re.match(r"^[A-Z]{1,5}(\.[A-Z])?$", code):
        return StockType.US

    # 港股: 5位数字，或带 hk 前缀
    code_lower = code.lower()
    if code_lower.startswith("hk"):
        return StockType.HK
    if re.match(r"^\d{5}$", code):
        return StockType.HK

    # A股: 6位数字
    if re.match(r"^\d{6}$", code):
        # ETF判断
        if code.startswith(("15", "16", "51", "56", "58", "59")):
            return StockType.ETF
        return StockType.A_SHARE

    return StockType.UNKNOWN


def is_us_code(code: str) -> bool:
    """是否为美股代码"""
    return detect_stock_type(code) == StockType.US


def is_hk_code(code: str) -> bool:
    """是否为港股代码"""
    return detect_stock_type(code) == StockType.HK


def is_etf_code(code: str) -> bool:
    """是否为ETF代码"""
    return detect_stock_type(code) == StockType.ETF


def normalize_stock_code(code: str) -> str:
    """标准化股票代码"""
    code = code.strip().upper()
    stock_type = detect_stock_type(code)

    if stock_type == StockType.HK:
        # 港股补零到5位
        return code.zfill(5)

    if stock_type == StockType.A_SHARE:
        # A股补零到6位
        return code.zfill(6)

    return code
