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


def convert_to_provider_format(code: str, provider: str) -> str:
    """
    Convert stock code to provider-specific format.

    This function unifies the stock code conversion logic across different data providers,
    handling the format differences for A-shares (Shanghai/Shenzhen markets).

    Provider formats:
    - yfinance: Shanghai .SS, Shenzhen .SZ, HK .HK, US no suffix
    - tushare: Shanghai .SH, Shenzhen .SZ
    - baostock: sh.XXXXXX, sz.XXXXXX

    Args:
        code: Original stock code (e.g., '600519', '000001', 'AAPL', 'hk00700')
        provider: Provider name ('yfinance', 'tushare', 'baostock')

    Returns:
        Provider-formatted stock code

    Examples:
        >>> convert_to_provider_format('600519', 'yfinance')
        '600519.SS'
        >>> convert_to_provider_format('600519', 'tushare')
        '600519.SH'
        >>> convert_to_provider_format('600519', 'baostock')
        'sh.600519'
        >>> convert_to_provider_format('000001', 'yfinance')
        '000001.SZ'
        >>> convert_to_provider_format('hk00700', 'yfinance')
        '0700.HK'
    """
    code = code.strip().upper()
    stock_type = detect_stock_type(code)

    # US stocks: return as-is for yfinance, not supported by tushare/baostock
    if stock_type == StockType.US:
        if provider == "yfinance":
            return code
        # tushare/baostock don't support US stocks
        return code

    # HK stocks: only yfinance has special handling
    if stock_type == StockType.HK:
        if provider == "yfinance":
            # Remove 'HK' prefix if present
            hk_code = code[2:] if code.startswith("HK") else code
            # Remove leading zeros, keep at least 1 digit
            hk_code = hk_code.lstrip("0") or "0"
            hk_code = hk_code.zfill(4)  # Pad to 4 digits
            return f"{hk_code}.HK"
        # tushare/baostock don't support HK stocks in standard way
        return code

    # A-shares: determine market by code prefix
    # Shanghai: 600xxx, 601xxx, 603xxx, 688xxx (STAR Market)
    # Shenzhen: 000xxx, 002xxx, 300xxx (ChiNext)

    # Remove existing suffix/prefix
    clean_code = code.replace(".SH", "").replace(".SS", "").replace(".SZ", "")
    clean_code = clean_code.replace(".sh", "").replace(".sz", "")

    # Remove existing baostock prefix
    if clean_code.startswith("SH.") or clean_code.startswith("SZ."):
        clean_code = clean_code[3:]

    # Determine market
    is_shanghai = clean_code.startswith(("600", "601", "603", "688"))

    if provider == "yfinance":
        if is_shanghai:
            return f"{clean_code}.SS"
        else:
            # Default to Shenzhen for unknown codes (matches existing behavior)
            return f"{clean_code}.SZ"

    elif provider == "tushare":
        if is_shanghai:
            return f"{clean_code}.SH"
        else:
            return f"{clean_code}.SZ"

    elif provider == "baostock":
        if is_shanghai:
            return f"sh.{clean_code}"
        else:
            return f"sz.{clean_code}"

    # Unknown provider, return as-is
    return code


def get_market_from_code(code: str) -> str:
    """
    Determine market from A-share stock code prefix.

    Args:
        code: Stock code (6 digits)

    Returns:
        'sh' for Shanghai, 'sz' for Shenzhen, or 'unknown'
    """
    code = code.strip().upper()

    # Remove existing suffix/prefix
    clean_code = code.replace(".SH", "").replace(".SS", "").replace(".SZ", "")
    clean_code = clean_code.replace(".sh", "").replace(".sz", "")
    if clean_code.startswith("SH.") or clean_code.startswith("SZ."):
        clean_code = clean_code[3:]

    if clean_code.startswith(("600", "601", "603", "688")):
        return "sh"
    elif clean_code.startswith(("000", "002", "300")):
        return "sz"

    return "unknown"
