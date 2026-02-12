"""Data sources for stock analysis."""

from stock_analyzer.infrastructure.data_sources.akshare_fetcher import AkshareFetcher
from stock_analyzer.infrastructure.data_sources.baostock_fetcher import BaostockFetcher
from stock_analyzer.infrastructure.data_sources.base import BaseFetcher, DataFetcherManager
from stock_analyzer.infrastructure.data_sources.efinance_fetcher import EfinanceFetcher
from stock_analyzer.infrastructure.data_sources.pytdx_fetcher import PytdxFetcher
from stock_analyzer.infrastructure.data_sources.tushare_fetcher import TushareFetcher
from stock_analyzer.infrastructure.data_sources.yfinance_fetcher import YfinanceFetcher

__all__ = [
    "BaseFetcher",
    "DataFetcherManager",
    "EfinanceFetcher",
    "AkshareFetcher",
    "TushareFetcher",
    "PytdxFetcher",
    "BaostockFetcher",
    "YfinanceFetcher",
]
