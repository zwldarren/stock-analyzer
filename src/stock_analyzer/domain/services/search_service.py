"""
搜索服务接口

定义搜索服务的抽象接口，遵循依赖倒置原则。
"""

from abc import ABC, abstractmethod
from typing import Any

from stock_analyzer.domain.models.search import SearchResponse


class ISearchService(ABC):
    """
    搜索服务接口

    负责搜索股票相关新闻、情报和市场信息。
    支持多个搜索提供商（Bocha、Tavily、SerpAPI等）。
    """

    @abstractmethod
    def search_comprehensive_intel(
        self, stock_code: str, stock_name: str, max_searches: int = 5, use_cache: bool = True
    ) -> dict[str, Any] | None:
        """
        综合情报搜索

        从多个维度搜索股票相关信息：
        - 公司新闻和公告
        - 行业动态
        - 市场舆情
        - 风险事件

        Args:
            stock_code: 股票代码
            stock_name: 股票名称
            max_searches: 最大搜索次数
            use_cache: 是否使用缓存

        Returns:
            dict[str, Any] | None: 各维度的搜索结果，失败返回None
        """

    @abstractmethod
    def format_intel_report(self, intel_results: dict[str, Any], stock_name: str) -> str:
        """
        格式化情报报告

        Args:
            intel_results: 搜索结果字典
            stock_name: 股票名称

        Returns:
            str: 格式化后的报告文本
        """

    @property
    @abstractmethod
    def is_available(self) -> bool:
        """
        检查搜索服务是否可用

        Returns:
            bool: 至少有一个搜索提供商可用
        """

    @abstractmethod
    def search_single_query(self, query: str, max_results: int = 10) -> dict[str, Any] | None:
        """
        执行单次搜索查询

        Args:
            query: 搜索关键词
            max_results: 最大结果数

        Returns:
            dict[str, Any] | None: 搜索结果
        """

    @abstractmethod
    def search_stock_news(
        self,
        stock_code: str,
        stock_name: str,
        max_results: int = 10,
        focus_keywords: list[str] | None = None,
        use_cache: bool = True,
    ) -> SearchResponse:
        """
        搜索股票相关新闻

        Args:
            stock_code: 股票代码
            stock_name: 股票名称
            max_results: 最大返回结果数
            focus_keywords: 重点关注的关键词列表
            use_cache: 是否使用缓存

        Returns:
            搜索结果对象
        """
