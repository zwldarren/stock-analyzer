"""
AI分析器接口

定义AI分析服务的抽象接口，遵循依赖倒置原则。
"""

from abc import ABC, abstractmethod
from typing import Any

from stock_analyzer.domain.models import AnalysisResult


class IAIAnalyzer(ABC):
    """
    AI分析器接口

    负责调用AI模型进行股票分析，支持多种AI提供商（Gemini、OpenAI等）。
    """

    @abstractmethod
    def analyze(self, context: dict[str, Any], news_context: str | None = None) -> AnalysisResult:
        """
        分析单只股票

        Args:
            context: 股票分析上下文数据（技术面、基本面等）
            news_context: 预先搜索的新闻内容（可选）

        Returns:
            AnalysisResult: 分析结果对象
        """

    @abstractmethod
    def is_available(self) -> bool:
        """
        检查分析器是否可用

        Returns:
            bool: AI服务是否可用（API Key是否配置正确）
        """

    @abstractmethod
    def batch_analyze(self, contexts: list[dict[str, Any]], delay_between: float = 2.0) -> list[AnalysisResult]:
        """
        批量分析多只股票

        Args:
            contexts: 上下文数据列表
            delay_between: 每次分析之间的延迟（秒）

        Returns:
            list[AnalysisResult]: 分析结果列表
        """

    @abstractmethod
    def generate_market_review(self, prompt: str, generation_config: dict[str, Any]) -> str | None:
        """
        生成市场复盘报告

        Args:
            prompt: 提示词
            generation_config: 生成配置

        Returns:
            生成的复盘报告文本，失败返回 None
        """
