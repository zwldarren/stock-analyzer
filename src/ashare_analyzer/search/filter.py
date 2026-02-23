"""
News relevance filter using AI.

Filters search results based on relevance to stock investment analysis
and freshness (timeliness).
"""

import asyncio
import concurrent.futures
import logging
from datetime import datetime

from ashare_analyzer.config import NewsFilterConfig
from ashare_analyzer.models import NewsFilterResponse, NewsFilterResult, NewsItemForFilter, SearchResult

logger = logging.getLogger(__name__)


def _run_async(coro):
    """Run async coroutine in sync context."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        # Already in async context, create task
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(asyncio.run, coro)
            return future.result()
    else:
        # No running loop, can use asyncio.run
        return asyncio.run(coro)


class NewsFilter:
    """
    AI-powered news relevance filter.

    Uses LLM tool call to evaluate news relevance and freshness in batch.
    Falls back to original results on any failure.
    """

    FILTER_TOOL = {
        "type": "function",
        "function": {
            "name": "filter_news_results",
            "description": (
                "批量判断新闻相关性。相关性判断：内容是否影响该股票投资决策"
                "（公司公告、业绩、行业政策、重大事件等）。"
                "时效性判断：fresh=1-2天内, acceptable=3-7天内, stale=超过7天或无日期信息"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "results": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "index": {"type": "integer", "description": "新闻索引"},
                                "is_relevant": {"type": "boolean", "description": "是否与股票投资分析相关"},
                                "freshness": {
                                    "type": "string",
                                    "enum": ["fresh", "acceptable", "stale"],
                                    "description": "时效性",
                                },
                            },
                            "required": ["index", "is_relevant", "freshness"],
                        },
                        "description": "过滤结果列表",
                    }
                },
                "required": ["results"],
            },
        },
    }

    FILTER_SYSTEM_PROMPT = """你是一个财经新闻相关性判断专家。请判断以下新闻与指定股票的投资分析是否相关。

判断标准：
1. 相关性：新闻内容是否影响该股票的投资决策
   - 相关：公司公告、业绩报告、行业政策、重大事件、高管变动、股权变更等
   - 不相关：仅提及股票名但主题无关、广告、无关行业新闻等

2. 时效性：
   - fresh: 1-2天内发布的新闻
   - acceptable: 3-7天内发布的新闻
   - stale: 超过7天或无法判断日期的新闻

请使用 filter_news_results 工具返回判断结果。"""

    def __init__(
        self,
        config: NewsFilterConfig,
        llm_client=None,
    ):
        """
        Initialize the news filter.

        Args:
            config: News filter configuration
            llm_client: LiteLLMClient instance (optional, will create if not provided)
        """
        self.config = config
        self._llm_client = llm_client

    def filter(
        self,
        results: list[SearchResult],
        stock_code: str,
        stock_name: str,
    ) -> list[SearchResult]:
        """
        Filter news results based on relevance and freshness.

        Args:
            results: List of search results to filter
            stock_code: Stock code (e.g., "600519")
            stock_name: Stock name (e.g., "贵州茅台")

        Returns:
            Filtered results, or original results if filtering fails
        """
        if not self.config.news_filter_enabled:
            return results

        if not results:
            return results

        # If no LLM client, return original results
        if self._llm_client is None:
            logger.warning("LLM client not available, returning original results")
            return results

        try:
            filtered = self._do_filter(results, stock_code, stock_name)

            # Ensure minimum results
            if len(filtered) < self.config.news_filter_min_results:
                logger.warning(
                    f"Filter returned only {len(filtered)} results, "
                    f"less than minimum {self.config.news_filter_min_results}, "
                    "returning original results"
                )
                return results

            return filtered

        except Exception as e:
            logger.warning(f"News filter failed: {e}, returning original results")
            return results

    def _do_filter(
        self,
        results: list[SearchResult],
        stock_code: str,
        stock_name: str,
    ) -> list[SearchResult]:
        """
        Perform the actual filtering using LLM.

        Args:
            results: List of search results
            stock_code: Stock code
            stock_name: Stock name

        Returns:
            Filtered list of SearchResult
        """
        # Prepare news items for LLM
        items = []
        for i, r in enumerate(results):
            items.append(
                NewsItemForFilter(
                    index=i,
                    title=r.title,
                    snippet=r.snippet,
                    source=r.source,
                    url=r.url,
                    published_date=r.published_date,
                )
            )

        # Build prompt
        current_date = datetime.now().strftime("%Y-%m-%d")
        prompt = self._build_filter_prompt(items, stock_code, stock_name, current_date)

        # Call LLM using async wrapper
        response = _run_async(self._call_llm(prompt))

        if response is None:
            raise ValueError("LLM returned no response")

        # Parse response and filter
        return self._apply_filter(results, response)

    def _build_filter_prompt(
        self,
        items: list[NewsItemForFilter],
        stock_code: str,
        stock_name: str,
        current_date: str,
    ) -> str:
        """Build the filter prompt."""
        items_json = "[\n"
        for item in items:
            items_json += f"  {item.model_dump_json()},\n"
        items_json += "]"

        return f"""股票：{stock_name} ({stock_code})
当前日期：{current_date}

新闻列表：
{items_json}

请判断每条新闻的相关性和时效性。"""

    async def _call_llm(self, prompt: str) -> NewsFilterResponse | None:
        """Call LLM with tool and return parsed response."""
        if self._llm_client is None:
            return None

        result = await self._llm_client.generate_with_tool(
            prompt=prompt,
            tool=self.FILTER_TOOL,
            generation_config={"temperature": 0.1},
            system_prompt=self.FILTER_SYSTEM_PROMPT,
        )

        if result is None:
            return None

        # Parse results
        results_data = result.get("results", [])
        filter_results = [NewsFilterResult(**r) for r in results_data]

        return NewsFilterResponse(results=filter_results)

    def _apply_filter(
        self,
        results: list[SearchResult],
        response: NewsFilterResponse,
    ) -> list[SearchResult]:
        """Apply filter response to results."""
        filtered = []

        for filter_result in response.results:
            idx = filter_result.index
            if idx < 0 or idx >= len(results):
                continue

            # Skip irrelevant or stale news
            if not filter_result.is_relevant:
                logger.debug(f"Filtering out irrelevant: {results[idx].title[:50]}")
                continue

            if filter_result.freshness == "stale":
                logger.debug(f"Filtering out stale: {results[idx].title[:50]}")
                continue

            filtered.append(results[idx])

        logger.info(f"News filter: {len(results)} -> {len(filtered)} results")
        return filtered
