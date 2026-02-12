"""
Search service.

Provides a unified search service interface, managing multiple search engines and search strategies.
"""

import logging
import re
import time
from datetime import datetime
from typing import Any

from stock_analyzer.domain.models import SearchResponse
from stock_analyzer.domain.services.search_service import ISearchService
from stock_analyzer.infrastructure.search import (
    ApiKeyProviderConfig,
    ProviderRegistry,
    SearxngProviderConfig,
)

logger = logging.getLogger(__name__)


class SearchService(ISearchService):
    """
    Search service.

    Features:
    1. Manages multiple search engines
    2. Automatic failover
    3. Result aggregation and formatting
    4. Automatic English keywords for HK/US stocks
    """

    # Enhanced search keyword templates (A-share Chinese)
    ENHANCED_SEARCH_KEYWORDS = [
        "{name} 股票 今日 股价",
        "{name} {code} 最新 行情 走势",
        "{name} 股票 分析 走势图",
        "{name} K线 技术分析",
        "{name} {code} 涨跌 成交量",
    ]

    # Enhanced search keyword templates (HK/US stocks English)
    ENHANCED_SEARCH_KEYWORDS_EN = [
        "{name} stock price today",
        "{name} {code} latest quote trend",
        "{name} stock analysis chart",
        "{name} technical analysis",
        "{name} {code} performance volume",
    ]

    def __init__(
        self,
        bocha_keys: list[str] | None = None,
        tavily_keys: list[str] | None = None,
        brave_keys: list[str] | None = None,
        serpapi_keys: list[str] | None = None,
        searxng_base_url: str = "",
        searxng_username: str | None = None,
        searxng_password: str | None = None,
    ):
        """
        Initialize the search service.

        Args:
            bocha_keys: List of Bocha search API keys.
            tavily_keys: List of Tavily API keys.
            brave_keys: List of Brave Search API keys.
            serpapi_keys: List of SerpAPI keys.
            searxng_base_url: SearXNG base URL.
            searxng_username: SearXNG Basic Auth username.
            searxng_password: SearXNG Basic Auth password.
        """
        self._providers = []

        # Use registry to create providers
        # Order: SearXNG -> Tavily -> Brave -> SerpAPI -> Bocha

        # 1. SearXNG (self-hosted, completely free)
        if searxng_base_url:
            provider = ProviderRegistry.create_provider(
                "searxng",
                SearxngProviderConfig(
                    base_url=searxng_base_url,
                    username=searxng_username,
                    password=searxng_password,
                ),
            )
            if provider:
                self._providers.append(provider)
                logger.info("已配置 searxng 搜索")

        # 2. Tavily (1000 requests/month free tier)
        if tavily_keys:
            provider = ProviderRegistry.create_provider("tavily", ApiKeyProviderConfig(api_keys=tavily_keys))
            if provider:
                self._providers.append(provider)
                logger.info(f"已配置 tavily 搜索，共 {len(tavily_keys)} 个 API Key")

        # 3. Brave Search (free tier available)
        if brave_keys:
            provider = ProviderRegistry.create_provider("brave", ApiKeyProviderConfig(api_keys=brave_keys))
            if provider:
                self._providers.append(provider)
                logger.info(f"已配置 brave 搜索，共 {len(brave_keys)} 个 API Key")

        # 4. SerpAPI (100 requests/month free tier)
        if serpapi_keys:
            provider = ProviderRegistry.create_provider("serpapi", ApiKeyProviderConfig(api_keys=serpapi_keys))
            if provider:
                self._providers.append(provider)
                logger.info(f"已配置 serpapi 搜索，共 {len(serpapi_keys)} 个 API Key")

        # 5. Bocha (paid only, Chinese optimized)
        if bocha_keys:
            provider = ProviderRegistry.create_provider("bocha", ApiKeyProviderConfig(api_keys=bocha_keys))
            if provider:
                self._providers.append(provider)
                logger.info(f"已配置 bocha 搜索，共 {len(bocha_keys)} 个 API Key")

    @staticmethod
    def _is_foreign_stock(stock_code: str) -> bool:
        """Check if the stock is HK or US stock."""
        code = stock_code.strip()
        # US stocks: 1-5 uppercase letters, may contain dot (e.g., BRK.B)
        if re.match(r"^[A-Za-z]{1,5}(\.[A-Za-z])?$", code):
            return True
        # HK stocks: starts with 'hk' prefix or 5-digit number
        lower = code.lower()
        if lower.startswith("hk"):
            return True
        return bool(code.isdigit() and len(code) == 5)

    @property
    def is_available(self) -> bool:
        """Check if any search engine is available."""
        return any(p.is_available for p in self._providers)

    def search_stock_news(
        self,
        stock_code: str,
        stock_name: str,
        max_results: int = 5,
        focus_keywords: list[str] | None = None,
    ) -> SearchResponse:
        """
        Search for stock-related news.

        Args:
            stock_code: Stock code.
            stock_name: Stock name.
            max_results: Maximum number of results to return.
            focus_keywords: List of keywords to focus on.

        Returns:
            SearchResponse object.
        """
        # Smart time range determination
        today_weekday = datetime.now().weekday()
        if today_weekday == 0:  # Monday
            search_days = 3
        elif today_weekday >= 5:  # Saturday(5), Sunday(6)
            search_days = 2
        else:  # Tuesday(1) - Friday(4)
            search_days = 1

        # Build search query (select language based on stock type)
        is_foreign = self._is_foreign_stock(stock_code)
        if focus_keywords:
            query = " ".join(focus_keywords)
        elif is_foreign:
            # Use English keywords for HK/US stocks
            query = f"{stock_name} {stock_code} stock latest news"
        else:
            # Use Chinese keywords for A-shares
            query = f"{stock_name} {stock_code} 股票 最新消息"

        logger.info(f"搜索股票新闻: {stock_name}({stock_code}), query='{query}', 时间范围: 近{search_days}天")

        # Try each search engine in order
        for provider in self._providers:
            if not provider.is_available:
                continue

            response = provider.search(query, max_results, days=search_days)

            if response.success and response.results:
                logger.info(f"使用 {provider.name} 搜索成功")
                return response
            else:
                logger.warning(f"{provider.name} 搜索失败: {response.error_message}，尝试下一个引擎")

        # All engines failed
        return SearchResponse(
            query=query,
            results=[],
            provider="None",
            success=False,
            error_message="所有搜索引擎都不可用或搜索失败",
        )

    def search_comprehensive_intel(
        self, stock_code: str, stock_name: str, max_searches: int = 5
    ) -> dict[str, SearchResponse]:
        """
        Multi-dimensional intelligence search.

        Search dimensions (5 total):
        1. Latest news - Recent news and events
        2. Market analysis - Research reports and ratings
        3. Risk check - Reductions, penalties, negative news
        4. Earnings expectations - Annual report forecasts, performance bulletins
        5. Industry analysis - Industry trends and competitors
        """
        results = {}
        search_count = 0

        # Select search keyword language based on stock type
        is_foreign = self._is_foreign_stock(stock_code)

        # Define search dimensions
        if is_foreign:
            search_dimensions = [
                {
                    "name": "latest_news",
                    "query": f"{stock_name} {stock_code} latest news events",
                    "desc": "最新消息",
                },
                {
                    "name": "market_analysis",
                    "query": f"{stock_name} analyst rating target price report",
                    "desc": "机构分析",
                },
                {
                    "name": "risk_check",
                    "query": f"{stock_name} risk insider selling lawsuit litigation",
                    "desc": "风险排查",
                },
                {
                    "name": "earnings",
                    "query": f"{stock_name} earnings revenue profit growth forecast",
                    "desc": "业绩预期",
                },
                {
                    "name": "industry",
                    "query": f"{stock_name} industry competitors market share outlook",
                    "desc": "行业分析",
                },
            ]
        else:
            search_dimensions = [
                {
                    "name": "latest_news",
                    "query": f"{stock_name} {stock_code} 最新 新闻 重大 事件",
                    "desc": "最新消息",
                },
                {
                    "name": "market_analysis",
                    "query": f"{stock_name} 研报 目标价 评级 深度分析",
                    "desc": "机构分析",
                },
                {
                    "name": "risk_check",
                    "query": f"{stock_name} 减持 处罚 违规 诉讼 利空 风险",
                    "desc": "风险排查",
                },
                {
                    "name": "earnings",
                    "query": f"{stock_name} 业绩预告 财报 营收 净利润 同比增长",
                    "desc": "业绩预期",
                },
                {
                    "name": "industry",
                    "query": f"{stock_name} 所在行业 竞争对手 市场份额 行业前景",
                    "desc": "行业分析",
                },
            ]

        logger.info(f"开始多维度情报搜索: {stock_name}({stock_code})")

        # Rotate through different search engines
        available_providers = [p for p in self._providers if p.is_available]
        if not available_providers:
            return results

        for search_count, dim in enumerate(search_dimensions):
            if search_count >= max_searches:
                break

            provider = available_providers[search_count % len(available_providers)]

            logger.info(f"[情报搜索] {dim['desc']}: 使用 {provider.name}")

            response = provider.search(dim["query"], max_results=5)
            results[dim["name"]] = response

            if response.success:
                logger.info(f"[情报搜索] {dim['desc']}: 获取 {len(response.results)} 条结果")
            else:
                logger.warning(f"[情报搜索] {dim['desc']}: 搜索失败 - {response.error_message}")

            # Brief delay to avoid rate limiting
            time.sleep(0.5)

        return results

    def format_intel_report(self, intel_results: dict[str, SearchResponse], stock_name: str) -> str:
        """Format intelligence search results into a report."""
        lines = [f"【{stock_name} 情报搜索结果】"]

        # Dimension display order
        display_order = ["latest_news", "market_analysis", "risk_check", "earnings", "industry"]

        for dim_name in display_order:
            if dim_name not in intel_results:
                continue

            resp = intel_results[dim_name]

            # Get dimension description
            dim_desc = dim_name
            if dim_name == "latest_news":
                dim_desc = "📰 最新消息"
            elif dim_name == "market_analysis":
                dim_desc = "📈 机构分析"
            elif dim_name == "risk_check":
                dim_desc = "⚠️ 风险排查"
            elif dim_name == "earnings":
                dim_desc = "📊 业绩预期"
            elif dim_name == "industry":
                dim_desc = "🏭 行业分析"

            lines.append(f"\n{dim_desc} (来源: {resp.provider}):")
            if resp.success and resp.results:
                for i, r in enumerate(resp.results[:4], 1):
                    date_str = f" [{r.published_date}]" if r.published_date else ""
                    lines.append(f"  {i}. {r.title}{date_str}")
                    snippet = r.snippet[:150] if len(r.snippet) > 20 else r.snippet
                    lines.append(f"     {snippet}...")
            else:
                lines.append("  未找到相关信息")

        return "\n".join(lines)

    def search_stock_price_fallback(
        self, stock_code: str, stock_name: str, max_attempts: int = 3, max_results: int = 5
    ) -> SearchResponse:
        """Enhance search when data sources fail."""
        if not self.is_available:
            return SearchResponse(
                query=f"{stock_name} 股价走势",
                results=[],
                provider="None",
                success=False,
                error_message="未配置搜索引擎 API Key",
            )

        logger.info(f"[增强搜索] 数据源失败，启动增强搜索: {stock_name}({stock_code})")

        all_results = []
        seen_urls = set()
        successful_providers = []

        # Search using multiple keyword templates
        is_foreign = self._is_foreign_stock(stock_code)
        keywords = self.ENHANCED_SEARCH_KEYWORDS_EN if is_foreign else self.ENHANCED_SEARCH_KEYWORDS
        for i, keyword_template in enumerate(keywords[:max_attempts]):
            query = keyword_template.format(name=stock_name, code=stock_code)

            logger.info(f"[增强搜索] 第 {i + 1}/{max_attempts} 次搜索: {query}")

            # Try each search engine in order
            for provider in self._providers:
                if not provider.is_available:
                    continue

                try:
                    response = provider.search(query, max_results=3)

                    if response.success and response.results:
                        # Deduplicate and add results
                        for result in response.results:
                            if result.url not in seen_urls:
                                seen_urls.add(result.url)
                                all_results.append(result)

                        if provider.name not in successful_providers:
                            successful_providers.append(provider.name)

                        logger.info(f"[增强搜索] {provider.name} 返回 {len(response.results)} 条结果")
                        break

                except Exception as e:
                    logger.warning(f"[增强搜索] {provider.name} 搜索异常: {e}")
                    continue

            # Brief delay between searches
            if i < max_attempts - 1:
                time.sleep(0.5)

        # Aggregate results
        if all_results:
            final_results = all_results[:max_results]
            provider_str = ", ".join(successful_providers) if successful_providers else "None"

            logger.info(f"[增强搜索] 完成，共获取 {len(final_results)} 条结果（来源: {provider_str}）")

            return SearchResponse(
                query=f"{stock_name}({stock_code}) 股价走势",
                results=final_results,
                provider=provider_str,
                success=True,
            )
        else:
            logger.warning("[增强搜索] 所有搜索均未返回结果")
            return SearchResponse(
                query=f"{stock_name}({stock_code}) 股价走势",
                results=[],
                provider="None",
                success=False,
                error_message="增强搜索未找到相关信息",
            )

    def search_single_query(self, query: str, max_results: int = 10) -> dict[str, Any] | None:
        """
        Execute a single search query.

        Args:
            query: Search query string.
            max_results: Maximum number of results.

        Returns:
            dict[str, Any] | None: Search results dictionary, None on failure.
        """
        # Try each search engine in order
        for provider in self._providers:
            if not provider.is_available:
                continue

            try:
                response = provider.search(query, max_results)

                if response.success and response.results:
                    # Convert to dictionary format
                    return {
                        "query": response.query,
                        "results": [
                            {
                                "title": r.title,
                                "snippet": r.snippet,
                                "url": r.url,
                                "published_date": r.published_date,
                            }
                            for r in response.results
                        ],
                        "provider": response.provider,
                        "success": response.success,
                    }
            except Exception as e:
                logger.warning(f"[单次搜索] {provider.name} 搜索异常: {e}")
                continue

        # All engines failed
        logger.warning(f"[单次搜索] 所有搜索引擎都失败: {query}")
        return None
