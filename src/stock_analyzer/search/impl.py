"""
Search engine implementations.

All concrete search engine provider implementations.
"""

import logging
from datetime import datetime
from urllib.parse import urlparse

import httpx
import serpapi

from stock_analyzer.models import SearchResponse, SearchResult
from stock_analyzer.search.base import (
    ApiKeyProviderConfig,
    ApiKeySearchProvider,
    BaseSearchProvider,
    SearxngProviderConfig,
)

logger = logging.getLogger(__name__)


def _extract_domain(url: str) -> str:
    """Extract domain from URL as source."""
    try:
        parsed = urlparse(url)
        domain = parsed.netloc.replace("www.", "")
        return domain or "未知来源"
    except Exception:
        logger.debug(f"Failed to extract domain from URL: {url}")
        return "未知来源"


class TavilySearchProvider(ApiKeySearchProvider):
    """
    Tavily Search Engine.

    Features:
    - AI/LLM optimized search API
    - 1000 requests/month on free tier
    - Returns structured search results
    """

    def __init__(self, config: ApiKeyProviderConfig):
        super().__init__(config, "Tavily")

    def _do_search_with_key(self, query: str, api_key: str, max_results: int, days: int = 7) -> SearchResponse:
        """Execute Tavily search."""
        try:
            from tavily import TavilyClient
        except ImportError:
            return SearchResponse(
                query=query,
                results=[],
                provider=self.name,
                success=False,
                error_message="tavily-python not installed, run: pip install tavily-python",
            )

        try:
            client = TavilyClient(api_key=api_key)

            # 执行搜索
            response = client.search(
                query=query,
                search_depth="advanced",
                max_results=max_results,
                include_answer=False,
                include_raw_content=False,
                days=days,
            )

            # 解析结果
            results = []
            for item in response.get("results", []):
                results.append(
                    SearchResult(
                        title=item.get("title", ""),
                        snippet=item.get("content", "")[:500],
                        url=item.get("url", ""),
                        source=_extract_domain(item.get("url", "")),
                        published_date=item.get("published_date"),
                    )
                )

            return SearchResponse(
                query=query,
                results=results,
                provider=self.name,
                success=True,
            )

        except Exception as e:
            error_msg = str(e)
            if "rate limit" in error_msg.lower() or "quota" in error_msg.lower():
                error_msg = f"API 配额已用尽: {error_msg}"

            return SearchResponse(query=query, results=[], provider=self.name, success=False, error_message=error_msg)


class SerpAPISearchProvider(ApiKeySearchProvider):
    """
    SerpAPI Search Engine.

    Features:
    - Supports Google, Bing, Baidu, and other search engines
    - 100 requests/month on free tier
    """

    def __init__(self, config: ApiKeyProviderConfig):
        super().__init__(config, "SerpAPI")

    def _do_search_with_key(self, query: str, api_key: str, max_results: int, days: int = 7) -> SearchResponse:
        """Execute SerpAPI search."""
        try:
            # 确定时间范围参数 tbs
            tbs = "qdr:w"
            if days <= 1:
                tbs = "qdr:d"
            elif days <= 7:
                tbs = "qdr:w"
            elif days <= 30:
                tbs = "qdr:m"
            else:
                tbs = "qdr:y"

            # 使用 Google 搜索
            params = {
                "engine": "google",
                "q": query,
                "api_key": api_key,
                "google_domain": "google.com.hk",
                "hl": "zh-cn",
                "gl": "cn",
                "tbs": tbs,
                "num": max_results,
            }

            search = serpapi.search(params)
            response = search.as_dict()

            # 解析结果
            results = []

            # 1. 解析 Knowledge Graph
            kg = response.get("knowledge_graph", {})
            if kg:
                title = kg.get("title", "知识图谱")
                desc = kg.get("description", "")

                details = []
                for key in ["type", "founded", "headquarters", "employees", "ceo"]:
                    val = kg.get(key)
                    if val:
                        details.append(f"{key}: {val}")

                snippet = f"{desc}\n" + " | ".join(details) if details else desc

                results.append(
                    SearchResult(
                        title=f"[知识图谱] {title}",
                        snippet=snippet,
                        url=kg.get("source", {}).get("link", ""),
                        source="Google Knowledge Graph",
                    )
                )

            # 2. 解析 Organic Results
            organic_results = response.get("organic_results", [])

            for item in organic_results[:max_results]:
                link = item.get("link", "")
                snippet = item.get("snippet", "")

                results.append(
                    SearchResult(
                        title=item.get("title", ""),
                        snippet=snippet[:1000],
                        url=link,
                        source=item.get("source", _extract_domain(link)),
                        published_date=item.get("date"),
                    )
                )

            return SearchResponse(
                query=query,
                results=results,
                provider=self.name,
                success=True,
            )

        except Exception as e:
            error_msg = str(e)
            return SearchResponse(query=query, results=[], provider=self.name, success=False, error_message=error_msg)


class BraveSearchProvider(ApiKeySearchProvider):
    """
    Brave Search Engine.

    Features:
    - Privacy-first independent search engine
    - Index of over 30 billion pages
    - Free tier available
    - Supports time range filtering

    Docs: https://brave.com/search/api/
    """

    API_ENDPOINT = "https://api.search.brave.com/res/v1/web/search"

    def __init__(self, config: ApiKeyProviderConfig):
        super().__init__(config, "Brave")

    def _do_search_with_key(self, query: str, api_key: str, max_results: int, days: int = 7) -> SearchResponse:
        """Execute Brave search."""
        try:
            # 请求头
            headers = {"X-Subscription-Token": api_key, "Accept": "application/json"}

            # 确定时间范围（freshness 参数）
            if days <= 1:
                freshness = "pd"  # Past day (24 hours)
            elif days <= 7:
                freshness = "pw"  # Past week
            elif days <= 30:
                freshness = "pm"  # Past month
            else:
                freshness = "py"  # Past year

            # 请求参数
            params = {
                "q": query,
                "count": min(max_results, 20),  # Brave max 20 results
                "freshness": freshness,
                "search_lang": "en",  # English content (US stocks preferred)
                "country": "US",  # US region preference
                "safesearch": "moderate",
            }

            # 执行搜索（GET 请求）
            response = httpx.get(self.API_ENDPOINT, headers=headers, params=params, timeout=10)

            # 检查HTTP状态码
            if response.status_code != 200:
                error_msg = self._parse_error(response)
                logger.warning(f"[Brave] 搜索失败: {error_msg}")
                return SearchResponse(
                    query=query, results=[], provider=self.name, success=False, error_message=error_msg
                )

            # 解析响应
            try:
                data = response.json()
            except ValueError as e:
                error_msg = f"响应JSON解析失败: {str(e)}"
                logger.error(f"[Brave] {error_msg}")
                return SearchResponse(
                    query=query, results=[], provider=self.name, success=False, error_message=error_msg
                )

            logger.debug(f"[Brave] 搜索完成，query='{query}'")
            logger.debug(f"[Brave] 原始响应: {data}")

            # 解析搜索结果
            results = []
            web_data = data.get("web", {})
            web_results = web_data.get("results", [])

            for item in web_results[:max_results]:
                # 解析发布日期（ISO 8601 格式）
                published_date = None
                age = item.get("age") or item.get("page_age")
                if age:
                    try:
                        dt = datetime.fromisoformat(age.replace("Z", "+00:00"))
                        published_date = dt.strftime("%Y-%m-%d")
                    except (ValueError, AttributeError):
                        published_date = age  # Use original value on parse failure

                results.append(
                    SearchResult(
                        title=item.get("title", ""),
                        snippet=item.get("description", "")[:500],  # Truncate to 500 chars
                        url=item.get("url", ""),
                        source=_extract_domain(item.get("url", "")),
                        published_date=published_date,
                    )
                )

            logger.debug(f"[Brave] 成功解析 {len(results)} 条结果")

            return SearchResponse(query=query, results=results, provider=self.name, success=True)

        except httpx.TimeoutException:
            error_msg = "请求超时"
            logger.error(f"[Brave] {error_msg}")
            return SearchResponse(query=query, results=[], provider=self.name, success=False, error_message=error_msg)
        except httpx.RequestError as e:
            error_msg = f"网络请求失败: {str(e)}"
            logger.error(f"[Brave] {error_msg}")
            return SearchResponse(query=query, results=[], provider=self.name, success=False, error_message=error_msg)
        except Exception as e:
            error_msg = f"未知错误: {str(e)}"
            logger.error(f"[Brave] {error_msg}")
            return SearchResponse(query=query, results=[], provider=self.name, success=False, error_message=error_msg)

    def _parse_error(self, response) -> str:
        """Parse error response."""
        try:
            if response.headers.get("content-type", "").startswith("application/json"):
                error_data = response.json()
                # Brave API error format
                if "message" in error_data:
                    return error_data["message"]
                if "error" in error_data:
                    return error_data["error"]
                return str(error_data)
            return response.text[:200]
        except Exception:
            logger.debug("Failed to extract error message from response")
            return f"HTTP {response.status_code}: {response.text[:200]}"


class BochaSearchProvider(ApiKeySearchProvider):
    """
    Bocha Search Engine.

    Features:
    - AI-optimized Chinese search API
    - Accurate results with complete abstracts
    - Supports time range filtering and AI summaries
    """

    def __init__(self, config: ApiKeyProviderConfig):
        super().__init__(config, "Bocha")

    def _do_search_with_key(self, query: str, api_key: str, max_results: int, days: int = 7) -> SearchResponse:
        """Execute Bocha search."""
        try:
            # 确定时间范围
            freshness = "oneWeek"
            if days <= 1:
                freshness = "oneDay"
            elif days <= 7:
                freshness = "oneWeek"
            elif days <= 30:
                freshness = "oneMonth"
            else:
                freshness = "oneYear"

            # 请求参数
            payload = {
                "query": query,
                "freshness": freshness,
                "summary": True,
                "count": min(max_results, 50),
            }

            headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

            # 执行搜索
            response = httpx.post(
                "https://api.bocha.cn/v1/web-search",
                headers=headers,
                json=payload,
                timeout=10,
            )

            if response.status_code != 200:
                error_message = response.text
                if response.status_code == 403:
                    error_msg = f"余额不足: {error_message}"
                elif response.status_code == 401:
                    error_msg = f"API KEY无效: {error_message}"
                elif response.status_code == 429:
                    error_msg = f"请求频率达到限制: {error_message}"
                else:
                    error_msg = f"HTTP {response.status_code}: {error_message}"

                return SearchResponse(
                    query=query,
                    results=[],
                    provider=self.name,
                    success=False,
                    error_message=error_msg,
                )

            data = response.json()

            if data.get("code") != 200:
                error_msg = data.get("msg") or f"API返回错误码: {data.get('code')}"
                return SearchResponse(
                    query=query,
                    results=[],
                    provider=self.name,
                    success=False,
                    error_message=error_msg,
                )

            # 解析搜索结果
            results = []
            web_pages = data.get("data", {}).get("webPages", {})
            value_list = web_pages.get("value", [])

            for item in value_list[:max_results]:
                snippet = item.get("summary") or item.get("snippet", "")
                if snippet:
                    snippet = snippet[:500]

                results.append(
                    SearchResult(
                        title=item.get("name", ""),
                        snippet=snippet,
                        url=item.get("url", ""),
                        source=item.get("siteName") or _extract_domain(item.get("url", "")),
                        published_date=item.get("datePublished"),
                    )
                )

            return SearchResponse(
                query=query,
                results=results,
                provider=self.name,
                success=True,
            )

        except httpx.TimeoutException:
            return SearchResponse(query=query, results=[], provider=self.name, success=False, error_message="请求超时")
        except httpx.RequestError as e:
            return SearchResponse(
                query=query, results=[], provider=self.name, success=False, error_message=f"网络请求失败: {str(e)}"
            )
        except Exception as e:
            return SearchResponse(
                query=query, results=[], provider=self.name, success=False, error_message=f"未知错误: {str(e)}"
            )


class SearxngSearchProvider(BaseSearchProvider):
    """
    SearXNG Search Engine.

    Features:
    - Open-source meta search engine, privacy-respecting
    - Supports self-hosted instances
    - Supports Basic Auth authentication
    - Can aggregate results from multiple search engines

    Docs: https://docs.searxng.org/
    """

    config_class = SearxngProviderConfig

    def __init__(self, config: SearxngProviderConfig):
        super().__init__(config, "SearXNG")
        self._base_url = config.base_url.rstrip("/")
        self._username = config.username
        self._password = config.password

    @property
    def is_available(self) -> bool:
        """Check if the provider is available."""
        return bool(self._base_url) and self._config.enabled

    def _get_auth(self) -> tuple[str, str] | None:
        """Get Basic Auth credentials."""
        if self._username and self._password:
            return (self._username, self._password)
        return None

    def _do_search(self, query: str, max_results: int, days: int = 7) -> SearchResponse:
        """Execute SearXNG search."""
        try:
            # 构建搜索URL
            search_url = f"{self._base_url}/search"

            # 确定时间范围
            time_range = "week"
            if days <= 1:
                time_range = "day"
            elif days <= 7:
                time_range = "week"
            elif days <= 30:
                time_range = "month"
            else:
                time_range = "year"

            # 请求参数
            params = {
                "q": query,
                "format": "json",
                "language": "zh-CN",
                "time_range": time_range,
                "safesearch": "0",
                "pageno": "1",
            }

            # 请求头
            headers = {
                "Accept": "application/json",
                "User-Agent": "StockAnalyzer/1.0",
            }

            # 添加Basic Auth
            auth = self._get_auth()
            basic_auth = None
            if auth:
                basic_auth = httpx.BasicAuth(auth[0], auth[1])
                logger.debug("[SearXNG] 使用Basic Auth认证")

            # 执行搜索
            if basic_auth:
                response = httpx.get(search_url, params=params, timeout=15, headers=headers, auth=basic_auth)
            else:
                response = httpx.get(search_url, params=params, timeout=15, headers=headers)

            # 检查HTTP状态码
            if response.status_code == 401:
                return SearchResponse(
                    query=query,
                    results=[],
                    provider=self.name,
                    success=False,
                    error_message="SearXNG 认证失败，请检查用户名和密码",
                )

            if response.status_code != 200:
                error_msg = f"HTTP {response.status_code}: {response.text[:200]}"
                logger.warning(f"[SearXNG] 搜索失败: {error_msg}")
                return SearchResponse(
                    query=query, results=[], provider=self.name, success=False, error_message=error_msg
                )

            # 解析响应
            try:
                data = response.json()
            except ValueError as e:
                error_msg = f"响应JSON解析失败: {str(e)}"
                logger.error(f"[SearXNG] {error_msg}")
                return SearchResponse(
                    query=query, results=[], provider=self.name, success=False, error_message=error_msg
                )

            logger.debug(f"[SearXNG] 搜索完成，query='{query}'")
            logger.debug(f"[SearXNG] 原始响应: {data}")

            # 解析搜索结果
            results = []
            search_results = data.get("results", [])

            for item in search_results[:max_results]:
                # 解析发布日期
                published_date = None
                published_str = item.get("publishedDate") or item.get("published")
                if published_str:
                    try:
                        # 尝试多种日期格式
                        for fmt in ["%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%a, %d %b %Y %H:%M:%S"]:
                            try:
                                dt = datetime.strptime(published_str, fmt)
                                published_date = dt.strftime("%Y-%m-%d")
                                break
                            except ValueError:
                                continue
                        if not published_date:
                            published_date = published_str[:10]  # Take first 10 chars as date
                    except Exception:
                        logger.debug(f"Failed to parse date: {published_str}")
                        published_date = published_str

                results.append(
                    SearchResult(
                        title=item.get("title", ""),
                        snippet=item.get("content", "")[:500],
                        url=item.get("url", ""),
                        source=item.get("source") or _extract_domain(item.get("url", "")),
                        published_date=published_date,
                    )
                )

            logger.debug(f"[SearXNG] 成功解析 {len(results)} 条结果")

            return SearchResponse(query=query, results=results, provider=self.name, success=True)

        except httpx.TimeoutException:
            error_msg = "请求超时"
            logger.error(f"[SearXNG] {error_msg}")
            return SearchResponse(query=query, results=[], provider=self.name, success=False, error_message=error_msg)
        except httpx.RequestError as e:
            error_msg = f"网络请求失败: {str(e)}"
            logger.error(f"[SearXNG] {error_msg}")
            return SearchResponse(query=query, results=[], provider=self.name, success=False, error_message=error_msg)
        except Exception as e:
            error_msg = f"未知错误: {str(e)}"
            logger.error(f"[SearXNG] {error_msg}")
            return SearchResponse(query=query, results=[], provider=self.name, success=False, error_message=error_msg)
