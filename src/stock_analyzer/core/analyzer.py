"""
Core analyzer module - simplified stock analysis workflow.

Replaces the complex CQRS Command pattern with direct function calls.
This module coordinates data preparation and delegates AI analysis to the agents module.
"""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from stock_analyzer.core.context_builders import (
    build_basic_context,
    build_chip_context,
    build_financial_context,
    build_growth_context,
    build_market_context,
    build_price_data,
    build_technical_context,
    build_technical_indicators,
    build_valuation_context,
    get_current_price,
)
from stock_analyzer.core.dependencies import get_ai_analyzer, get_data_service, get_db, get_search_service
from stock_analyzer.domain.models import AnalysisResult

logger = logging.getLogger(__name__)


def analyze_stock(
    stock_code: str,
    save_context_snapshot: bool = True,
) -> AnalysisResult:
    """
    Analyze a single stock using the multi-agent system.

    This function coordinates data preparation (Core layer) and
    delegates AI analysis to the agents layer.

    Args:
        stock_code: Stock code to analyze
        save_context_snapshot: Whether to save context to database

    Returns:
        AnalysisResult with trading decision and reasoning
    """
    logger.info(f"开始分析股票: {stock_code}")

    # 1. Get data service
    data_service = get_data_service()

    # 2. Build analysis context
    context, news_context = _build_analysis_context(stock_code, data_service)

    if not context.get("raw_data"):
        logger.error(f"无法获取股票数据: {stock_code}")
        return AnalysisResult(
            code=stock_code,
            name=f"Stock{stock_code}",
            sentiment_score=50,
            trend_prediction="震荡",
            operation_advice="持有",
            decision_type="hold",
            confidence_level="低",
            analysis_summary="Failed to retrieve data",
            success=False,
            error_message="No historical data available",
        )

    # 3. Execute AI analysis through agents layer
    analyzer = get_ai_analyzer()
    result = analyzer.analyze(context, news_context=news_context)

    # 4. Save result to database
    if result and result.success:
        db = get_db()
        db.save_analysis_history(
            result=result,
            query_id="",
            news_content=news_context,
            save_snapshot=save_context_snapshot,
        )
        logger.info(f"分析完成: {result.operation_advice}")
    else:
        logger.error(f"分析失败: {result.error_message if result else '未知错误'}")

    return result


def batch_analyze(
    stock_codes: list[str],
    max_workers: int = 3,
    save_context_snapshot: bool = True,
) -> list[AnalysisResult]:
    """
    Batch analyze multiple stocks concurrently.

    Args:
        stock_codes: List of stock codes to analyze
        max_workers: Maximum concurrent workers
        save_context_snapshot: Whether to save context to database

    Returns:
        List of successful AnalysisResult objects
    """
    logger.info(f"开始批量分析 {len(stock_codes)} 只股票")

    results: list[AnalysisResult] = []

    # Use thread pool for concurrent analysis
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_code = {executor.submit(analyze_stock, code, save_context_snapshot): code for code in stock_codes}

        for future in as_completed(future_to_code):
            code = future_to_code[future]
            try:
                result = future.result()
                if result and result.success:
                    results.append(result)
                    logger.info(f"[{code}] 分析完成: {result.operation_advice}")
                else:
                    logger.warning(f"[{code}] 分析失败")
            except Exception as e:
                logger.error(f"[{code}] 分析出错: {e}")

    return results


def _build_analysis_context(
    stock_code: str,
    data_service: Any,
) -> tuple[dict[str, Any], str | None]:
    """
    Build complete analysis context for AI agents.

    Core layer responsibility: Data preparation and context building.
    This function gathers all necessary data for the agents layer to perform analysis.

    Args:
        stock_code: Stock code to analyze
        data_service: Data service instance

    Returns:
        Tuple of (context dict, news context string)
    """
    from stock_analyzer.domain import StockNameResolver

    # 1. Get basic data
    realtime_quote = data_service.get_realtime_quote(stock_code)
    stock_name = ""
    if realtime_quote and realtime_quote.name:
        stock_name = realtime_quote.name
    else:
        # 使用 StockNameResolver 从数据源解析
        stock_name = StockNameResolver(data_manager=data_service).resolve(stock_code)

    daily_data, _ = data_service.get_daily_data(stock_code, days=90)
    if not stock_name:
        stock_name = f"Stock{stock_code}"

    # 2. Get supplementary data
    chip_data = None
    try:
        chip_data = data_service.get_chip_distribution(stock_code)
    except Exception as e:
        logger.debug(f"[{stock_code}] 获取筹码分布失败: {e}")

    # 3. News search
    news_context = None
    search_service = get_search_service()
    if search_service and search_service.is_available:
        try:
            intel_results = search_service.search_comprehensive_intel(
                stock_code=stock_code, stock_name=stock_name, max_searches=5
            )
            if intel_results:
                news_context = search_service.format_intel_report(intel_results, stock_name)
        except Exception as e:
            logger.warning(f"[{stock_code}] 新闻搜索失败: {e}")

    # 4. Build context using focused builders
    context = build_basic_context(stock_code, stock_name, daily_data, realtime_quote)

    # Add technical context
    technical_context = build_technical_context(daily_data)
    context.update(technical_context)

    # Add current price
    current_price = get_current_price(realtime_quote, daily_data)
    if current_price > 0:
        context["current_price"] = current_price

    # Add price data
    price_data = build_price_data(daily_data)
    if price_data:
        context["price_data"] = price_data

    # Add technical indicators
    technical_indicators = build_technical_indicators(daily_data)
    if technical_indicators:
        context["technical_data"] = technical_indicators

    # Add valuation context
    valuation_data = build_valuation_context(realtime_quote, daily_data, current_price, data_service, stock_code)
    if valuation_data:
        context["valuation_data"] = valuation_data
        if "industry_name" in valuation_data:
            context["industry"] = valuation_data["industry_name"]

    # Add financial context
    financial_data = build_financial_context(realtime_quote, daily_data)
    if financial_data:
        context["financial_data"] = financial_data

    # Add growth context
    growth_data = build_growth_context(daily_data, realtime_quote)
    if growth_data:
        context["growth_data"] = growth_data

    # Add market context
    market_data = build_market_context(daily_data)
    if market_data:
        context["market_data"] = market_data

    # Add chip context
    chip_context = build_chip_context(chip_data)
    if chip_context:
        context["chip"] = chip_context

    return context, news_context
