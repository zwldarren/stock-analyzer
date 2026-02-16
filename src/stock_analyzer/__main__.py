"""
===================================
A股自选股智能分析系统 - 主调度程序 (Simplified)
===================================

使用方式：
    python -m stock_analyzer              # 正常运行
    python -m stock_analyzer --debug      # 调试模式
    python -m stock_analyzer --dry-run    # 仅获取数据不分析
"""

import atexit
import logging
import os
import sys
from datetime import datetime

import click

from stock_analyzer.analysis import batch_analyze
from stock_analyzer.config import Config, check_config_valid, get_config, get_config_safe
from stock_analyzer.dependencies import get_data_manager, get_notification_service
from stock_analyzer.utils.console_display import get_display
from stock_analyzer.utils.logging_config import get_console, setup_logging

logger = logging.getLogger(__name__)


def _cleanup_resources() -> None:
    try:
        from stock_analyzer.ai.clients import shutdown_llm_http_clients

        shutdown_llm_http_clients()
    except Exception:
        pass


atexit.register(_cleanup_resources)


def _print_banner() -> None:
    """使用 Rich 输出启动横幅，替代 logger.info 避免格式混乱"""
    console = get_console()
    console.print()
    console.rule("[bold cyan]A股自选股智能分析系统[/bold cyan]", style="cyan")
    console.print(f"  运行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", style="dim")
    console.print()


def _print_analysis_header(stock_codes: list[str], max_workers: int, dry_run: bool) -> None:
    """使用 Rich 输出分析配置摘要"""
    console = get_console()
    mode = "仅获取数据" if dry_run else "完整分析"
    console.print(
        f"📋 分析 [bold]{len(stock_codes)}[/bold] 只股票: [cyan]{', '.join(stock_codes)}[/cyan]"
        f"  (并发: {max_workers}, 模式: {mode})"
    )
    console.print()


@click.group(invoke_without_command=True)
@click.option("--debug", is_flag=True, help="启用调试模式，输出详细日志")
@click.option("--dry-run", is_flag=True, help="仅获取数据，不进行 AI 分析")
@click.option("--stocks", type=str, help="指定要分析的股票代码，逗号分隔（覆盖配置文件）")
@click.option("--no-notify", is_flag=True, help="不发送推送通知")
@click.option(
    "--single-notify",
    is_flag=True,
    help="启用单股推送模式：每分析完一只股票立即推送，而不是汇总推送",
)
@click.option("--workers", type=int, default=None, help="并发线程数（默认使用配置值）")
@click.option("--schedule", is_flag=True, help="启用定时任务模式，每日定时执行")
@click.pass_context
def cli(
    ctx: click.Context,
    debug: bool,
    dry_run: bool,
    stocks: str | None,
    no_notify: bool,
    single_notify: bool,
    workers: int | None,
    schedule: bool,
) -> int:
    """A股自选股智能分析系统"""
    # 如果没有子命令，运行主程序
    if ctx.invoked_subcommand is None:
        return run_main(
            debug,
            dry_run,
            stocks,
            no_notify,
            single_notify,
            workers,
            schedule,
        )
    return 0


def run_main(
    debug: bool,
    dry_run: bool,
    stocks: str | None,
    no_notify: bool,
    single_notify: bool,
    workers: int | None,
    schedule: bool,
) -> int:
    """运行主程序逻辑"""
    # 检查配置
    config, errors = get_config_safe()
    is_valid, missing = check_config_valid(config)

    if not is_valid:
        from rich.console import Console

        console = Console()
        console.print("\n[bold yellow]⚠️ 未检测到有效配置[/bold yellow]")
        console.print("\n[dim]缺少以下必需配置:[/dim]")
        for item in missing:
            console.print(f"  - {item}")
        console.print("\n[dim]请运行以下命令完成初始化:[/dim]")
        console.print("  [bold cyan]stock-analyzer init[/bold cyan]")
        return 1

    # 加载配置（在设置日志前加载，以获取日志目录）
    config = get_config()

    # 应用系统配置：代理设置
    # GitHub Actions 环境自动跳过代理配置
    if os.getenv("GITHUB_ACTIONS") != "true":
        if config.system.http_proxy:
            os.environ["http_proxy"] = config.system.http_proxy
            logger.debug(f"已设置 http_proxy: {config.system.http_proxy}")
        if config.system.https_proxy:
            os.environ["https_proxy"] = config.system.https_proxy
            logger.debug(f"已设置 https_proxy: {config.system.https_proxy}")

    effective_debug = debug or config.system.debug
    setup_logging(debug=effective_debug, log_dir=config.logging.log_dir)

    _print_banner()

    # 验证配置
    warnings = config.validate_config()
    for warning in warnings:
        logger.warning(warning)

    # 解析股票列表
    stock_codes = None
    if stocks:
        stock_codes = [code.strip() for code in stocks.split(",") if code.strip()]
        logger.info(f"使用命令行指定的股票列表: {stock_codes}")

    try:
        # 模式1: 定时任务模式
        if schedule or config.schedule.schedule_enabled:
            logger.info("模式: 定时任务")
            logger.info(f"每日执行时间: {config.schedule.schedule_time}")

            from stock_analyzer.scheduler import run_with_schedule

            def scheduled_task():
                run_full_analysis(
                    config,
                    stock_codes,
                    dry_run,
                    no_notify,
                    single_notify,
                    workers,
                    debug,
                )

            run_with_schedule(
                task=scheduled_task,
                schedule_time=config.schedule.schedule_time,
                run_immediately=True,
            )
            return 0

        # 模式2: 正常单次运行
        run_full_analysis(
            config,
            stock_codes,
            dry_run,
            no_notify,
            single_notify,
            workers,
            debug,
        )

        console = get_console()
        console.print()
        console.rule("[dim]执行完成[/dim]", style="dim")
        console.print()

        return 0

    except KeyboardInterrupt:
        logger.info("用户中断，程序退出")
        return 130

    except Exception as e:
        logger.exception(f"程序执行失败: {e}")
        return 1


def run_full_analysis(
    config: Config,
    stock_codes: list[str] | None,
    dry_run: bool,
    no_notify: bool,
    single_notify: bool,
    workers: int | None,
    debug: bool = False,
):
    """
    执行完整的分析流程 (Simplified version)

    这是定时任务调用的主函数
    """
    try:
        # 命令行参数 --single-notify 覆盖配置（#55）
        if single_notify:
            config.notification_message.single_stock_notify = True

        # 获取股票列表
        if stock_codes is None:
            config.refresh_stock_list()
            stock_codes = config.stock_list

        if not stock_codes:
            logger.error("未配置自选股列表，请在 .env 文件中设置 STOCK_LIST")
            return []

        max_workers = workers or config.system.max_workers

        # 使用 Rich 输出分析头（不经过 logger）
        _print_analysis_header(stock_codes, max_workers, dry_run)

        # 批量预取实时行情
        _prefetch_realtime_quotes(stock_codes)

        results = []

        if dry_run:
            # dry_run模式：仅获取数据
            results = _run_dry_mode(stock_codes, max_workers)
        else:
            # 正常分析模式
            results = _run_analysis_mode(
                stock_codes,
                config,
                max_workers,
                no_notify,
            )

        # 输出最终报告
        if results:
            display = get_display()
            display.show_final_report(results)

        return results

    except Exception as e:
        logger.exception(f"分析流程执行失败: {e}")
        return []


def _prefetch_realtime_quotes(stock_codes: list[str]) -> None:
    """批量预取实时行情数据以优化性能"""
    try:
        data_manager = get_data_manager()
        prefetch_count = data_manager.prefetch_realtime_quotes(stock_codes)
        if prefetch_count > 0:
            logger.debug(f"已启用批量预取架构：一次拉取全市场数据，{len(stock_codes)} 只股票共享缓存")
    except Exception as e:
        logger.debug(f"批量预取实时行情失败: {e}")


def _run_dry_mode(stock_codes: list[str], max_workers: int) -> list:
    """dry_run模式：仅获取数据，不进行分析"""
    logger.info("Dry-run模式：仅获取数据")

    from concurrent.futures import ThreadPoolExecutor, as_completed

    data_manager = get_data_manager()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_code = {executor.submit(data_manager.get_daily_data, code, 90): code for code in stock_codes}

        for future in as_completed(future_to_code):
            code = future_to_code[future]
            try:
                daily_data, source = future.result()
                if daily_data is not None and not daily_data.empty:
                    logger.info(f"[{code}] 数据获取成功: {len(daily_data)} 条")
                else:
                    logger.warning(f"[{code}] 数据获取失败")
            except Exception as e:
                logger.error(f"[{code}] 任务执行失败: {e}")

    # dry_run模式下返回空列表（没有分析结果）
    return []


def _run_analysis_mode(
    stock_codes: list[str],
    config: Config,
    max_workers: int,
    no_notify: bool,
) -> list:
    """正常分析模式"""
    # 批量分析
    results = batch_analyze(
        stock_codes=stock_codes,
        max_workers=max_workers,
    )

    # 发送通知
    if results and not no_notify:
        _send_notifications(results, config)

    return results


def _send_notifications(results: list, config: Config) -> None:
    """发送分析结果通知"""
    try:
        notifier = get_notification_service()

        logger.info("生成决策仪表盘日报...")
        report = notifier.generate_dashboard_report(results)
        filepath = notifier.save_report_to_file(report)
        logger.info(f"决策仪表盘日报已保存: {filepath}")

        if not notifier.is_available():
            logger.info("通知渠道未配置，跳过推送")
            return

        # 发送到各个渠道
        _send_to_channels(notifier, report, results)

    except Exception as e:
        logger.error(f"发送通知失败: {e}")


def _send_to_channels(notifier, report: str, results: list) -> None:
    """发送报告到各个通知渠道"""
    context_success = notifier.send_to_context(report)

    # 发送完整报告
    success = notifier.send(report) or context_success

    if success:
        logger.info("决策仪表盘推送成功")
    else:
        logger.warning("决策仪表盘推送失败")


# 主入口点
def main() -> int:
    """程序主入口"""
    return cli()


if __name__ == "__main__":
    sys.exit(main())
