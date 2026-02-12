"""交互式配置向导 - 使用 questionary 引导用户完成初始配置"""

from typing import Any

import click
import questionary
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from stock_analyzer.config import check_config_valid, get_config_safe
from stock_analyzer.config.config import get_project_root
from stock_analyzer.infrastructure import save_config_to_db_only

console = Console()

# 通知渠道选项
NOTIFICATION_CHANNELS = [
    ("邮件", "email"),
    ("Telegram", "telegram"),
    ("Discord Webhook", "discord"),
    ("自定义 Webhook", "custom_webhook"),
    ("暂不设置", "none"),
]


def print_welcome() -> None:
    """打印欢迎信息"""
    welcome_text = Text()
    welcome_text.append("🚀 股票分析器 - 初始化向导\n", style="bold cyan")
    welcome_text.append("本向导将帮助您完成基础配置\n", style="dim")
    welcome_text.append("您可以随时按 Ctrl+C 退出", style="dim")

    console.print(Panel(welcome_text, border_style="cyan"))


def print_step(step_num: int, total: int, title: str) -> None:
    """打印步骤标题"""
    console.print(f"\n[bold blue][{step_num}/{total}][/bold blue] {title}")


def ask_ai_config() -> dict[str, Any]:
    """询问 AI 模型配置"""
    print_step(1, 4, "配置 AI 模型")

    console.print("\n[dim]litellm 格式: provider/model-name[/dim]")
    console.print("[dim]例如: openai/gpt-5, deepseek/deepseek-chat, gemini/gemini-3-pro[/dim]")

    # 直接询问模型名称
    model = questionary.text(
        "请输入模型名称 (litellm 格式):",
        default="deepseek/deepseek-chat",
    ).ask()

    if not model:
        console.print("[yellow]警告: 未输入模型名称[/yellow]")
        return {}

    # API 密钥
    api_key = questionary.password("请输入 API 密钥:").ask()

    # 可选：Base URL
    base_url = None
    use_custom_url = questionary.confirm(
        "是否需要设置自定义 API 地址?",
        default=False,
    ).ask()

    if use_custom_url:
        base_url = questionary.text("请输入 API Base URL:").ask()

    config = {
        "LLM_MODEL": model,
        "LLM_API_KEY": api_key,
        "LLM_BASE_URL": base_url,
    }

    # 高级 AI 参数配置
    console.print("\n[dim]AI 生成参数配置 (可选，使用默认值直接回车)[/dim]")

    # 温度参数
    temperature_str = questionary.text(
        "温度参数 (0-2, 默认 0.7，越低越确定，越高越创造性):",
        default="0.7",
    ).ask()
    if temperature_str and temperature_str != "0.7":
        try:
            temp_val = float(temperature_str)
            if 0 <= temp_val <= 2:
                config["LLM_TEMPERATURE"] = temperature_str
        except ValueError:
            pass

    # 最大 token
    max_tokens_str = questionary.text(
        "最大输出 Token 数 (默认 8192):",
        default="8192",
    ).ask()
    if max_tokens_str and max_tokens_str != "8192":
        config["LLM_MAX_TOKENS"] = max_tokens_str

    # 备选模型配置
    use_fallback = questionary.confirm(
        "是否配置备选模型? (主模型失败时自动切换)",
        default=False,
    ).ask()

    if use_fallback:
        console.print("\n[dim]备选模型配置 (litellm 格式)[/dim]")
        fallback_model = questionary.text(
            "请输入备选模型名称:",
            default="gemini/gemini-3-flash-preview",
        ).ask()
        if fallback_model:
            config["LLM_FALLBACK_MODEL"] = fallback_model
            fallback_api_key = questionary.password("请输入备选模型 API Key:").ask()
            if fallback_api_key:
                config["LLM_FALLBACK_API_KEY"] = fallback_api_key
            use_fallback_url = questionary.confirm("备选模型是否需要自定义 API 地址?", default=False).ask()
            if use_fallback_url:
                fallback_url = questionary.text("请输入备选模型 API Base URL:").ask()
                if fallback_url:
                    config["LLM_FALLBACK_BASE_URL"] = fallback_url

    return config


def ask_stock_list() -> dict[str, Any]:
    """询问股票列表"""
    print_step(2, 4, "配置股票列表")

    console.print("\n[dim]提示: 输入股票代码，多个代码用逗号分隔[/dim]")
    console.print("[dim]例如: 600519,000858,300750[/dim]")

    stock_input = questionary.text("请输入要分析的股票代码:").ask()

    if not stock_input:
        console.print("[yellow]警告: 未输入股票代码，后续可通过命令行 --stocks 参数指定[/yellow]")
        return {}

    # 简单验证和格式化
    stocks = [s.strip() for s in stock_input.split(",") if s.strip()]

    if not stocks:
        console.print("[yellow]警告: 未输入股票代码[/yellow]")
        return {}

    console.print(f"[green]已配置 {len(stocks)} 只股票[/green]")
    return {"STOCK_LIST": ",".join(stocks)}


def _configure_single_channel(channel_type: str) -> dict[str, str]:
    """配置单个通知渠道的参数"""
    config: dict[str, str] = {}

    if channel_type == "telegram":
        token = questionary.text("请输入 Telegram Bot Token:").ask()
        chat_id = questionary.text("请输入 Telegram Chat ID:").ask()
        if token and chat_id:
            config["TELEGRAM_BOT_TOKEN"] = token
            config["TELEGRAM_CHAT_ID"] = chat_id
        thread_id = questionary.text("请输入 Telegram Message Thread ID (可选，直接回车跳过):").ask()
        if thread_id:
            config["TELEGRAM_MESSAGE_THREAD_ID"] = thread_id

    elif channel_type == "email":
        config["EMAIL_SENDER"] = questionary.text("请输入发件人邮箱:").ask() or ""
        config["EMAIL_PASSWORD"] = questionary.password("请输入邮箱密码/授权码:").ask() or ""
        receivers = questionary.text("请输入收件人邮箱 (多个用逗号分隔):").ask()
        if receivers:
            config["EMAIL_RECEIVERS"] = receivers

    elif channel_type == "discord":
        url = questionary.text("请输入 Discord Webhook URL:").ask()
        if url:
            config["DISCORD_WEBHOOK_URL"] = url

    elif channel_type == "custom_webhook":
        urls = []
        while True:
            url = questionary.text("请输入 Webhook URL:").ask()
            if url:
                urls.append(url)
                if not questionary.confirm("是否添加更多 Webhook URL?", default=False).ask():
                    break
            else:
                break
        if urls:
            config["CUSTOM_WEBHOOK_URLS"] = ",".join(urls)
        use_bearer = questionary.confirm("是否需要 Bearer Token 认证?", default=False).ask()
        if use_bearer:
            config["CUSTOM_WEBHOOK_BEARER_TOKEN"] = questionary.password("请输入 Bearer Token:").ask()

    return config


def ask_notification() -> dict[str, Any]:
    """询问通知配置 - 支持多选通知渠道"""
    print_step(3, 4, "配置通知渠道 (可选)")

    console.print("\n[dim]配置通知渠道后，分析结果将自动推送到指定平台[/dim]")
    console.print("[dim]支持配置多个通知渠道，实现冗余备份[/dim]")

    should_config = questionary.confirm(
        "是否配置通知渠道?",
        default=False,
    ).ask()

    if not should_config:
        return {}

    # 选择通知渠道 - 使用 checkbox 支持多选
    choices = [name for name, _ in NOTIFICATION_CHANNELS if _ != "none"]
    selected_names = questionary.checkbox(
        "请选择通知渠道 (空格选择，回车确认):",
        choices=choices,
    ).ask()

    if not selected_names:
        console.print("[yellow]未选择任何通知渠道[/yellow]")
        return {}

    # 为每个选中的渠道配置参数
    all_config: dict[str, Any] = {}
    name_to_type = {name: ctype for name, ctype in NOTIFICATION_CHANNELS}

    for channel_name in selected_names:
        channel_type = name_to_type.get(channel_name)
        if not channel_type or channel_type == "none":
            continue

        console.print(f"\n[bold cyan]配置 {channel_name}:[/bold cyan]")
        channel_config = _configure_single_channel(channel_type)
        all_config.update(channel_config)

    if all_config:
        enabled_channels = [name for name in selected_names if name != "暂不设置"]
        console.print(f"[green]已配置 {len(enabled_channels)} 个通知渠道[/green]")

    return all_config


def _configure_search_provider(provider: str) -> dict[str, str]:
    """配置单个搜索引擎的 API key，支持多 key"""
    config: dict[str, str] = {}

    if provider == "searxng":
        config["SEARXNG_BASE_URL"] = (
            questionary.text("请输入 SearXNG 实例地址 (如 https://searx.example.com):").ask() or ""
        )
        use_auth = questionary.confirm("是否需要 Basic Auth 认证?", default=False).ask()
        if use_auth:
            config["SEARXNG_USERNAME"] = questionary.text("请输入用户名:").ask() or ""
            config["SEARXNG_PASSWORD"] = questionary.password("请输入密码:").ask() or ""
    else:
        # 支持配置多个 API key 用于轮换
        keys: list[str] = []
        while True:
            key = questionary.password(f"请输入 {provider} API Key:").ask()
            if key:
                keys.append(key)
                if not questionary.confirm(f"是否添加更多 {provider} API Key (用于轮换)?", default=False).ask():
                    break
            else:
                break

        if keys:
            key_str = ",".join(keys)
            provider_lower = provider.lower()
            if provider_lower == "tavily":
                config["TAVILY_API_KEYS"] = key_str
            elif provider_lower == "brave":
                config["BRAVE_API_KEYS"] = key_str
            elif provider_lower == "serpapi":
                config["SERPAPI_API_KEYS"] = key_str
            elif provider_lower == "bocha":
                config["BOCHA_API_KEYS"] = key_str

    return config


def ask_advanced_options() -> dict[str, Any]:
    """询问高级选项 - 支持多选搜索引擎"""
    print_step(4, 4, "高级选项 (可选)")

    config: dict[str, Any] = {}

    # 搜索引擎配置 - 使用 checkbox 支持多选
    should_config_search = questionary.confirm(
        "是否配置搜索引擎 API? (用于获取新闻情报)",
        default=False,
    ).ask()

    if should_config_search:
        search_choices = ["Tavily", "Brave", "SerpAPI", "Bocha", "SearXNG (自托管)"]
        selected_providers = questionary.checkbox(
            "请选择搜索引擎 (空格选择，回车确认):",
            choices=search_choices,
        ).ask()

        if selected_providers:
            for provider in selected_providers:
                console.print(f"\n[bold cyan]配置 {provider}:[/bold cyan]")
                provider_config = _configure_search_provider(provider)
                config.update(provider_config)

            console.print(f"[green]已配置 {len(selected_providers)} 个搜索引擎[/green]")

    # 数据源配置
    should_config_tushare = questionary.confirm(
        "是否配置 Tushare Token? (专业数据源，可选)",
        default=False,
    ).ask()

    if should_config_tushare:
        config["TUSHARE_TOKEN"] = questionary.password("请输入 Tushare Token:").ask()

    # 代理配置
    should_config_proxy = questionary.confirm(
        "是否需要配置代理?",
        default=False,
    ).ask()

    if should_config_proxy:
        http_proxy = questionary.text(
            "请输入 HTTP 代理地址 (可选，直接回车跳过):",
            default="",
        ).ask()
        https_proxy = questionary.text(
            "请输入 HTTPS 代理地址 (可选，直接回车跳过):",
            default="",
        ).ask()
        if http_proxy:
            config["HTTP_PROXY"] = http_proxy
        if https_proxy:
            config["HTTPS_PROXY"] = https_proxy

    # 定时任务配置
    console.print("\n[dim]定时任务配置 (可选)[/dim]")
    enable_schedule = questionary.confirm(
        "是否启用定时任务? (每天自动执行分析)",
        default=False,
    ).ask()

    if enable_schedule:
        config["SCHEDULE_ENABLED"] = "true"
        schedule_time = questionary.text(
            "请输入每日执行时间 (HH:MM 格式，默认 18:00):",
            default="18:00",
        ).ask()
        if schedule_time and schedule_time != "18:00":
            config["SCHEDULE_TIME"] = schedule_time

    return config


def run_setup_wizard() -> bool:
    """
    运行交互式配置向导

    Returns:
        是否成功完成配置
    """
    try:
        print_welcome()

        # 收集所有配置
        all_config = {}

        # 步骤 1: AI 配置
        ai_config = ask_ai_config()
        all_config.update(ai_config)

        # 步骤 2: 股票列表
        stock_config = ask_stock_list()
        all_config.update(stock_config)

        # 步骤 3: 通知配置
        notify_config = ask_notification()
        all_config.update(notify_config)

        # 步骤 4: 高级选项
        advanced_config = ask_advanced_options()
        all_config.update(advanced_config)

        # 保存配置
        console.print("\n[bold cyan]正在保存配置到数据库...[/bold cyan]")

        # 过滤掉 None 值
        all_config = {k: v for k, v in all_config.items() if v is not None}

        # 保存到数据库（TUI 配置只保存到数据库，不保存到 .env）
        db_path = get_project_root() / "data" / "stock_analysis.db"
        db_url = f"sqlite:///{db_path}"
        save_config_to_db_only(all_config, db_url)

        console.print("[green]✅ 配置已保存到数据库[/green]")

        # 验证配置
        console.print("\n[bold cyan]验证配置...[/bold cyan]")

        # 清除缓存，重新加载配置
        from stock_analyzer.config.config import get_config

        get_config.cache_clear()

        config, errors = get_config_safe()
        is_valid, missing = check_config_valid(config)

        if is_valid:
            console.print("\n[bold green]✅ 配置完成并验证通过！[/bold green]")
            console.print(f"[dim]配置已保存到数据库: {db_path}[/dim]")
            console.print("\n[bold]现在可以运行:[/bold] stock-analyzer")
            return True
        else:
            console.print("\n[yellow]⚠️ 配置已保存，但缺少以下必需项:[/yellow]")
            for item in missing:
                console.print(f"  - {item}")
            return False

    except KeyboardInterrupt:
        console.print("\n\n[yellow]配置已取消[/yellow]")
        return False
    except Exception as e:
        console.print(f"\n\n[red]配置失败: {e}[/red]")
        return False


@click.command(name="init")
@click.option(
    "--skip-check",
    is_flag=True,
    help="跳过配置检查，直接运行向导",
)
def init_command(skip_check: bool) -> int:
    """运行配置初始化向导"""
    # 检查现有配置
    if not skip_check:
        config, errors = get_config_safe()
        is_valid, missing = check_config_valid(config)

        if is_valid:
            console.print("[green]✅ 检测到有效配置，无需初始化[/green]")
            should_run = questionary.confirm(
                "是否重新运行配置向导?",
                default=False,
            ).ask()
            if not should_run:
                return 0

    # 运行向导
    success = run_setup_wizard()
    return 0 if success else 1


def check_and_prompt_config() -> bool:
    """
    检查配置并在需要时提示用户运行初始化向导

    Returns:
        配置是否有效
    """
    config, errors = get_config_safe()
    is_valid, missing = check_config_valid(config)

    if is_valid:
        return True

    # 配置无效，提示用户
    console.print("\n[bold yellow]⚠️ 未检测到有效配置[/bold yellow]")

    if missing:
        console.print("\n[dim]缺少以下必需配置:[/dim]")
        for item in missing:
            console.print(f"  - {item}")

    console.print("\n[dim]您可以通过以下方式配置:[/dim]")
    console.print("  1. 运行初始化向导: stock-analyzer init")
    console.print("  2. 手动创建 .env 文件")

    should_run = questionary.confirm(
        "是否立即运行初始化向导?",
        default=True,
    ).ask()

    if should_run:
        return run_setup_wizard()

    return False
