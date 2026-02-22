# A股分析器

基于大语言模型（LLM）的智能A股分析系统，采用多智能体架构。自动分析您的自选股列表，并将"决策仪表盘"推送到 Discord、Telegram 或 Email。

[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

**[English](README.md)**

## 功能特点

- **多智能体架构**：专业化的 AI 智能体协同工作，提供全面分析
  - 技术分析智能体 - 图表形态、技术指标、趋势分析
  - 基本面分析智能体 - 财务报表、估值指标
  - 新闻情绪智能体 - 新闻聚合与情绪评分
  - 筹码分布智能体 - 机构/散户资金流向分析
  - 风险管理智能体 - 仓位管理、风险评估
  - 投资组合经理智能体 - 最终决策综合

- **多数据源**：聚合 Akshare、Baostock、Tushare、eFinance 等数据源

- **灵活的通知渠道**：支持 Discord、Telegram、Email 推送

- **多 AI 提供商**：通过 LiteLLM 支持 100+ LLM 提供商（DeepSeek、OpenAI、Gemini、Claude 等）

- **定时执行**：自动每日分析

## 快速开始

### 环境要求

- Python 3.13+
- [uv](https://docs.astral.sh/uv/) 包管理器

### 安装

```bash
# 克隆仓库
git clone https://github.com/zwldarren/ashare-analyzer.git
cd ashare-analyzer

# 安装依赖
uv sync

# 复制环境配置
cp .env.example .env
```

### 配置

编辑 `.env` 文件：

```bash
# 自选股列表（逗号分隔）
STOCK_LIST=600519,300750,002594

# AI 模型（LiteLLM 格式：provider/model-name）
LLM_MODEL=deepseek/deepseek-reasoner
LLM_API_KEY=your_api_key_here
```

完整配置选项见[配置详情](#配置详情)。

### 使用方法

```bash
# 运行分析
uv run ashare-analyzer

# 调试模式（详细日志）
uv run ashare-analyzer --debug

# 分析指定股票
uv run ashare-analyzer --stocks 600519,300750

# 定时模式（每日在配置时间自动执行）
uv run ashare-analyzer --schedule

# 试运行模式（仅获取数据，不进行 AI 分析）
uv run ashare-analyzer --dry-run

# 跳过通知推送
uv run ashare-analyzer --no-notify
```

## 配置详情

### AI 模型配置

通过 LiteLLM 格式支持 100+ 提供商：

| 提供商 | 模型示例 | API Key 获取 |
|--------|----------|--------------|
| DeepSeek | `deepseek/deepseek-reasoner` | [platform.deepseek.com](https://platform.deepseek.com/) |
| OpenAI | `openai/gpt-5.2` | [platform.openai.com](https://platform.openai.com/) |
| Gemini | `gemini/gemini-3.1-pro-preview` | [aistudio.google.com](https://aistudio.google.com/) |
| Claude | `anthropic/claude-sonnet-4-6` | [console.anthropic.com](https://console.anthropic.com/) |
| Azure | `azure/gpt-5.2` | Azure Portal |

完整提供商列表：[LiteLLM Providers](https://docs.litellm.ai/docs/providers)

### 备选模型

配置备选模型实现自动故障转移：

```bash
LLM_FALLBACK_MODEL=gemini/gemini-3-flash-preview
LLM_FALLBACK_API_KEY=your_fallback_key
```

### 通知渠道

配置一个或多个通知渠道：

```bash
# Discord
DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/...

# Telegram
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id

# 邮件 (SMTP)
EMAIL_SMTP_HOST=smtp.gmail.com
EMAIL_SMTP_PORT=587
EMAIL_SMTP_USER=your_email@gmail.com
EMAIL_SMTP_PASSWORD=your_app_password
EMAIL_RECIPIENTS=recipient@example.com
```

### 搜索引擎（用于获取新闻）

```bash
# Tavily（推荐）
TAVILY_API_KEY=your_tavily_key

# SerpAPI（备选）
SERPAPI_API_KEY=your_serpapi_key
```

### 定时任务配置

```bash
# 启用定时任务
SCHEDULE_ENABLED=true

# 执行时间（24小时制，默认 18:00）
SCHEDULE_TIME=18:00
```

## 开发

```bash
# 格式化代码
ruff format

# 代码检查
ruff check --fix

# 类型检查
ty check .

# 运行测试
uv run pytest

# 运行测试并生成覆盖率报告
uv run pytest --cov=ashare_analyzer
```

## Docker 部署

```bash
# 构建镜像
docker build -t ashare-analyzer -f docker/Dockerfile .

# 运行容器
docker run -it --env-file .env ashare-analyzer

# 使用 docker-compose
cd docker && docker-compose up -d
```

## 命令行选项

| 选项 | 说明 |
|------|------|
| `--debug` | 启用调试模式，输出详细日志 |
| `--dry-run` | 仅获取数据，不进行 AI 分析 |
| `--stocks <codes>` | 指定股票代码（逗号分隔，覆盖配置文件） |
| `--no-notify` | 不发送推送通知 |
| `--single-notify` | 单股推送模式：每分析完一只股票立即推送 |
| `--workers <n>` | 并发线程数 |
| `--schedule` | 启用定时任务模式 |

## 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件。

## 致谢

- [ZhuLinsen/daily_stock_analysis](https://github.com/ZhuLinsen/daily_stock_analysis) - 原始项目来源
- [virattt/ai-hedge-fund](https://github.com/virattt/ai-hedge-fund) - 多智能体架构灵感
- 所有开源库的贡献者

---

⭐ 如果这个项目对您有帮助，请给个 Star！