# Ashare Analyzer

An intelligent A-share stock analysis system powered by LLMs with multi-agent architecture. Automatically analyzes your watchlist and delivers a "Decision Dashboard" to Discord, Telegram, or Email.

[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

**[中文文档](README_CN.md)**

## Features

- **Multi-Agent Architecture**: Specialized AI agents collaborate to provide comprehensive analysis
  - Technical Analysis Agent - Chart patterns, indicators, trend analysis
  - Fundamental Analysis Agent - Financial statements, valuation metrics
  - News Sentiment Agent - News aggregation and sentiment scoring
  - Chip Distribution Agent - Institutional/retail fund flow analysis
  - Risk Management Agent - Position sizing, risk assessment
  - Portfolio Manager Agent - Final decision synthesis

- **Multi-Source Data**: Aggregates data from Akshare, Baostock, Tushare, eFinance, etc.

- **Flexible Notifications**: Push results to Discord, Telegram, or Email

- **Multiple AI Providers**: Supports 100+ LLM providers via LiteLLM (DeepSeek, OpenAI, Gemini, Claude, etc.)

- **Scheduled Execution**: Run daily analysis automatically

## Quick Start

### Prerequisites

- Python 3.13+
- [uv](https://docs.astral.sh/uv/) for dependency management

### Installation

```bash
# Clone the repository
git clone https://github.com/zwldarren/ashare-analyzer.git
cd ashare-analyzer

# Install dependencies
uv sync

# Copy environment configuration
cp .env.example .env
```

### Configuration

Edit `.env` file with your configurations:

```bash
# Stock watchlist (comma-separated)
STOCK_LIST=600519,300750,002594

# AI Model (LiteLLM format: provider/model-name)
LLM_MODEL=deepseek/deepseek-reasoner
LLM_API_KEY=your_api_key_here
```

See [Configuration](#configuration-details) for all options.

### Usage

```bash
# Run analysis
uv run ashare-analyzer

# Debug mode (verbose logging)
uv run ashare-analyzer --debug

# Analyze specific stocks
uv run ashare-analyzer --stocks 600519,300750

# Scheduled mode (runs daily at configured time)
uv run ashare-analyzer --schedule

# Dry run (fetch data only, no AI analysis)
uv run ashare-analyzer --dry-run

# Skip notifications
uv run ashare-analyzer --no-notify
```

## Configuration Details

### AI Model Configuration

Supports 100+ providers via LiteLLM format:

| Provider | Model Example | API Key Source |
|----------|---------------|----------------|
| DeepSeek | `deepseek/deepseek-reasoner` | [platform.deepseek.com](https://platform.deepseek.com/) |
| OpenAI | `openai/gpt-5.2` | [platform.openai.com](https://platform.openai.com/) |
| Gemini | `gemini/gemini-3.1-pro-preview` | [aistudio.google.com](https://aistudio.google.com/) |
| Claude | `anthropic/claude-sonnet-4-6` | [console.anthropic.com](https://console.anthropic.com/) |

Full provider list: [LiteLLM Providers](https://docs.litellm.ai/docs/providers)


### Notification Channels

Configure one or more notification channels:

```bash
# Discord
DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/...

# Telegram
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id

# Email (SMTP)
EMAIL_SMTP_HOST=smtp.gmail.com
EMAIL_SMTP_PORT=587
EMAIL_SMTP_USER=your_email@gmail.com
EMAIL_SMTP_PASSWORD=your_app_password
EMAIL_RECIPIENTS=recipient@example.com
```

### Search Engines (for news)

```bash
# Tavily (recommended)
TAVILY_API_KEY=your_tavily_key

# SerpAPI (alternative)
SERPAPI_API_KEY=your_serpapi_key
```

### News Filter Configuration

The news filter uses AI to filter out low-relevance and stale news results.

| Environment Variable | Description | Default |
|------------------|-------------|---------|
| `NEWS_FILTER_ENABLED` | Enable/disable news filter | `true` |
| `NEWS_FILTER_MIN_RESULTS` | Minimum results after filtering | `3` |
| `NEWS_FILTER_MODEL` | LLM model for filtering (optional, falls back to LLM_MODEL) | - |

Example:
```bash
NEWS_FILTER_ENABLED=true
NEWS_FILTER_MIN_RESULTS=3
NEWS_FILTER_MODEL=deepseek/deepseek-chat
```

## Development

```bash
# Format code
ruff format

# Lint code
ruff check --fix

# Type check
ty check .

# Run tests
uv run pytest

# Run tests with coverage
uv run pytest --cov=ashare_analyzer
```

## Docker

```bash
# Build image
docker build -t ashare-analyzer -f docker/Dockerfile .

# Run container
docker run -it --env-file .env ashare-analyzer

# Using docker-compose
cd docker && docker-compose up -d
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [ZhuLinsen/daily_stock_analysis](https://github.com/ZhuLinsen/daily_stock_analysis) - Original project this was forked from
- [virattt/ai-hedge-fund](https://github.com/virattt/ai-hedge-fund) - Multi-agent architecture inspiration
- All the open-source libraries that made this possible

---

⭐ Star this repo if you find it helpful!
