# Stock Analyzer

An intelligent stock analysis system powered by LLMs (Large Language Models) that automatically analyzes your watchlist of A-shares, and pushes a "Decision Dashboard" to Discord / Telegram / Email.

[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)


## 🚀 Quick Start

### Prerequisites

- Python 3.13 or higher
- [uv](https://docs.astral.sh/uv/) for dependency management

### Installation

```bash
# Clone the repository
git clone git@github.com:zwldarren/stock-analyzer.git
cd stock-analyzer

# Install dependencies
uv sync

# Copy environment configuration
cp .env.example .env
```

### Configuration

Edit `.env` file with your configurations:

```bash
STOCK_LIST=600519,300750,002594
```

### Usage

```bash
# Run analysis
uv run stock-analyzer

# Debug mode
uv run stock-analyzer --debug

# Analyze specific stocks
uv run stock-analyzer --stocks 600519,000001

# Scheduled mode (runs daily)
uv run stock-analyzer --schedule

# Dry run (no AI analysis)
uv run stock-analyzer --dry-run

# No notifications
uv run stock-analyzer --no-notify
```

## 🔧 Development

```bash
# Format code
ruff format

# Lint code
ruff check --fix

# Type check
ty check .

# Run tests
pytest
```

## 📦 Docker

```bash
# Build image
docker build -t stock-analyzer -f docker/Dockerfile .

# Run container
docker run -it --env-file .env stock-analyzer
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [ZhuLinsen/daily_stock_analysis](https://github.com/ZhuLinsen/daily_stock_analysis) - Original project this was forked from
- [virattt/ai-hedge-fund](https://github.com/virattt/ai-hedge-fund)
- All the open-source libraries that made this possible

---

⭐ Star this repo if you find it helpful!
