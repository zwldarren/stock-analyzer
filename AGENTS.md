# AGENTS.md - Agentic Coding Guidelines

## Build/Lint/Test Commands

### Installation & Environment
```bash
# Install dependencies
uv sync
```

### Lint, Format & Type Check
```bash
# Format code with ruff
ruff format

# Lint with ruff
ruff check --fix

# Type checking with ty
ty check .
```

### Running the Application
```bash
# Use uv run to execute (entry point is stock-analyzer command)
uv run stock-analyzer --help                    # See all options
uv run stock-analyzer --stocks 600519           # Analyze single stock
uv run stock-analyzer --market-review           # Market review only
uv run stock-analyzer --dry-run --no-notify     # Test without AI/notifications
uv run stock-analyzer --debug                   # Debug mode with verbose logs
uv run stock-analyzer --schedule                # Enable scheduled execution
```

### Docker
```bash
# Build and run Docker container
docker build -t stock-analyzer -f docker/Dockerfile .
docker run -it --env-file .env stock-analyzer
```

## Code Style Guidelines

### Imports (enforced by ruff)
- **Order**: stdlib → third-party → local (with blank line separation)
- **Style**: Absolute imports preferred over relative
- **Example**:
  ```python
  import json
  import logging
  from dataclasses import dataclass
  from typing import Any

  import pandas as pd
  import requests
  from tenacity import retry, stop_after_attempt

  from stock_analyzer.config import get_config
  from stock_analyzer.data_provider.base import BaseFetcher
  ```

### Formatting & Linting (ruff)
- **Line length**: 120 characters (configured in pyproject.toml)
- **Quotes**: Double quotes for strings
- **Trailing commas**: Required for multi-line structures
- Use `ruff format` to auto-format and `ruff check` to lint

### Type Checking (ty)
- Run `ty check .` to validate type hints
- Use type hints for function parameters and return types
- `from typing import Any, Optional, Union`
- Use `dict`, `list` instead of `Dict`, `List` (Python 3.9+)

### Naming Conventions
- Functions/variables: `snake_case`
- Classes: `PascalCase`
- Constants: `UPPER_CASE`
- Private methods: `_leading_underscore`

### Error Handling
- Use domain exceptions from `stock_analyzer.domain.exceptions`
- Use `@handle_errors` decorator for graceful degradation
- Use `safe_execute()` for safe function calls with defaults
- **Exception Hierarchy**:
  - `StockAnalyzerException` (base)
    - `DataFetchError` (data fetching failures)
      - `RateLimitError`, `DataSourceUnavailableError`
    - `StorageError`, `ValidationError`, `AnalysisError`
    - `NotificationError`, `ConfigurationError`
- **Example**:
  ```python
  from stock_analyzer.domain.exceptions import handle_errors, DataFetchError

  @handle_errors("获取股票数据失败", default_return=None)
  def fetch_stock_data(code: str) -> dict | None:
      # 可能抛出异常的代码
      return api.get_data(code)
  ```

### Logging
- Use appropriate levels: `logger.debug()`, `logger.info()`, `logger.warning()`, `logger.error()`
- Include context in log messages (stock code, operation)
- Logging messages should be Chinese for user-facing logs, English for developer/debug logs
- **Example**:
  ```python
  logger.info(f"开始分析股票: {stock_code}")
  logger.warning(f"数据源 {provider} 不可用，切换到备选")
  logger.error(f"分析失败: {e}")
  ```

### Documentation
- **Docstrings**: Use triple quotes for modules, classes, and functions (English)
- **Comments**: Use Chinese comments for inline explanations (project is bilingual)
- **Keep it practical**: Focus on "why" not "what"
- **Example**:
  ```python
  def calculate_trend(self, data: pd.DataFrame) -> TrendResult:
      """
      Calculate market trend based on technical indicators.

      Args:
          data: Historical price data with OHLCV columns

      Returns:
          TrendResult with trend status and signals
      """
      # 计算移动平均线
      ma5 = data['close'].rolling(5).mean()

      # 判断多头排列
      if ma5.iloc[-1] > ma10.iloc[-1] > ma20.iloc[-1]:
          return TrendResult(status="bullish")
      return TrendResult(status="bearish")
  ```

## AI/ML Patterns

- **API clients**: Support multiple providers (Gemini, OpenAI, DeepSeek)
- **Rate limiting**: Implement delays between API calls
- **Prompts**: Stored as constants in `ai/prompts.py`, version controlled
- **Retry logic**: Use exponential backoff for API failures

## Testing Guidelines

### Test Markers
- `@pytest.mark.live`: Tests that call real data provider APIs (may consume quota)
- `@pytest.mark.slow`: Tests that are slow to run
- Use `pytest --run-live` to include live tests
- Use `pytest -m "not slow"` to skip slow tests

### Test Structure
- Unit tests: `tests/unit/`
- Integration tests: `tests/integration/`
- E2E tests: `tests/e2e/`
- Fixtures: `tests/fixtures/`

### Writing Tests
- Use descriptive test names: `test_rsi_calculation_with_valid_data`
- Mock external APIs in unit tests
- Use fixtures from `tests/conftest.py` for common setup
