# Multi-stage Dockerfile for stock-analyzer using uv
# Reference: https://docs.astral.sh/uv/guides/integration/docker/

# Stage 1: Builder
FROM python:3.14-slim AS builder

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set Shanghai timezone
ENV TZ=Asia/Shanghai
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Set working directory
WORKDIR /app

# Enable bytecode compilation for better startup performance
ENV UV_COMPILE_BYTECODE=1

# Use copy link mode to avoid hardlink issues with cache mounts
ENV UV_LINK_MODE=copy

# Disable development dependencies
ENV UV_NO_DEV=1

# Install only dependencies first (for better layer caching)
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-install-project --no-editable

# Copy the project code
COPY . /app

# Sync the project (install the package itself) with non-editable mode
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-editable

# Stage 2: Runtime
FROM python:3.14-slim AS runtime

# Install uv in runtime stage as well (needed for uv run)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set Shanghai timezone
ENV TZ=Asia/Shanghai

# Create non-root user for security (with home directory for uv cache)
RUN groupadd -r appuser && useradd -r -m -g appuser appuser

# Copy the virtual environment from builder
COPY --from=builder --chown=appuser:appuser /app/.venv /app/.venv

# Set environment variables
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1
ENV PROJECT_ROOT=/app

# Switch to non-root user
USER appuser

# Set working directory
WORKDIR /app

ENTRYPOINT ["uv", "run", "stock-analyzer"]
CMD ["--help"]
