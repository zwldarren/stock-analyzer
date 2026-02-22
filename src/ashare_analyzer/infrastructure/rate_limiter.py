"""异步限流器 - 令牌桶算法"""

import asyncio
import time


class AsyncRateLimiter:
    """
    Token bucket rate limiter for async context.

    Usage:
        limiter = AsyncRateLimiter(rate=2.0)  # 2 requests per second
        await limiter.acquire()
        response = await session.get(url)
    """

    def __init__(self, rate: float, burst: int = 1):
        """
        Args:
            rate: Tokens per second (QPS)
            burst: Max burst capacity (bucket size)
        """
        self._rate = rate
        self._burst = burst
        self._tokens = float(burst)
        self._last_update = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        """Acquire a token, waiting if necessary."""
        async with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_update
            self._tokens = min(self._burst, self._tokens + elapsed * self._rate)
            self._last_update = now

            if self._tokens < 1.0:
                wait_time = (1.0 - self._tokens) / self._rate
                await asyncio.sleep(wait_time)
                self._tokens = 0.0
                self._last_update = time.monotonic()
            else:
                self._tokens -= 1.0
