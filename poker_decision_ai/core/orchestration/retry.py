"""Retry helper with exponential backoff + jitter (used by the engine)."""
from __future__ import annotations

import asyncio
import random
from typing import Any, Awaitable, Callable

from ..errors import FrameworkError


async def retry_async(
    fn: Callable[[], Awaitable[Any]],
    *,
    attempts: int = 2,
    base_delay: float = 0.2,
    max_delay: float = 2.0,
    on_retry: Callable[[int, Exception], Any] | None = None,
) -> Any:
    """Call fn() up to `attempts` times; retry only FrameworkError-derived failures.

    Non-framework exceptions (bugs) propagate immediately - retrying a bug is
    pointless and hides problems.
    """
    last: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            return await fn()
        except FrameworkError as e:
            if not getattr(e, "retryable", False) or attempt >= attempts:
                raise
            last = e
            if on_retry:
                on_retry(attempt, e)
            delay = min(max_delay, base_delay * (2 ** (attempt - 1))) * (0.5 + random.random())
            await asyncio.sleep(delay)
    assert last is not None  # pragma: no cover
    raise last
