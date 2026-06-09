"""In-memory sliding-window rate limiter.

Thread-safe for single-process deployments (FastAPI + uvicorn workers = 1).
For multi-replica production, replace with a Redis-backed implementation.
"""

from collections import defaultdict, deque
from time import monotonic


class RateLimiter:
    """Sliding-window token-bucket limiter keyed by arbitrary string (email, IP, …)."""

    def __init__(self, limit: int, window_sec: float) -> None:
        self.limit = limit
        self.window = window_sec
        self._hits: dict[str, deque[float]] = defaultdict(deque)

    def check(self, key: str) -> tuple[bool, int]:
        """Check whether *key* is within the rate limit.

        Removes stale timestamps before counting.

        Returns:
            ``(allowed, retry_after_sec)`` — if *allowed* is True, ``retry_after``
            is 0; otherwise it is the number of seconds the caller must wait.
        """
        now = monotonic()
        hits = self._hits[key]
        while hits and now - hits[0] > self.window:
            hits.popleft()
        if len(hits) >= self.limit:
            retry = int(self.window - (now - hits[0])) + 1
            return False, retry
        hits.append(now)
        return True, 0


# ---------------------------------------------------------------------------
# Module-level singletons
# ---------------------------------------------------------------------------

# Advisor: 10 requests per 60 seconds per email
advisor_limiter = RateLimiter(limit=10, window_sec=60)
