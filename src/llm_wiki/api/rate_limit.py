"""In-memory sliding-window rate limiter for API endpoints (LW-19).

Intentionally simple: no Redis, no external dependencies. Suitable for a
single API replica (current deployment model). For horizontal scaling with
multiple API containers, replace ``InMemoryRateLimiter`` with a Redis-backed
implementation (planned as LW-19.1).

Thread-safety: all state is protected by a ``threading.Lock`` because
FastAPI may route requests through different threads when using sync
dependencies alongside async handlers.
"""

from __future__ import annotations

import threading
import time
from collections import deque


_GC_INTERVAL = 100  # prune empty deques every N calls to avoid memory leaks


class InMemoryRateLimiter:
    """Sliding-window rate limiter keyed by an arbitrary string (e.g., IP address).

    Tracks the timestamps of accepted requests per key. On each ``check()``,
    timestamps older than *window_seconds* are discarded before counting.

    Args:
        max_requests: Maximum number of requests allowed per *window_seconds*.
        window_seconds: Length of the sliding window in seconds.
    """

    def __init__(self, max_requests: int, window_seconds: int) -> None:
        self._max = max_requests
        self._window = window_seconds
        self._windows: dict[str, deque[float]] = {}
        self._lock = threading.Lock()
        self._calls_since_gc = 0

    def check(self, key: str) -> tuple[bool, int]:
        """Check if *key* is within the rate limit and record a new request.

        If the request is allowed, the current timestamp is appended to the
        key's window. If not, the window is unchanged (the rejected request
        does not consume a slot).

        Args:
            key: Identifier for the request source (e.g., IP address).

        Returns:
            ``(allowed, retry_after_seconds)`` where *retry_after_seconds*
            is 0 when allowed, or the number of seconds until the oldest
            in-window request expires (upper bound).
        """
        now = time.monotonic()
        cutoff = now - self._window

        with self._lock:
            if key not in self._windows:
                self._windows[key] = deque()
            dq = self._windows[key]

            # Evict timestamps outside the sliding window
            while dq and dq[0] <= cutoff:
                dq.popleft()

            if len(dq) < self._max:
                dq.append(now)
                self._calls_since_gc += 1
                if self._calls_since_gc >= _GC_INTERVAL:
                    self._gc()
                return True, 0

            # Reject: compute how long until the oldest slot expires
            oldest = dq[0]
            retry_after = max(1, int(oldest - cutoff) + 1)
            return False, retry_after

    def reset(self, key: str | None = None) -> None:
        """Clear rate-limit state.  Useful in tests.

        Args:
            key: Specific key to reset. If ``None``, clears all keys.
        """
        with self._lock:
            if key is None:
                self._windows.clear()
            else:
                self._windows.pop(key, None)

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _gc(self) -> None:
        """Remove keys with empty deques to prevent unbounded memory growth.

        Must be called under ``self._lock``.
        """
        now = time.monotonic()
        cutoff = now - self._window
        empty_keys = [
            k for k, dq in self._windows.items()
            if not dq or dq[-1] <= cutoff
        ]
        for k in empty_keys:
            del self._windows[k]
        self._calls_since_gc = 0
