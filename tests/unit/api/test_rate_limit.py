"""Unit tests for InMemoryRateLimiter (LW-19)."""

from __future__ import annotations

import threading
import time
from unittest.mock import patch

import pytest

from llm_wiki.api.rate_limit import InMemoryRateLimiter


# ---------------------------------------------------------------------------
# Basic sliding-window behaviour
# ---------------------------------------------------------------------------


def test_first_n_requests_allowed() -> None:
    limiter = InMemoryRateLimiter(max_requests=3, window_seconds=60)
    for _ in range(3):
        allowed, retry = limiter.check("client-a")
        assert allowed
        assert retry == 0


def test_nth_plus_one_request_denied() -> None:
    limiter = InMemoryRateLimiter(max_requests=3, window_seconds=60)
    for _ in range(3):
        limiter.check("client-a")
    allowed, retry = limiter.check("client-a")
    assert not allowed
    assert retry >= 1


def test_retry_after_is_positive_on_rejection() -> None:
    limiter = InMemoryRateLimiter(max_requests=1, window_seconds=60)
    limiter.check("x")
    allowed, retry_after = limiter.check("x")
    assert not allowed
    assert retry_after > 0


def test_window_expiry_allows_new_requests() -> None:
    """After the window passes, the counter resets."""
    limiter = InMemoryRateLimiter(max_requests=2, window_seconds=60)

    fake_time = [0.0]

    def _mock_time() -> float:
        return fake_time[0]

    with patch("llm_wiki.api.rate_limit.time.monotonic", side_effect=_mock_time):
        limiter.check("ip")
        limiter.check("ip")
        assert limiter.check("ip") == (False, 60)  # over limit

        fake_time[0] = 61.0  # advance past the window
        allowed, retry = limiter.check("ip")
        assert allowed, "Should be allowed after window expires"


def test_different_keys_are_isolated() -> None:
    limiter = InMemoryRateLimiter(max_requests=1, window_seconds=60)
    limiter.check("alpha")
    # alpha is now at limit — beta should still be allowed
    allowed_alpha, _ = limiter.check("alpha")
    allowed_beta, _ = limiter.check("beta")
    assert not allowed_alpha
    assert allowed_beta


def test_reset_single_key() -> None:
    limiter = InMemoryRateLimiter(max_requests=1, window_seconds=60)
    limiter.check("x")
    assert limiter.check("x")[0] is False  # at limit

    limiter.reset("x")
    assert limiter.check("x")[0] is True   # reset → allowed again


def test_reset_all_keys() -> None:
    limiter = InMemoryRateLimiter(max_requests=1, window_seconds=60)
    limiter.check("a")
    limiter.check("b")
    limiter.reset()
    assert limiter.check("a")[0] is True
    assert limiter.check("b")[0] is True


def test_reset_nonexistent_key_is_noop() -> None:
    limiter = InMemoryRateLimiter(max_requests=3, window_seconds=60)
    limiter.reset("ghost")  # should not raise


def test_rejected_request_does_not_consume_slot() -> None:
    """A rejected request must not count toward the window."""
    limiter = InMemoryRateLimiter(max_requests=2, window_seconds=60)
    limiter.check("k")
    limiter.check("k")
    # 2 of 2 consumed — now reject several
    for _ in range(5):
        limiter.check("k")
    limiter.reset("k")
    # Fresh state: should be able to send exactly 2 again
    assert limiter.check("k")[0] is True
    assert limiter.check("k")[0] is True
    assert limiter.check("k")[0] is False


# ---------------------------------------------------------------------------
# Thread-safety
# ---------------------------------------------------------------------------


def test_thread_safety_exact_limit() -> None:
    """100 threads each issuing 10 requests — exactly max_requests should succeed."""
    max_req = 50
    limiter = InMemoryRateLimiter(max_requests=max_req, window_seconds=60)
    results: list[bool] = []
    lock = threading.Lock()

    def _worker() -> None:
        for _ in range(10):
            allowed, _ = limiter.check("shared-key")
            with lock:
                results.append(allowed)

    threads = [threading.Thread(target=_worker) for _ in range(100)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    allowed_count = sum(results)
    assert allowed_count == max_req, (
        f"Expected exactly {max_req} allowed, got {allowed_count}"
    )
