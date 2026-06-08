"""Unit tests for quality/budget.py (LW-19)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from llm_wiki.quality.budget import (
    BudgetExceeded,
    BudgetGuard,
    BudgetSnapshot,
    compute_budget_snapshot,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_NOW = datetime(2026, 6, 9, 12, 0, 0, tzinfo=timezone.utc)


def _write_log(path: Path, entries: list[dict]) -> None:  # type: ignore[type-arg]
    with path.open("w", encoding="utf-8") as fh:
        for e in entries:
            fh.write(json.dumps(e) + "\n")


def _entry(
    cost: float,
    input_tokens: int = 100,
    output_tokens: int = 50,
    date: str = "2026-06-09",
) -> dict:  # type: ignore[type-arg]
    return {
        "file_id": "test",
        "agent_type": "writer",
        "model": "gpt-5.4-mini",
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cached_input_tokens": 0,
        "cost_usd": cost,
        "timestamp": f"{date}T10:00:00+00:00",
        "duration_ms": 500,
    }


# ---------------------------------------------------------------------------
# compute_budget_snapshot
# ---------------------------------------------------------------------------


def test_snapshot_aggregates_today_correctly(tmp_path: Path) -> None:
    log = tmp_path / "usage.log"
    _write_log(log, [
        _entry(0.01, input_tokens=100, output_tokens=50),   # today
        _entry(0.02, input_tokens=200, output_tokens=80),   # today
        _entry(0.05, date="2026-06-08"),                    # yesterday — excluded from today
    ])
    snap = compute_budget_snapshot(log, now=_NOW)
    assert snap.cost_today_usd == pytest.approx(0.03, abs=1e-6)
    assert snap.tokens_today == 430   # (100+50) + (200+80)


def test_snapshot_cost_this_month(tmp_path: Path) -> None:
    log = tmp_path / "usage.log"
    _write_log(log, [
        _entry(0.01, date="2026-06-01"),
        _entry(0.02, date="2026-06-09"),   # today
        _entry(0.10, date="2026-05-31"),   # last month — excluded
    ])
    snap = compute_budget_snapshot(log, now=_NOW)
    assert snap.cost_this_month_usd == pytest.approx(0.03, abs=1e-6)


def test_snapshot_skips_malformed_lines(tmp_path: Path) -> None:
    log = tmp_path / "usage.log"
    with log.open("w") as fh:
        fh.write('{"cost_usd": 0.01, "input_tokens": 10, "output_tokens": 5, "timestamp": "2026-06-09T10:00:00+00:00"}\n')
        fh.write("NOT VALID JSON\n")
        fh.write("\n")  # blank line
        fh.write('{"cost_usd": 0.02, "input_tokens": 20, "output_tokens": 10, "timestamp": "2026-06-09T11:00:00+00:00"}\n')

    snap = compute_budget_snapshot(log, now=_NOW)
    assert snap.cost_today_usd == pytest.approx(0.03, abs=1e-6)
    assert snap.tokens_today == 45   # 15 + 30


def test_snapshot_returns_zeros_when_file_absent(tmp_path: Path) -> None:
    snap = compute_budget_snapshot(tmp_path / "nonexistent.log", now=_NOW)
    assert snap.cost_today_usd == 0.0
    assert snap.tokens_today == 0
    assert snap.cost_this_month_usd == 0.0


def test_snapshot_timestamp_is_now(tmp_path: Path) -> None:
    snap = compute_budget_snapshot(tmp_path / "nonexistent.log", now=_NOW)
    assert snap.timestamp == _NOW


def test_snapshot_performance(tmp_path: Path) -> None:
    """compute_budget_snapshot should handle 10k lines in under 100ms."""
    import time

    log = tmp_path / "usage.log"
    entries = [_entry(0.001) for _ in range(10_000)]
    _write_log(log, entries)

    t0 = time.perf_counter()
    snap = compute_budget_snapshot(log, now=_NOW)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    assert elapsed_ms < 100, f"Took {elapsed_ms:.1f}ms — expected < 100ms"
    assert snap.cost_today_usd == pytest.approx(10.0, rel=0.01)


# ---------------------------------------------------------------------------
# BudgetGuard
# ---------------------------------------------------------------------------


def _guard(
    tmp_path: Path,
    entries: list[dict],  # type: ignore[type-arg]
    cost_limit: float | None = None,
    token_limit: int | None = None,
) -> BudgetGuard:
    log = tmp_path / "usage.log"
    _write_log(log, entries)
    return BudgetGuard(
        usage_log_path=log,
        daily_cost_limit_usd=cost_limit,
        daily_token_limit=token_limit,
    )


def test_guard_passes_when_limits_none(tmp_path: Path) -> None:
    guard = _guard(tmp_path, [_entry(999.0)])  # massive spend but no limits
    snap = guard.check()
    assert isinstance(snap, BudgetSnapshot)


def test_guard_passes_when_under_cost_limit(tmp_path: Path) -> None:
    guard = _guard(tmp_path, [_entry(0.01)], cost_limit=1.00)
    snap = guard.check()
    assert snap.cost_today_usd == pytest.approx(0.01, abs=1e-6)


def test_guard_raises_when_cost_exceeded(tmp_path: Path) -> None:
    guard = _guard(tmp_path, [_entry(2.00)], cost_limit=1.00)
    with pytest.raises(BudgetExceeded, match="daily cost limit"):
        guard.check()


def test_guard_raises_when_cost_exactly_at_limit(tmp_path: Path) -> None:
    guard = _guard(tmp_path, [_entry(1.00)], cost_limit=1.00)
    with pytest.raises(BudgetExceeded):
        guard.check()


def test_guard_passes_when_under_token_limit(tmp_path: Path) -> None:
    # 150 tokens per entry, cost_limit=None
    guard = _guard(tmp_path, [_entry(0.001, input_tokens=100, output_tokens=50)], token_limit=500)
    snap = guard.check()
    assert snap.tokens_today == 150


def test_guard_raises_when_tokens_exceeded(tmp_path: Path) -> None:
    entries = [_entry(0.001, input_tokens=300, output_tokens=200)] * 5  # 2500 tokens
    guard = _guard(tmp_path, entries, token_limit=1000)
    with pytest.raises(BudgetExceeded, match="daily token limit"):
        guard.check()


def test_guard_cost_limit_none_token_limit_set(tmp_path: Path) -> None:
    """Token limit fires even when cost limit is None."""
    guard = _guard(
        tmp_path,
        [_entry(0.001, input_tokens=800, output_tokens=300)],
        cost_limit=None,
        token_limit=1000,
    )
    with pytest.raises(BudgetExceeded, match="token"):
        guard.check()


def test_guard_empty_log_always_passes(tmp_path: Path) -> None:
    log = tmp_path / "usage.log"
    log.write_text("")
    guard = BudgetGuard(log, daily_cost_limit_usd=0.001, daily_token_limit=1)
    # Empty log = zero spend = should NOT raise
    snap = guard.check()
    assert snap.cost_today_usd == 0.0
