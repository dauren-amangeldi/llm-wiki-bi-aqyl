"""Daily cost and token budget enforcement (LW-19).

Design:
- No database, no cache. Budget is computed by scanning ``data/usage.log``
  on every call. At 1 000 files/day the log has ~10 k lines (~500 KB) and
  reads take <10 ms — negligible compared to an LLM API round-trip.
- All timestamps are compared in UTC to match the existing convention in
  ``api/routes.py::get_stats``.
- ``BudgetGuard`` is constructed once per ``LLMClient`` instance and shares
  its settings. A ``None`` limit means "disabled".
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import structlog

logger = structlog.get_logger(__name__)


@dataclass(frozen=True)
class BudgetSnapshot:
    """Aggregated spend for today and this calendar month (UTC)."""

    cost_today_usd: float
    tokens_today: int
    cost_this_month_usd: float
    timestamp: datetime  # UTC moment when the snapshot was taken


def compute_budget_snapshot(
    usage_log_path: Path,
    now: datetime | None = None,
) -> BudgetSnapshot:
    """Aggregate today's and this month's spend from ``usage.log``.

    Reads the log synchronously — callers in ``LLMClient`` are either
    already on a sync path (``embed``) or call this before starting the
    async LLM request (``complete``).

    Tolerates malformed lines: each bad line is logged at WARNING and
    skipped so a corrupt entry does not disable the whole check.

    Args:
        usage_log_path: Path to the JSONL usage log.
        now: UTC "now" override for testing. Defaults to ``datetime.now(UTC)``.

    Returns:
        ``BudgetSnapshot`` with aggregated totals. Returns all-zeros if the
        file does not exist.
    """
    if now is None:
        now = datetime.now(timezone.utc)

    today = now.date()
    this_month = (now.year, now.month)

    cost_today = 0.0
    tokens_today = 0
    cost_this_month = 0.0

    if not usage_log_path.exists():
        return BudgetSnapshot(
            cost_today_usd=0.0,
            tokens_today=0,
            cost_this_month_usd=0.0,
            timestamp=now,
        )

    for raw_line in usage_log_path.read_text(encoding="utf-8").splitlines():
        raw_line = raw_line.strip()
        if not raw_line:
            continue
        try:
            record = json.loads(raw_line)
            ts = datetime.fromisoformat(record["timestamp"])
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            cost = float(record.get("cost_usd", 0.0))
            tokens = int(record.get("input_tokens", 0)) + int(record.get("output_tokens", 0))
            if ts.date() == today:
                cost_today += cost
                tokens_today += tokens
            if (ts.year, ts.month) == this_month:
                cost_this_month += cost
        except (json.JSONDecodeError, KeyError, ValueError, TypeError):
            logger.warning("budget_snapshot_parse_error", line=raw_line[:120])

    return BudgetSnapshot(
        cost_today_usd=round(cost_today, 6),
        tokens_today=tokens_today,
        cost_this_month_usd=round(cost_this_month, 6),
        timestamp=now,
    )


class BudgetExceeded(RuntimeError):
    """Raised by ``BudgetGuard.check()`` when a daily limit is already crossed.

    Raised *before* the LLM call so no tokens are wasted.  The pipeline
    catches this as an ordinary exception and transitions the file to FAILED
    via the existing error path.
    """


class BudgetGuard:
    """Checks daily spend before every LLM call.

    Stateless apart from the configured limits and the path to ``usage.log``.
    Thread-safe: ``check()`` only reads the log and has no mutable state.

    Args:
        usage_log_path: Path to the JSONL usage log.
        daily_cost_limit_usd: Maximum spend in USD per calendar day (UTC).
            ``None`` disables the cost check.
        daily_token_limit: Maximum total tokens per calendar day (UTC).
            ``None`` disables the token check.
    """

    def __init__(
        self,
        usage_log_path: Path,
        daily_cost_limit_usd: float | None,
        daily_token_limit: int | None,
    ) -> None:
        self._path = usage_log_path
        self._cost_limit = daily_cost_limit_usd
        self._token_limit = daily_token_limit

    def check(self) -> BudgetSnapshot:
        """Raise ``BudgetExceeded`` if today's spend already crosses any limit.

        Returns the snapshot when all limits are clear (useful for callers
        that want to emit a warning at the 80% threshold).

        Raises:
            BudgetExceeded: When ``cost_today_usd >= daily_cost_limit_usd``
                or ``tokens_today >= daily_token_limit``.
        """
        from llm_wiki.config import settings

        snapshot = compute_budget_snapshot(self._path)

        logger.debug(
            "budget_check",
            cost_today_usd=snapshot.cost_today_usd,
            cost_limit_usd=self._cost_limit,
            tokens_today=snapshot.tokens_today,
            token_limit=self._token_limit,
        )

        # --- Cost check ---
        if self._cost_limit is not None:
            ratio = snapshot.cost_today_usd / self._cost_limit if self._cost_limit > 0 else 0.0
            if snapshot.cost_today_usd >= self._cost_limit:
                logger.error(
                    "budget_exceeded",
                    kind="cost",
                    cost_today_usd=snapshot.cost_today_usd,
                    cost_limit_usd=self._cost_limit,
                )
                raise BudgetExceeded(
                    f"daily cost limit ${self._cost_limit:.4f} exceeded: "
                    f"spent ${snapshot.cost_today_usd:.4f} today"
                )
            if ratio >= settings.budget_warning_threshold_pct:
                logger.warning(
                    "budget_warning",
                    kind="cost",
                    cost_today_usd=snapshot.cost_today_usd,
                    cost_limit_usd=self._cost_limit,
                    pct_used=round(ratio * 100, 1),
                )

        # --- Token check ---
        if self._token_limit is not None and self._token_limit > 0:
            ratio = snapshot.tokens_today / self._token_limit
            if snapshot.tokens_today >= self._token_limit:
                logger.error(
                    "budget_exceeded",
                    kind="tokens",
                    tokens_today=snapshot.tokens_today,
                    token_limit=self._token_limit,
                )
                raise BudgetExceeded(
                    f"daily token limit {self._token_limit:,} exceeded: "
                    f"used {snapshot.tokens_today:,} tokens today"
                )
            if ratio >= settings.budget_warning_threshold_pct:
                logger.warning(
                    "budget_warning",
                    kind="tokens",
                    tokens_today=snapshot.tokens_today,
                    token_limit=self._token_limit,
                    pct_used=round(ratio * 100, 1),
                )

        return snapshot
