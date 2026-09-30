"""One usage ledger for text, embeddings, OCR, transcription and images.

Only numeric usage and identifiers are retained. Missing usage/pricing is null,
never a fabricated zero. ``cost_usd`` is an estimate from configured tariffs.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import structlog
from filelock import FileLock

from llm_wiki.observability import context, error_fields, new_id

logger = structlog.get_logger(__name__)


def number(value: Any) -> int | float | None:
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _get(value: Any, key: str) -> Any:
    return value.get(key) if isinstance(value, dict) else getattr(value, key, None)


def response_usage(response: Any) -> dict[str, Any]:
    usage = _get(response, "usage")
    input_tokens = number(_get(usage, "prompt_tokens"))
    if input_tokens is None:
        input_tokens = number(_get(usage, "input_tokens"))
    output_tokens = number(_get(usage, "completion_tokens"))
    if output_tokens is None:
        output_tokens = number(_get(usage, "output_tokens"))
    # Embeddings expose prompt_tokens + total_tokens but no output_tokens.
    if (
        input_tokens is not None
        and output_tokens is None
        and number(_get(usage, "total_tokens")) == input_tokens
    ):
        output_tokens = 0
    details = _get(usage, "prompt_tokens_details") or _get(usage, "input_tokens_details")
    cached = number(_get(details, "cached_tokens")) or 0
    output_details = _get(usage, "completion_tokens_details") or _get(
        usage, "output_tokens_details"
    )
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cached_input_tokens": cached,
        # Already included in output_tokens; never add these again to totals.
        "reasoning_tokens": number(_get(output_details, "reasoning_tokens")),
        "image_input_tokens": number(_get(details, "image_tokens")),
        "text_input_tokens": number(_get(details, "text_tokens")),
        "audio_seconds": number(_get(response, "duration")) or number(_get(usage, "seconds")),
    }


def estimate_cost(
    model: str, usage: dict[str, Any], *, media: bool = False
) -> tuple[float | None, str]:
    from llm_wiki.config import settings

    prices = settings.price_table.get(model)
    if not prices:
        return None, "unknown_model"
    seconds = usage.get("audio_seconds")
    if seconds is not None and "per_minute" in prices:
        return round(seconds / 60 * prices["per_minute"], 8), "estimated"
    incoming, outgoing = usage.get("input_tokens"), usage.get("output_tokens")
    if incoming is None or outgoing is None:
        return None, "missing_usage"
    if "input" not in prices or "output" not in prices:
        return None, "missing_rates"
    cached = min(incoming, usage.get("cached_input_tokens") or 0)
    if media and usage.get("image_input_tokens") is not None:
        image_tokens, text_tokens = usage["image_input_tokens"], usage.get("text_input_tokens")
        # Image input and text input can have different tariffs. Require both.
        if text_tokens is None or "image_input" not in prices or cached:
            return None, "missing_media_rates"
        input_cost = text_tokens * prices["input"] + image_tokens * prices["image_input"]
    else:
        input_cost = (incoming - cached) * prices["input"] + cached * prices.get(
            "cached_input", prices["input"]
        )
    return round((input_cost + outgoing * prices["output"]) / 1_000_000, 8), (
        "estimated_uncached_upper_bound" if cached and "cached_input" not in prices else "estimated"
    )


def write_usage(record: dict[str, Any], path: Path | None = None) -> dict[str, Any]:
    from llm_wiki.config import settings

    record = {**context(), **record}
    record.setdefault("timestamp", datetime.now(UTC).isoformat())
    record.setdefault("call_id", new_id())
    record.setdefault("operation_id", record.get("request_id") or record["call_id"])
    if "cost_status" not in record:
        record["cost_usd"], record["cost_status"] = estimate_cost(
            record["model"], record, media=record.get("agent_type") == "image"
        )
    # Exactly this event is the spend ledger in Elastic. Do not sum other
    # timing/error events, which describe the same call.
    logger.info("llm_usage", **record)
    path = path or settings.usage_log_path
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with FileLock(str(path) + ".lock", timeout=5), path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
    except Exception as exc:
        # Retrying a paid provider call because the local ledger is unavailable
        # would charge twice. The stderr usage event remains available.
        logger.error("usage_persistence_failed", call_id=record["call_id"], **error_fields(exc))
    return record


@contextmanager
def media_call(
    model: str, agent_type: str, file_id: str, *, usage_log_path: Path | None = None, **fields: Any
) -> Iterator[dict[str, Any]]:
    """Capture actual SDK usage immediately, before downstream parsing can fail."""
    from llm_wiki.config import settings
    from llm_wiki.quality.budget import BudgetExceeded, BudgetGuard

    call_id = new_id()
    started = time.perf_counter()
    result: dict[str, Any] = {}
    logger.info(
        "ai_call_started",
        call_id=call_id,
        model=model,
        agent_type=agent_type,
        file_id=file_id,
        **fields,
    )
    try:
        BudgetGuard(
            usage_log_path or settings.usage_log_path,
            settings.daily_cost_limit_usd,
            settings.daily_token_limit,
        ).check()
        yield result
    except BaseException as exc:
        logger.warning(
            "ai_call_finished",
            call_id=call_id,
            model=model,
            agent_type=agent_type,
            file_id=file_id,
            outcome=(
                "blocked"
                if isinstance(exc, BudgetExceeded)
                else "cancelled"
                if isinstance(exc, asyncio.CancelledError)
                else "failed"
            ),
            duration_ms=round((time.perf_counter() - started) * 1000, 2),
            cost_status="not_called" if isinstance(exc, BudgetExceeded) else "unknown_on_failure",
            **fields,
            **error_fields(exc),
        )
        raise
    else:
        record = write_usage(
            {
                "call_id": call_id,
                "model": model,
                "agent_type": agent_type,
                "file_id": file_id,
                "provider": "openai",
                "outcome": "success",
                "duration_ms": round((time.perf_counter() - started) * 1000, 2),
                **response_usage(result.get("response")),
                **fields,
            },
            usage_log_path,
        )
        logger.info(
            "ai_call_finished",
            call_id=call_id,
            model=model,
            agent_type=agent_type,
            file_id=file_id,
            outcome="success",
            duration_ms=record["duration_ms"],
            **fields,
        )
