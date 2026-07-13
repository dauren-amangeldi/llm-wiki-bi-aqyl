"""Hidden ops dashboard — internal cost/usage/error metrics.

Gated by ``OPS_DASHBOARD_TOKEN`` (a shared secret, not tied to Keycloak or
any user session). Every route here 404s instead of 401/403 when the token
is missing or wrong, so the endpoint's existence isn't revealed to probing.

ponytail: reads the whole usage.log per request (JSONL, append-only). Fine
at current scale (single small file, admin-only, infrequent polling) — swap
for a DB-backed usage table if this file grows large enough to matter.
"""

import json
from collections import defaultdict
from datetime import UTC, datetime, timedelta

from fastapi import Depends, HTTPException, Request
from pydantic import BaseModel
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from llm_wiki.api.deps import get_db
from llm_wiki.api.v1 import router
from llm_wiki.config import settings
from llm_wiki.storage.metadata import FileRecord


def require_ops_token(request: Request) -> None:
    """Raise 404 unless ``X-Ops-Token`` matches the configured secret."""
    token = settings.ops_dashboard_token
    if not token or request.headers.get("X-Ops-Token") != token:
        raise HTTPException(status_code=404, detail="Not found")


class AgentTypeBreakdown(BaseModel):
    agent_type: str
    calls: int
    cost_usd: float
    avg_duration_ms: float
    input_tokens: int
    output_tokens: int


class ModelBreakdown(BaseModel):
    model: str
    calls: int
    cost_usd: float


class DailyCost(BaseModel):
    date: str
    cost_usd: float
    calls: int


class RecentCall(BaseModel):
    timestamp: str
    agent_type: str
    model: str
    cost_usd: float
    duration_ms: int
    file_id: str


class OpsSummary(BaseModel):
    cost_today_usd: float
    cost_this_month_usd: float
    total_files: int
    files_by_status: dict[str, int]
    avg_cost_per_ingestion_usd: float
    by_agent_type: list[AgentTypeBreakdown]
    by_model: list[ModelBreakdown]
    cost_by_day: list[DailyCost]
    recent_calls: list[RecentCall]


def _read_usage_log() -> list[dict]:
    """Parse every well-formed line of usage.log; skip corrupt ones."""
    path = settings.usage_log_path
    if not path.exists():
        return []
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records


@router.get(
    "/ops/summary",
    response_model=OpsSummary,
    dependencies=[Depends(require_ops_token)],
    include_in_schema=False,
)
async def ops_summary(db: AsyncSession = Depends(get_db)) -> OpsSummary:
    """Aggregate cost/usage/error metrics for the hidden ops dashboard."""
    now = datetime.now(UTC)
    today = now.date()
    records = _read_usage_log()

    cost_today = 0.0
    cost_this_month = 0.0
    by_agent: dict[str, dict] = defaultdict(
        lambda: {"calls": 0, "cost_usd": 0.0, "duration_ms": 0, "input_tokens": 0, "output_tokens": 0}
    )
    by_model: dict[str, dict] = defaultdict(lambda: {"calls": 0, "cost_usd": 0.0})
    by_day: dict[str, dict] = defaultdict(lambda: {"cost_usd": 0.0, "calls": 0})
    cutoff = now - timedelta(days=14)

    for r in records:
        try:
            ts = datetime.fromisoformat(r["timestamp"])
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=UTC)
            cost = float(r.get("cost_usd", 0.0))
            agent_type = r.get("agent_type", "unknown")
            model = r.get("model", "unknown")
        except (KeyError, ValueError, TypeError):
            continue

        if ts.date() == today:
            cost_today += cost
        if (ts.year, ts.month) == (now.year, now.month):
            cost_this_month += cost

        a = by_agent[agent_type]
        a["calls"] += 1
        a["cost_usd"] += cost
        a["duration_ms"] += int(r.get("duration_ms", 0))
        a["input_tokens"] += int(r.get("input_tokens", 0))
        a["output_tokens"] += int(r.get("output_tokens", 0))

        m = by_model[model]
        m["calls"] += 1
        m["cost_usd"] += cost

        if ts >= cutoff:
            day_key = ts.date().isoformat()
            d = by_day[day_key]
            d["cost_usd"] += cost
            d["calls"] += 1

    recent = sorted(records, key=lambda r: r.get("timestamp", ""), reverse=True)[:20]

    total_files_result = await db.execute(select(func.count(FileRecord.file_id)))
    total_files = total_files_result.scalar_one() or 0

    status_counts_result = await db.execute(
        select(FileRecord.status, func.count(FileRecord.file_id)).group_by(FileRecord.status)
    )
    files_by_status = dict(status_counts_result.all())

    avg_cost_result = await db.execute(
        select(func.avg(FileRecord.cost_usd)).where(FileRecord.status == "DONE")
    )
    avg_cost_raw = avg_cost_result.scalar_one()
    avg_cost = round(float(avg_cost_raw), 4) if avg_cost_raw is not None else 0.0

    return OpsSummary(
        cost_today_usd=round(cost_today, 4),
        cost_this_month_usd=round(cost_this_month, 4),
        total_files=total_files,
        files_by_status=files_by_status,
        avg_cost_per_ingestion_usd=avg_cost,
        by_agent_type=[
            AgentTypeBreakdown(
                agent_type=k,
                calls=v["calls"],
                cost_usd=round(v["cost_usd"], 4),
                avg_duration_ms=round(v["duration_ms"] / v["calls"], 1) if v["calls"] else 0.0,
                input_tokens=v["input_tokens"],
                output_tokens=v["output_tokens"],
            )
            for k, v in sorted(by_agent.items(), key=lambda kv: -kv[1]["cost_usd"])
        ],
        by_model=[
            ModelBreakdown(model=k, calls=v["calls"], cost_usd=round(v["cost_usd"], 4))
            for k, v in sorted(by_model.items(), key=lambda kv: -kv[1]["cost_usd"])
        ],
        cost_by_day=[
            DailyCost(date=k, cost_usd=round(v["cost_usd"], 4), calls=v["calls"])
            for k, v in sorted(by_day.items())
        ],
        recent_calls=[
            RecentCall(
                timestamp=r.get("timestamp", ""),
                agent_type=r.get("agent_type", "unknown"),
                model=r.get("model", "unknown"),
                cost_usd=round(float(r.get("cost_usd", 0.0)), 6),
                duration_ms=int(r.get("duration_ms", 0)),
                file_id=r.get("file_id", ""),
            )
            for r in recent
        ],
    )
