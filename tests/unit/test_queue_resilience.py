"""Устойчивость очередей после инцидента 2026-08-20 (прод).

Три защиты, каждая закрывает свой круг «карусели»:
  - janitor ``sweep_stuck_generations`` — зависшие pending-артефакты и файлы
    переводятся в failed, UI не показывает «генерируется» вечно;
  - дедуп ``_start_generation`` — повторный клик по «сгенерировать» не ставит
    вторую тяжёлую LLM-задачу, пока жива первая;
  - poison-pill cap — файл, чья обработка раз за разом убивает воркер
    (OOM/hard limit + reject_on_worker_lost = бесконечные redelivery мимо
    max_retries), после INGEST_MAX_DELIVERIES доставок помечается FAILED.
"""

from __future__ import annotations

import asyncio
import os
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, patch

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from llm_wiki.config import settings
from llm_wiki.storage.metadata import ArtifactRecord, FileRecord

TEST_DATABASE_URL = os.environ.get(
    "TEST_DATABASE_URL",
    "postgresql+psycopg://llmwiki:devpassword@postgres:5432/llmwiki",
)


@pytest.fixture(autouse=True)
def _point_settings_at_test_db(monkeypatch: pytest.MonkeyPatch) -> None:
    """Задачи строят собственные engine из settings.database_url —
    в тестах он обязан смотреть в тестовую БД."""
    monkeypatch.setattr(settings, "database_url", TEST_DATABASE_URL)


# ---------------------------------------------------------------------------
# Конфигурация очередей
# ---------------------------------------------------------------------------


def test_queue_priority_puts_user_facing_first() -> None:
    """Прод-воркер без -Q слушает все очереди; при queue_order_strategy=
    priority порядок task_queues = строгий приоритет (BRPOP берёт первую
    непустую). Артефакты и light обязаны стоять раньше ingest — иначе
    генерации снова встанут за завалом PDF."""
    from llm_wiki.orchestrator.tasks import celery_app

    opts = celery_app.conf.broker_transport_options
    assert opts.get("queue_order_strategy") == "priority"
    names = [q.name for q in celery_app.conf.task_queues]
    assert names.index("artifacts") < names.index("ingest")
    assert names.index("light") < names.index("ingest")


# ---------------------------------------------------------------------------
# Janitor
# ---------------------------------------------------------------------------


async def test_sweep_fails_stale_generations_and_keeps_fresh(
    db_session: AsyncSession,
) -> None:
    from llm_wiki.orchestrator.tasks import sweep_stuck_generations

    now = datetime.now(timezone.utc)
    db_session.add(
        ArtifactRecord(
            artifact_id="a-stale-started", document_id="d1", kind="report",
            versions=[], status="pending",
            started_at=now - timedelta(minutes=20),
        )
    )
    db_session.add(
        ArtifactRecord(
            artifact_id="a-fresh-started", document_id="d2", kind="report",
            versions=[], status="pending",
            started_at=now - timedelta(minutes=2),
        )
    )
    db_session.add(
        ArtifactRecord(
            artifact_id="a-stale-queued", document_id="d3", kind="test",
            versions=[], status="pending",
            created_at=now - timedelta(hours=1),
            updated_at=now - timedelta(hours=1),
        )
    )
    db_session.add(
        FileRecord(
            file_id="f-stale", original_name="old.pdf", status="PROCESSING",
            created_at=now - timedelta(hours=4),
        )
    )
    db_session.add(
        FileRecord(
            file_id="f-fresh", original_name="new.pdf", status="PROCESSING",
            created_at=now - timedelta(minutes=10),
        )
    )
    await db_session.commit()

    # apply() крутит собственный asyncio.Runner — уводим в поток, чтобы не
    # столкнуться с уже работающим циклом pytest-asyncio.
    result = await asyncio.to_thread(sweep_stuck_generations.apply)
    swept = result.get()
    assert swept == {"artifacts_swept": 2, "files_swept": 1}

    db_session.expire_all()
    stale_started = await db_session.get(ArtifactRecord, "a-stale-started")
    fresh_started = await db_session.get(ArtifactRecord, "a-fresh-started")
    stale_queued = await db_session.get(ArtifactRecord, "a-stale-queued")
    f_stale = await db_session.get(FileRecord, "f-stale")
    f_fresh = await db_session.get(FileRecord, "f-fresh")

    assert stale_started is not None and stale_started.status == "failed"
    assert "прервана" in (stale_started.error or "")
    assert fresh_started is not None and fresh_started.status == "pending"
    assert stale_queued is not None and stale_queued.status == "failed"
    assert f_stale is not None and f_stale.status == "FAILED"
    assert f_fresh is not None and f_fresh.status == "PROCESSING"


# ---------------------------------------------------------------------------
# Дедуп повторных кликов «сгенерировать»
# ---------------------------------------------------------------------------


async def test_start_generation_dedupes_live_pending(
    db_session: AsyncSession,
) -> None:
    from llm_wiki.api.v1 import artifacts as artifacts_api
    from llm_wiki.orchestrator import tasks as tasks_mod

    calls: list[tuple] = []

    with patch.object(
        tasks_mod.generate_artifact,
        "apply_async",
        side_effect=lambda *a, **kw: calls.append((a, kw)),
    ):
        first = await artifacts_api._start_generation(
            db_session, kind="report", document_id="doc-dd", language="ru"
        )
        assert first["status"] == "pending"
        assert len(calls) == 1

        # Второй клик, пока генерация жива → тот же id, БЕЗ новой задачи.
        second = await artifacts_api._start_generation(
            db_session, kind="report", document_id="doc-dd", language="ru"
        )
        assert second["artifact_id"] == first["artifact_id"]
        assert len(calls) == 1

        # Состарим строку за окно дедупа — регенерация снова разрешена.
        record = await db_session.get(ArtifactRecord, first["artifact_id"])
        assert record is not None
        old = datetime.now(timezone.utc) - timedelta(
            seconds=artifacts_api._PENDING_DEDUP_WINDOW_S + 60
        )
        record.created_at = old
        record.updated_at = old
        await db_session.commit()

        third = await artifacts_api._start_generation(
            db_session, kind="report", document_id="doc-dd", language="ru"
        )
        assert third["artifact_id"] == first["artifact_id"]
        assert len(calls) == 2


# ---------------------------------------------------------------------------
# Poison-pill cap для ingest
# ---------------------------------------------------------------------------


async def test_ingest_delivery_cap_marks_file_failed(
    db_session: AsyncSession,
) -> None:
    from llm_wiki.orchestrator.tasks import INGEST_MAX_DELIVERIES, process_file_task

    db_session.add(
        FileRecord(
            file_id="f-poison", original_name="boom.pdf", status="PROCESSING",
            ingest_attempts=INGEST_MAX_DELIVERIES,
        )
    )
    await db_session.commit()

    pipeline_mock = AsyncMock()
    with patch("llm_wiki.orchestrator.pipeline.process_file", new=pipeline_mock):
        await asyncio.to_thread(process_file_task.apply, args=["f-poison"])

    pipeline_mock.assert_not_called()
    db_session.expire_all()
    record = await db_session.get(FileRecord, "f-poison")
    assert record is not None and record.status == "FAILED"
    assert "снят с очереди" in (record.error or "")


async def test_ingest_counter_resets_on_success(
    db_session: AsyncSession,
) -> None:
    from llm_wiki.orchestrator.tasks import process_file_task

    db_session.add(
        FileRecord(
            file_id="f-lucky", original_name="ok.pdf", status="PROCESSING",
            ingest_attempts=2,  # пара прошлых падений — ещё в пределах бюджета
        )
    )
    await db_session.commit()

    with patch(
        "llm_wiki.orchestrator.pipeline.process_file", new=AsyncMock(return_value=None)
    ):
        await asyncio.to_thread(process_file_task.apply, args=["f-lucky"])

    db_session.expire_all()
    record = await db_session.get(FileRecord, "f-lucky")
    assert record is not None
    assert record.ingest_attempts == 0
