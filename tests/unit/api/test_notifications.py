"""Б1: лента уведомлений — упсёрт in-place, видимость, unread, live-строки.

Контракт: терминальные события хранятся (одна строка на сущность — ретрай
переписывает «Ошибка» в «Готово», не наслаивая), «в работе» derive-ится из
статусов files/artifacts на момент GET. Отметки чтения — на сервере, per-user.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from datetime import datetime, timedelta, timezone

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from llm_wiki.api.deps import get_db
from llm_wiki.main import app
from llm_wiki.storage import notifications as notif
from llm_wiki.storage.metadata import (
    ArtifactRecord,
    CaseRecord,
    FileRecord,
    NotificationRead,
    NotificationRecord,
    ensure_column_migrations,
    update_file_status,
)

USER = "demo@bi.group"
OTHER = "someone@bi.group"


@pytest.mark.asyncio
@pytest.mark.parametrize("sync", [False, True])
async def test_delivery_is_idempotent_and_older_outcome_cannot_replace_retry(
    db_session: AsyncSession, sync: bool,
) -> None:
    at = datetime(2026, 9, 7, 5, 20, tzinfo=timezone.utc)
    args = dict(section="materials", family="generation", event="done",
                entity_id="clock", title="Material", occurred_at=at, occurrence_key="attempt-1")

    async def deliver(**changes) -> None:
        if sync:
            await asyncio.to_thread(notif._upsert_event_sync, **(args | changes))
        else:
            await notif.upsert_event(db_session, **(args | changes))

    await deliver()
    row = (await notif.list_events(db_session, USER))[0][0]
    original_id, inserted_at, delivered_at = row.id, row.created_at, row.updated_at
    await notif.mark_read(db_session, USER, mark_all=True)
    await deliver()
    row, read = (await notif.list_events(db_session, USER))[0]
    assert (row.id, row.created_at, row.updated_at, row.occurred_at, read) == (
        original_id, inserted_at, delivered_at, at, True,
    )

    # A genuinely new processing attempt with the SAME outcome must be visible.
    await deliver(occurred_at=at + timedelta(hours=1), occurrence_key="attempt-2")
    row, read = (await notif.list_events(db_session, USER))[0]
    assert row.id == original_id and row.created_at == inserted_at
    assert row.occurred_at == at + timedelta(hours=1) and not read
    await notif.mark_read(db_session, USER, mark_all=True)
    # A delayed failure from the previous attempt must not move time backwards.
    await deliver(event="failed", occurred_at=at, occurrence_key="old-failure")
    row, read = (await notif.list_events(db_session, USER))[0]
    assert row.event == "done" and row.occurred_at == at + timedelta(hours=1) and read


@pytest.mark.asyncio
async def test_concurrent_deliveries_create_one_notification(db_engine) -> None:
    factory = async_sessionmaker(db_engine, expire_on_commit=False)

    async def deliver() -> None:
        async with factory() as session:
            await notif.upsert_event(session, section="materials", family="generation",
                                     event="done", entity_id="concurrent", title="Material",
                                     occurred_at=datetime(2026, 9, 7, tzinfo=timezone.utc),
                                     occurrence_key="same-attempt")

    await asyncio.gather(*(deliver() for _ in range(5)))
    async with factory() as session:
        assert len((await session.scalars(select(NotificationRecord))).all()) == 1


@pytest.mark.asyncio
async def test_stale_emitter_cannot_borrow_a_newer_sources_timestamp(db_session) -> None:
    at = datetime(2026, 9, 7, tzinfo=timezone.utc)
    db_session.add(FileRecord(file_id="recovered", original_name="a.pdf", status="DONE", finished_at=at))
    db_session.add(ArtifactRecord(artifact_id="recovered-art", document_id="recovered", kind="report",
                                  status="ready", finished_at=at))
    await db_session.commit()
    await notif.notify_file_done(db_session, "recovered")
    await notif.notify_artifact_event(db_session, artifact_id="recovered-art", document_id="recovered",
                                      kind="report", event="done")
    await notif.mark_read(db_session, USER, mark_all=True)
    await notif.notify_file_failed(db_session, "recovered", "old failure")
    await notif.notify_artifact_event(db_session, artifact_id="recovered-art", document_id="recovered",
                                      kind="report", event="failed", detail="old failure")
    await asyncio.to_thread(notif.notify_file_failed_sync, "recovered", "old failure")
    await asyncio.to_thread(notif.notify_artifact_failed_sync, "recovered-art", "old failure")
    rows = await notif.list_events(db_session, USER)
    assert len(rows) == 2
    assert all(row.event == "done" and row.occurred_at == at and read for row, read in rows)


@pytest.mark.asyncio
async def test_finished_at_survives_rename_and_duplicate_status_but_resets_on_retry(db_session) -> None:
    fr = FileRecord(file_id="finished", original_name="a.pdf", status="WRITTEN")
    db_session.add(fr)
    await db_session.commit()
    await update_file_status(db_session, fr.file_id, "DONE")
    await db_session.refresh(fr)
    finished = fr.finished_at
    assert finished is not None
    fr.display_name = "Renamed"
    await db_session.commit()
    await update_file_status(db_session, fr.file_id, "DONE")
    await db_session.refresh(fr)
    assert fr.finished_at == finished
    await notif.notify_file_done(db_session, fr.file_id)
    row, _ = (await notif.list_events(db_session, USER))[0]
    assert row.occurred_at == finished
    await notif.mark_read(db_session, USER, mark_all=True)
    await notif.notify_file_done(db_session, fr.file_id)
    assert (await notif.unread_counts(db_session, USER))["materials"] == 0
    await update_file_status(db_session, fr.file_id, "RECEIVED")
    await db_session.refresh(fr)
    assert fr.finished_at is None
    await update_file_status(db_session, fr.file_id, "FAILED")
    await db_session.refresh(fr)
    assert fr.finished_at > finished


@pytest.mark.asyncio
async def test_api_uses_source_event_time_and_sorts_by_it(client, db_session) -> None:
    at = datetime(2026, 9, 7, 5, 20, tzinfo=timezone.utc)
    db_session.add(ArtifactRecord(artifact_id="source-time", document_id="doc", kind="report",
                                  status="ready", finished_at=at))
    await db_session.commit()
    await notif.notify_artifact_event(db_session, artifact_id="source-time", document_id="doc",
                                      kind="report", event="done", requested_by=USER)
    await notif.upsert_event(db_session, section="materials", family="generation", event="done",
                             entity_id="late-delivery", title="Late", occurred_at=at - timedelta(hours=1))
    data = (await client.get("/api/v1/notifications")).json()
    assert [r["entity_id"] for r in data["items"]] == ["source-time", "late-delivery"]
    assert data["items"][0]["occurred_at"] == data["items"][0]["created_at"] == at.isoformat()


@pytest.mark.asyncio
async def test_case_rename_preserves_event_but_reattachment_is_a_new_occurrence(client, db_session, monkeypatch) -> None:
    from llm_wiki.api.v1 import cases

    monkeypatch.setattr(cases, "_dispatch_autotag", lambda _: None)
    at = datetime(2026, 9, 1, tzinfo=timezone.utc)
    db_session.add(FileRecord(file_id="case-file", original_name="a.pdf", status="DONE", finished_at=at))
    db_session.add(CaseRecord(id="case-clock", title="Case", doc_ids=["case-file"], owner=USER,
                              created_at=at, materials_updated_at=at))
    await db_session.commit()
    await notif.notify_case_ready_if_done(db_session, "case-clock")
    await notif.mark_read(db_session, USER, mark_all=True)
    response = await client.put("/api/v1/cases/case-clock", json={"title": "Renamed"})
    assert response.status_code == 200
    await notif.notify_case_ready_if_done(db_session, "case-clock")
    row, read = (await notif.list_events(db_session, USER))[0]
    assert row.occurred_at == at and read
    for docs in ([], ["case-file"]):
        response = await client.put("/api/v1/cases/case-clock", json={"title": "Renamed", "doc_ids": docs})
        assert response.status_code == 200
    row, read = (await notif.list_events(db_session, USER))[0]
    case = await db_session.get(CaseRecord, "case-clock", populate_existing=True)
    assert row.occurred_at == case.materials_updated_at > at and not read
    assert len(await notif.list_events(db_session, USER)) == 1


@pytest.mark.asyncio
async def test_legacy_timestamp_migration_is_repeatable(db_engine) -> None:
    async with db_engine.begin() as conn:
        await conn.execute(FileRecord.__table__.insert().values(
            file_id="legacy", original_name="old.pdf", status="DONE",
            created_at=datetime(2026, 9, 1, tzinfo=timezone.utc),
            updated_at=datetime(2026, 9, 2, tzinfo=timezone.utc),
        ))
        for table, column in (("files", "finished_at"), ("cases", "materials_updated_at"),
                              ("notifications", "occurred_at"), ("notifications", "occurrence_key")):
            await conn.execute(text(f"ALTER TABLE {table} DROP COLUMN {column}"))
        await ensure_column_migrations(conn)
        first = (await conn.execute(text("SELECT finished_at FROM files WHERE file_id='legacy'"))).scalar_one()
        await conn.execute(text("UPDATE files SET updated_at='2026-09-03T00:00:00Z' WHERE file_id='legacy'"))
        await ensure_column_migrations(conn)
        second = (await conn.execute(text("SELECT finished_at FROM files WHERE file_id='legacy'"))).scalar_one()
        assert first == second == datetime(2026, 9, 2, tzinfo=timezone.utc)


@pytest_asyncio.fixture
async def client(db_engine) -> AsyncGenerator[AsyncClient, None]:
    factory: async_sessionmaker[AsyncSession] = async_sessionmaker(
        bind=db_engine, expire_on_commit=False, autoflush=False
    )

    async def _override_get_db() -> AsyncGenerator[AsyncSession, None]:
        async with factory() as session:
            yield session

    app.dependency_overrides[get_db] = _override_get_db
    async with AsyncClient(
        transport=ASGITransport(app=app),  # type: ignore[arg-type]
        base_url="http://test",
        headers={"X-User-Email": USER},
    ) as c:
        yield c
    app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# Storage: upsert in-place
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_upsert_in_place_and_read_reset(db_session: AsyncSession) -> None:
    """Повторное событие той же сущности обновляет строку и сбрасывает чтение."""
    await notif.upsert_event(
        db_session, section="materials", family="generation", event="failed",
        entity_id="f-1", title="Отчёт.pdf", detail="упс",
    )
    rows = (await db_session.scalars(select(NotificationRecord))).all()
    assert len(rows) == 1
    # Пользователь прочитал «Ошибку»…
    await notif.mark_read(db_session, USER, mark_all=True)
    assert (await notif.unread_counts(db_session, USER))["materials"] == 0

    # …ретрай успешен: та же строка становится «Готово» и снова непрочитана.
    await notif.upsert_event(
        db_session, section="materials", family="generation", event="done",
        entity_id="f-1", title="Отчёт.pdf",
    )
    rows = (await db_session.scalars(select(NotificationRecord))).all()
    assert len(rows) == 1
    assert rows[0].event == "done"
    assert rows[0].detail is None
    assert (await notif.unread_counts(db_session, USER))["materials"] == 1


@pytest.mark.asyncio
async def test_personal_events_hidden_from_others(db_session: AsyncSession) -> None:
    """Личное (recipient) видно только адресату; broadcast — всем."""
    await notif.upsert_event(
        db_session, section="materials", family="generation", event="done",
        entity_id="f-mine", title="Моё", recipient=USER,
    )
    await notif.upsert_event(
        db_session, section="cases", family="privacy", event="published",
        entity_id="case-1", title="Общий кейс", actor=OTHER,
    )
    mine = await notif.list_events(db_session, USER)
    assert {r.entity_id for r, _ in mine} == {"f-mine", "case-1"}
    others = await notif.list_events(db_session, OTHER)
    assert {r.entity_id for r, _ in others} == {"case-1"}
    # Счётчики раздельные per-user.
    await notif.mark_read(db_session, OTHER, mark_all=True)
    assert (await notif.unread_counts(db_session, OTHER))["cases"] == 0
    assert (await notif.unread_counts(db_session, USER))["cases"] == 1


@pytest.mark.asyncio
async def test_notify_file_done_emits_case_done(db_session: AsyncSession) -> None:
    """Последний DONE-материал кейса добавляет «Кейс обработан» в «Кейсы»."""
    db_session.add(FileRecord(file_id="f-a", original_name="a.pdf", status="DONE"))
    db_session.add(FileRecord(file_id="f-b", original_name="b.pdf", status="DONE",
                              display_name="Стратегия Грузии"))
    db_session.add(CaseRecord(id="case-9", title="Выход на рынок",
                              doc_ids=["f-a", "f-b"], owner=USER))
    await db_session.commit()

    await notif.notify_file_done(db_session, "f-b")

    rows = (await db_session.scalars(select(NotificationRecord))).all()
    by_section = {r.section: r for r in rows}
    assert by_section["materials"].title == "Стратегия Грузии"
    assert by_section["materials"].meta["case_id"] == "case-9"
    assert by_section["cases"].entity_id == "case-9"
    assert by_section["cases"].event == "done"
    assert by_section["cases"].meta["materials"] == 2


@pytest.mark.asyncio
async def test_case_done_waits_for_all_materials(db_session: AsyncSession) -> None:
    """Пока хоть один материал кейса в работе — событие кейса не эмитится."""
    db_session.add(FileRecord(file_id="f-c", original_name="c.pdf", status="DONE"))
    db_session.add(FileRecord(file_id="f-d", original_name="d.pdf", status="SEARCHED"))
    db_session.add(CaseRecord(id="case-w", title="В работе",
                              doc_ids=["f-c", "f-d"], owner=USER))
    await db_session.commit()

    await notif.notify_file_done(db_session, "f-c")

    sections = {r.section for r in (await db_session.scalars(select(NotificationRecord))).all()}
    assert sections == {"materials"}


# ---------------------------------------------------------------------------
# API: GET /notifications (items + live + unread), POST /read
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_notifications_live_and_read_flow(
    client: AsyncClient, db_session: AsyncSession
) -> None:
    # Событие: артефакт готов (личное).
    db_session.add(FileRecord(file_id="f-doc", original_name="doc.pdf",
                              status="DONE", display_name="Материал"))
    await db_session.commit()
    await notif.notify_artifact_event(
        db_session, artifact_id="art-1", document_id="f-doc", kind="report",
        event="done", requested_by=USER,
    )
    # Живая генерация: файл на шаге WRITTEN (3/4) внутри кейса.
    db_session.add(FileRecord(file_id="f-live", original_name="live.pdf",
                              status="WRITTEN", owner=USER))
    db_session.add(CaseRecord(id="case-live", title="Живой кейс",
                              doc_ids=["f-live", "f-doc"], owner=USER))
    # Живой артефакт: pending без started_at → «в очереди».
    db_session.add(ArtifactRecord(artifact_id="art-q", document_id="f-doc",
                                  kind="test", status="pending", requested_by=USER))
    await db_session.commit()

    resp = await client.get("/api/v1/notifications")
    assert resp.status_code == 200
    data = resp.json()

    assert [i["entity_id"] for i in data["items"]] == ["art-1"]
    item = data["items"][0]
    assert item["title"] == "Материал"
    assert item["meta"]["kind"] == "report"
    assert item["read"] is False
    assert data["unread"]["artifacts"] == 1

    live = {(row["section"], row["entity_id"]): row for row in data["live"]}
    file_row = live[("materials", "f-live")]
    assert (file_row["step"], file_row["total_steps"], file_row["stage"]) == (3, 4, "write")
    assert file_row["case_id"] == "case-live"
    case_row = live[("cases", "case-live")]
    assert (case_row["done"], case_row["total"]) == (1, 2)
    assert live[("artifacts", "art-q")]["state"] == "queued"

    # «Прочитать все»
    resp = await client.post("/api/v1/notifications/read", json={"all": True})
    assert resp.json()["marked"] == 1
    data = (await client.get("/api/v1/notifications")).json()
    assert data["unread"] == {"cases": 0, "materials": 0, "artifacts": 0}
    assert data["items"][0]["read"] is True


@pytest.mark.asyncio
async def test_cards_generate_emits_artifact_done(
    client: AsyncClient, db_session: AsyncSession
) -> None:
    """Карточки генерятся синхронно (не через Celery) — раньше по ним события
    «артефакт готов» не было вовсе. Теперь _generate_and_store эмитит его сам."""
    from unittest.mock import AsyncMock, patch

    # created_pages непустой — иначе fail-fast guard (источник без содержимого)
    # отклонит запрос 422 до генерации.
    db_session.add(FileRecord(file_id="f-card", original_name="c.pdf", status="DONE",
                              display_name="Материал для карточек",
                              created_pages=["page-card"]))
    await db_session.commit()

    with patch("llm_wiki.api.v1.artifacts.generate_content",
               new=AsyncMock(return_value={"cards": []})), \
         patch("llm_wiki.llm.client.LLMClient") as _llm:
        _llm.return_value.aclose = AsyncMock()
        resp = await client.post(
            "/api/v1/cards/generate",
            json={"document_id": "f-card", "languages": ["ru"]},
        )
    assert resp.status_code == 200

    rows = (await db_session.scalars(
        select(NotificationRecord).where(NotificationRecord.section == "artifacts")
    )).all()
    assert len(rows) == 1
    row = rows[0]
    assert (row.event, row.meta["kind"]) == ("done", "card")
    assert row.meta["document_id"] == "f-card"
    assert row.title == "Материал для карточек"
    assert row.recipient == USER


@pytest.mark.asyncio
async def test_cards_generate_empty_case_422_no_event(
    client: AsyncClient, db_session: AsyncSession
) -> None:
    """Fail-fast: карточки по кейсу БЕЗ материалов — 422 мгновенно, генерация
    не начинается, поэтому и события в ленту НЕТ (юзер уже увидел причину
    тостом; полный контракт — в test_artifact_failfast.py)."""
    db_session.add(CaseRecord(id="case-empty", title="Пустой кейс", doc_ids=[], owner=USER))
    await db_session.commit()

    resp = await client.post(
        "/api/v1/cards/generate",
        json={"document_id": "case-empty", "languages": ["ru"]},
    )
    assert resp.status_code == 422
    assert "нет материалов" in resp.json()["detail"].lower()

    rows = (await db_session.scalars(
        select(NotificationRecord).where(NotificationRecord.section == "artifacts")
    )).all()
    assert rows == []


@pytest.mark.asyncio
async def test_create_case_from_existing_materials_emits_ready(
    client: AsyncClient, db_session: AsyncSession
) -> None:
    """Пункт 2: кейс из УЖЕ готовых (дедуп) материалов сразу даёт «кейс готов»
    (пайплайн по ним не запускается), а по самим материалам — ничего."""
    db_session.add(FileRecord(file_id="f-existing", original_name="e.pdf", status="DONE"))
    await db_session.commit()

    resp = await client.post(
        "/api/v1/cases",
        json={"title": "Кейс из готовых", "doc_ids": ["f-existing"],
              "sensitive": False, "tags": []},
    )
    assert resp.status_code == 201

    rows = (await db_session.scalars(select(NotificationRecord))).all()
    # Ровно одно событие — по кейсу; по существующему материалу ничего.
    assert len(rows) == 1
    row = rows[0]
    assert (row.section, row.family, row.event) == ("cases", "generation", "done")
    assert row.entity_id == resp.json()["id"]


@pytest.mark.asyncio
async def test_update_case_attaching_ready_materials_emits_ready(
    client: AsyncClient, db_session: AsyncSession
) -> None:
    """Флоу модалки: пустой кейс → PUT с готовыми материалами → «кейс готов».
    Материалы в работе не триггерят преждевременный «готов»."""
    db_session.add(FileRecord(file_id="f-ok", original_name="a.pdf", status="DONE"))
    db_session.add(FileRecord(file_id="f-busy", original_name="b.pdf", status="SEARCHED"))
    db_session.add(CaseRecord(id="case-u", title="Кейс", doc_ids=[], owner=USER))
    await db_session.commit()

    # Прикрепляем ещё обрабатывающийся материал — «готов» не должен прийти.
    r1 = await client.put(
        "/api/v1/cases/case-u",
        json={"title": "Кейс", "doc_ids": ["f-busy"], "sensitive": False, "tags": []},
    )
    assert r1.status_code == 200
    assert (await db_session.scalars(select(NotificationRecord))).all() == []

    # Теперь состав — только готовые материалы → «кейс готов».
    r2 = await client.put(
        "/api/v1/cases/case-u",
        json={"title": "Кейс", "doc_ids": ["f-ok"], "sensitive": False, "tags": []},
    )
    assert r2.status_code == 200
    rows = (await db_session.scalars(select(NotificationRecord))).all()
    assert len(rows) == 1
    assert (rows[0].section, rows[0].event) == ("cases", "done")


@pytest.mark.asyncio
async def test_case_privacy_flip_emits_broadcast(
    client: AsyncClient, db_session: AsyncSession
) -> None:
    """PUT /cases/{id} со сменой sensitive пишет социальное событие с актором."""
    db_session.add(CaseRecord(id="case-p", title="Секрет", sensitive=True, owner=USER))
    await db_session.commit()

    resp = await client.put(
        "/api/v1/cases/case-p",
        json={"title": "Секрет", "doc_ids": [], "sensitive": False, "tags": []},
    )
    assert resp.status_code == 200

    rows = (await db_session.scalars(select(NotificationRecord))).all()
    assert len(rows) == 1
    row = rows[0]
    assert (row.section, row.family, row.event) == ("cases", "privacy", "published")
    assert row.actor == USER
    assert row.recipient is None  # broadcast

    # Обратный флип обновляет ту же строку (не спамит ленту).
    await client.put(
        "/api/v1/cases/case-p",
        json={"title": "Секрет", "doc_ids": [], "sensitive": True, "tags": []},
    )
    # API писал в другой сессии — сбрасываем identity map, иначе stale-объект.
    db_session.expire_all()
    rows = (await db_session.scalars(select(NotificationRecord))).all()
    assert len(rows) == 1
    assert rows[0].event == "privated"
    # QA D8: «стал приватным» — только владельцу (кейс другим уже не виден,
    # broadcast вёл бы кликом в «Недоступно»).
    assert rows[0].recipient == USER


@pytest.mark.asyncio
async def test_delete_case_purges_its_notifications(
    client: AsyncClient, db_session: AsyncSession
) -> None:
    """Каскадное удаление кейса не оставляет призрачных уведомлений."""
    db_session.add(FileRecord(file_id="f-x", original_name="x.pdf", status="DONE"))
    db_session.add(CaseRecord(id="case-x", title="Сносимый", doc_ids=["f-x"], owner=USER))
    await db_session.commit()
    await notif.notify_file_done(db_session, "f-x")
    await notif.notify_artifact_event(
        db_session, artifact_id="art-x", document_id="f-x", kind="report",
        event="done", requested_by=USER,
    )
    await notif.mark_read(db_session, USER, mark_all=True)
    assert len((await db_session.scalars(select(NotificationRecord))).all()) == 3

    resp = await client.delete("/api/v1/cases/case-x")
    assert resp.status_code == 200

    assert (await db_session.scalars(select(NotificationRecord))).all() == []
    assert (await db_session.scalars(select(NotificationRead))).all() == []
