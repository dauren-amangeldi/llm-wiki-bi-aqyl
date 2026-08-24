"""Б1: лента уведомлений — упсёрт in-place, видимость, unread, live-строки.

Контракт: терминальные события хранятся (одна строка на сущность — ретрай
переписывает «Ошибка» в «Готово», не наслаивая), «в работе» derive-ится из
статусов files/artifacts на момент GET. Отметки чтения — на сервере, per-user.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy import select
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
)

USER = "demo@bi.group"
OTHER = "someone@bi.group"


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
