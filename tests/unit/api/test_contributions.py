from datetime import datetime, timezone

from llm_wiki.api.v1.contributions import get_contributions, month_start, public_name
from llm_wiki.storage.metadata import CaseRecord, FileRecord, User


def add_cases(db, owner, count, *, sensitive=False, created_at=None):
    for i in range(count):
        fid = f"{owner}-{sensitive}-{i}"
        db.add(FileRecord(file_id=fid, original_name="ready.md", status="DONE"))
        db.add(CaseRecord(id=fid, title="Case", owner=owner, doc_ids=[fid],
                          sensitive=sensitive, created_at=created_at or datetime.now(timezone.utc)))


async def test_empty_and_private_only_are_not_contributions(db_session):
    add_cases(db_session, "me@bi.group", 4, sensitive=True)
    add_cases(db_session, "private-colleague@bi.group", 8, sensitive=True)
    add_cases(db_session, "anon", 3)
    add_cases(db_session, None, 2)
    await db_session.commit()
    result = await get_contributions("all", db_session, "me@bi.group")
    assert result == {"myRank": None, "myCases": 0, "rankDelta": None, "top": [],
                      "hasCasesEver": False, "timezone": "Asia/Almaty"}


async def test_shared_cases_ties_and_names_without_email(db_session):
    add_cases(db_session, "alice@bi.group", 2)
    add_cases(db_session, "bob@bi.group", 2)
    add_cases(db_session, "me@bi.group", 1)
    add_cases(db_session, "me@bi.group", 20, sensitive=True)
    db_session.add_all([User(id="alice@bi.group", name="Alice Example"),
                        User(id="bob@bi.group", name="bob@bi.group")])
    await db_session.commit()
    result = await get_contributions("all", db_session, "me@bi.group")
    assert (result["myRank"], result["myCases"], result["hasCasesEver"]) == (3, 1, True)
    assert [r["rank"] for r in result["top"]] == [1, 1, 3]
    assert result["top"][0]["name"] == "Alice E."
    assert result["top"][1]["name"] == ""
    assert result["top"][2]["isMe"] is True
    assert "@" not in str(result)


async def test_current_user_outside_top_ten_is_still_ranked(db_session):
    for i in range(12):
        add_cases(db_session, f"user-{i:02d}", 12-i)
    await db_session.commit()
    result = await get_contributions("all", db_session, "user-11")
    assert len(result["top"]) == 10
    assert result["myRank"] == 12 and result["myCases"] == 1
    assert not any(r["isMe"] for r in result["top"])


async def test_month_uses_created_at_and_preserves_all_time_participation(db_session, monkeypatch):
    from llm_wiki.api.v1 import contributions

    class FixedTime(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime(2026, 9, 14, tzinfo=timezone.utc)

    monkeypatch.setattr(contributions, "datetime", FixedTime)
    add_cases(db_session, "me", 1, created_at=datetime(2026, 8, 31, 18, 59, tzinfo=timezone.utc))
    add_cases(db_session, "other", 1, created_at=datetime(2026, 8, 31, 19, 0, tzinfo=timezone.utc))
    await db_session.commit()
    result = await get_contributions("month", db_session, "me")
    assert result["myRank"] is None and result["hasCasesEver"] is True
    assert len(result["top"]) == 1


def test_calendar_month_boundary_and_safe_display_name():
    start = month_start(datetime(2026, 8, 31, 20, tzinfo=timezone.utc))
    assert start.astimezone(timezone.utc) == datetime(2026, 8, 31, 19, tzinfo=timezone.utc)
    assert public_name("Alice Example") == ("Alice E.", "AE")
    assert public_name("somebody@bi.group") == ("", "")
    assert public_name(None) == ("", "")
