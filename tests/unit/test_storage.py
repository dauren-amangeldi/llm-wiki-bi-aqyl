"""Unit tests for the storage layer.

Covers: atomic writes, file locking, IndexStorage (read/add/move/backlinks),
append_log_entry (idempotency, format), and metadata CRUD.
"""

import threading
from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncSession

from llm_wiki.storage.filesystem import atomic_write, ensure_dirs
from llm_wiki.storage.index import Heading, IndexStorage
from llm_wiki.storage.log import append_log_entry, _entry_exists
from llm_wiki.storage.metadata import (
    append_state_history,
    create_file_record,
    ensure_dev_user,
    get_file_record,
    get_or_create_user,
    update_file_status,
)


# ===========================================================================
# filesystem.py
# ===========================================================================


def test_atomic_write_creates_file(tmp_path: Path) -> None:
    """atomic_write should create a file with the given content."""
    target = tmp_path / "output.md"
    atomic_write(target, "# Hello\n")
    assert target.exists()
    assert target.read_text() == "# Hello\n"


def test_atomic_write_overwrites_existing(tmp_path: Path) -> None:
    """atomic_write should replace an existing file atomically."""
    target = tmp_path / "page.md"
    target.write_text("old content")
    atomic_write(target, "new content")
    assert target.read_text() == "new content"


def test_atomic_write_no_temp_file_left(tmp_path: Path) -> None:
    """No .tmp files should remain after a successful atomic_write."""
    target = tmp_path / "page.md"
    atomic_write(target, "content")
    leftover = list(tmp_path.glob("*.tmp"))
    assert leftover == [], f"Unexpected temp files: {leftover}"


def test_ensure_dirs_creates_nested(tmp_path: Path) -> None:
    """ensure_dirs should create nested directories if they do not exist."""
    deep = tmp_path / "a" / "b" / "c"
    ensure_dirs(deep)
    assert deep.is_dir()


def test_ensure_dirs_idempotent(tmp_path: Path) -> None:
    """ensure_dirs should not raise when directories already exist."""
    d = tmp_path / "existing"
    d.mkdir()
    ensure_dirs(d)  # second call must not raise


# ===========================================================================
# index.py — IndexStorage
# ===========================================================================


def test_index_read_headings_empty(db_engine) -> None:  # type: ignore[no-untyped-def]  # noqa: ANN001
    """read_headings on an empty index returns an empty list."""
    assert IndexStorage().read_headings() == []


def test_index_add_and_read_pages(db_engine) -> None:  # type: ignore[no-untyped-def]  # noqa: ANN001
    """add_page persists a (slug, title, level, section) row readable via read_pages."""
    storage = IndexStorage()
    storage.add_page("transformers", "AI Models", title="Transformers")
    assert storage.read_pages() == [("transformers", "Transformers", 2, "AI Models")]


def test_index_read_headings_has_section_and_entry(db_engine) -> None:  # type: ignore[no-untyped-def]  # noqa: ANN001
    """read_headings emits a section header (no slug) then the page entry (slug)."""
    storage = IndexStorage()
    storage.add_page("gpt", "AI Models", title="GPT")
    headings = storage.read_headings()
    assert Heading(level=2, text="AI Models", slug=None) in headings
    assert any(h.slug == "gpt" for h in headings)


def test_index_add_page_idempotent(db_engine) -> None:  # type: ignore[no-untyped-def]  # noqa: ANN001
    """add_page called twice with the same slug does not create duplicates."""
    storage = IndexStorage()
    storage.add_page("transformers", "AI Models")
    storage.add_page("transformers", "AI Models")
    assert len(storage.read_pages()) == 1


def test_index_add_page_updates_on_conflict(db_engine) -> None:  # type: ignore[no-untyped-def]  # noqa: ANN001
    """Re-adding a slug updates its title/section."""
    storage = IndexStorage()
    storage.add_page("p", "Old", title="P")
    storage.add_page("p", "New", title="P2")
    assert storage.read_pages() == [("p", "P2", 2, "New")]


def test_index_move_page(db_engine) -> None:  # type: ignore[no-untyped-def]  # noqa: ANN001
    """move_page changes an entry's section."""
    storage = IndexStorage()
    storage.add_page("target", "Old Section")
    storage.move_page("target", "New Section")
    assert storage.read_pages()[0][3] == "New Section"


def test_index_remove_page(db_engine) -> None:  # type: ignore[no-untyped-def]  # noqa: ANN001
    """remove_page deletes the entry."""
    storage = IndexStorage()
    storage.add_page("gone", "X")
    storage.remove_page("gone")
    assert storage.read_pages() == []


def test_index_get_backlinks_returns_empty(db_engine) -> None:  # type: ignore[no-untyped-def]  # noqa: ANN001
    """Index-level co-occurrence is no longer tracked; get_backlinks is a stub."""
    assert IndexStorage().get_backlinks("anything") == []


# ===========================================================================
# log.py — append_log_entry
# ===========================================================================


def test_log_entry_creates_content(tmp_path: Path) -> None:
    """append_log_entry writes the expected Markdown block."""
    log = tmp_path / "log.md"
    log.write_text("# Ingestion Log\n", encoding="utf-8")
    append_log_entry(log, "file-abc", "report.pdf", ["transformers"], ["llm"], 0.0123)
    content = log.read_text()
    assert "file-abc" in content
    assert "report.pdf" in content
    assert "transformers" in content
    assert "llm" in content
    assert "$0.0123" in content


def test_log_entry_idempotent(tmp_path: Path) -> None:
    """append_log_entry called twice with the same file_id writes only one entry."""
    log = tmp_path / "log.md"
    log.write_text("# Ingestion Log\n", encoding="utf-8")
    append_log_entry(log, "file-xyz", "doc.pdf", [], [], 0.0)
    append_log_entry(log, "file-xyz", "doc.pdf", [], [], 0.0)
    content = log.read_text()
    assert content.count("file-xyz") == 1


def test_log_entry_no_existing_file(tmp_path: Path) -> None:
    """append_log_entry creates log.md if it doesn't exist yet."""
    log = tmp_path / "log.md"
    assert not log.exists()
    append_log_entry(log, "new-file", "new.pdf", [], [], 0.0)
    assert log.exists()
    assert "new-file" in log.read_text()


def test_entry_exists_false_for_missing_file(tmp_path: Path) -> None:
    """_entry_exists returns False when the log file doesn't exist."""
    assert not _entry_exists(tmp_path / "nonexistent.md", "any-id")


# ===========================================================================
# metadata.py — async CRUD
# ===========================================================================


async def test_create_file_record(db_session: AsyncSession) -> None:
    """create_file_record inserts a row and returns it."""
    record = await create_file_record(db_session, "id-001", "report.pdf")
    assert record.file_id == "id-001"
    assert record.original_name == "report.pdf"
    assert record.status == "RECEIVED"
    assert record.state_history == []


async def test_get_file_record_found(db_session: AsyncSession) -> None:
    """get_file_record returns the row for a known file_id."""
    await create_file_record(db_session, "id-002", "file.pdf")
    fetched = await get_file_record(db_session, "id-002")
    assert fetched is not None
    assert fetched.file_id == "id-002"


async def test_get_file_record_not_found(db_session: AsyncSession) -> None:
    """get_file_record returns None for an unknown file_id."""
    result = await get_file_record(db_session, "nonexistent")
    assert result is None


async def test_update_file_status(db_session: AsyncSession) -> None:
    """update_file_status changes the status field."""
    await create_file_record(db_session, "id-003", "x.pdf")
    await update_file_status(db_session, "id-003", "DONE")
    record = await get_file_record(db_session, "id-003")
    assert record is not None
    assert record.status == "DONE"


async def test_append_state_history(db_session: AsyncSession) -> None:
    """append_state_history adds one entry per call to the JSON list."""
    await create_file_record(db_session, "id-004", "y.pdf")
    await append_state_history(db_session, "id-004", "STORED")
    await append_state_history(db_session, "id-004", "SEARCHED")
    record = await get_file_record(db_session, "id-004")
    assert record is not None
    states = [e["state"] for e in record.state_history]
    assert states == ["STORED", "SEARCHED"]


async def test_append_state_history_noop_for_unknown(db_session: AsyncSession) -> None:
    """append_state_history silently ignores an unknown file_id."""
    await append_state_history(db_session, "ghost-id", "STORED")  # must not raise


# ===========================================================================
# metadata.py — users (LW-N1)
# ===========================================================================


async def test_get_or_create_user_is_idempotent(db_session: AsyncSession) -> None:
    """get_or_create_user returns the same row on repeated calls."""
    first = await get_or_create_user(db_session, "user-a", "Alice", "admin")
    second = await get_or_create_user(db_session, "user-a", "Alice", "admin")
    assert first.id == second.id == "user-a"
    assert first.role == "admin"


async def test_ensure_dev_user(db_session: AsyncSession) -> None:
    """ensure_dev_user creates the stable dev-user row."""
    user = await ensure_dev_user(db_session)
    assert user.id == "dev-user"
    assert user.role == "admin"
