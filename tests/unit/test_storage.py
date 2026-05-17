"""Unit tests for the storage layer.

Covers: atomic writes, file locking, IndexStorage (read/add/move/backlinks),
append_log_entry (idempotency, format), and metadata CRUD.
"""

import threading
from pathlib import Path

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from llm_wiki.storage.filesystem import atomic_write, ensure_dirs
from llm_wiki.storage.index import Heading, IndexStorage
from llm_wiki.storage.log import append_log_entry, _entry_exists
from llm_wiki.storage.metadata import (
    Base,
    FileRecord,
    append_state_history,
    create_file_record,
    get_file_record,
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


def _make_index(tmp_path: Path, content: str = "# Wiki Index\n") -> IndexStorage:
    idx = tmp_path / "index.md"
    idx.write_text(content, encoding="utf-8")
    return IndexStorage(idx)


def test_index_read_headings_empty_file(tmp_path: Path) -> None:
    """read_headings on an empty file returns an empty list."""
    storage = IndexStorage(tmp_path / "index.md")
    assert storage.read_headings() == []


def test_index_read_headings_parses_levels(tmp_path: Path) -> None:
    """read_headings extracts heading levels and text correctly."""
    content = "# Top\n## Second\n### Third\n"
    storage = _make_index(tmp_path, content)
    headings = storage.read_headings()
    assert len(headings) == 3
    assert headings[0] == Heading(level=1, text="Top")
    assert headings[1] == Heading(level=2, text="Second")
    assert headings[2] == Heading(level=3, text="Third")


def test_index_read_headings_extracts_slug(tmp_path: Path) -> None:
    """read_headings should populate the slug field when [[slug]] is present."""
    content = "## [[my-page]]\n"
    storage = _make_index(tmp_path, content)
    headings = storage.read_headings()
    assert headings[0].slug == "my-page"


def test_index_add_page_creates_section(tmp_path: Path) -> None:
    """add_page creates a new section when none exists."""
    storage = _make_index(tmp_path)
    storage.add_page("transformers", "AI Models")
    content = (tmp_path / "index.md").read_text()
    assert "## AI Models" in content
    assert "[[transformers]]" in content


def test_index_add_page_under_existing_section(tmp_path: Path) -> None:
    """add_page appends under an existing section."""
    content = "# Wiki Index\n\n## AI Models\n- [[bert]]\n"
    storage = _make_index(tmp_path, content)
    storage.add_page("gpt", "AI Models")
    result = (tmp_path / "index.md").read_text()
    # Both slugs must be present and gpt must appear after bert
    assert "[[bert]]" in result
    assert "[[gpt]]" in result
    assert result.index("[[bert]]") < result.index("[[gpt]]")


def test_index_add_page_idempotent(tmp_path: Path) -> None:
    """add_page called twice with the same slug does not create duplicates."""
    storage = _make_index(tmp_path)
    storage.add_page("transformers", "AI Models")
    storage.add_page("transformers", "AI Models")
    content = (tmp_path / "index.md").read_text()
    assert content.count("[[transformers]]") == 1


def test_index_add_page_concurrent(tmp_path: Path) -> None:
    """Concurrent add_page calls must not corrupt the index."""
    index_path = tmp_path / "index.md"
    index_path.write_text("# Wiki Index\n", encoding="utf-8")
    storage = IndexStorage(index_path)
    errors: list[Exception] = []

    def add(slug: str) -> None:
        try:
            storage.add_page(slug, "Concurrent")
        except Exception as exc:
            errors.append(exc)

    slugs = [f"page-{i}" for i in range(8)]
    threads = [threading.Thread(target=add, args=(s,)) for s in slugs]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == [], f"Exceptions during concurrent writes: {errors}"
    content = index_path.read_text(encoding="utf-8")
    for slug in slugs:
        assert f"[[{slug}]]" in content, f"Missing [[{slug}]] after concurrent writes"


def test_index_move_page(tmp_path: Path) -> None:
    """move_page relocates an entry from one section to another."""
    content = "# Wiki\n\n## Old Section\n- [[target]]\n\n## New Section\n- [[other]]\n"
    storage = _make_index(tmp_path, content)
    storage.move_page("target", "New Section")
    result = (tmp_path / "index.md").read_text()
    assert "[[target]]" in result
    # Entry must now appear inside New Section, not Old Section
    new_start = result.index("## New Section")
    assert result.index("[[target]]") > new_start
    # Should appear only once
    assert result.count("[[target]]") == 1


def test_index_get_backlinks_empty(tmp_path: Path) -> None:
    """get_backlinks returns empty list when there are no cross-references."""
    storage = _make_index(tmp_path, "# Wiki\n\n## X\n- [[page-a]]\n")
    assert storage.get_backlinks("page-b") == []


def test_index_get_backlinks_finds_inline_refs(tmp_path: Path) -> None:
    """get_backlinks returns slugs that share a line with the target."""
    content = "# Wiki\n\n## X\n- [[page-a]] and [[page-b]]\n- [[page-c]]\n"
    storage = _make_index(tmp_path, content)
    backlinks = storage.get_backlinks("page-b")
    assert "page-a" in backlinks


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


@pytest.fixture
async def db_session(tmp_path: Path) -> AsyncSession:  # type: ignore[misc]
    """Yield an async SQLAlchemy session backed by an in-memory SQLite DB."""
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path}/test.db")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory: async_sessionmaker[AsyncSession] = async_sessionmaker(
        bind=engine, expire_on_commit=False
    )
    async with factory() as session:
        yield session
    await engine.dispose()


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
