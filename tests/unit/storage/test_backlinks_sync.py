"""Unit tests for llm_wiki.storage.backlinks_sync (LW-13).

Covers: new-page sync, update diff, self-reference, missing target,
idempotency, concurrent writes, and no-write-when-unchanged.

Wiki pages live in the object store (LocalObjectStore at a per-test temp dir,
provided by the autouse conftest fixture).
"""

from __future__ import annotations

import threading
from unittest.mock import patch

import pytest

from llm_wiki.storage import wiki_store
from llm_wiki.storage.backlinks_sync import sync_backlinks_for_page
from llm_wiki.utils.backlinks import extract_backlinks


@pytest.fixture(autouse=True)
def _wiki_db(db_engine):  # type: ignore[no-untyped-def]  # noqa: ANN001
    """backlinks_sync reads/writes wiki pages in Postgres — clean DB per test."""
    yield


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_wiki(pages: dict[str, str]) -> None:
    """Populate the wiki store (Postgres) with the given {slug: content} pages."""
    for slug, content in pages.items():
        wiki_store.save_page(slug, wiki_store.extract_page_title(content, slug), content)


def _read(slug: str) -> str:
    return wiki_store.get_page(slug) or ""


# ===========================================================================
# Basic injection
# ===========================================================================


def test_new_page_adds_backlink_to_target() -> None:
    """Creating page A that links [[B]] → B gets ## Backlinks with [[A]]."""
    _make_wiki({"target-b": "# Target B\n\nSome content.\n"})

    sync_backlinks_for_page(
        source_slug="page-a",
        new_content="# Page A\n\nLinks to [[target-b]].\n",
        previous_outgoing=(),
        file_id="test-f1",
    )

    assert "page-a" in extract_backlinks(_read("target-b"))


def test_new_page_multiple_targets() -> None:
    """All targets of a new page receive a backlink injection."""
    _make_wiki({"bb": "# B\n\n", "cc": "# C\n\n"})

    sync_backlinks_for_page(
        source_slug="aa",
        new_content="# A\n\n[[bb]] and [[cc]].\n",
        previous_outgoing=(),
    )

    assert "aa" in extract_backlinks(_read("bb"))
    assert "aa" in extract_backlinks(_read("cc"))


# ===========================================================================
# Update diff
# ===========================================================================


def test_update_removes_old_adds_new() -> None:
    """When A's link changes from [[B]] to [[C]], B loses backlink, C gains it."""
    _make_wiki({
        "bb": "# B\n\n## Backlinks\n\n- [[page-a]]\n",
        "cc": "# C\n\nContent.\n",
    })

    sync_backlinks_for_page(
        source_slug="page-a",
        new_content="# A\n\nNow links to [[cc]] instead.\n",
        previous_outgoing=["bb"],
    )

    assert "page-a" not in extract_backlinks(_read("bb"))
    assert "page-a" in extract_backlinks(_read("cc"))


def test_update_partial_change() -> None:
    """Link added to C while link to B is retained — only C is updated."""
    _make_wiki({
        "bb": "# B\n\n## Backlinks\n\n- [[source]]\n",
        "cc": "# C\n\nContent.\n",
    })

    sync_backlinks_for_page(
        source_slug="source",
        new_content="# S\n\n[[bb]] and [[cc]].\n",
        previous_outgoing=["bb"],
    )

    assert "source" in extract_backlinks(_read("bb"))
    assert "source" in extract_backlinks(_read("cc"))


# ===========================================================================
# Self-reference
# ===========================================================================


def test_self_reference_ignored() -> None:
    """A page that links to itself does not get a ## Backlinks entry for itself."""
    _make_wiki({"self-page": "# Self\n\nContent here.\n"})

    sync_backlinks_for_page(
        source_slug="self-page",
        new_content="# Self\n\nLinks to [[self-page]] itself.\n",
        previous_outgoing=(),
    )

    content = _read("self-page")
    assert "## Backlinks" not in content or "self-page" not in extract_backlinks(content)


# ===========================================================================
# Missing target
# ===========================================================================


def test_missing_target_no_exception() -> None:
    """Linking to a non-existent page logs a warning but does not raise."""
    result = sync_backlinks_for_page(
        source_slug="source",
        new_content="# Source\n\n[[nonexistent]].\n",
        previous_outgoing=(),
    )

    assert "nonexistent" in result["added"]  # detected as added
    assert not wiki_store.page_exists("nonexistent")  # no page created


# ===========================================================================
# Idempotency
# ===========================================================================


def test_idempotent_double_call() -> None:
    """Calling sync_backlinks_for_page twice with the same args → same stored state."""
    _make_wiki({"target": "# Target\n\nContent.\n"})
    args = dict(
        source_slug="source",
        new_content="# Source\n\n[[target]].\n",
        previous_outgoing=(),
    )

    sync_backlinks_for_page(**args)
    after_first = _read("target")

    sync_backlinks_for_page(**args)
    after_second = _read("target")

    assert after_first == after_second


# ===========================================================================
# No-write when unchanged
# ===========================================================================


def test_no_write_when_unchanged() -> None:
    """The store is NOT written when the content would not change."""
    _make_wiki({"target": "# Target\n\n## Backlinks\n\n- [[source]]\n"})

    with patch("llm_wiki.storage.wiki_store.save_page") as mock_write:
        sync_backlinks_for_page(
            source_slug="source",
            new_content="# Source\n\n[[target]].\n",
            previous_outgoing=["target"],  # unchanged
        )
        mock_write.assert_not_called()


def test_no_write_when_no_diff() -> None:
    """No writes if previous_outgoing == new outgoing (nothing changed)."""
    _make_wiki({"page-b": "# B\n\n## Backlinks\n\n- [[page-a]]\n"})

    with patch("llm_wiki.storage.wiki_store.save_page") as mock_write:
        sync_backlinks_for_page(
            source_slug="page-a",
            new_content="# A\n\n[[page-b]].\n",
            previous_outgoing=["page-b"],  # identical
        )
        mock_write.assert_not_called()


# ===========================================================================
# Return value
# ===========================================================================


def test_return_value_added_removed() -> None:
    """sync_backlinks_for_page returns correct added/removed lists."""
    _make_wiki({
        "old-target": "# Old\n\n## Backlinks\n\n- [[page-a]]\n",
        "new-target": "# New\n\nContent.\n",
    })

    result = sync_backlinks_for_page(
        source_slug="page-a",
        new_content="# A\n\n[[new-target]].\n",
        previous_outgoing=["old-target"],
    )

    assert result["added"] == ["new-target"]
    assert result["removed"] == ["old-target"]


def test_return_value_empty_when_no_diff() -> None:
    """Return value has empty lists when nothing changed."""
    _make_wiki({"bb": "# B\n\n## Backlinks\n\n- [[aa]]\n"})

    result = sync_backlinks_for_page(
        source_slug="aa",
        new_content="# A\n\n[[bb]].\n",
        previous_outgoing=["bb"],
    )
    assert result == {"added": [], "removed": []}


# ===========================================================================
# Concurrency
# ===========================================================================


def test_concurrent_writes_to_same_target() -> None:
    """8 threads all inject a backlink into the same target — all present, no dups."""
    _make_wiki({"popular": "# Popular\n\nContent.\n"})
    n_threads = 8
    errors: list[Exception] = []

    def worker(i: int) -> None:
        try:
            sync_backlinks_for_page(
                source_slug=f"source-{i:02d}",
                new_content=f"# Source {i}\n\n[[popular]].\n",
                previous_outgoing=(),
            )
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"Thread errors: {errors}"

    backlinks = extract_backlinks(_read("popular"))
    assert len(backlinks) == n_threads
    assert len(set(backlinks)) == n_threads
    assert sorted(backlinks) == sorted(f"source-{i:02d}" for i in range(n_threads))
