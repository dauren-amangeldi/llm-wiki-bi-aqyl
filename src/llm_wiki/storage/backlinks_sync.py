"""Bidirectional backlink synchronisation across wiki pages (LW-13).

When page A is (re)written, this module compares A's outgoing ``[[links]]``
before and after the write and propagates the diff into the ``## Backlinks``
section of every affected target page B.

Concurrency model
-----------------
A per-slug ``threading.Lock`` (obtained from ``_lock_for``) protects every
individual target-page write.  We deliberately do NOT use a single global
lock — that would serialise every backlink update and kill throughput when
Celery ``--concurrency`` is raised in Sprint 3.

Note: ``IndexStorage`` already holds its own ``FileLock`` for ``index.md``.
Backlink sync never touches ``index.md``, so no new race conditions are
introduced here.

Invariant
---------
Even if the Writer Agent accidentally modifies or drops the ``## Backlinks``
section (despite the prompt instruction to keep it intact), the next
``sync_backlinks_for_page`` call for *any page that links to the affected
page* will re-inject the correct entry.  The section thus converges to the
correct state on the next pipeline run.
"""

from __future__ import annotations

import threading
from collections.abc import Iterable

import structlog

from llm_wiki.storage.object_store import get_object_store, wiki_key
from llm_wiki.utils.backlinks import (
    extract_outgoing_links,
    inject_backlink,
    remove_backlink,
)

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Per-slug lock registry
# ---------------------------------------------------------------------------

_LOCKS: dict[str, threading.Lock] = {}
_LOCKS_GUARD = threading.Lock()


def _lock_for(slug: str) -> threading.Lock:
    """Return (and lazily create) a dedicated ``threading.Lock`` for *slug*.

    The guard ensures at-most-one lock object per slug even under concurrent
    first-use scenarios.

    Args:
        slug: Wiki page slug whose file we are about to write.

    Returns:
        A ``threading.Lock`` dedicated to *slug*.
    """
    with _LOCKS_GUARD:
        lock = _LOCKS.get(slug)
        if lock is None:
            lock = threading.Lock()
            _LOCKS[slug] = lock
        return lock


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def sync_backlinks_for_page(
    source_slug: str,
    new_content: str,
    previous_outgoing: Iterable[str] = (),
    file_id: str = "",
) -> dict[str, list[str]]:
    """Reconcile backlinks for a page that was just (re)written.

    Computes the diff between *previous_outgoing* and the outgoing links
    extracted from *new_content*, then for each changed target page:

    * **Added link** (A now links to T, but didn't before):
      ``inject_backlink`` is called on ``wiki/{T}.md``.
    * **Removed link** (A used to link to T, but no longer does):
      ``remove_backlink`` is called on ``wiki/{T}.md``.

    All writes use ``atomic_write`` and are guarded by a per-slug lock.
    Self-references (``source_slug == target_slug``) are silently ignored.
    Missing target files are logged as warnings and skipped — not an error.
    The function is idempotent: calling it twice with the same arguments
    leaves on-disk state unchanged after the first call.

    Args:
        wiki_dir: Directory containing ``wiki/{slug}.md`` files.
        source_slug: Slug of the page that was just saved.
        new_content: Current content of *source_slug* after the Writer Agent.
        previous_outgoing: Outgoing links the page had **before** this write.
            Pass an empty iterable for brand-new pages (Scenario A).
            For updates (Scenario B) pass
            ``extract_outgoing_links(old_content)`` captured before
            ``_save_wiki_page`` was called.
        file_id: Correlation ID threaded through structured log events.

    Returns:
        ``{"added": [<target_slugs>], "removed": [<target_slugs>]}`` —
        lists of target slugs that received an inject or remove operation
        respectively.  Useful for structured logging and test assertions.
    """
    bound_log = logger.bind(file_id=file_id, source_slug=source_slug)

    new_outgoing: set[str] = set(extract_outgoing_links(new_content))
    prev_outgoing: set[str] = set(previous_outgoing)

    # Self-references are never meaningful as backlinks
    new_outgoing.discard(source_slug)
    prev_outgoing.discard(source_slug)

    added = sorted(new_outgoing - prev_outgoing)
    removed = sorted(prev_outgoing - new_outgoing)

    for target in added:
        _apply_to_target(
            target_slug=target,
            source_slug=source_slug,
            operation="add",
            log=bound_log,
        )

    for target in removed:
        _apply_to_target(
            target_slug=target,
            source_slug=source_slug,
            operation="remove",
            log=bound_log,
        )

    if added or removed:
        bound_log.info(
            "backlinks_synced",
            added=added,
            removed=removed,
        )

    return {"added": added, "removed": removed}


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _apply_to_target(
    target_slug: str,
    source_slug: str,
    operation: str,
    log: structlog.types.FilteringBoundLogger,  # type: ignore[type-arg]
) -> None:
    """Apply a single inject or remove operation to *target_slug*'s page.

    Acquires the per-slug lock before reading and potentially writing the
    target page in the object store.  Does nothing if the page does not exist
    or if the operation would produce no change.

    Args:
        target_slug: Slug of the page whose ``## Backlinks`` section to update.
        source_slug: Slug to inject or remove.
        operation: ``"add"`` or ``"remove"``.
        log: Bound structlog logger (already carries ``file_id``/``source_slug``).
    """
    store = get_object_store()
    key = wiki_key(target_slug)

    with _lock_for(target_slug):
        old_content = store.get_text(key)
        if old_content is None:
            log.warning(
                "backlink_target_missing",
                target=target_slug,
                operation=operation,
            )
            return

        if operation == "add":
            new_target_content = inject_backlink(old_content, source_slug)
        else:
            new_target_content = remove_backlink(old_content, source_slug)

        if new_target_content == old_content:
            log.debug(
                "backlink_unchanged",
                target=target_slug,
                operation=operation,
            )
            return  # no write needed — idempotent

        store.put_text(key, new_target_content)
        log.debug(
            "backlink_file_updated",
            target=target_slug,
            operation=operation,
        )
