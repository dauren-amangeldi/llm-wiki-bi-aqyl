"""Owner-scoped visibility for private wiki pages (wiki_store).

A ``sensitive`` page with an ``owner`` is a normal wiki page for that owner but
invisible to everyone else. These tests lock the two visibility rules:

* point lookups (get_page / get_page_meta) — unfiltered for internal callers
  (``caller=None``), owner-only for a concrete caller;
* enumerations (list_pages / keyword_search / get_all_pages) — public-only by
  default (protects the linter/auditor), owner's private pages included when a
  caller is given.
"""

from __future__ import annotations

from llm_wiki.storage import wiki_store

ALICE = "alice@bi.group"
BOB = "bob@bi.group"


async def test_private_page_owner_scoping(db_engine) -> None:
    wiki_store.save_page(
        "pub-x", "Public X", "# Public X\nОбщий документ.", sensitive=False
    )
    wiki_store.save_page(
        "private-y",
        "Secret Y",
        "# Secret Y\nСекретный отчёт зебра.",
        sensitive=True,
        owner=ALICE,
    )

    # ── Point lookups ──────────────────────────────────────────────────────
    assert wiki_store.get_page("private-y", caller=ALICE) is not None  # owner
    assert wiki_store.get_page("private-y", caller=BOB) is None  # other user
    assert wiki_store.get_page("private-y") is not None  # internal, unfiltered
    assert wiki_store.get_page("pub-x", caller=BOB) is not None  # public

    meta_owner = wiki_store.get_page_meta("private-y", caller=ALICE)
    assert meta_owner is not None and meta_owner.sensitive is True
    assert wiki_store.get_page_meta("private-y", caller=BOB) is None
    assert wiki_store.page_exists("private-y", caller=ALICE) is True
    assert wiki_store.page_exists("private-y", caller=BOB) is False

    # ── Enumerations ───────────────────────────────────────────────────────
    alice_slugs = {p.slug for p in wiki_store.list_pages(caller=ALICE)}
    bob_slugs = {p.slug for p in wiki_store.list_pages(caller=BOB)}
    anon_slugs = {p.slug for p in wiki_store.list_pages()}
    assert "private-y" in alice_slugs
    assert "private-y" not in bob_slugs
    assert "private-y" not in anon_slugs
    for slugs in (alice_slugs, bob_slugs, anon_slugs):
        assert "pub-x" in slugs

    # get_all_pages defaults to public-only so bulk linter/auditor never reads
    # private content; the owner can opt in with a caller.
    default_all = {s for s, _ in wiki_store.get_all_pages()}
    owner_all = {s for s, _ in wiki_store.get_all_pages(caller=ALICE)}
    assert "private-y" not in default_all
    assert "private-y" in owner_all

    # ── Keyword search ─────────────────────────────────────────────────────
    alice_hits = {h.slug for h in wiki_store.keyword_search("зебра", caller=ALICE)}
    bob_hits = {h.slug for h in wiki_store.keyword_search("зебра", caller=BOB)}
    anon_hits = {h.slug for h in wiki_store.keyword_search("зебра")}
    assert "private-y" in alice_hits
    assert "private-y" not in bob_hits
    assert "private-y" not in anon_hits
