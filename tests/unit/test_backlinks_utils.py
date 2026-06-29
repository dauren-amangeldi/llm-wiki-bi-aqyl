"""Unit tests for llm_wiki.utils.backlinks (LW-13).

Covers inject_backlink, remove_backlink, extract_backlinks, and
extract_outgoing_links.
"""

from __future__ import annotations


from llm_wiki.utils.backlinks import (
    extract_backlinks,
    extract_outgoing_links,
    inject_backlink,
    remove_backlink,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_BODY = "# Transformers\n\nSome content about [[attention-mechanism]] and [[bert]].\n"

_WITH_BACKLINKS = (
    "# Transformers\n\nSome content.\n\n"
    "## Backlinks\n\n"
    "- [[gpt]]\n"
    "- [[llama]]\n"
)


# ===========================================================================
# inject_backlink
# ===========================================================================


def test_inject_creates_section_when_absent() -> None:
    """inject_backlink adds ## Backlinks section if page has none."""
    result = inject_backlink(_BODY, "gpt")
    assert "## Backlinks" in result
    assert "- [[gpt]]" in result


def test_inject_section_ends_with_newline() -> None:
    """The appended section ends with a trailing newline."""
    result = inject_backlink(_BODY, "gpt")
    assert result.endswith("\n")


def test_inject_blank_line_before_new_section() -> None:
    """A blank line separates the body from the new ## Backlinks section."""
    result = inject_backlink("# Page\n\nBody text.\n", "src")
    lines = result.splitlines()
    bl_idx = lines.index("## Backlinks")
    assert lines[bl_idx - 1] == ""  # blank line before header


def test_inject_appends_sorted_to_existing_section() -> None:
    """inject_backlink inserts a new slug in alphabetical order."""
    content = "# Page\n\n## Backlinks\n\n- [[zzz]]\n"
    result = inject_backlink(content, "aaa")
    lines = result.splitlines()
    bl_idx = lines.index("## Backlinks")
    bullets = [l for l in lines[bl_idx:] if l.startswith("- [[")]
    assert bullets[0] == "- [[aaa]]"
    assert bullets[1] == "- [[zzz]]"


def test_inject_sorted_middle() -> None:
    """New slug is inserted between existing slugs when alphabetically in the middle."""
    content = "# P\n\n## Backlinks\n\n- [[aaa]]\n- [[zzz]]\n"
    result = inject_backlink(content, "mmm")
    bullets = [l for l in result.splitlines() if l.startswith("- [[")]
    assert bullets == ["- [[aaa]]", "- [[mmm]]", "- [[zzz]]"]


def test_inject_idempotent() -> None:
    """Calling inject_backlink twice with the same slug returns content unchanged."""
    first = inject_backlink(_BODY, "gpt")
    second = inject_backlink(first, "gpt")
    assert first == second


def test_inject_does_not_touch_other_sections() -> None:
    """inject_backlink does not modify any other ## section."""
    content = "# Page\n\n## References\n\n- some ref\n"
    result = inject_backlink(content, "src")
    assert "## References" in result
    assert "- some ref" in result


def test_inject_not_confused_by_backstory_section() -> None:
    """## Backstory section is not parsed as ## Backlinks."""
    content = "# Page\n\n## Backstory\n\nSome history.\n"
    result = inject_backlink(content, "src")
    # Should CREATE a new ## Backlinks section, not modify ## Backstory
    assert "## Backlinks" in result
    assert "## Backstory" in result
    assert "- [[src]]" in result
    # ## Backstory must still contain "Some history."
    assert "Some history." in result


def test_inject_multiple_distinct_sources() -> None:
    """Multiple injections from different sources all appear, sorted."""
    content = "# Page\n\nBody.\n"
    for src in ["charlie", "alice", "bob"]:
        content = inject_backlink(content, src)
    bullets = [l for l in content.splitlines() if l.startswith("- [[")]
    assert bullets == ["- [[alice]]", "- [[bob]]", "- [[charlie]]"]


# ===========================================================================
# remove_backlink
# ===========================================================================


def test_remove_removes_bullet() -> None:
    """remove_backlink removes an existing bullet from the section."""
    result = remove_backlink(_WITH_BACKLINKS, "gpt")
    assert "- [[gpt]]" not in result
    assert "- [[llama]]" in result


def test_remove_removes_empty_section() -> None:
    """When the last bullet is removed, the entire ## Backlinks section disappears."""
    content = "# Page\n\nBody.\n\n## Backlinks\n\n- [[only-one]]\n"
    result = remove_backlink(content, "only-one")
    assert "## Backlinks" not in result
    assert "only-one" not in result


def test_remove_no_dangling_empty_section() -> None:
    """After removing last bullet, no empty '## Backlinks' header is left."""
    # Use a valid 2-char slug (regex requires at least 2 chars: start + end char).
    content = "# Page\n\n## Backlinks\n\n- [[pg]]\n"
    result = remove_backlink(content, "pg")
    assert "## Backlinks" not in result


def test_remove_idempotent_when_absent() -> None:
    """remove_backlink is a no-op when the slug is not in the section."""
    result = remove_backlink(_WITH_BACKLINKS, "nonexistent")
    assert result == _WITH_BACKLINKS


def test_remove_idempotent_when_no_section() -> None:
    """remove_backlink is a no-op when there is no ## Backlinks section."""
    result = remove_backlink(_BODY, "gpt")
    assert result == _BODY


def test_remove_preserves_other_sections() -> None:
    """remove_backlink does not alter sections other than ## Backlinks."""
    content = (
        "# Page\n\n"
        "## References\n\n- ref1\n\n"
        "## Backlinks\n\n- [[src]]\n\n"
        "## See Also\n\n- other\n"
    )
    result = remove_backlink(content, "src")
    assert "## References" in result
    assert "- ref1" in result
    assert "## See Also" in result
    assert "- other" in result


def test_remove_then_inject_cycle() -> None:
    """remove followed by inject returns to the original state."""
    after_remove = remove_backlink(_WITH_BACKLINKS, "gpt")
    restored = inject_backlink(after_remove, "gpt")
    # Both "gpt" and "llama" must be present and sorted
    bullets = [l for l in restored.splitlines() if l.startswith("- [[")]
    assert "- [[gpt]]" in bullets
    assert "- [[llama]]" in bullets
    assert bullets == sorted(bullets)


# ===========================================================================
# extract_backlinks
# ===========================================================================


def test_extract_empty_when_no_section() -> None:
    """extract_backlinks returns [] when ## Backlinks section is absent."""
    assert extract_backlinks(_BODY) == []


def test_extract_returns_slugs_in_order() -> None:
    """extract_backlinks returns slugs in the order they appear in the file."""
    assert extract_backlinks(_WITH_BACKLINKS) == ["gpt", "llama"]


def test_extract_ignores_junk_lines() -> None:
    """Non-bullet prose inside ## Backlinks section is silently ignored."""
    content = (
        "# Page\n\n"
        "## Backlinks\n\n"
        "- [[real-slug]]\n"
        "Some LLM-written paragraph here.\n"
        "- [[another-slug]]\n"
    )
    result = extract_backlinks(content)
    assert result == ["real-slug", "another-slug"]


def test_extract_stops_at_next_heading() -> None:
    """Slugs from sections after ## Backlinks are not included."""
    content = (
        "# Page\n\n"
        "## Backlinks\n\n"
        "- [[correct]]\n\n"
        "## References\n\n"
        "- [[should-not-appear]]\n"
    )
    result = extract_backlinks(content)
    assert result == ["correct"]
    assert "should-not-appear" not in result


def test_extract_roundtrip_with_inject() -> None:
    """extract_backlinks returns exactly what inject_backlink inserted."""
    content = "# Page\n\nBody.\n"
    for slug in ["zebra", "alpha", "middle"]:
        content = inject_backlink(content, slug)
    extracted = extract_backlinks(content)
    # Sorted because inject keeps them sorted
    assert extracted == ["alpha", "middle", "zebra"]


# ===========================================================================
# extract_outgoing_links (sanity / regression)
# ===========================================================================


def test_outgoing_dedup_preserves_first_occurrence() -> None:
    """extract_outgoing_links deduplicates, keeping first-occurrence order."""
    content = "[[bert]] and [[gpt]] again [[bert]]"
    result = extract_outgoing_links(content)
    assert result == ["bert", "gpt"]


def test_outgoing_empty_for_no_links() -> None:
    """extract_outgoing_links returns [] when no [[links]] are present."""
    assert extract_outgoing_links("No links here.") == []


def test_outgoing_ignores_malformed_links() -> None:
    """Links that don't match the slug regex are not returned."""
    # Uppercase is not valid; single-char slugs don't match the regex
    content = "[[GPT]] and [[x]] and [[valid-slug]]"
    result = extract_outgoing_links(content)
    assert result == ["valid-slug"]
