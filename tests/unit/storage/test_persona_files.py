"""Unit tests for the Twins persona .md/TOML-frontmatter loader."""

from __future__ import annotations

from pathlib import Path

from llm_wiki.storage.persona_files import PERSONAS_DIR, load_persona_files, parse_persona_file

_SAMPLE = """+++
id = "test_persona"
name = "Тест"
inspiration = "test fixture"
real_name = "Test Person"
track = "tech"
pinned = true
lens = "test lens"
avatar_init = "TP"

[domain_weights]
tech = 0.7
real_estate = 0.2
finance = 0.1
+++

Ты — тестовая персона. Системный промпт во втором абзаце.
"""


def test_parse_persona_file_splits_frontmatter_and_body() -> None:
    data = parse_persona_file(_SAMPLE)
    assert data["id"] == "test_persona"
    assert data["track"] == "tech"
    assert data["pinned"] is True
    assert data["domain_weights"] == {"tech": 0.7, "real_estate": 0.2, "finance": 0.1}
    assert data["system_prompt"] == "Ты — тестовая персона. Системный промпт во втором абзаце."


def test_load_persona_files_reads_all_md_files_in_directory(tmp_path: Path) -> None:
    (tmp_path / "a.md").write_text(_SAMPLE, encoding="utf-8")
    other = _SAMPLE.replace("test_persona", "other_persona").replace('name = "Тест"', 'name = "Другой"')
    (tmp_path / "b.md").write_text(other, encoding="utf-8")

    rows = load_persona_files(tmp_path)

    assert {r["id"] for r in rows} == {"test_persona", "other_persona"}


def test_real_personas_directory_has_eleven_files_with_unique_ids() -> None:
    rows = load_persona_files(PERSONAS_DIR)
    assert len(rows) == 11
    ids = [r["id"] for r in rows]
    assert len(set(ids)) == 11
    assert "musk" in ids
    assert "data_metrics" in ids


def test_load_persona_files_wraps_parse_errors_with_filename(tmp_path: Path) -> None:
    # Malformed file: missing closing +++
    bad_file = tmp_path / "malformed.md"
    bad_file.write_text("+++\nid = 'test'\n", encoding="utf-8")

    try:
        load_persona_files(tmp_path)
        assert False, "Expected ValueError to be raised"
    except ValueError as exc:
        error_msg = str(exc)
        assert "malformed.md" in error_msg, f"Filename not in error: {error_msg}"
        assert "failed to parse" in error_msg.lower()


def test_real_personas_all_have_a_real_name() -> None:
    rows = load_persona_files(PERSONAS_DIR)
    for row in rows:
        assert row["real_name"], f"{row['id']} is missing real_name"
    ids_to_names = {r["id"]: r["real_name"] for r in rows}
    assert ids_to_names["musk"] == "Elon Musk"
    assert ids_to_names["data_metrics"] == "DJ Patil"
