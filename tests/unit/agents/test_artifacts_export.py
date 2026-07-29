"""Unit tests for artifact export (docx / pptx / pdf byte generation)."""

from __future__ import annotations

import io
import zipfile

import pytest

from llm_wiki.agents.artifacts_export import (
    ExportError,
    _first_existing,
    _REGULAR_FONT_CANDIDATES,
    export_artifact,
    supported_formats,
)

REPORT = {
    "executive_summary": "Резюме материала о летнем лагере.",
    "metrics": [{"label": "Формат", "value": "Лагерь"}],
    "sections": [{"heading": "Замысел", "body": "Тело раздела с кириллицей."}],
}
TEST = {
    "questions": [
        {"prompt": "Что это?", "options": ["Вариант А", "Вариант Б"], "correct": 1,
         "explanation": "Потому что Б."},
    ]
}
CARD = {
    "title": "Карточка знаний",
    "summary": "Краткий итог.",
    "key_points": [{"label": "Инсайт", "text": "Важная мысль."}],
    "recommendations": ["Сделать шаг X"],
    "tags": ["обучение", "практика"],
}
PRESENTATION = {
    "title": "Дека о лагере",
    "slides": [{"heading": "Слайд 1", "bullets": ["раз", "два"], "notes": "заметки"}],
}

_HAS_PDF_FONT = _first_existing(_REGULAR_FONT_CANDIDATES) is not None
needs_font = pytest.mark.skipif(not _HAS_PDF_FONT, reason="no Unicode TTF on this host")


def _is_zip_with(data: bytes, member_prefix: str) -> bool:
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        return any(n.startswith(member_prefix) for n in z.namelist())


# --- registry -------------------------------------------------------------


def test_supported_formats() -> None:
    assert supported_formats("report") == {"pdf", "docx"}
    assert supported_formats("presentation") == {"pdf", "pptx"}
    assert supported_formats("infographic") == set()
    assert supported_formats("podcast") == set()


@pytest.mark.parametrize("kind,fmt", [
    ("report", "pptx"), ("presentation", "docx"), ("infographic", "pdf"), ("card", "xml"),
])
def test_unsupported_pair_raises(kind: str, fmt: str) -> None:
    with pytest.raises(ExportError):
        export_artifact(kind, {}, fmt)


# --- docx (OOXML zip) -----------------------------------------------------


@pytest.mark.parametrize("kind,content", [
    ("report", REPORT), ("test", TEST), ("card", CARD),
])
def test_docx_is_valid_ooxml(kind: str, content: dict) -> None:
    data, media = export_artifact(kind, content, "docx")
    assert media.endswith("wordprocessingml.document")
    assert data[:2] == b"PK"  # zip magic
    assert _is_zip_with(data, "word/document.xml")


# --- pptx (OOXML zip) -----------------------------------------------------


def test_pptx_has_slides() -> None:
    data, media = export_artifact("presentation", PRESENTATION, "pptx")
    assert media.endswith("presentationml.presentation")
    assert data[:2] == b"PK"
    assert _is_zip_with(data, "ppt/slides/slide")


# --- pdf (needs a Unicode font) -------------------------------------------


@needs_font
@pytest.mark.parametrize("kind,content", [
    ("report", REPORT), ("test", TEST), ("card", CARD), ("presentation", PRESENTATION),
])
def test_pdf_magic(kind: str, content: dict) -> None:
    data, media = export_artifact(kind, content, "pdf")
    assert media == "application/pdf"
    assert data[:4] == b"%PDF"
    assert len(data) > 500  # non-trivial document


@needs_font
def test_pdf_handles_empty_content() -> None:
    # missing keys must not crash — just a near-empty document
    data, _ = export_artifact("report", {}, "pdf")
    assert data[:4] == b"%PDF"
