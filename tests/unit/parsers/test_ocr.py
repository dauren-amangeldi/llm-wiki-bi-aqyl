"""Unit tests for the scanned-PDF OCR fallback (renders real pages, mocks OpenAI)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from llm_wiki.config import settings
from llm_wiki.parsers.ocr import OCRError, ocr_pdf


def _image_pdf(tmp_path: Path, pages: int = 1) -> Path:
    """Build an image-only PDF (no text layer) so ocr_pdf's renderer has real input."""
    from PIL import Image, ImageDraw

    imgs = []
    for i in range(pages):
        img = Image.new("RGB", (600, 300), "white")
        ImageDraw.Draw(img).text((20, 140), f"page {i}", fill="black")
        imgs.append(img)
    out = tmp_path / "scan.pdf"
    imgs[0].save(str(out), "PDF", save_all=True, append_images=imgs[1:])
    return out


def _mock_openai(page_texts: list[str]) -> MagicMock:
    """Fake openai.OpenAI returning page_texts from chat.completions.create in order."""
    seq = iter(page_texts)

    def create(**_kw: object) -> MagicMock:
        resp = MagicMock()
        resp.choices = [MagicMock(message=MagicMock(content=next(seq)))]
        return resp

    client = MagicMock()
    client.chat.completions.create.side_effect = create
    return client


def test_ocr_pdf_concatenates_pages(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "openai_api_key", "sk-test")
    pdf = _image_pdf(tmp_path, pages=2)
    fake = _mock_openai(["Страница один", "Страница два"])
    with patch("openai.OpenAI", return_value=fake):
        out = ocr_pdf(pdf, file_id="t")
    assert out == "Страница один\n\nСтраница два"


def test_ocr_skips_blank_pages(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "openai_api_key", "sk-test")
    pdf = _image_pdf(tmp_path, pages=2)
    fake = _mock_openai(["", "Только вторая"])  # first page unreadable
    with patch("openai.OpenAI", return_value=fake):
        out = ocr_pdf(pdf, file_id="t")
    assert out == "Только вторая"


def test_ocr_respects_max_pages(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "openai_api_key", "sk-test")
    monkeypatch.setattr(settings, "ocr_max_pages", 1)
    pdf = _image_pdf(tmp_path, pages=3)
    fake = _mock_openai(["only one call"])  # would StopIteration if it OCR'd >1 page
    with patch("openai.OpenAI", return_value=fake):
        out = ocr_pdf(pdf, file_id="t")
    assert out == "only one call"


def test_ocr_requires_api_key(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "openai_api_key", "")
    pdf = _image_pdf(tmp_path)
    with pytest.raises(OCRError, match="API_KEY"):
        ocr_pdf(pdf)


def test_ocr_raises_when_no_text_produced(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "openai_api_key", "sk-test")
    pdf = _image_pdf(tmp_path)
    fake = _mock_openai([""])  # vision returns nothing on every page
    with patch("openai.OpenAI", return_value=fake):
        with pytest.raises(OCRError, match="no text"):
            ocr_pdf(pdf)


def test_ocr_wraps_api_errors(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "openai_api_key", "sk-test")
    pdf = _image_pdf(tmp_path)
    client = MagicMock()
    client.chat.completions.create.side_effect = RuntimeError("boom")
    with patch("openai.OpenAI", return_value=client):
        with pytest.raises(OCRError, match="page 1"):
            ocr_pdf(pdf)


# ---------------------------------------------------------------------------
# ocr_image — фото конспектов/заметок (jpeg/png/webp) как материалы
# ---------------------------------------------------------------------------


def _photo(tmp_path: Path, name: str = "notes.png") -> Path:
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (400, 200), "white")
    ImageDraw.Draw(img).text((20, 90), "конспект", fill="black")
    out = tmp_path / name
    img.save(str(out))
    return out


def test_ocr_image_transcribes_photo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from llm_wiki.parsers.ocr import ocr_image

    monkeypatch.setattr(settings, "openai_api_key", "sk-test")
    photo = _photo(tmp_path)
    fake = _mock_openai(["Лекция 3: фотосинтез. Хлорофилл поглощает свет."])
    with patch("openai.OpenAI", return_value=fake):
        out = ocr_image(photo, file_id="t")
    assert "фотосинтез" in out
    # data:image/png — mime по расширению файла, не захардкоженный png от рендера
    sent = fake.chat.completions.create.call_args.kwargs["messages"][0]["content"][1]
    assert sent["image_url"]["url"].startswith("data:image/png;base64,")


def test_ocr_image_jpeg_mime(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from llm_wiki.parsers.ocr import ocr_image

    monkeypatch.setattr(settings, "openai_api_key", "sk-test")
    photo = _photo(tmp_path, "notes.jpg")
    fake = _mock_openai(["текст"])
    with patch("openai.OpenAI", return_value=fake):
        ocr_image(photo, file_id="t")
    sent = fake.chat.completions.create.call_args.kwargs["messages"][0]["content"][1]
    assert sent["image_url"]["url"].startswith("data:image/jpeg;base64,")


def test_ocr_image_no_text_raises_human_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from llm_wiki.parsers.ocr import ocr_image

    monkeypatch.setattr(settings, "openai_api_key", "sk-test")
    photo = _photo(tmp_path)
    fake = _mock_openai([""])  # модель ничего не прочла
    with patch("openai.OpenAI", return_value=fake):
        with pytest.raises(OCRError, match="читаемый текст"):
            ocr_image(photo, file_id="t")


def test_ocr_image_too_large_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from llm_wiki.parsers import ocr as ocr_mod

    monkeypatch.setattr(settings, "openai_api_key", "sk-test")
    monkeypatch.setattr(ocr_mod, "_MAX_IMAGE_BYTES", 10)  # крошечный лимит для теста
    photo = _photo(tmp_path)
    with pytest.raises(OCRError, match="20 МБ"):
        ocr_mod.ocr_image(photo, file_id="t")
