"""Unit tests for wiki list-snippet generation (frontmatter stripping)."""

from __future__ import annotations

from llm_wiki.api.v1.wiki import _plain_snippet, _strip_frontmatter


def test_strip_frontmatter_removes_leading_yaml_block() -> None:
    content = (
        "---\n"
        "title: Dr. John's Products, Ltd.: запуск SpinBrush\n"
        "tags: [entrepreneurship, product-launch]\n"
        "summary: Кейс о запуске SpinBrush.\n"
        "---\n\n"
        "# Dr. John's Products, Ltd.: запуск SpinBrush\n\n"
        "## Обзор кейса\n\nDr. John's Products, Ltd. вывела на рынок SpinBrush."
    )
    stripped = _strip_frontmatter(content)
    assert "---" not in stripped
    assert "tags:" not in stripped
    assert stripped.startswith("# Dr. John's Products")


def test_strip_frontmatter_is_noop_without_frontmatter() -> None:
    content = "# A plain page\n\nNo frontmatter here."
    assert _strip_frontmatter(content) == content


def test_plain_snippet_does_not_leak_frontmatter() -> None:
    content = (
        "---\n"
        "title: Dr. John's Products, Ltd.: запуск SpinBrush\n"
        "tags: [entrepreneurship, product-launch, consumer-products, distribution, "
        "oral-care, marketing-mix, case-questions, valuation, strategic-analysis, hbr]\n"
        "---\n\n"
        "# Dr. John's Products, Ltd.: запуск SpinBrush\n\n"
        "## Обзор кейса\n\n"
        "Dr. John's Products, Ltd. — компания Джона Ошера — вывела на рынок дешёвую "
        "электрическую зубную щётку SpinBrush, которая быстро получила сильный отклик."
    )
    snippet = _plain_snippet(content)
    assert "---" not in snippet
    assert "tags:" not in snippet
    assert "entrepreneurship" not in snippet
    assert snippet.startswith("Dr. John's Products, Ltd.")


def test_plain_snippet_still_strips_markdown_noise() -> None:
    content = "# Heading\n\nSome **bold** and [[wiki-link]] text here."
    snippet = _plain_snippet(content)
    assert snippet == "Heading Some bold and wiki-link text here."
