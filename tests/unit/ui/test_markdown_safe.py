"""Unit tests for llm_wiki.ui.markdown_safe (100% coverage required)."""

from pathlib import Path

from llm_wiki.ui.markdown_safe import escape_dollars_for_streamlit


def test_escapes_currency() -> None:
    assert escape_dollars_for_streamlit("цена $6.00") == r"цена \$6.00"


def test_escapes_multiple_dollars() -> None:
    inp = "от $5 до $10"
    assert escape_dollars_for_streamlit(inp) == r"от \$5 до \$10"


def test_leaves_already_escaped_alone() -> None:
    """Already-escaped \\$ must not be doubled on a second call."""
    inp = r"уже экранировано \$5"
    assert escape_dollars_for_streamlit(inp) == r"уже экранировано \$5"


def test_idempotent() -> None:
    """Calling the function twice produces the same result as calling it once."""
    inp = "price is $9.99 and $4.50"
    once = escape_dollars_for_streamlit(inp)
    twice = escape_dollars_for_streamlit(once)
    assert once == twice


def test_no_dollars_unchanged() -> None:
    inp = "обычный текст без валюты"
    assert escape_dollars_for_streamlit(inp) == inp


def test_empty_string() -> None:
    assert escape_dollars_for_streamlit("") == ""


def test_dollar_at_start() -> None:
    assert escape_dollars_for_streamlit("$100 вверх") == r"\$100 вверх"


def test_dollar_at_end() -> None:
    assert escape_dollars_for_streamlit("стоит 100$") == r"стоит 100\$"


def test_consecutive_dollars() -> None:
    assert escape_dollars_for_streamlit("$$display$$") == r"\$\$display\$\$"


def test_handles_real_case() -> None:
    """Regression on the Dr. John's Products case from the bug report."""
    inp = (
        "розничную цену менее $6.00, а при продаже через таких ритейлеров, "
        "как Wal-Mart и Target, — менее $5.00"
    )
    out = escape_dollars_for_streamlit(inp)
    assert r"\$6.00" in out
    assert r"\$5.00" in out
    assert "Wal-Mart" in out      # hyphen stays a hyphen
    assert " — " in out           # em-dash untouched
    # No bare (unescaped) dollar signs remain
    import re as _re
    assert _re.search(r"(?<!\\)\$", out) is None


def test_currency_test_fixture() -> None:
    """Regression fixture: all dollars in currency_test.md get escaped."""
    fixture = (
        Path(__file__).parent.parent.parent
        / "fixtures"
        / "sample_wikis"
        / "currency_test.md"
    )
    content = fixture.read_text(encoding="utf-8")
    out = escape_dollars_for_streamlit(content)
    # No bare dollar signs remain
    import re
    bare_dollars = re.findall(r"(?<!\\)\$", out)
    assert bare_dollars == [], f"Bare dollars still present: {bare_dollars}"
    # But escaped ones are there
    assert r"\$" in out
