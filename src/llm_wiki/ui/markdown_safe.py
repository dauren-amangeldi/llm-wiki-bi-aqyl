"""Utilities for safely rendering wiki content in Streamlit.

Streamlit's ``st.markdown`` renders ``$...$`` and ``$$...$$`` as LaTeX math
by default.  Wiki pages are never expected to contain math; dollar signs mean
currency.  Without escaping, content like ``"price is $6.00"`` renders as
broken LaTeX, eating spaces and converting hyphens to the math-minus glyph.

Usage:

    from llm_wiki.ui.markdown_safe import escape_dollars_for_streamlit

    st.markdown(escape_dollars_for_streamlit(content))
"""

import re

# Match any `$` that is NOT already preceded by a backslash.
# Negative lookbehind ``(?<!\\)`` handles existing ``\\$`` escapes so they
# are not doubled on repeated calls.
_UNESCAPED_DOLLAR: re.Pattern[str] = re.compile(r"(?<!\\)\$")


def escape_dollars_for_streamlit(content: str) -> str:
    """Escape bare ``$`` signs so Streamlit does not treat them as LaTeX delimiters.

    Wiki pages contain dollar signs as currency symbols, not math markers.
    This function makes every unescaped ``$`` into ``\\$`` so Streamlit's
    Markdown renderer passes them through literally.

    Already-escaped ``\\$`` sequences are left untouched — the function is
    safe to call multiple times on the same string.

    Args:
        content: Raw Markdown content of a wiki page (or index/log/issues file).

    Returns:
        Markdown string with all bare ``$`` replaced by ``\\$``, ready for
        ``st.markdown()``.
    """
    return _UNESCAPED_DOLLAR.sub(r"\\$", content)
