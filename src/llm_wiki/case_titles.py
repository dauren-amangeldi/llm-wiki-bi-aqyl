"""Clean automatically generated case titles, never manually entered names."""

import re

_MONTH = r"(?:0?[1-9]|1[0-2])"
_DAY = r"(?:0?[1-9]|[12][0-9]|3[01])"
_DATE_PREFIX = re.compile(
    rf"^(?:20[0-9]{{2}}[-.]{_MONTH}[-.]{_DAY}|"
    rf"{_DAY}[-.]{_MONTH}[-.]20[0-9]{{2}}|"
    rf"{_MONTH}[-.]{_DAY}|{_DAY}[-.]{_MONTH})(?:\s+|$)"
)
_FILE_EXTENSION = re.compile(
    r"\.(?:pdf|docx?|txt|md|jpe?g|png|webp|mp3|ogg|wav|m4a|webm)$", re.IGNORECASE
)


def clean_automatic_case_title(title: str) -> str:
    """Remove filename separators and leading upload dates, preserving names/numbers.

    Keep casing and meaningful hyphens (BI Group, GPT-4, Price-Value), and
    standalone years/IDs. Split legacy multi-file titles before removing dates.
    """
    parts = []
    for part in title.split("·"):
        part = " ".join(part.replace("_", " ").split())
        part = _FILE_EXTENSION.sub("", part)
        part = _DATE_PREFIX.sub("", part).strip()
        if part:
            parts.append(part)
    return " · ".join(parts)[:120].rstrip()
