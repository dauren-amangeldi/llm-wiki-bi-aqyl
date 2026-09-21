"""Text normalization at the boundary between parsers and PostgreSQL."""


def sanitize_extracted_text(text: str) -> str:
    """Remove NUL bytes, which PostgreSQL text columns cannot store.

    PDF text layers can contain these characters even when extraction succeeds.
    Preserve all other Unicode characters and whitespace in the source.
    """
    return text.replace("\x00", "")
