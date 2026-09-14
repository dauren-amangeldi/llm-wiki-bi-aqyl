"""Shared language policy for council turns, independent of persona prompts."""

LANGUAGES = {"ru": "Russian", "en": "English", "kk": "Kazakh"}


def normalize_language(language: str) -> str:
    code = language.lower().replace("_", "-").split("-", 1)[0]
    return code if code in LANGUAGES else "ru"


def with_response_language(system: str, language: str) -> str:
    """Enforce the language resolved once by the router for the whole user turn."""
    target = LANGUAGES[normalize_language(language)]
    return system + (
        "\n\nRESPONSE LANGUAGE POLICY (overrides persona language defaults):\n"
        f"Write your entire reply ONLY in {target}. This turn's response language "
        "has already been selected from the user's question. Translate information "
        "from documents into that language. Keep it throughout replies to other "
        "experts, regardless of the language of sources, persona instructions or "
        "earlier conversation. "
        "Apply this to all natural-language output fields; preserve JSON keys, "
        "participant IDs, source slugs and exact source titles."
    )
