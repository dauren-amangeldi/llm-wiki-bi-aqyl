"""Resolve council references against the case's actual wiki pages."""

import re
from collections.abc import Mapping
from typing import Any

WIKILINK = re.compile(r"\[\[([^\[\]\n]+)\]\]")
PLACEHOLDER = re.compile(r"\[source\s+document\]", re.IGNORECASE)


def normalize_citations(content: dict[str, Any], sources: Mapping[str, str]) -> dict[str, Any]:
    """Keep real references; never invent a target for an old placeholder."""
    text = str(content.get("text") or "")
    cite = str(content.get("cite") or "")
    anchors: list[str] = []

    def resolve(raw: str) -> str | None:
        slug = raw.split("|", 1)[0].split("#", 1)[0].strip()
        return slug if slug in sources else None

    def inline(match: re.Match[str]) -> str:
        anchor = resolve(match.group(1))
        if not anchor:
            return ""
        if anchor not in anchors:
            anchors.append(anchor)
        return f"[[{anchor}]]"

    clean = PLACEHOLDER.sub("", WIKILINK.sub(inline, text)).strip()
    for match in WIKILINK.finditer(cite):
        anchor = resolve(match.group(1))
        if anchor and anchor not in anchors:
            anchors.append(anchor)
    # Legacy replies sometimes stored a plain slug or an unambiguous page title.
    if cite in sources and cite not in anchors:
        anchors.append(cite)
    elif cite and not anchors:
        matches = [slug for slug, title in sources.items() if title.casefold() == cite.casefold()]
        if len(matches) == 1:
            anchors.append(matches[0])
    for item in content.get("citations") or []:
        if isinstance(item, dict):
            anchor = resolve(str(item.get("anchor") or ""))
            if anchor and anchor not in anchors:
                anchors.append(anchor)
    return {
        **content,
        "text": clean,
        "cite": "",
        "citations": [{"anchor": slug, "title": sources[slug]} for slug in anchors],
        "citation_unavailable": bool(
            content.get("citation_unavailable")
            or content.get("citations")
            or cite.strip()
            or PLACEHOLDER.search(text)
            or WIKILINK.search(text)
        )
        and not anchors,
    }
