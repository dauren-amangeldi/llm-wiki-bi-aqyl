"""Loads Twins persona definitions from editable `.md` files (BI-AQYL-TWINS).

Each file is TOML frontmatter (between `+++` delimiters) followed by the
persona's system prompt as plain text. Editing a file and restarting the
service updates the persona in the database — no code change, no migration.
Uses `tomllib` (stdlib since Python 3.11, this project's floor) — no new
dependency for a project that doesn't otherwise use YAML/TOML.
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

PERSONAS_DIR = Path(__file__).resolve().parent.parent / "personas" / "twins"


def parse_persona_file(text: str) -> dict[str, Any]:
    """Parse one persona file: TOML frontmatter + body as `system_prompt`."""
    if not text.startswith("+++"):
        raise ValueError("persona file must start with a '+++' TOML frontmatter block")
    _, frontmatter, body = text.split("+++", 2)
    data = tomllib.loads(frontmatter)
    data["system_prompt"] = body.strip()
    return data


def load_persona_files(directory: Path = PERSONAS_DIR) -> list[dict[str, Any]]:
    """Load and parse every `.md` persona file in *directory*, sorted by filename."""
    files = sorted(directory.glob("*.md"))
    results = []
    for f in files:
        try:
            results.append(parse_persona_file(f.read_text(encoding="utf-8")))
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"failed to parse persona file {f.name}: {exc}") from exc
    return results
