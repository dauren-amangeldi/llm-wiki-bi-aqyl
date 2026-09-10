"""One slide geometry for the browser preview and editable PowerPoint export."""

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

WIDTH, HEIGHT = 960, 540
FONT = "Arial"


@lru_cache(maxsize=128)
def _font(size: int, bold: bool):
    from PIL import ImageFont

    suffix = "-Bold" if bold else ""
    paths = (
        f"/usr/share/fonts/truetype/dejavu/DejaVuSans{suffix}.ttf",
        f"/System/Library/Fonts/Supplemental/Arial{' Bold' if bold else ''}.ttf",
    )
    path = next((p for p in paths if Path(p).exists()), None)
    return ImageFont.truetype(path, size) if path else ImageFont.load_default(size=size)


def _lines(text: str, width: float, size: int, bold: bool) -> list[str]:
    font = _font(size, bold)
    lines: list[str] = []
    for paragraph in text.splitlines() or [""]:
        line = ""
        for word in paragraph.split():
            candidate = f"{line} {word}".strip()
            if font.getlength(candidate) <= width:
                line = candidate
                continue
            if line:
                lines.append(line)
            line = word
            # Split only oversized words, using logarithmic metric lookups.
            while font.getlength(line) > width:
                low, high = 1, len(line)
                while low < high:
                    mid = (low + high + 1) // 2
                    if font.getlength(line[:mid]) <= width:
                        low = mid
                    else:
                        high = mid - 1
                lines.append(line[:low])
                line = line[low:]
        lines.append(line)
    return lines


def presentation_layout(content: dict[str, Any]) -> dict[str, Any]:
    raw = content.get("slides")
    return _layout_slides(json.dumps(raw if isinstance(raw, list) else [], ensure_ascii=False))


@lru_cache(maxsize=32)
def _layout_slides(serialized: str) -> dict[str, Any]:
    raw = json.loads(serialized)
    slides = []
    for item in raw if isinstance(raw, list) else []:
        if not isinstance(item, dict):
            continue
        title = next(
            (item[k] for k in ("heading", "title") if isinstance(item.get(k), str) and item[k]), ""
        )
        bullets = (
            [b for b in item.get("bullets", []) if isinstance(b, str) and b.strip()]
            if isinstance(item.get("bullets"), list)
            else []
        )
        body = item.get("body") if isinstance(item.get("body"), str) else ""
        if not title and not bullets and not body:
            continue
        rows = bullets or ([body] if body else [])
        # Fit without clipping or silently dropping generated text.
        size = 23
        while True:
            heading_size = round(size * 1.5)
            title_lines = _lines(title, 832, heading_size, True) if title else []
            row_lines = [_lines(row, 812 if bullets else 832, size, False) for row in rows]
            height = len(title_lines) * heading_size * 1.2 + (24 if title_lines and rows else 0)
            height += sum(len(lines) * size * 1.4 + size * 0.45 for lines in row_lines)
            if height <= 420 or size <= 6:
                break
            size -= 1
        # Exceptionally dense stored slides still fit; same scale in both outputs.
        scale = min(1, 420 / max(height, 1))
        y = (HEIGHT - height * scale) / 2 if not slides else 60
        blocks = []
        for line in title_lines:
            blocks.append(
                {
                    "text": line,
                    "x": 64,
                    "y": round(y, 2),
                    "size": round(heading_size * scale, 2),
                    "bold": True,
                }
            )
            y += heading_size * 1.2 * scale
        if title_lines and rows:
            y += 24 * scale
        marks = []
        for lines in row_lines:
            if bullets:
                marks.append(
                    {"x": 64, "y": round(y + size * 0.55 * scale, 2), "size": round(6 * scale, 2)}
                )
            for line in lines:
                blocks.append(
                    {
                        "text": line,
                        "x": 84 if bullets else 64,
                        "y": round(y, 2),
                        "size": round(size * scale, 2),
                        "bold": False,
                    }
                )
                y += size * 1.4 * scale
            y += size * 0.45 * scale
        slides.append(
            {
                "title": title,
                "notes": item.get("notes") if isinstance(item.get("notes"), str) else "",
                "background": ["07859E", "126B89"] if not slides else ["FFFFFF", "FFFFFF"],
                "foreground": "FFFFFF" if not slides else "172B3A",
                "accent": "A5DEE8" if not slides else "07859E",
                "blocks": blocks,
                "marks": marks,
            }
        )
    return {"width": WIDTH, "height": HEIGHT, "font": FONT, "slides": slides}
