"""Explicit, bounded six-image quality comparison. Never called by the app."""

import argparse
import asyncio
import json
import time
from pathlib import Path

from llm_wiki.agents.visual_presentation import artwork_prompt, draw_artwork, render_slide
from llm_wiki.config import settings

COPY = {
    "ru": (
        "Архитектура доверия",
        "Три этапа устойчивого роста",
        [
            "1 500 объектов — единый стандарт",
            "Проверка качества перед масштабированием",
            "Прозрачные правила и ответственность",
        ],
    ),
    "kk": (
        "Сенім архитектурасы",
        "Тұрақты өсудің үш кезеңі",
        [
            "1 500 нысан — бірыңғай стандарт",
            "Сапаны тексеру және нәтижені бағалау",
            "Әділдік, қауіпсіздік, ашықтық",
        ],
    ),
    "en": (
        "The architecture of trust",
        "Three stages of sustainable growth",
        [
            "1,500 sites — one shared standard",
            "Validate quality before scaling",
            "Clear rules and accountability",
        ],
    ),
}


async def main(output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    config = {"model": settings.visual_image_model, "size": "1536x864", "quality": "medium"}
    plan = {
        "style_block": "Calm white background, deep navy and cobalt, translucent blue architectural glass blocks, coherent soft studio lighting."
    }
    report = []
    for lang, (title, subtitle, points) in COPY.items():
        slide = {
            "title": title,
            "subtitle": subtitle,
            "points": points,
            "source_refs": ["sample"],
            "layout": "editorial",
            "speaker_notes": "Synthetic typography sample; not a real business claim.",
            "image_brief": "Three clear blue architectural modules forming a stable bridge, each supports the next, white background, no lettering.",
        }
        for mode in ("hybrid", "full"):
            destination = output / f"{lang}-{mode}.png"
            if destination.exists():
                continue
            prompt = (
                artwork_prompt(plan, slide)
                if mode == "hybrid"
                else (
                    "Create a finished 16:9 executive presentation slide. "
                    + plan["style_block"]
                    + " Large readable text on left, architectural glass illustration on right. "
                    + f"Render these exact texts in {lang}, preserve every character and number: "
                    + json.dumps(
                        {"title": title, "subtitle": subtitle, "points": points}, ensure_ascii=False
                    )
                    + " No extra text. All copy inside a 5% safe margin."
                )
            )
            started = time.monotonic()
            try:
                data = await draw_artwork(prompt, f"visual-pilot-{lang}-{mode}", config)
                if mode == "hybrid":
                    (output / f"{lang}-artwork.png").write_bytes(data)
                    data = render_slide(slide, data, 1, 1, lang)
                destination.write_bytes(data)
                result = {
                    "language": lang,
                    "mode": mode,
                    "seconds": round(time.monotonic() - started, 2),
                    "bytes": len(data),
                    "ok": True,
                }
            except Exception as exc:
                result = {
                    "language": lang,
                    "mode": mode,
                    "seconds": round(time.monotonic() - started, 2),
                    "ok": False,
                    "error_type": type(exc).__name__,
                }
            report.append(result)
            (output / "results.json").write_text(
                json.dumps({"config": config, "samples": report}, indent=2)
            )
            print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    asyncio.run(main(args.output))
