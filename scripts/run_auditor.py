"""CLI for running the LLM Auditor (LW-15).

Defaults to ``--sync`` mode (chat completions) for fast iteration during
development.  Use ``--batch`` for production/cost-optimised runs.

Examples (inside container):

    # Sync-mode: immediate results, small sample
    docker compose exec api uv run python scripts/run_auditor.py \\
        --sync --sample 10 --dry-run

    # Specific slugs
    docker compose exec api uv run python scripts/run_auditor.py \\
        --sync --slugs llm,agents,transformers

    # Run against a contradiction fixture
    docker compose exec api uv run python scripts/run_auditor.py \\
        --sync --wiki-dir tests/fixtures/sample_wikis/contradiction/ --dry-run

    # Production batch run (polling included, up to 24 h)
    docker compose exec api uv run python scripts/run_auditor.py --batch

Exit codes:
    0 — no issues found (or aborted by cost guard)
    1 — issues found
    2 — internal error
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the LLM wiki Auditor.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--sync",
        dest="mode",
        action="store_const",
        const="sync",
        help="Use ordinary chat completions (fast, slightly more expensive).",
    )
    mode_group.add_argument(
        "--batch",
        dest="mode",
        action="store_const",
        const="batch",
        help="Use OpenAI Batch API (-50%% cost, up to 24 h).",
    )
    parser.set_defaults(mode="sync")  # safe default — never accidentally batch

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not write to issues.md.",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        metavar="N",
        help="Randomly sample N pages instead of processing all.",
    )
    parser.add_argument(
        "--slugs",
        default=None,
        help="Comma-separated list of page slugs to audit.",
    )
    parser.add_argument(
        "--json",
        dest="output_json",
        action="store_true",
        help="Output machine-readable JSON.",
    )
    parser.add_argument(
        "--max-cost-usd",
        type=float,
        default=None,
        help="Abort if estimated cost exceeds this value (USD).",
    )
    parser.add_argument(
        "--wiki-dir",
        default=None,
        help="Override path to wiki pages directory.",
    )
    return parser.parse_args()


def _estimate_cost(n_pages: int, mode: str) -> float:
    """Return a rough USD cost estimate.

    Rate: ~$0.003/page sync, ~$0.0015/page batch.
    """
    rate = 0.0015 if mode == "batch" else 0.003
    return round(n_pages * rate, 4)


def _confirm(prompt: str) -> bool:
    """Ask the user to type 'yes' to continue."""
    try:
        answer = input(prompt + " [yes/no]: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        return False
    return answer == "yes"


def main() -> None:
    args = _parse_args()

    repo_root = Path(__file__).parent.parent
    src_dir = repo_root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

    from llm_wiki.agents.auditor import AuditorAgent
    from llm_wiki.config import settings
    from llm_wiki.llm.client import LLMClient
    from llm_wiki.quality.issues_writer import upsert_section
    from llm_wiki.quality.models import IssueSection

    # ----------------------------------------------------------------- paths
    wiki_dir: Path = Path(args.wiki_dir) if args.wiki_dir else settings.wiki_dir

    # -------------------------------------------------------- load wiki pages
    all_pages: list[tuple[str, str]] = []
    if wiki_dir.exists():
        for md_file in sorted(wiki_dir.glob("*.md")):
            all_pages.append((md_file.stem, md_file.read_text(encoding="utf-8")))

    if not all_pages:
        print("No wiki pages found in " + str(wiki_dir))
        sys.exit(0)

    # ---------------------------------------------------------------- filter
    slugs_filter = (
        {s.strip() for s in args.slugs.split(",") if s.strip()} if args.slugs else None
    )
    if slugs_filter:
        all_pages = [(s, c) for s, c in all_pages if s in slugs_filter]

    if args.sample and len(all_pages) > args.sample:
        import random
        all_pages = random.sample(all_pages, args.sample)

    n_pages = len(all_pages)
    if n_pages == 0:
        print("No matching pages to audit.")
        sys.exit(0)

    # --------------------------------------------------------- cost estimate
    estimated_cost = _estimate_cost(n_pages, args.mode)
    print(f"Pages to audit: {n_pages}")
    print(f"Mode: {args.mode}")
    print(f"Estimated cost: ~${estimated_cost:.4f} USD")

    if args.max_cost_usd is not None and estimated_cost > args.max_cost_usd:
        print(
            f"\nABORTED: estimated cost ${estimated_cost:.4f} exceeds "
            f"--max-cost-usd ${args.max_cost_usd:.4f}"
        )
        sys.exit(0)

    if estimated_cost > 1.0:
        if not _confirm(
            f"\nEstimated cost ${estimated_cost:.4f} exceeds $1.00. Proceed?"
        ):
            print("Aborted by user.")
            sys.exit(0)

    print()

    # ----------------------------------------- build related pairs (optional)
    related_pairs: list[tuple[str, str]] = []
    try:
        from llm_wiki.llm.embeddings import EmbeddingStore

        llm_tmp = LLMClient()
        emb_store = EmbeddingStore(
            llm_client=llm_tmp
        )
        page_slugs = {slug for slug, _ in all_pages}
        for slug, _ in all_pages:
            hits = emb_store.query(slug, top_k=5, file_id="run-auditor")
            for hit in hits:
                if hit.similarity >= 0.6 and hit.slug != slug and hit.slug in page_slugs:
                    pair = tuple(sorted([slug, hit.slug]))
                    if pair not in related_pairs:
                        related_pairs.append(pair)  # type: ignore[arg-type]
    except Exception as exc:  # noqa: BLE001
        print(f"Warning: could not build related pairs from embeddings: {exc}")

    # ------------------------------------------------------------------ run
    llm = LLMClient()

    async def _run() -> list:
        agent = AuditorAgent(llm)
        result = await agent.run(
            wiki_pages=all_pages,
            related_pairs=related_pairs,
            mode=args.mode,
        )
        await llm.aclose()
        return result

    try:
        issues = asyncio.run(_run())
    except Exception as exc:
        print(f"ERROR: Auditor raised an exception: {exc}", file=sys.stderr)
        sys.exit(2)

    # ---------------------------------------------------------------- write
    updated = False
    if not args.dry_run and issues:
        try:
            upsert_section(
                issues_path=settings.issues_path,
                section=IssueSection.LLM_FLAGGED,
                issues=issues,
            )
            updated = True
        except Exception as exc:
            print(f"WARNING: could not write to issues.md: {exc}", file=sys.stderr)

    # --------------------------------------------------------------- output
    by_kind: dict[str, int] = {}
    for issue in issues:
        by_kind[issue.kind.value] = by_kind.get(issue.kind.value, 0) + 1

    if args.output_json:
        output = {
            "issues": [
                {
                    "kind": i.kind.value,
                    "section": i.section.value,
                    "page_slug": i.page_slug,
                    "description": i.description,
                    "related_slugs": list(i.related_slugs),
                }
                for i in issues
            ],
            "summary": by_kind,
            "issues_md_updated": updated,
        }
        print(json.dumps(output, ensure_ascii=False, indent=2))
    else:
        if issues:
            summary_parts = [f"{count} {kind}" for kind, count in sorted(by_kind.items())]
            print("Found: " + ", ".join(summary_parts))
            print()
            for issue in sorted(issues, key=lambda i: (i.kind, i.page_slug)):
                related = (
                    " → " + ", ".join(issue.related_slugs) if issue.related_slugs else ""
                )
                print(f"  [{issue.kind}] {issue.page_slug}: {issue.description}{related}")
        else:
            print("No issues found.")

        if args.dry_run:
            print("\n(dry-run: issues.md not modified)")
        elif updated:
            print(f"\nissues.md updated: {settings.issues_path}")

    sys.exit(1 if issues else 0)


if __name__ == "__main__":
    main()
