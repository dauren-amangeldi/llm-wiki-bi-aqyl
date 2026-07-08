"""CLI for running the deterministic wiki Linter (LW-14).

Examples (inside container):

    # Preview findings without touching issues.md
    docker compose exec api uv run python scripts/run_linter.py --dry-run

    # Run all checks and update issues.md
    docker compose exec api uv run python scripts/run_linter.py

    # Only specific checks
    docker compose exec api uv run python scripts/run_linter.py \\
        --checks dead_link,orphan_page

    # Machine-readable JSON output (for CI)
    docker compose exec api uv run python scripts/run_linter.py --json

    # Run against a fixture wiki (for regression tests)
    docker compose exec api uv run python scripts/run_linter.py \\
        --wiki-dir tests/fixtures/sample_wikis/dead_links/ --dry-run

Exit codes:
    0 — no issues found (or --dry-run completed cleanly)
    1 — one or more issues found
    2 — internal error
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the deterministic wiki Linter.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not write to issues.md; only print findings to stdout.",
    )
    parser.add_argument(
        "--checks",
        default="dead_link,orphan_page,stale_date",
        help="Comma-separated subset of checks to run "
             "(default: dead_link,orphan_page,stale_date).",
    )
    parser.add_argument(
        "--json",
        dest="output_json",
        action="store_true",
        help="Output machine-readable JSON instead of human text.",
    )
    parser.add_argument(
        "--wiki-dir",
        default=None,
        help="Override path to wiki pages directory (default: data/wiki/).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    # ------------------------------------------------------------------ setup
    # Add project src/ to sys.path so the script works both inside and outside
    # the Docker container (as long as it runs from the repo root).
    repo_root = Path(__file__).parent.parent
    src_dir = repo_root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

    from llm_wiki.config import settings
    from llm_wiki.quality.issues_writer import upsert_section
    from llm_wiki.quality.linter import run_linter
    from llm_wiki.quality.models import IssueKind, IssueSection
    from llm_wiki.storage.index import IndexStorage

    # ----------------------------------------------------------------- paths
    wiki_dir: Path = Path(args.wiki_dir) if args.wiki_dir else settings.wiki_dir
    issues_path: Path = settings.issues_path

    # --------------------------------------------------------- load wiki pages
    wiki_pages: dict[str, str] = {}
    if args.wiki_dir and wiki_dir.exists():
        # Explicit --wiki-dir: read markdown fixtures from disk.
        for md_file in sorted(wiki_dir.glob("*.md")):
            wiki_pages[md_file.stem] = md_file.read_text(encoding="utf-8")
    else:
        # Default: wiki pages live in Postgres.
        from llm_wiki.storage import wiki_store

        wiki_pages = dict(wiki_store.get_all_pages())

    if not wiki_pages:
        _print_or_json(
            output_json=args.output_json,
            message="No wiki pages found in " + str(wiki_dir),
            issues=[],
            summary={},
        )
        sys.exit(0)

    # --------------------------------------------------- index root sections
    index_storage = IndexStorage()
    index_root_sections: set[str] = {
        h.text.lower().replace(" ", "-")
        for h in index_storage.read_headings()
        if h.slug is None
    }

    # ------------------------------------------------------------------ run
    requested = {c.strip() for c in args.checks.split(",") if c.strip()}
    valid_linter_kinds = {
        IssueKind.DEAD_LINK.value,
        IssueKind.ORPHAN_PAGE.value,
        IssueKind.STALE_DATE.value,
    }
    active_checks = requested & valid_linter_kinds if requested else valid_linter_kinds

    current_year = datetime.now(timezone.utc).year
    try:
        all_issues = run_linter(
            wiki_pages=wiki_pages,
            index_root_sections=index_root_sections,
            current_year=current_year,
        )
    except Exception as exc:
        print(f"ERROR: Linter raised an exception: {exc}", file=sys.stderr)
        sys.exit(2)

    issues = [i for i in all_issues if i.kind.value in active_checks]

    # --------------------------------------------------------------- write
    updated = False
    if not args.dry_run:
        try:
            upsert_section(
                issues_path=issues_path,
                section=IssueSection.AUTO_DETECTED,
                issues=issues,
            )
            updated = True
        except Exception as exc:
            print(f"WARNING: Could not write to issues.md: {exc}", file=sys.stderr)

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
        summary_parts = [f"{count} {kind}" for kind, count in sorted(by_kind.items())]
        if summary_parts:
            print("Found: " + ", ".join(summary_parts))
        else:
            print("No issues found.")

        if issues and not args.output_json:
            print()
            for issue in sorted(issues, key=lambda i: (i.kind, i.page_slug)):
                related = (
                    " → " + ", ".join(issue.related_slugs) if issue.related_slugs else ""
                )
                print(f"  [{issue.kind}] {issue.page_slug}: {issue.description}{related}")

        if not args.dry_run:
            if updated:
                print(f"\nissues.md updated: {issues_path}")
            else:
                print("\n(issues.md not updated due to write error)")
        else:
            print("\n(dry-run: issues.md not modified)")

    sys.exit(1 if issues else 0)


def _print_or_json(
    output_json: bool,
    message: str,
    issues: list,
    summary: dict,
) -> None:
    if output_json:
        print(json.dumps({"issues": issues, "summary": summary}, ensure_ascii=False))
    else:
        print(message)


if __name__ == "__main__":
    main()
