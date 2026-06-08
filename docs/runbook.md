# Runbook

> What to do when things break. Updated as incidents occur.

## Service Down: api

```bash
docker compose logs api --tail=50
docker compose restart api
```

## Pipeline Stuck: file never reaches DONE

1. Check Celery worker: `docker compose logs worker --tail=50`
2. Check Redis queue depth: `docker compose exec redis redis-cli LLEN celery`
3. Manually re-queue: `docker compose exec api uv run python -c "from llm_wiki.orchestrator.tasks import process_file_task; process_file_task.delay('<file_id>')"`

## Rolling Back an Ingestion

Implemented in Sprint 3. For now: manually delete `data/wiki/{slug}.md` and
remove the entry from `data/index.md`.

## Reindexing ChromaDB

Two separate scripts manage the two ChromaDB collections:

**`headings` collection** — page titles, used by SearchAgent and IndexStorage:
```bash
docker compose exec api uv run python scripts/reindex.py
# Preview without writing:
docker compose exec api uv run python scripts/reindex.py --dry-run
```

**`chunks` collection** — page body fragments, used by AnswerAgent (LW-20.1):
```bash
docker compose exec api uv run python scripts/reindex_chunks.py
# Preview without writing:
docker compose exec api uv run python scripts/reindex_chunks.py --dry-run
```

Run `reindex_chunks.py`:
- On first deploy (if wiki pages existed before LW-20.1 was deployed).
- After changing `EMBEDDING_MODEL` or `EMBEDDING_DIMENSIONS`.
- After a bulk import that bypassed the normal ingestion pipeline.

The ingestion pipeline auto-syncs chunks for each page it writes, so routine
use does not require manual reindexing.

## High LLM Cost Alert

Check `data/usage.log` for recent entries:
```bash
docker compose exec api tail -n 100 /app/data/usage.log | python -m json.tool
```

## Manually Inspecting Wiki Quality

The quality system maintains `data/issues.md` with two sections:
- `## 🔎 Auto-detected` — deterministic Linter findings (dead links, orphans, stale dates)
- `## 🤖 LLM-flagged` — semantic Auditor findings (contradictions, duplicates, suspected stale)

### Run the Linter manually

```bash
# Preview findings without modifying issues.md
docker compose exec api uv run python scripts/run_linter.py --dry-run

# Run all checks and update issues.md (same code as post-ingest pipeline step)
docker compose exec api uv run python scripts/run_linter.py

# Specific checks only
docker compose exec api uv run python scripts/run_linter.py --checks dead_link

# JSON output (for CI or scripting)
docker compose exec api uv run python scripts/run_linter.py --json

# Test against a regression fixture
docker compose exec api uv run python scripts/run_linter.py \
  --wiki-dir tests/fixtures/sample_wikis/dead_links/ --dry-run
```

Exit code `1` when issues are found — use this in pre-commit hooks or CI.

### Run the Auditor manually

```bash
# Quick smoke test — sync mode, 5 random pages
docker compose exec api uv run python scripts/run_auditor.py --sync --sample 5 --dry-run

# Specific pages
docker compose exec api uv run python scripts/run_auditor.py \
  --sync --slugs kubernetes,docker,microservices

# Full wiki, sync mode, write results
docker compose exec api uv run python scripts/run_auditor.py --sync

# Production run via Batch API (-50% cost, ~24 h)
docker compose exec api uv run python scripts/run_auditor.py --batch

# Cost guard — abort if over $0.50
docker compose exec api uv run python scripts/run_auditor.py \
  --sync --max-cost-usd 0.50

# Run against a contradiction fixture for prompt debugging
docker compose exec api uv run python scripts/run_auditor.py \
  --sync --wiki-dir tests/fixtures/sample_wikis/contradiction/ --dry-run
```

The script **always prints the estimated cost before making LLM calls** and
asks for confirmation if the estimate exceeds $1.00.

### Trigger via API

```bash
# Deterministic Linter (synchronous, instant)
curl -X POST http://localhost:8000/api/v1/lint/run
curl -X POST "http://localhost:8000/api/v1/lint/run?dry_run=true&checks=dead_link,orphan_page"

# LLM Auditor (async Celery task)
TASK=$(curl -s -X POST http://localhost:8000/api/v1/audit/run \
  -H "Content-Type: application/json" \
  -d '{"mode": "sync", "sample": 10}' | python -c "import sys,json; print(json.load(sys.stdin)['task_id'])")

# Poll for result
curl http://localhost:8000/api/v1/audit/$TASK
```

### Weekly automatic audit

Celery Beat runs the Auditor every Monday at 03:00 UTC via the
`weekly-audit` beat schedule entry.  To check if it ran:

```bash
docker compose logs beat --tail=20
docker compose logs worker --tail=50 | grep weekly_audit
```

To trigger immediately without waiting for the schedule:
```bash
docker compose exec api uv run python -c "
from llm_wiki.orchestrator.tasks import run_weekly_audit
run_weekly_audit.delay(mode='sync')
"
```
