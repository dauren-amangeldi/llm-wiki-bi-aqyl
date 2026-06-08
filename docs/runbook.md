# Runbook

> What to do when things break. Each section: **Symptoms → Diagnosis → Fix → Verification**.
> Updated as incidents occur.

---

## Table of contents

1. [Quick reference](#quick-reference)
2. [Service-level incidents](#service-level-incidents)
   - [API down](#api-down)
   - [Worker (Celery) down](#worker-celery-down)
   - [Redis down](#redis-down)
   - [ChromaDB corruption / model mismatch](#chromadb-corruption--model-mismatch)
   - [SQLite locked](#sqlite-locked)
3. [Pipeline-level incidents](#pipeline-level-incidents)
   - [File stuck in RECEIVED / STORED / SEARCHED / WRITTEN / LINTED](#file-stuck-in-received--stored--searched--written--linted)
   - [Pipeline keeps retrying and failing](#pipeline-keeps-retrying-and-failing)
   - [File never reaches DONE](#file-never-reaches-done)
   - [Partial write: page on disk but missing from index.md (or vice versa)](#partial-write-page-on-disk-but-missing-from-indexmd-or-vice-versa)
4. [Data integrity](#data-integrity)
   - [Roll back a single ingestion](#roll-back-a-single-ingestion)
   - [Replay pipeline from a specific step](#replay-pipeline-from-a-specific-step)
   - [Reindex headings collection](#reindex-headings-collection)
   - [Reindex chunks collection (LW-20.1)](#reindex-chunks-collection-lw-201)
   - [Recover from accidental wiki/ deletion](#recover-from-accidental-wiki-deletion)
   - [Recover from accidental index.md corruption](#recover-from-accidental-indexmd-corruption)
5. [Cost & quota](#cost--quota)
   - [Daily cost spike — diagnosis from usage.log](#daily-cost-spike--diagnosis-from-usagelog)
   - [How to read usage.log](#how-to-read-usagelog)
   - [Emergency: disable ingestion](#emergency-disable-ingestion)
6. [Quality system](#quality-system)
   - [Run Linter manually](#run-linter-manually)
   - [Run Auditor manually](#run-auditor-manually)
   - [Interpret issues.md](#interpret-issuesmd)
   - [Auditor batch job stuck](#auditor-batch-job-stuck)
7. [Q&A endpoint (LW-20)](#qa-endpoint-lw-20)
   - [Ask endpoint returns "low confidence" for everything](#ask-endpoint-returns-low-confidence-for-everything)
   - [Ask endpoint slow](#ask-endpoint-slow)
8. [Deployment](#deployment)
   - [Cold start checklist](#cold-start-checklist)
   - [After changing EMBEDDING_MODEL or EMBEDDING_DIMENSIONS](#after-changing-embedding_model-or-embedding_dimensions)
   - [After restoring a backup of /data](#after-restoring-a-backup-of-data)
   - [Bump a single image without downtime](#bump-a-single-image-without-downtime)
9. [Appendix: common error signatures](#appendix-common-error-signatures)

---

## Quick reference

State of the system in 30 seconds:

```bash
# Are all services up?
docker compose ps

# Does the API respond?
curl -s http://localhost:8000/api/v1/stats | jq

# How deep is the Celery queue?
docker compose exec redis redis-cli LLEN celery

# Any files stuck in FAILED in the last 24 h?
docker compose exec api uv run python -c "
import sqlite3, datetime
from llm_wiki.config import settings
con = sqlite3.connect(str(settings.data_dir / 'metadata.db'))
cutoff = (datetime.datetime.utcnow() - datetime.timedelta(days=1)).isoformat()
rows = con.execute(
    \"SELECT file_id, original_name, status FROM files WHERE status='FAILED' AND created_at > ?\",
    (cutoff,)
).fetchall()
print(rows if rows else 'No failures')
"

# How much did we spend today?
curl -s http://localhost:8000/api/v1/stats | jq .cost_today_usd
```

### Data directory map

```
/data/
├── raw/            — original uploaded files (PDF, MD) — never modified after upload
├── wiki/           — generated wiki pages (.md) — source of truth for content
├── chroma/         — vector DB (two collections: headings, chunks)
├── index.md        — wiki structure map, source of truth for sections
├── log.md          — append-only ingestion event log
├── issues.md       — Linter + Auditor quality report
├── usage.log       — JSONL cost record for every LLM call
└── metadata.db     — SQLite: file records + state history
```

### Key environment variables

| Variable | Required | Effect |
|---|---|---|
| `OPENAI_API_KEY` | Yes (if `LLM_PROVIDER=openai`) | Authentication for OpenAI calls |
| `LLM_PROVIDER` | Yes | `ollama` \| `openai` \| `anthropic` |
| `WIKI_LANGUAGE` | No (default `en`) | Language for wiki generation and Streamlit UI (`ru` \| `kk` \| `en`) |
| `EMBEDDING_MODEL` | No | Changing this requires running both reindex scripts |
| `EMBEDDING_DIMENSIONS` | No | Must match the chosen embedding model |
| `LOG_LEVEL` | No (default `INFO`) | Log verbosity: `DEBUG` \| `INFO` \| `WARNING` \| `ERROR` |
| `DAILY_TOKEN_BUDGET` | No | **TODO: requires LW-19 budget alerts implementation** |
| `INGESTION_RATE_LIMIT_PER_MIN` | No | **TODO: requires LW-19 budget alerts implementation** |

---

## Service-level incidents

### API down

**Symptoms:** `curl http://localhost:8000/api/v1/stats` returns connection refused or 5xx. Streamlit viewer shows "Connection error".

**Diagnosis:**
```bash
docker compose ps api
docker compose logs api --tail=50
```

Expected when healthy: `api` shows `Up` and `(healthy)` in `docker compose ps`.

**Fix:**
```bash
# 1. Restart the api container
docker compose restart api

# 2. If it keeps crashing, check for import errors or missing env vars
docker compose logs api --tail=100 | jq 'select(.level == "error" or .level == "critical")'

# 3. If the DB is missing or corrupt, reinitialise (non-destructive for existing data)
docker compose exec api uv run python scripts/init_wiki.py
```

**Verification:**
```bash
curl -s http://localhost:8000/api/v1/stats | jq .page_count
# Expected: a number (0 on empty wiki)
```

**Root cause hint:** Most common cause is a missing or malformed `.env` variable (e.g., `OPENAI_API_KEY` with trailing whitespace).

---

### Worker (Celery) down

**Symptoms:** Files are accepted by `POST /files` (status 202) but stay in `RECEIVED` indefinitely. `GET /files/{id}` shows `status: RECEIVED` even after minutes.

**Diagnosis:**
```bash
docker compose ps worker
docker compose logs worker --tail=50
docker compose exec redis redis-cli LLEN celery  # tasks piling up
```

**Fix:**
```bash
# 1. Restart the worker
docker compose restart worker

# 2. If the worker crashes immediately, check for missing dependencies or config
docker compose logs worker --tail=100 | jq 'select(.level == "error")'

# 3. Force-process a stuck file by ID (re-enqueue)
docker compose exec api uv run python -c "
from llm_wiki.orchestrator.tasks import process_file_task
process_file_task.delay('<FILE_ID>')
print('Enqueued.')
"
```

**Verification:**
```bash
# Watch logs until the file transitions to DONE
docker compose logs worker -f | grep -m1 '"state_transition"'
```

---

### Redis down

**Symptoms:** `POST /files` returns 5xx. Celery worker logs show `redis.exceptions.ConnectionError`. All async operations fail.

**Diagnosis:**
```bash
docker compose ps redis
docker compose exec redis redis-cli PING  # Expected: PONG
```

**Fix:**
```bash
# 1. Restart Redis
docker compose restart redis

# 2. Restart worker and beat so they reconnect
docker compose restart worker beat
```

**Verification:**
```bash
docker compose exec redis redis-cli PING   # PONG
curl -s -o /dev/null -w "%{http_code}" -X POST http://localhost:8000/api/v1/files \
  -F "file=@/dev/null"  # Should return 422 (bad file), not 5xx
```

**Root cause hint:** Redis container ran out of memory or was OOM-killed by the host OS.

---

### ChromaDB corruption / model mismatch

**Symptoms:** API or worker logs contain `EmbeddingModelMismatchError`. `POST /files` fails at the SEARCHED step. `POST /api/v1/ask` returns 500.

**Diagnosis:**
```bash
docker compose logs api --tail=50 | jq 'select(.exc_type == "EmbeddingModelMismatchError")'
docker compose logs worker --tail=50 | jq 'select(.exc_type == "EmbeddingModelMismatchError")'
```

The error message will say which collection (`headings` or `chunks`) and which model hash caused the mismatch.

**Fix:**

This happens when `EMBEDDING_MODEL` or `EMBEDDING_DIMENSIONS` changed in `.env` without rebuilding the collections.

```bash
# 1. Stop the worker so no new writes hit the stale collections
docker compose stop worker

# 2. Rebuild the headings collection (SearchAgent / IndexStorage)
docker compose exec api uv run python scripts/reindex.py

# 3. Rebuild the chunks collection (AnswerAgent — LW-20.1)
docker compose exec api uv run python scripts/reindex_chunks.py

# 4. Restart services
docker compose start worker
```

**Verification:**
```bash
docker compose exec api uv run python -c "
from llm_wiki.config import settings
from llm_wiki.llm.embeddings import EmbeddingStore
from llm_wiki.llm.client import LLMClient
llm = LLMClient()
store = EmbeddingStore(chroma_path=settings.chroma_dir, llm_client=llm)
print('headings OK, count:', store.count())
"
```

**Root cause hint:** Changing `EMBEDDING_MODEL` in `.env` without running the reindex scripts. See [Cold start checklist](#cold-start-checklist) for a prevention procedure.

---

### SQLite locked

**Symptoms:** API logs show `OperationalError: database is locked`. Usually appears during high-concurrency ingestion.

**Diagnosis:**
```bash
docker compose logs api --tail=30 | jq 'select(.event | contains("locked"))'
```

**Fix:**
```bash
# SQLite WAL mode handles most concurrency. A simple API restart clears stale connections.
docker compose restart api
```

If the lock persists after restart:
```bash
# Connect directly and check for open transactions
docker compose exec api sqlite3 /app/data/metadata.db ".tables"
# If that hangs, the DB file is held by a zombie process — restart the full stack
docker compose down && docker compose up -d
```

**Verification:**
```bash
curl -s http://localhost:8000/api/v1/stats | jq .page_count
```

**Root cause hint:** A long-running Celery task held a read lock without committing. SQLite in WAL mode is usually not affected, but a crash mid-transaction can leave a lock file (`metadata.db-wal`, `metadata.db-shm`).

---

## Pipeline-level incidents

### File stuck in RECEIVED / STORED / SEARCHED / WRITTEN / LINTED

**Symptoms:** `GET /api/v1/files/{file_id}` shows a status that hasn't advanced in >5 minutes.

**Diagnosis:**
```bash
# Check what state the file is in
curl -s http://localhost:8000/api/v1/files/<FILE_ID> | jq '{status, state_history}'

# Check worker logs for the file
docker compose logs worker --tail=200 | jq 'select(.file_id == "<FILE_ID>")'

# Check Celery queue depth
docker compose exec redis redis-cli LLEN celery
```

**Fix:**

If the worker is alive but the file is stuck (queue depth is 0 but no progress):
```bash
# Re-enqueue the task — the pipeline is idempotent, completed steps are skipped
docker compose exec api uv run python -c "
from llm_wiki.orchestrator.tasks import process_file_task
process_file_task.delay('<FILE_ID>')
print('Re-enqueued.')
"
```

If you need to force-reset to a specific step, see [Replay pipeline from a specific step](#replay-pipeline-from-a-specific-step).

**Verification:**
```bash
# Poll until status changes
watch -n5 "curl -s http://localhost:8000/api/v1/files/<FILE_ID> | jq .status"
```

---

### Pipeline keeps retrying and failing

**Symptoms:** Worker logs show repeated `pipeline_failed` events for the same `file_id`. The file's `state_history` grows but never reaches DONE. Celery retries the task multiple times.

**Diagnosis:**
```bash
# Find all error events for the file
docker compose logs worker | jq 'select(.level == "error" and .file_id == "<FILE_ID>")'

# Check the exact step where it fails
curl -s http://localhost:8000/api/v1/files/<FILE_ID> | jq .state_history
```

Common root causes and fixes:

| Symptom in logs | Cause | Fix |
|---|---|---|
| `AuthenticationError` / `Invalid API key` | Wrong or expired `OPENAI_API_KEY` | Update `.env`, `docker compose up -d api worker` |
| `RateLimitError` persisting | Daily budget hit (LW-19 not implemented yet) | Check `usage.log`, wait or reduce concurrency |
| `EmbeddingModelMismatchError` | Model changed without reindex | See [ChromaDB corruption / model mismatch](#chromadb-corruption--model-mismatch) |
| `FileNotFoundError: Raw file for ... not found` | Raw file deleted from `data/raw/` | Upload the file again via `POST /files` |
| `JSONDecodeError` in writer step | LLM returned malformed JSON | Retry once; if persistent, check model (`OLLAMA_MODEL` too small) |

**Fix (force-stop retries):**

If you want to stop the retries and mark the file as permanently failed:
```bash
docker compose exec api uv run python -c "
import asyncio
from llm_wiki.api.deps import _engine
from sqlalchemy.ext.asyncio import async_sessionmaker, AsyncSession
from sqlalchemy import update
from llm_wiki.storage.metadata import FileRecord

async def mark_failed():
    session_factory = async_sessionmaker(bind=_engine, expire_on_commit=False)
    async with session_factory() as session:
        await session.execute(
            update(FileRecord)
            .where(FileRecord.file_id == '<FILE_ID>')
            .values(status='FAILED')
        )
        await session.commit()
    print('Marked as FAILED.')

asyncio.run(mark_failed())
"
```

**Verification:**
```bash
curl -s http://localhost:8000/api/v1/files/<FILE_ID> | jq .status
# Expected: "FAILED"
```

---

### File never reaches DONE

See [File stuck in RECEIVED / STORED / SEARCHED / WRITTEN / LINTED](#file-stuck-in-received--stored--searched--written--linted) and [Pipeline keeps retrying and failing](#pipeline-keeps-retrying-and-failing).

Additional check — LINTED step sometimes fails silently (non-fatal) and pipeline continues to LOGGED. If the status is LOGGED but not DONE, the LOGGED → DONE transition failed:

```bash
docker compose logs worker | jq 'select(.file_id == "<FILE_ID>")' | tail -20
```

---

### Partial write: page on disk but missing from index.md (or vice versa)

**Symptoms:** Streamlit viewer shows a page that is not in the index tree, or the index tree has an entry whose page file does not exist. Linter reports orphan pages or dead links pointing to existing files.

**Diagnosis:**

```bash
# Find .md files in wiki/ that have no entry in index.md
docker compose exec api uv run python -c "
from llm_wiki.config import settings
import re

index_text = (settings.index_path).read_text(encoding='utf-8')
index_slugs = set(re.findall(r'\[\[([^\]]+)\]\]', index_text))
wiki_slugs = {p.stem for p in settings.wiki_dir.glob('*.md')}

only_on_disk = wiki_slugs - index_slugs
only_in_index = index_slugs - wiki_slugs
print('On disk but not in index.md:', only_on_disk or 'none')
print('In index.md but no .md file:', only_in_index or 'none')
"
```

**Fix:**

For a slug that is **on disk but not in index.md** (the common case after a crash mid-pipeline):
```bash
# Add the missing entry to the index manually — find which section it belongs to
# and add a line: - [[slug]] Title under the correct ## Section heading in data/index.md
# Then rebuild the headings ChromaDB entry:
docker compose exec api uv run python -c "
from llm_wiki.config import settings
from llm_wiki.storage.index import IndexStorage
from llm_wiki.llm.embeddings import EmbeddingStore
from llm_wiki.llm.client import LLMClient

slug = '<SLUG>'
title = '<TITLE>'
section = 'General'  # adjust as needed

llm = LLMClient()
store = EmbeddingStore(chroma_path=settings.chroma_dir, llm_client=llm)
index = IndexStorage(settings.index_path, embedding_store=store)
index.add_page(slug, section, title=title)
print(f'Added {slug} to index.')
"
```

For a slug that is **in index.md but the .md file is missing**:
```bash
# Remove the dangling index entry — edit data/index.md manually to remove the [[slug]] line
# Then remove the stale ChromaDB embedding:
docker compose exec api uv run python -c "
from llm_wiki.config import settings
from llm_wiki.llm.embeddings import EmbeddingStore
from llm_wiki.llm.client import LLMClient

slug = '<SLUG>'
llm = LLMClient()
store = EmbeddingStore(chroma_path=settings.chroma_dir, llm_client=llm)
store.delete(slug)
print(f'Deleted embedding for {slug}.')
"
```

**Verification:**
```bash
docker compose exec api uv run python -c "
from llm_wiki.config import settings
import re
index_text = settings.index_path.read_text()
index_slugs = set(re.findall(r'\[\[([^\]]+)\]\]', index_text))
wiki_slugs = {p.stem for p in settings.wiki_dir.glob('*.md')}
print('Mismatch:', wiki_slugs.symmetric_difference(index_slugs) or 'none')
"
```

---

## Data integrity

### Roll back a single ingestion

> The Writer Agent merges content into existing pages. For **created** pages rollback is clean. For **updated** pages the old content is gone — you can only remove the additions manually.

**Step 1 — Find the file_id:**
```bash
# By original filename
docker compose exec api uv run python -c "
import asyncio
from llm_wiki.api.deps import _engine
from sqlalchemy.ext.asyncio import async_sessionmaker
from llm_wiki.storage.metadata import FileRecord
from sqlalchemy import select

async def find():
    session_factory = async_sessionmaker(bind=_engine, expire_on_commit=False)
    async with session_factory() as session:
        rows = await session.execute(
            select(FileRecord).where(FileRecord.original_name.ilike('%<FILENAME>%'))
        )
        for r in rows.scalars():
            print(r.file_id, r.original_name, r.status, r.created_pages, r.updated_pages)

asyncio.run(find())
"
```

**Step 2 — Identify affected pages:**
```bash
docker compose exec api uv run python -c "
import asyncio
from llm_wiki.api.deps import _engine
from sqlalchemy.ext.asyncio import async_sessionmaker
from llm_wiki.storage.metadata import FileRecord
from sqlalchemy import select

FILE_ID = '<FILE_ID>'

async def show():
    session_factory = async_sessionmaker(bind=_engine, expire_on_commit=False)
    async with session_factory() as session:
        r = await session.get(FileRecord, FILE_ID)
        if r:
            print('Created pages:', list(r.created_pages or []))
            print('Updated pages:', list(r.updated_pages or []))
        else:
            print('File not found')

asyncio.run(show())
"
```

**Step 3 — Roll back created pages** (full removal):
```bash
docker compose exec api uv run python -c "
from llm_wiki.config import settings
from llm_wiki.llm.embeddings import EmbeddingStore
from llm_wiki.llm.chunk_store import ChunkStore
from llm_wiki.llm.client import LLMClient

CREATED_SLUGS = ['<slug1>', '<slug2>']  # from step 2

llm = LLMClient()
embedding_store = EmbeddingStore(chroma_path=settings.chroma_dir, llm_client=llm)
chunk_store = ChunkStore(chroma_path=settings.chroma_dir, llm_client=llm)

for slug in CREATED_SLUGS:
    # Delete wiki file
    wiki_file = settings.wiki_dir / f'{slug}.md'
    if wiki_file.exists():
        wiki_file.unlink()
        print(f'Deleted {wiki_file}')

    # Remove from headings Chroma
    embedding_store.delete(slug)
    print(f'Deleted heading embedding for {slug}')

    # Remove chunks from Chroma
    chunk_store.delete_page(slug)
    print(f'Deleted chunks for {slug}')

print('Done. You must also manually remove [[slug]] lines from data/index.md.')
print('Run the Linter to confirm no dead links remain.')
"
```

**Step 4 — For updated pages** (manual review required):
```bash
# The Writer Agent merged new content into these pages — full rollback is not automatic.
# Review and edit each page manually:
docker compose exec api cat /app/data/wiki/<SLUG>.md
```

> **Warning:** Updated pages are modified in-place by the Writer Agent. There is no automatic undo. Future improvement: git-tracked wiki directory would enable `git checkout data/wiki/<slug>.md`. **TODO: requires LW-sprint3 wiki versioning.**

**Step 5 — Mark the record rolled back in SQLite:**
```bash
docker compose exec api uv run python -c "
import asyncio
from llm_wiki.api.deps import _engine
from sqlalchemy.ext.asyncio import async_sessionmaker
from sqlalchemy import update
from llm_wiki.storage.metadata import FileRecord

FILE_ID = '<FILE_ID>'

async def mark():
    session_factory = async_sessionmaker(bind=_engine, expire_on_commit=False)
    async with session_factory() as session:
        await session.execute(
            update(FileRecord).where(FileRecord.file_id == FILE_ID).values(status='ROLLED_BACK')
        )
        await session.commit()
    print('Marked as ROLLED_BACK.')

asyncio.run(mark())
"
```

**Verification:**
```bash
# Linter should report no dead links pointing to the removed slugs
docker compose exec api uv run python scripts/run_linter.py --dry-run
```

---

### Replay pipeline from a specific step

The pipeline records every completed state in `state_history`. On re-run it skips states already in that list. To replay from step X, delete state_history entries after (and including) X.

**Example: replay from WRITTEN (re-run the Writer Agent without re-running Search):**

```bash
docker compose exec api uv run python -c "
import asyncio, json
from llm_wiki.api.deps import _engine
from sqlalchemy.ext.asyncio import async_sessionmaker
from sqlalchemy import update, select
from llm_wiki.storage.metadata import FileRecord

FILE_ID = '<FILE_ID>'
# States to KEEP (everything before the replay point)
KEEP_STATES = {'RECEIVED', 'STORED', 'SEARCHED'}

async def trim():
    session_factory = async_sessionmaker(bind=_engine, expire_on_commit=False)
    async with session_factory() as session:
        r = await session.get(FileRecord, FILE_ID)
        if not r:
            print('File not found'); return
        original = r.state_history or []
        trimmed = [e for e in original if e.get('state') in KEEP_STATES]
        print(f'Trimmed history: {[e[\"state\"] for e in trimmed]}')
        await session.execute(
            update(FileRecord)
            .where(FileRecord.file_id == FILE_ID)
            .values(state_history=trimmed, status='SEARCHED')
        )
        await session.commit()
    print('History trimmed. Re-enqueue the task to replay from WRITTEN.')

asyncio.run(trim())
"

# Re-enqueue
docker compose exec api uv run python -c "
from llm_wiki.orchestrator.tasks import process_file_task
process_file_task.delay('<FILE_ID>')
print('Re-enqueued.')
"
```

---

### Reindex headings collection

When to run: first deploy, after changing `EMBEDDING_MODEL` / `EMBEDDING_DIMENSIONS`, after bulk wiki import.

```bash
docker compose exec api uv run python scripts/reindex.py

# Preview without writing
docker compose exec api uv run python scripts/reindex.py --dry-run
```

---

### Reindex chunks collection (LW-20.1)

When to run: first deploy (if wiki pages pre-existed), after changing `EMBEDDING_MODEL` / `EMBEDDING_DIMENSIONS`, after bulk import.

```bash
docker compose exec api uv run python scripts/reindex_chunks.py

# Preview chunk count without writing to Chroma or calling the API
docker compose exec api uv run python scripts/reindex_chunks.py --dry-run
```

The ingestion pipeline auto-syncs chunks for every page it writes, so routine use does not require manual reindexing.

---

### Recover from accidental wiki/ deletion

**If `data/wiki/` is empty or missing:**

```bash
ls /app/data/wiki/   # inside container — confirms deletion
```

Options, from best to worst:

1. **Restore from backup** (if you snapshot `/data/` periodically): restore the snapshot volume and restart.

2. **Re-run ingestion for all raw files** — raw files in `data/raw/` are preserved.
   ```bash
   # List all raw files that have been ingested
   docker compose exec api uv run python -c "
   import asyncio
   from llm_wiki.api.deps import _engine
   from sqlalchemy.ext.asyncio import async_sessionmaker
   from sqlalchemy import select
   from llm_wiki.storage.metadata import FileRecord

   async def list_files():
       sm = async_sessionmaker(bind=_engine, expire_on_commit=False)
       async with sm() as session:
           rows = await session.execute(select(FileRecord))
           for r in rows.scalars():
               print(r.file_id, r.original_name, r.status)

   asyncio.run(list_files())
   "
   # Then re-enqueue each file_id that had status DONE
   ```
   > **Warning:** The Writer Agent generates wiki pages non-deterministically. Re-ingesting the same raw files will produce different wiki pages than the originals.

3. **`scripts/replay_all.py`** — **TODO: not implemented. Tracked as a future sprint task.**

After recovery:
```bash
docker compose exec api uv run python scripts/reindex.py
docker compose exec api uv run python scripts/reindex_chunks.py
```

---

### Recover from accidental index.md corruption

**Symptoms:** Streamlit index tree is empty or malformed. `IndexStorage.read_headings()` returns 0 entries.

**Diagnosis:**
```bash
docker compose exec api head -30 /app/data/index.md
```

**Fix:**

If `index.md` is structurally broken (e.g., truncated mid-line):
```bash
# Rebuild from wiki/ files — adds every .md file to a single "General" section
docker compose exec api uv run python -c "
from llm_wiki.config import settings
from llm_wiki.storage.index import IndexStorage

index = IndexStorage(settings.index_path)
# Read existing headings (may be empty or broken)
# Add back every wiki page under General
for path in sorted(settings.wiki_dir.glob('*.md')):
    slug = path.stem
    title = slug.replace('-', ' ').title()
    # Read first # heading for a better title
    for line in path.read_text(encoding='utf-8').splitlines():
        if line.startswith('# '):
            title = line[2:].strip()
            break
    index.add_page(slug, 'General', title=title)
    print(f'Added {slug}')
print('index.md rebuilt.')
"
# Then rebuild ChromaDB headings
docker compose exec api uv run python scripts/reindex.py
```

**Verification:**
```bash
docker compose exec api uv run python -c "
from llm_wiki.config import settings
from llm_wiki.storage.index import IndexStorage
index = IndexStorage(settings.index_path)
headings = index.read_headings()
print(f'{len(headings)} headings in index')
"
```

---

## Cost & quota

### When budget alert fires (LW-19)

**Symptoms:** Worker logs show `budget_exceeded` at level `error`. Files start failing at the SEARCHED or WRITTEN step with `BudgetExceeded`. `POST /files` returns 503 with "Daily budget exceeded".

**Find the alert:**
```bash
docker compose logs api worker | jq 'select(.event == "budget_exceeded")'
# Shows: kind (cost|tokens), cost_today_usd, cost_limit_usd
```

**Check current spend:**
```bash
curl -s http://localhost:8000/api/v1/stats | jq '{cost_today_usd, budget_cost_limit_usd, budget_cost_used_pct}'
```

**Fix — raise the limit for today:**
```bash
# Edit .env: DAILY_COST_LIMIT_USD=10.0
docker compose up -d api worker
```

**Root cause — find who spent the budget:**
See [Daily cost spike — diagnosis from usage.log](#daily-cost-spike--diagnosis-from-usagelog) below for the jq breakdown by agent type.

---

### Daily cost spike — diagnosis from usage.log

**Symptoms:** `curl /api/v1/stats | jq .cost_today_usd` is higher than expected.

**Diagnosis:**

```bash
# Break down cost by agent type (writer, search, answer, embed, audit)
docker compose exec api sh -c "
cat /app/data/usage.log | python3 -c \"
import sys, json
from collections import defaultdict
totals = defaultdict(float)
counts = defaultdict(int)
for line in sys.stdin:
    line = line.strip()
    if not line: continue
    r = json.loads(line)
    k = r.get('agent_type', 'unknown')
    totals[k] += r.get('cost_usd', 0)
    counts[k] += 1
for k in sorted(totals, key=totals.get, reverse=True):
    print(f'{k:20s}  calls={counts[k]:5d}  cost=\${totals[k]:.4f}')
\"
"

# Top 10 most expensive individual ingestions
docker compose exec api sh -c "
cat /app/data/usage.log | python3 -c \"
import sys, json
from collections import defaultdict
by_file = defaultdict(float)
by_name = {}
for line in sys.stdin:
    line = line.strip()
    if not line: continue
    r = json.loads(line)
    fid = r.get('file_id', '')
    by_file[fid] += r.get('cost_usd', 0)
for fid, cost in sorted(by_file.items(), key=lambda x: x[1], reverse=True)[:10]:
    print(f'\${cost:.4f}  {fid}')
\"
"
```

**Fix:**

- If cost is from `audit` calls: the weekly Auditor ran (expected on Monday). Check `data/issues.md` for results.
- If cost is from `writer` calls on unexpected ingestions: check `data/log.md` for recent entries.
- If the spike is ongoing: see [Emergency: disable ingestion](#emergency-disable-ingestion).

**Verification:**
```bash
curl -s http://localhost:8000/api/v1/stats | jq '{cost_today_usd, cost_total_usd}'
```

---

### How to read usage.log

Each line is a JSON record:

```json
{
  "ts": "2026-06-09T01:00:00Z",
  "file_id": "01HXYZ...",
  "agent_type": "writer",
  "model": "gpt-4o-mini",
  "prompt_tokens": 1200,
  "completion_tokens": 400,
  "cost_usd": 0.00048
}
```

Fields: `ts` (ISO timestamp), `file_id` (correlation ID), `agent_type` (`writer`/`search`/`answer`/`embed`/`audit`), `model`, token counts, `cost_usd`.

Quick inspection:
```bash
# Last 20 calls
docker compose exec api tail -n 20 /app/data/usage.log | jq .

# Total spend today
docker compose exec api sh -c "
TODAY=$(date -u +%Y-%m-%d)
cat /app/data/usage.log | python3 -c \"
import sys, json
total = sum(json.loads(l).get('cost_usd', 0) for l in sys.stdin if l.strip() and '$TODAY' in l)
print(f'Today: \${total:.4f}')
\"
"
```

---

### Emergency: disable ingestion

Use one of these escalating options depending on severity:

**Option 1 — Stop the worker (soft pause).**
`POST /files` continues to accept files; they queue in Redis. Good for short pauses (< 1 h).
```bash
docker compose stop worker
# Resume:
docker compose start worker
```

**Option 2 — Drain the Redis queue (discard pending tasks).**
Stops all queued ingestions. Files already accepted are lost from the queue but exist in `metadata.db` as `RECEIVED`. They can be re-enqueued later.
```bash
docker compose exec redis redis-cli DEL celery
```

**Option 3 — Environment-based kill switch.**
Set `INGESTION_ENABLED=false` in `.env` and restart the API to return 503 from `POST /files`.
```bash
# Edit .env: INGESTION_ENABLED=false
docker compose up -d api
# Verify:
curl -s -o /dev/null -w "%{http_code}" -X POST http://localhost:8000/api/v1/files \
  -F "file=@/dev/null"
# Expected: 503
```

**Option 4 — Full stack pause.**
```bash
docker compose stop worker beat api
# Resume:
docker compose start api worker beat
```

---

## Quality system

### Run Linter manually

```bash
# Preview findings without modifying issues.md
docker compose exec api uv run python scripts/run_linter.py --dry-run

# Run all checks and update issues.md (same code as the post-ingest pipeline step)
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

**Via API:**
```bash
# Synchronous, instant
curl -X POST http://localhost:8000/api/v1/lint/run
curl -X POST "http://localhost:8000/api/v1/lint/run?dry_run=true&checks=dead_link,orphan_page"
```

---

### Run Auditor manually

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

# Debug against a contradiction fixture
docker compose exec api uv run python scripts/run_auditor.py \
  --sync --wiki-dir tests/fixtures/sample_wikis/contradiction/ --dry-run
```

The script **always prints the estimated cost before making LLM calls** and asks for confirmation if the estimate exceeds $1.00.

**Via API:**
```bash
# LLM Auditor (async Celery task)
TASK=$(curl -s -X POST http://localhost:8000/api/v1/audit/run \
  -H "Content-Type: application/json" \
  -d '{"mode": "sync", "sample": 10}' | python3 -c "import sys,json; print(json.load(sys.stdin)['task_id'])")

# Poll for result
curl http://localhost:8000/api/v1/audit/$TASK
```

**Weekly automatic audit:**

Celery Beat runs the Auditor every Monday at 03:00 UTC via the `weekly-audit` beat schedule entry.
```bash
docker compose logs beat --tail=20
docker compose logs worker --tail=50 | grep weekly_audit

# Trigger immediately without waiting for the schedule
docker compose exec api uv run python -c "
from llm_wiki.orchestrator.tasks import run_weekly_audit
run_weekly_audit.delay(mode='sync')
"
```

---

### Interpret issues.md

`data/issues.md` has two sections:

| Section | Source | How to act |
|---|---|---|
| `## 🔎 Auto-detected` | Deterministic Linter (runs after every ingest) | Fix dead links / orphan pages in the referenced wiki files |
| `## 🤖 LLM-flagged` | LLM Auditor (weekly or manual) | Review each finding — some are false positives; dismiss by editing the wiki page |

Issues are deduplicated by a `slug:kind` key. Re-running the Linter or Auditor overwrites each section in-place.

---

### Auditor batch job stuck

**Symptoms:** `GET /api/v1/audit/{task_id}` returns `PENDING` for > 2 hours after a batch run was started.

**Diagnosis:**
```bash
# Check if the Celery task is still running
docker compose logs worker --tail=100 | grep audit

# Check if OpenAI batch job was submitted (the job ID is logged)
docker compose logs worker | jq 'select(.event | test("batch"))' | tail -20
```

**Fix:**
```bash
# Re-run in sync mode as a fallback
docker compose exec api uv run python scripts/run_auditor.py --sync --sample 20
```

---

## Q&A endpoint (LW-20)

### Ask endpoint returns "low confidence" for everything

**Symptoms:** Every `POST /api/v1/ask` response has `"confidence": "low"` and `"sources": []`, even for questions with obvious answers in the wiki.

**Diagnosis:**
```bash
# Check if the chunks collection has any data
docker compose exec api uv run python -c "
from llm_wiki.config import settings
from llm_wiki.llm.chunk_store import ChunkStore
from llm_wiki.llm.client import LLMClient
llm = LLMClient()
store = ChunkStore(chroma_path=settings.chroma_dir, llm_client=llm)
print('Chunk count:', store.count())
"

# Also check headings collection
docker compose exec api uv run python -c "
from llm_wiki.config import settings
from llm_wiki.llm.embeddings import EmbeddingStore
from llm_wiki.llm.client import LLMClient
llm = LLMClient()
store = EmbeddingStore(chroma_path=settings.chroma_dir, llm_client=llm)
print('Heading count:', store.count())
"
```

**Fix:**

- If chunk count is 0: run `reindex_chunks.py` (LW-20.1 was deployed but chunks were never indexed).
  ```bash
  docker compose exec api uv run python scripts/reindex_chunks.py
  ```
- If heading count is also 0: run both reindex scripts.
- If counts look correct but confidence is still low: the question may genuinely not be covered by the wiki. Try with a question whose answer you know is in a specific page.

**Verification:**
```bash
# Ask about something definitely in the wiki
curl -s -X POST http://localhost:8000/api/v1/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What pages are in the wiki?", "top_k": 5}' | jq .confidence
```

---

### Ask endpoint slow

**Symptoms:** `POST /api/v1/ask` takes > 15 seconds. Streamlit shows a timeout spinner.

**Diagnosis:**

The endpoint has two latency sources: embedding API call and LLM completion call.

```bash
# Enable DEBUG logs to see timing
docker compose logs api --tail=50 | jq 'select(.file_id == "ask")'
```

Look for `ask_chunk_query_failed` (embedding failure), or a long gap between the embedding call and the LLM call.

**Fix:**

- **Embedding API slow:** The embedding call is synchronous (wrapped in an async executor). Check OpenAI status page. No code fix needed — it will resolve when the API is healthy.
- **LLM call slow:** Normal for large context or slow models. Reduce `top_k` in the request, or switch to a faster model (`LLM_PROVIDER`, `LLM_MODEL`).
- **Ollama model cold-started:** The first request after a long idle period may take 10–30 s while Ollama loads the model into VRAM. Subsequent requests are fast.

```bash
# Test embedding latency in isolation
docker compose exec api uv run python -c "
import time
from llm_wiki.llm.client import LLMClient
llm = LLMClient()
t0 = time.time()
v = llm.embed(['test query'])
print(f'Embed latency: {time.time()-t0:.2f}s, dim={len(v[0])}')
"
```

---

## Deployment

### Cold start checklist

1. **Verify `.env` file** contains all required variables:
   ```bash
   # Minimum required set for OpenAI provider:
   grep -E "^(OPENAI_API_KEY|LLM_PROVIDER|WIKI_LANGUAGE)" .env
   ```
   Required: `OPENAI_API_KEY` (if `LLM_PROVIDER=openai`), `LLM_PROVIDER`.

2. **Start all services and wait for healthchecks:**
   ```bash
   docker compose up -d
   sleep 30
   docker compose ps  # all should show (healthy)
   ```

3. **Initialise data directory** (first time only, idempotent):
   ```bash
   docker compose exec api uv run python scripts/init_wiki.py
   ```

4. **If `data/index.md` was imported from a backup, rebuild ChromaDB:**
   ```bash
   docker compose exec api uv run python scripts/reindex.py
   docker compose exec api uv run python scripts/reindex_chunks.py
   ```

5. **Smoke test:**
   ```bash
   # Upload a test file
   curl -s -X POST http://localhost:8000/api/v1/files \
     -F "file=@tests/fixtures/sample.md" | jq .file_id

   # Wait ~30 s, check status
   curl -s http://localhost:8000/api/v1/files/<FILE_ID> | jq .status

   # Ask a question
   curl -s -X POST http://localhost:8000/api/v1/ask \
     -H "Content-Type: application/json" \
     -d '{"question": "What is in the wiki?"}' | jq .confidence
   ```

---

### After changing EMBEDDING_MODEL or EMBEDDING_DIMENSIONS

1. Update `.env` with the new values.
2. Stop the worker (prevents writes to the stale collection):
   ```bash
   docker compose stop worker
   ```
3. Rebuild both collections:
   ```bash
   docker compose exec api uv run python scripts/reindex.py
   docker compose exec api uv run python scripts/reindex_chunks.py
   ```
4. Restart the worker:
   ```bash
   docker compose start worker
   ```
5. Verify:
   ```bash
   docker compose logs worker --tail=20 | jq 'select(.level == "error")'
   # Expected: no EmbeddingModelMismatchError
   ```

---

### After restoring a backup of /data

1. Stop all services: `docker compose down`
2. Restore the `/data/` volume from backup.
3. Start services: `docker compose up -d`
4. Rebuild both ChromaDB collections (Chroma state in `data/chroma/` may be stale or from a different model):
   ```bash
   docker compose exec api uv run python scripts/reindex.py
   docker compose exec api uv run python scripts/reindex_chunks.py
   ```
5. Run the Linter to check data consistency:
   ```bash
   docker compose exec api uv run python scripts/run_linter.py --dry-run
   ```
6. Verify the wiki page count matches expectations:
   ```bash
   curl -s http://localhost:8000/api/v1/stats | jq .page_count
   ```

---

### Bump a single image without downtime

```bash
# Rebuild only the api image and replace the running container
docker compose up -d --no-deps --build api

# Worker (no downtime — in-flight tasks complete before the old worker exits)
docker compose up -d --no-deps --build worker

# Viewer (Streamlit)
docker compose up -d --no-deps --build viewer
```

---

## Appendix: common error signatures

### `EmbeddingModelMismatchError`

```
EmbeddingModelMismatchError: headings collection was built with model 'text-embedding-ada-002',
but config says 'text-embedding-3-small'. Run: reindex.py
```

**Cause:** `EMBEDDING_MODEL` changed in `.env` after the ChromaDB collection was already populated.

**Fix:** See [After changing EMBEDDING_MODEL or EMBEDDING_DIMENSIONS](#after-changing-embedding_model-or-embedding_dimensions).

---

### `RuntimeError: Event loop is closed`

```
RuntimeError: Event loop is closed
  File ".../httpx/_transports/asyncio.py", line ...
```

**Cause:** `LLMClient.aclose()` was not called within the same event loop that created the `httpx` client, or the loop was torn down before cleanup completed. This was a known issue in Celery tasks (fixed in LW-20 `aclose()` rewrite).

**Fix:** Ensure `await llm.aclose()` is called in the `finally:` block of every Celery task and FastAPI route that uses `LLMClient`. Do not share a single `LLMClient` instance across event loop boundaries.

---

### `FileLock timeout` on index.md

```
filelock.Timeout: The file lock '.../index.md.lock' could not be acquired
```

**Cause:** Two Celery workers tried to write to `index.md` simultaneously. The lock timeout (default 10 s) was exceeded.

**Symptoms:** One of the concurrent ingestion tasks fails at the WRITTEN step.

**Fix:**
```bash
# Remove the stale lock file (safe when no write is actually in progress)
docker compose exec api rm -f /app/data/index.md.lock
# Re-enqueue the failed task
docker compose exec api uv run python -c "
from llm_wiki.orchestrator.tasks import process_file_task
process_file_task.delay('<FILE_ID>')
"
```

**Prevention:** Run with `CELERY_WORKER_CONCURRENCY=1` (default) to serialise all writes. Parallel concurrency requires a distributed lock (future improvement).

---

### `openai.RateLimitError` persisting

```
openai.RateLimitError: Error code: 429 - Rate limit exceeded for ...
```

**Cause:** The OpenAI account hit its per-minute or daily token limit.

**Diagnosis:**
```bash
# How much have we spent today?
curl -s http://localhost:8000/api/v1/stats | jq .cost_today_usd

# Which agent type is burning tokens?
docker compose exec api sh -c "cat /app/data/usage.log | \
  python3 -c \"import sys,json; [print(json.loads(l).get('agent_type'), json.loads(l).get('cost_usd')) for l in sys.stdin if l.strip()]\""
```

**Fix:**
1. Wait for the rate limit window to reset (usually 1 minute for RPM, midnight UTC for daily quota).
2. If the daily quota is hit, stop the worker until the next day: `docker compose stop worker`.
3. **Budget alerts (LW-19)** will add automatic enforcement. **TODO: requires LW-19 implementation.**

**Prevention:** Set `DAILY_TOKEN_BUDGET` in `.env` once LW-19 is implemented.
