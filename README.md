# LLM Wiki

LLM-powered wiki ingestion system. Upload a PDF or Markdown file → agents extract knowledge,
synthesize wiki pages, maintain backlinks, and run weekly consistency checks.

## Quick Start

The default provider is **OpenAI**. Copy the env template and add your API key.

```bash
# 1. Configure: copy the template and fill in OPENAI_API_KEY
cp .env.example .env
# edit OPENAI_API_KEY in .env

# 2. Start all services (api, worker, beat, redis)
docker compose up --build

# 3. Initialize the data directory (first time only)
docker compose exec api uv run python scripts/init_wiki.py

# 3b. Backfill chunk index for existing wiki pages (after deploy or if /ask is empty)
docker compose exec api uv run python scripts/reindex_chunks.py

# 4. Upload a file
curl -X POST http://localhost:8000/api/v1/files \
  -F "file=@/path/to/document.pdf"

# 5. Check processing status
curl http://localhost:8000/api/v1/files/{file_id}
```

The API is served at **http://localhost:8000** (OpenAPI docs at `/docs`). The
user-facing UI is the separate frontend repository.

### Staging

```bash
cp .env.staging.example .env.staging.local   # fill in OPENAI_API_KEY
docker compose -f docker-compose.yml -f docker-compose.staging.yml up --build
```

## Development Commands (all inside Docker)

```bash
# Run tests
docker compose exec api uv run pytest

# Lint
docker compose exec api uv run ruff check .

# Type check
docker compose exec api uv run mypy --strict src/

# Open a shell
docker compose exec api bash

# Run quality checks against current wiki
docker compose exec api uv run python scripts/run_linter.py
docker compose exec api uv run python scripts/run_auditor.py --sync --sample 5
```

## Architecture

```
User → POST /files → FastAPI → Celery Queue
                              ↓
                        Orchestrator (state machine)
                        RECEIVED → STORED → SEARCHED → WRITTEN → LINTED → LOGGED → DONE
                              ↓              ↓             ↓          ↓
                         parse file    Search Agent   Writer Agent  Linter
                         (pypdf/md)    (LLM)          (LLM)         (pure Python)
                                           ↓
                                       ChromaDB (LW-11)

Weekly Celery Beat (Mon 03:00 UTC):
                        Auditor Agent (LLM, Batch API) → issues.md ## LLM-flagged
```

## Task Map

| ID | Task | Status |
|----|------|--------|
| LW-1 | Repository skeleton | ✅ Done |
| LW-2 | Storage layer | ✅ Done |
| LW-3 | Parsers (PDF + MD) | ✅ Done |
| LW-4 | LLM client wrapper | ✅ Done |
| LW-5 | POST /files endpoint + Celery task | ✅ Done |
| LW-6 | Search Agent v1 | ✅ Done |
| LW-7 | Writer Agent — create page | ✅ Done |
| LW-8 | Writer Agent — update pages | ✅ Done |
| LW-9 | Orchestrator (state machine) | ✅ Done |
| LW-10 | GET /files/{id} status endpoint | ✅ Done |
| LW-10.1 | Wiki viewer (read-only UI for wiki, index, log) — removed; replaced by the frontend app | ✅ Done |
| LW-11 | ChromaDB + embeddings infrastructure | ✅ Done |
| LW-12.1 | SHA-256 file deduplication (POST /files) | ✅ Done |
| LW-12 | Search Agent v2 (embedding pre-filter + LLM re-rank) | ✅ Done |
| LW-13 | Backlink mechanics (bidirectional ## Backlinks sync) | ✅ Done |
| LW-14 | Deterministic Linter (dead links, orphan pages, stale dates) | ✅ Done |
| LW-15 | LLM Auditor (contradictions, duplicates, suspected stale) + Celery Beat | ✅ Done |
| LW-16 | GET /wiki/{slug}, /log, /stats | ✅ Done |
| LW-17 | Observability (structlog JSON logs + request_id) — OTel deferred | ✅ Done |
| LW-18 | Runbook | ✅ Done |
| LW-19 | Rate limiting + budget alerts (in-memory, single-replica) | ✅ Done |
| LW-20 | POST /api/v1/ask (AnswerAgent) | ✅ Done |
| LW-20.1 | Chunk-level retrieval for AnswerAgent (ChunkStore) | ✅ Done |

## API Docs

Available at http://localhost:8000/docs after starting the stack.

## Embedding Index

Two ChromaDB collections live in `data/chroma/`:

| Collection | One entry per… | Used by | Purpose |
|---|---|---|---|
| `headings` | Page title | `SearchAgent`, `IndexStorage` | Classify *which pages* a new document relates to |
| `chunks` | ~500-token body fragment | `AnswerAgent` | Retrieve *which paragraph* answers a user question |

They are intentionally separate: title embeddings and body-chunk embeddings are not
comparable (different text length, different semantics), and the SearchAgent needs
page-level IDs while the AnswerAgent needs passage-level text.

**Rebuild the headings index** (after first import or model change):
```bash
docker compose exec api uv run python scripts/reindex.py
docker compose exec api uv run python scripts/reindex.py --dry-run
```

**Rebuild the chunks index** (LW-20.1 — run on first deploy and after bulk wiki edits):
```bash
docker compose exec api uv run python scripts/reindex_chunks.py
docker compose exec api uv run python scripts/reindex_chunks.py --dry-run
```

> **Note:** If you change `EMBEDDING_MODEL` or `EMBEDDING_DIMENSIONS`, run *both*
> `reindex.py` and `reindex_chunks.py` to rebuild both collections.
>
> See [docs/lw-20.1-chunks.md](docs/lw-20.1-chunks.md) for the architectural rationale.

## Wiki Quality System (LW-14 + LW-15)

The quality system has two layers:

| Layer | What | When | Cost |
|-------|------|------|------|
| **Linter** (LW-14) | Dead links, orphan pages, stale dates | After every ingest | Zero (pure Python) |
| **Auditor** (LW-15) | Contradictions, duplicates, suspected stale | Weekly (Mon 03:00 UTC) | ~$0.003/page sync, ~$0.0015/page batch |

Both write to `data/issues.md` in separate sections (`## 🔎 Auto-detected` and `## 🤖 LLM-flagged`).

### Manual Linter run

```bash
# Preview findings without modifying issues.md
docker compose exec api uv run python scripts/run_linter.py --dry-run

# Run all checks and update issues.md
docker compose exec api uv run python scripts/run_linter.py

# Only specific checks
docker compose exec api uv run python scripts/run_linter.py --checks dead_link,orphan_page

# JSON output for CI
docker compose exec api uv run python scripts/run_linter.py --json

# Run against a fixture wiki
docker compose exec api uv run python scripts/run_linter.py \
  --wiki-dir tests/fixtures/sample_wikis/dead_links/ --dry-run
```

### Manual Auditor run

```bash
# Sync mode (immediate, for debugging) — DEFAULT
docker compose exec api uv run python scripts/run_auditor.py --sync --sample 10 --dry-run

# Specific pages
docker compose exec api uv run python scripts/run_auditor.py --sync --slugs llm,agents

# Production batch run (OpenAI Batch API, -50% cost, ~24 h)
docker compose exec api uv run python scripts/run_auditor.py --batch

# Guard against expensive accidental runs
docker compose exec api uv run python scripts/run_auditor.py --sync --max-cost-usd 0.50
```

### API endpoints

```bash
# Run Linter via API (synchronous, < 1 s)
curl -X POST http://localhost:8000/api/v1/lint/run
curl -X POST http://localhost:8000/api/v1/lint/run?dry_run=true

# Enqueue Auditor via API (async Celery task)
curl -X POST http://localhost:8000/api/v1/audit/run \
  -H "Content-Type: application/json" \
  -d '{"mode": "sync", "dry_run": true, "sample": 5}'

# Poll Auditor task status
curl http://localhost:8000/api/v1/audit/{task_id}
```

## Q&A endpoint

Ask the wiki a question and get a synthesised answer with citations:

```bash
curl -X POST http://localhost:8000/api/v1/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Что такое LoRA?", "top_k": 5}'
```

The AnswerAgent never invents facts — if the wiki does not cover the question, the response has `confidence: "low"` and an empty `sources` list. All citations in the answer body are validated against the retrieved sources; hallucinated `[[slug]]` references are stripped automatically.

**Retrieval (LW-20.1):** AnswerAgent retrieves over chunked page bodies (`chunks` collection) so that questions answered deep in a page body are found even when the page title does not match the query. The heading-only path (LW-20) is kept as a backward-compatible fallback when the chunks collection is empty.

## Logging

All services emit structured JSON logs to stdout. View them with:

```bash
docker compose logs -f api | jq
docker compose logs -f worker | jq
```

Every HTTP request gets a `request_id` (returned in the `X-Request-ID` response header
and present on every log line for that request). Every Celery file-ingestion task
auto-binds `file_id` to the log context, so filtering all logs for one file is trivial:

```bash
docker compose logs worker | jq 'select(.file_id == "01HXYZ...")'
```

Log level controlled by the `LOG_LEVEL` env var (default `INFO`).

## Cost Tracking

Every LLM call is logged to `data/usage.log` as JSON-lines with tokens and USD cost.
View aggregate stats: `GET /api/v1/stats`.

## Rate limiting & budget (LW-19)

Configure in `.env`:

| Variable | Default | Effect |
|---|---|---|
| `INGESTION_ENABLED` | `true` | Kill switch — set `false` to return 503 on `POST /files` |
| `INGESTION_RATE_LIMIT_PER_MIN` | `10` | Max uploads per minute per source IP |
| `ASK_RATE_LIMIT_PER_MIN` | `30` | Max `/ask` requests per minute per source IP |
| `DAILY_COST_LIMIT_USD` | _(empty = disabled)_ | Hard cap on total LLM spend per UTC day |
| `DAILY_TOKEN_LIMIT` | _(empty = disabled)_ | Hard cap on total tokens per UTC day |

When a daily limit is exceeded, `LLMClient` raises `BudgetExceeded` before any API call is made. The ingestion task transitions to `FAILED`. A structured log event `budget_exceeded` at level `error` is emitted — this is the alerting mechanism until Slack/webhook integration is added.

`POST /files` also checks the budget upfront and returns 503 immediately if the limit is already crossed, so no Celery task is enqueued.

Rate limiting is in-memory per API replica — sufficient for single-container deploys. For horizontal scaling, swap `InMemoryRateLimiter` for a Redis-backed implementation (planned as LW-19.1).
