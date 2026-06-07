# LLM Wiki

LLM-powered wiki ingestion system. Upload a PDF or Markdown file → agents extract knowledge,
synthesize wiki pages, maintain backlinks, and run weekly consistency checks.

## Quick Start (Testing — local Ollama)

```bash
# 1. Start all services (api, worker, beat, redis, ollama).
#    ollama-init pulls the model automatically on first run — no manual step needed.
docker compose -f docker-compose.yml -f docker-compose.testing.yml \
  --env-file .env.testing up --build

# 2. Initialize the data directory (first time only)
docker compose exec api uv run python scripts/init_wiki.py

# 3. Upload a file
curl -X POST http://localhost:8000/api/v1/files \
  -F "file=@/path/to/document.pdf"

# 4. Check processing status  (LW-10, coming soon)
curl http://localhost:8000/api/v1/files/{file_id}
```

> **Model size:** the testing profile defaults to `qwen2.5-coder:3b` (~2 GB).
> Override with `OLLAMA_MODEL=qwen2.5-coder:14b` in `.env.testing` for higher quality.

## Quick Start (Staging — OpenAI)

```bash
# 1. Copy the template and fill in your API key
cp .env.staging.example .env.staging.local
# edit OPENAI_API_KEY in .env.staging.local

# 2. Start (no --env-file flag needed — the staging override loads it directly)
docker compose -f docker-compose.yml -f docker-compose.staging.yml up --build
```

## 🌐 Wiki Viewer

A lightweight Streamlit UI for browsing the wiki without entering the Docker container.

### Access

After `docker compose up`, open: **http://localhost:8501**

### Features

- **Wiki Index** — clickable tree of all ingested pages
- **Wiki Page** — rendered markdown with internal link navigation
- **Changelog** — ingestion history from `log.md`, filterable by type
- **Stats** — page count, raw file count, total wiki size, cost summary
- Read-only — does not modify any data

> **Note:** This is a temporary dev/demo tool. A full Next.js UI is planned for Sprint 3.

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
| LW-10.1 | Streamlit wiki viewer (read-only UI for wiki, index, log) | ✅ Done |
| LW-11 | ChromaDB + embeddings infrastructure | ✅ Done |
| LW-12.1 | SHA-256 file deduplication (POST /files) | ✅ Done |
| LW-12 | Search Agent v2 (embedding pre-filter + LLM re-rank) | ✅ Done |
| LW-13 | Backlink mechanics (bidirectional ## Backlinks sync) | ✅ Done |
| LW-14 | Deterministic Linter (dead links, orphan pages, stale dates) | ✅ Done |
| LW-15 | LLM Auditor (contradictions, duplicates, suspected stale) + Celery Beat | ✅ Done |
| LW-16 | GET /wiki/{slug}, /log, /stats + Streamlit deep-linking | ✅ Done |
| LW-17 | Observability (structlog JSON logs + request_id) — OTel deferred | ✅ Done |
| LW-18 | Runbook | 🔲 |
| LW-20 | POST /api/v1/ask + Streamlit Q&A (AnswerAgent) | ✅ Done |
| LW-19 | Rate limiting + budget alerts | 🔲 |

## API Docs

Available at http://localhost:8000/docs after starting the stack.

## Embedding Index

Wiki headings are indexed into ChromaDB (`data/chroma/`) using `text-embedding-3-small`.
The Search Agent pre-filters candidates via cosine similarity before LLM re-ranking.

**Rebuild the index** (after first import or after changing `EMBEDDING_MODEL`):
```bash
docker compose exec api uv run python scripts/reindex.py
# Preview without writing:
docker compose exec api uv run python scripts/reindex.py --dry-run
```

> **Note:** If you change `EMBEDDING_MODEL` or `EMBEDDING_DIMENSIONS`, the service will refuse
> to start until you run `reindex.py` to rebuild the collection with the new model.

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

Or use the Streamlit UI: open http://localhost:8501 and click **❓ Спросить** in the sidebar.

The AnswerAgent never invents facts — if the wiki does not cover the question, the response has `confidence: "low"` and an empty `sources` list. All citations in the answer body are validated against the retrieved sources; hallucinated `[[slug]]` references are stripped automatically.

**Known limitation (v1):** retrieval uses heading-only embeddings (LW-11) plus a lightweight keyword fallback over full page bodies (with Russian/Kazakh stop-word filtering and prefix matching). Questions whose answer lives in the body of a page with an unrelated title may still miss. A chunk-level RAG upgrade is tracked as LW-20.1.

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
