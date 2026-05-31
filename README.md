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
```

## Architecture

```
User → POST /files → FastAPI → Celery Queue
                              ↓
                        Orchestrator (state machine)
                        RECEIVED → STORED → SEARCHED → WRITTEN → LOGGED → DONE
                              ↓              ↓             ↓
                         parse file    Search Agent   Writer Agent
                         (pypdf/md)    (LLM)          (LLM)
                                           ↓
                                       ChromaDB (LW-11)
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
| LW-11 | ChromaDB + embeddings | 🔲 |
| LW-12 | Search Agent v2 (embedding pre-filter) | 🔲 |
| LW-13 | Backlink mechanics | 🔲 |
| LW-14 | Lint Agent v1 (rule-based) | 🔲 |
| LW-15 | Lint Agent v2 (LLM checks) + Celery Beat | 🔲 |
| LW-16 | GET /wiki, /log, /stats endpoints | 🔲 |
| LW-17 | Observability (structlog + OpenTelemetry) | 🔲 |
| LW-18 | Runbook | 🔲 |
| LW-19 | Rate limiting + budget alerts | 🔲 |

## API Docs

Available at http://localhost:8000/docs after starting the stack.

## Cost Tracking

Every LLM call is logged to `data/usage.log` as JSON-lines with tokens and USD cost.
View aggregate stats: `GET /api/v1/stats` (LW-16).
