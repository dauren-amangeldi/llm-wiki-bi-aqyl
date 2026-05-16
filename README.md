# LLM Wiki

LLM-powered wiki ingestion system. Upload a PDF or Markdown file → agents extract knowledge,
synthesize wiki pages, maintain backlinks, and run weekly consistency checks.

## Quick Start (Testing — local Ollama)

```bash
# 1. Start all services (api, worker, beat, redis, ollama)
docker compose -f docker-compose.yml -f docker-compose.testing.yml \
  --env-file .env.testing up --build

# 2. Pull the LLM model (first time only)
docker compose exec ollama ollama pull qwen2.5-coder:14b

# 3. Initialize the data directory
docker compose exec api uv run python scripts/init_wiki.py

# 4. Upload a file
curl -X POST http://localhost:8000/api/v1/files \
  -F "file=@/path/to/document.pdf"

# 5. Check processing status
curl http://localhost:8000/api/v1/files/{file_id}
```

## Quick Start (Staging — OpenAI)

```bash
# Copy and fill in your API key
cp .env.staging .env.staging.local
# edit OPENAI_API_KEY in .env.staging.local

docker compose -f docker-compose.yml -f docker-compose.staging.yml \
  --env-file .env.staging.local up --build
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
```

## Architecture

```
User → POST /files → FastAPI → Celery Queue
                              ↓
                        Orchestrator (state machine)
                        RECEIVED → STORED → SEARCHED → WRITTEN → LOGGED → DONE
                              ↓              ↓             ↓
                         parse file    Search Agent   Writer Agent
                         (pypdf/md)    (GPT Mini)     (GPT Mini)
                                           ↓
                                       ChromaDB
```

## Task Map (Sprint 1–2)

| ID | Task | Status |
|----|------|--------|
| LW-1 | Repository skeleton | ✅ Done |
| LW-2 | Storage layer | 🔲 |
| LW-3 | Parsers (PDF + MD) | 🔲 |
| LW-4 | LLM client wrapper | 🔲 |
| LW-5 | POST /files endpoint | 🔲 |
| LW-6 | Search Agent v1 | 🔲 |
| LW-7 | Writer Agent (create) | 🔲 |
| LW-8 | Writer Agent (update) | 🔲 |
| LW-9 | Orchestrator | 🔲 |
| LW-10 | GET /files/{id} | 🔲 |

## API Docs

Available at http://localhost:8000/docs after starting the stack.

## Cost Tracking

Every LLM call is logged to `data/usage.log` as JSON-lines with tokens and USD cost.
View aggregate stats: `GET /api/v1/stats` (LW-16).
