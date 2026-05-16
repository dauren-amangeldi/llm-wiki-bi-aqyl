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

```bash
docker compose exec api uv run python scripts/reindex.py
```

## High LLM Cost Alert

Check `data/usage.log` for recent entries:
```bash
docker compose exec api tail -n 100 /app/data/usage.log | python -m json.tool
```
