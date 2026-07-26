#!/usr/bin/env bash
#
# Reset the LOCAL dev stack to a clean, empty state:
#   1. wipe the object store  (/app/data — raw uploads + any leftover files)
#   2. flush Redis            (Celery queue / result backend)
#   3. drop & recreate the Postgres schema (removes ALL rows AND tables)
#   4. restart api/worker/beat so the app rebuilds the schema (create_all)
#      and re-seeds skills + the allowed_users whitelist
#
# LOCAL ONLY. Operates on the docker-compose containers named "<project>-*"
# and localhost:8000 — it cannot reach any deployed / prod environment.
#
# Usage:   scripts/reset_local.sh
# Env:     COMPOSE_PROJECT (default "llm-wiki"), API_PORT (default 8000)
set -euo pipefail

PROJECT="${COMPOSE_PROJECT:-llm-wiki}"
PORT="${API_PORT:-8000}"
API="${PROJECT}-api-1"
WORKER="${PROJECT}-worker-1"
BEAT="${PROJECT}-beat-1"
PG="${PROJECT}-postgres-1"
REDIS="${PROJECT}-redis-1"

# Safety guard: only proceed if the local Postgres container is actually running.
if ! docker ps --format '{{.Names}}' | grep -qx "$PG"; then
  echo "✗ '$PG' is not running. Start the local stack first: docker compose up -d" >&2
  exit 1
fi

echo "→ 1/4  object store (/app/data)"
docker exec "$API" sh -c 'rm -rf /app/data/* 2>/dev/null || true' 2>/dev/null \
  && echo "   cleared" || echo "   (api down — dirs recreated on start)"

echo "→ 2/4  Redis (Celery queue)"
docker exec "$REDIS" redis-cli FLUSHALL >/dev/null && echo "   flushed"

echo "→ 3/4  Postgres schema (drop + recreate)"
docker exec "$PG" sh -c \
  'PGPASSWORD=$POSTGRES_PASSWORD psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" \
     -c "DROP SCHEMA public CASCADE; CREATE SCHEMA public;"' >/dev/null \
  && echo "   reset"

echo "→ 4/4  restart api/worker/beat (rebuild schema + seed)"
docker restart "$API" "$WORKER" "$BEAT" >/dev/null

printf "   waiting for api"
for _ in $(seq 1 40); do
  if [ "$(curl -s -o /dev/null -w '%{http_code}' "http://localhost:${PORT}/healthz" 2>/dev/null)" = "200" ]; then
    echo " ✓ healthy"
    break
  fi
  printf "."
  sleep 1
done

echo
docs=$(curl -s "http://localhost:${PORT}/api/v1/documents" -H 'X-User-Email: demo@bi.group')
cases=$(curl -s "http://localhost:${PORT}/api/v1/cases" -H 'X-User-Email: demo@bi.group')
echo "✓ Local stack reset."
echo "  documents: ${docs}"
echo "  cases:     ${cases}"
