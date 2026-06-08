#!/usr/bin/env bash
set -euo pipefail

# Resolve repo root and operate from there.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

COMPOSE_FILE="deploy/docker-compose.yml"

echo "Starting postgres and redis via $COMPOSE_FILE"
docker compose -f "$COMPOSE_FILE" up -d postgres redis

# Wait up to 30 seconds for postgres to be healthy.
CONTAINER_ID="$(docker compose -f "$COMPOSE_FILE" ps -q postgres)"
if [ -z "$CONTAINER_ID" ]; then
  echo "Error: could not resolve postgres container id." >&2
  exit 1
fi

echo -n "Waiting for postgres to be healthy"
HEALTHY=0
for _ in $(seq 1 30); do
  STATUS="$(docker inspect --format '{{.State.Health.Status}}' "$CONTAINER_ID" 2>/dev/null || echo "starting")"
  if [ "$STATUS" = "healthy" ]; then
    HEALTHY=1
    break
  fi
  echo -n "."
  sleep 1
done
echo ""

if [ "$HEALTHY" -ne 1 ]; then
  echo "Error: postgres did not become healthy within 30 seconds." >&2
  exit 1
fi

echo "Postgres is healthy."
echo "DATABASE_URL=postgres://finsight:finsight_dev@localhost:55432/finsight?sslmode=disable"
