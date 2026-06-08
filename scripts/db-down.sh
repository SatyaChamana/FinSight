#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

COMPOSE_FILE="deploy/docker-compose.yml"

WITH_VOLUMES=0
for arg in "$@"; do
  case "$arg" in
    --volumes)
      WITH_VOLUMES=1
      ;;
    *)
      echo "Unknown argument: $arg" >&2
      echo "Usage: $(basename "$0") [--volumes]" >&2
      exit 1
      ;;
  esac
done

if [ "$WITH_VOLUMES" -eq 1 ]; then
  echo "Stopping stack and removing volumes."
  docker compose -f "$COMPOSE_FILE" down -v
else
  echo "Stopping stack (preserving volumes)."
  docker compose -f "$COMPOSE_FILE" down
fi
