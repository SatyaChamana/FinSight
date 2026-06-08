#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

: "${DATABASE_URL:=postgres://finsight:finsight_dev@localhost:55432/finsight?sslmode=disable}"

MIGRATIONS_DIR="internal/store/migrations"
MODE="${1:-up}"

case "$MODE" in
  up|down)
    ;;
  *)
    echo "Error: unknown mode '$MODE'. Use 'up' or 'down'." >&2
    exit 1
    ;;
esac

if [ ! -d "$MIGRATIONS_DIR" ]; then
  echo "Error: migrations directory '$MIGRATIONS_DIR' does not exist." >&2
  exit 1
fi

FILES=()
if [ "$MODE" = "up" ]; then
  # Lexicographic ascending order. Portable to bash 3.2 (macOS default).
  while IFS= read -r f; do FILES+=("$f"); done < <(find "$MIGRATIONS_DIR" -maxdepth 1 -type f -name '*.up.sql' | LC_ALL=C sort)
else
  # Lexicographic descending order.
  while IFS= read -r f; do FILES+=("$f"); done < <(find "$MIGRATIONS_DIR" -maxdepth 1 -type f -name '*.down.sql' | LC_ALL=C sort -r)
fi

if [ "${#FILES[@]}" -eq 0 ]; then
  echo "Error: no migration files found in '$MIGRATIONS_DIR' for mode '$MODE'." >&2
  exit 1
fi

for f in "${FILES[@]}"; do
  echo "applying $(basename "$f")" >&2
  psql "$DATABASE_URL" -v ON_ERROR_STOP=1 -f "$f"
done
