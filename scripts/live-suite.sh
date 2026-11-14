#!/usr/bin/env bash
# One-command live suite (docs/07 §3): throwaway schema + Redis prefix + data dir, real
# `caracal serve` and `caracal worker` by PID, then `pytest -m live tests/live`.
# Always cleans up: kills the PIDs, drops the schema, deletes the prefix keys and the data dir.
# Exit code is pytest's. Run from anywhere; the script cd's to the repository root.
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

if [ -f .env ]; then
  # Real shell environment wins over .env: source .env first, then re-export the overrides below.
  set -a
  . ./.env
  set +a
fi

SCRATCH="${JUDGE_SCRATCH:-$(mktemp -d)}"
TAG="caracal_live_$(openssl rand -hex 4)"
PORT="${CARACAL_LIVE_PORT:-11497}"
DATA_DIR="$SCRATCH/live-data/$TAG"
LOG_DIR="$SCRATCH/live-$TAG"
SERVER_PID=""
WORKER_PID=""
RC=0

export CARACAL_DB_SCHEMA="$TAG"
export REDIS_PREFIX="$TAG:"
export CARACAL_DATA_DIR="$DATA_DIR"
export CARACAL_PORT="$PORT"
export CARACAL_PUBLIC_BASE_URL="http://127.0.0.1:$PORT"
export CARACAL_LIVE_BASE_URL="http://127.0.0.1:$PORT"
export CARACAL_LOG_LEVEL="${CARACAL_LOG_LEVEL:-INFO}"

PG_URL="${DATABASE_URL/+psycopg/}"
mkdir -p "$LOG_DIR"

cleanup() {
  local status=$?
  trap - EXIT INT TERM
  if [ -n "$WORKER_PID" ]; then
    kill -TERM "$WORKER_PID" 2>/dev/null || true
  fi
  if [ -n "$SERVER_PID" ]; then
    kill -TERM "$SERVER_PID" 2>/dev/null || true
  fi
  if [ -n "$WORKER_PID" ]; then
    wait "$WORKER_PID" 2>/dev/null || true
  fi
  if [ -n "$SERVER_PID" ]; then
    wait "$SERVER_PID" 2>/dev/null || true
  fi
  psql "$PG_URL" -qAtc "DROP SCHEMA IF EXISTS \"$TAG\" CASCADE" >/dev/null 2>&1 || true
  .venv/bin/python - "$TAG" <<'PY' || true
import os
import sys

import redis

prefix = f"{sys.argv[1]}:"
client = redis.Redis.from_url(os.environ["REDIS_URL"])
keys = list(client.scan_iter(match=f"{prefix}*", count=1000))
if keys:
    client.delete(*keys)
print(f"removed {len(keys)} redis keys under {prefix}")
PY
  rm -rf "$DATA_DIR"
  exit "$status"
}
trap cleanup EXIT INT TERM

echo "live suite: schema=$TAG prefix=$TAG: port=$PORT data=$DATA_DIR"
psql "$PG_URL" -qAtc "CREATE SCHEMA \"$TAG\"" || { echo "could not create schema" >&2; exit 2; }
if ! .venv/bin/caracal migrate >"$LOG_DIR/migrate.log" 2>&1; then
  echo "migration failed:" >&2
  cat "$LOG_DIR/migrate.log" >&2
  exit 2
fi
echo "migrate: ok"

.venv/bin/caracal serve >"$LOG_DIR/serve.log" 2>&1 &
SERVER_PID=$!
.venv/bin/caracal worker >"$LOG_DIR/worker.log" 2>&1 &
WORKER_PID=$!

READY=0
for _ in $(seq 1 60); do
  if curl -fsS "http://127.0.0.1:$PORT/healthz" >/dev/null 2>&1; then
    READY=1
    break
  fi
  sleep 0.5
done
if [ "$READY" -ne 1 ]; then
  echo "serve did not become healthy; logs:" >&2
  cat "$LOG_DIR/serve.log" "$LOG_DIR/worker.log" >&2
  exit 2
fi

set +e
.venv/bin/python -m pytest -m live tests/live "$@"
RC=$?
set -e

echo "live suite: pytest exit=$RC (logs in $LOG_DIR)"
exit "$RC"
