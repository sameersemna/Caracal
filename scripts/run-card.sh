#!/usr/bin/env bash
# Run one Kilo worker round on one inbox card.
# Usage: scripts/run-card.sh agents/inbox/NN-<ID>.md [max_restarts]
# The worker is restarted (same card) if it exits without writing its outbox report,
# e.g. after a crash or context limit. Output goes to agents/logs/NN.log.
set -euo pipefail
cd "$(dirname "$0")/.."

card="${1:?usage: run-card.sh agents/inbox/NN-ID.md [max_restarts]}"
max="${2:-4}"
[ -f "$card" ] || { echo "no such card: $card" >&2; exit 1; }
nn="$(basename "$card" | cut -d- -f1)"
report="agents/outbox/${nn}-report.md"
log="agents/logs/${nn}.log"
mkdir -p agents/logs agents/outbox

export KILO_CONFIG="${KILO_CONFIG:-/home/sameer/Public/Shared/Work/Services/MCP/hybrid-vision-mcp/.kilo/profiles/ollama_deepseek.jsonc}"
# Kilo's profile reads {env:OLLAMA_API_KEY} for its own model calls. If unset, take the ollama.com key from .env.
if [ -z "${OLLAMA_API_KEY:-}" ] && [ -f .env ]; then
  OLLAMA_API_KEY="$(grep -E '^OLLAMA_CLOUD_API_KEY=' .env | head -1 | cut -d= -f2- | sed -E 's/^"(.*)"$/\1/; s/^'"'"'(.*)'"'"'$/\1/')"
  export OLLAMA_API_KEY
fi
[ -n "${OLLAMA_API_KEY:-}" ] || { echo "OLLAMA_API_KEY is not set and .env has no OLLAMA_CLOUD_API_KEY" >&2; exit 1; }
# The worker must never reach real services from tests: point every live endpoint at a dead port.
# (Process env overrides .env in pydantic-settings.) Kilo's own model traffic uses KILO_CONFIG, not these.
export WHISPER_URL=http://127.0.0.1:9 OLLAMA_CLOUD_URL=http://127.0.0.1:9 OLLAMA_LOCAL_URL=http://127.0.0.1:9
export OLLAMA_CLOUD_API_KEY=
export DATABASE_URL=postgresql+asyncpg://nobody:nobody@127.0.0.1:9/none REDIS_URL=redis://127.0.0.1:9/0
export BRAVE_API_KEY= EXA_API_KEY= HF_TOKEN=

for i in $(seq 1 "$max"); do
  if git cat-file -e "HEAD:$report" 2>/dev/null || [ -f "$report" ]; then
    echo "report present: $report"; break
  fi
  echo "=== $(date -Is) round $nn attempt $i (card $card)" | tee -a "$log"
  kilo run --auto --agent=caracal-worker --dir "$PWD" \
    "You are on round ${nn}. Your card is ${card}. Follow your instructions from Step 1. If a previous attempt of this round left work on the task branch, continue it instead of starting over. Do not stop until ${report} is written and committed." \
    </dev/null 2>&1 | tee -a "$log" || true
  sleep 5
done

if [ -f "$report" ]; then
  echo "=== worker finished: $(head -1 "$report")" | tee -a "$log"
else
  echo "=== worker did not produce $report after $max attempts" | tee -a "$log"; exit 3
fi
