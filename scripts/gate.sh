#!/usr/bin/env bash
# Caracal quality gate (docs/07 §1). Exits 0 only when every check passes.
set -euo pipefail
cd "$(dirname "$0")/.."

mkdir -p tests/unit tests/contract

echo "==> ruff check"
uv run ruff check src tests scripts

echo "==> ruff format --check"
uv run ruff format --check src tests scripts

echo "==> mypy (strict)"
uv run mypy src

echo "==> repository rules"
uv run python scripts/check_rules.py

echo "==> pytest (unit + contract, live excluded)"
uv run pytest tests/unit tests/contract -q

echo "GATE: PASS"
