#!/bin/bash
# Single entrypoint for all lint/type checks (ruff + ty) via uv.
# Usage: bash scripts/lint.sh
set -e
cd "$(dirname "$0")/.."
uv run --group dev ruff check src scripts tests
uv run --group dev ruff format --check src scripts tests
uv run --group dev ty check
echo "lint OK"
