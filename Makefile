.PHONY: install test smoke lint gate-detector

install:
	uv sync

test:
	PYTHONPATH=src uv run python -m unittest discover -s tests -t . -v

smoke:
	PYTHONPATH=src uv run python scripts/run_smoke.py --config configs/smoke/smoke.yaml --out artifacts/smoke-cpu

lint:
	bash scripts/lint.sh

# M03.1 laptop-verifiable gate (optional `ml` group: torch CPU + transformers).
# Needs the dataset cache from scripts/carve_splits.py --cache-dir /tmp/pii-redteam-cache.
gate-detector:
	uv run --group ml python scripts/modernbert_cpu_gate.py
