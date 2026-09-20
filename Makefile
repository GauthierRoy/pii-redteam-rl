.PHONY: install test smoke lint

install:
	uv sync

test:
	PYTHONPATH=src uv run python -m unittest discover -s tests -t . -v

smoke:
	PYTHONPATH=src uv run python scripts/run_smoke.py --config configs/smoke/smoke.yaml --out artifacts/smoke-cpu

lint:
	bash scripts/lint.sh
