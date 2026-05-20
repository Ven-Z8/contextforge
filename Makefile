.PHONY: install test lint bench evals clean

install:
	uv sync --extra dev

test:
	uv run pytest tests/unit tests/integration -v

lint:
	uv run ruff check src/ tests/

bench:
	uv run python scripts/benchmark.py

evals:
	uv run pytest tests/evals -v

clean:
	rm -rf .venv __pycache__ .pytest_cache dist
