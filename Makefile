.PHONY: install test lint bench bench-fast bench-ragas bench-public bench-qdrant evals clean

install:
	uv sync --extra dev

test:
	uv run pytest tests/unit tests/integration -v

lint:
	uv run ruff check src/ tests/

bench:
	uv sync --extra benchmark --extra local
	uv run python scripts/benchmark.py --n 10 --skip-ragas

bench-fast:
	uv sync --extra benchmark --extra local
	uv run python scripts/benchmark.py --n 10 --fast-eval

bench-ragas:
	uv sync --extra benchmark --extra local
	uv run python scripts/benchmark.py --n 10

bench-public:
	uv sync --extra benchmark --extra local
	uv run python benchmarks/eval.py --dataset natural_questions --n 100

bench-qdrant:
	uv sync --extra benchmark --extra local --extra qdrant
	uv run python benchmarks/eval.py --dataset natural_questions --n 100 --include-qdrant --output docs/qdrant-benchmarks.md

evals:
	uv run pytest tests/evals -v

clean:
	rm -rf .venv __pycache__ .pytest_cache dist
