.PHONY: check lint types test demo study

check: lint types test

lint:
	uv run ruff check src tests

types:
	uv run mypy src

test:
	uv run pytest

demo:
	uv run asgn demo

study:
	uv run asgn study --seeds 10
