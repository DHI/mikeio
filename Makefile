SHELL := /bin/bash

LIB = mikeio

check:
	@$(MAKE) -j3 lint typecheck test

build: typecheck test
	uv build

lint:
	uv run ruff check .

format:
	uv run ruff format $(LIB)/

test:
	uv run pytest -n auto

typecheck:
	uv run mypy .

coverage: 
	uv run pytest --cov-report html --cov=$(LIB) tests/

perftest:
	uv run pytest tests/performance/ --durations=0

docs: FORCE
	set -e; \
	cd docs; \
	uv run quartodoc build; \
	uv run quarto render; \
	if [ ! -f _site/index.html ]; then \
        echo "Error: index.html not found. Quarto render failed."; \
        exit 1; \
    fi; \
    cd -


FORCE:


