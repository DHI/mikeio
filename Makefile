SHELL := /bin/bash

LIB = mikeio

check: lint typecheck test

build: typecheck test
	python -m build

lint:
	ruff check .

format:
	ruff format $(LIB)/

test:
	pytest

typecheck:
	mypy .

coverage: 
	pytest --cov-report html --cov=$(LIB) tests/

perftest:
	pytest tests/performance/ --durations=0

docs: FORCE
	set -e; \
	cd docs; \
	quartodoc build; \
	quarto render; \
	if [ ! -f _site/index.html ]; then \
        echo "Error: index.html not found. Quarto render failed."; \
        exit 1; \
    fi; \
    cd -


FORCE:


