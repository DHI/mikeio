# Run all checks: lint, typecheck, test
check: lint typecheck test

# Build package (after typecheck and test)
build: typecheck test
    uv build

# Lint with ruff
lint:
    uv run ruff check .

# Auto-fix formatting
format:
    uv run ruff format mikeio/

# Run tests
test:
    uv run pytest

# Type check with mypy
typecheck:
    uv run mypy .

# Generate HTML coverage report
coverage:
    uv run pytest --cov-report html --cov=mikeio tests/

# Run performance tests
perftest:
    uv run pytest tests/performance/ --durations=0

# Build documentation
docs:
    cd docs && uv run quartodoc build && uv run quarto render
    test -f docs/_site/index.html || { echo "Error: index.html not found. Quarto render failed."; exit 1; }
    cd docs && uv run python generate_llms_txt.py
