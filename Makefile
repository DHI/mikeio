LIB = mikeio

check: lint typecheck test

build: typecheck test
	python -m build

lint:
	ruff check .

format:
	ruff format $(LIB)/

test:
	pytest --disable-warnings

typecheck:
	mypy $(LIB)/

coverage: 
	pytest --cov-report html --cov=$(LIB) tests/

doctest:
	# only test a specific set of files for now
	pytest mikeio/dfs/*.py mikeio/dfsu/*.py mikeio/eum/*.py mikeio/pfs/*.py mikeio/spatial/_grid_geometry.py --doctest-modules
	rm -f *.dfs* # remove temporary files, created from doctests

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


