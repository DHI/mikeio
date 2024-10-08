LIB = mikeio

check: lint typecheck test

build: typecheck test
	python -m build

lint:
	ruff check $(LIB)/

format:
	ruff format $(LIB)/

pylint:
	pylint --disable=all --enable=attribute-defined-outside-init mikeio/

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
	cd docs && quarto add --no-prompt .
	cd docs && quartodoc build
	cd docs && quartodoc interlinks
	quarto render docs

FORCE:


