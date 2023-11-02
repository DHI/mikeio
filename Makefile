LIB = mikeio

LIB = mikeio

check: lint typecheck test

build: typecheck test
	python -m build

lint:
	ruff .

test:
	pytest --disable-warnings

typecheck:
	mypy $(LIB)/ --config-file pyproject.toml

coverage: 
	pytest --cov-report html --cov=$(LIB) tests/

doctest:
	# only test a specific set of files for now
	pytest mikeio/dfs/*.py mikeio/dfsu/*.py mikeio/eum/*.py mikeio/pfs/*.py mikeio/spatial/_grid_geometry.py --doctest-modules
	rm -f *.dfs* # remove temporary files, created from doctests

perftest:
	pytest tests/performance/ --durations=0

docs: FORCE
	cd docs; make html ;cd -

FORCE:


