LIB = mikeio

check: lint typecheck test

lint:
	ruff .

build: check
	python -m build

test:
	pytest --disable-warnings

doctest:
	# only test a specific set of files for now
	pytest mikeio/dfs/*.py mikeio/dfsu/*.py mikeio/eum/*.py mikeio/pfs/*.py mikeio/spatial/_grid_geometry.py --doctest-modules
	rm -f *.dfs* # remove temporary files, created from doctests

perftest:
	pytest tests/performance/ --durations=0

typecheck:
	mypy $(LIB)/

coverage: 
	pytest --cov-report html --cov=$(LIB) tests/

docs: FORCE
	cd docs; make html ;cd -

FORCE:
