
build: test
	#python setup.py sdist bdist_wheel
	python -m build

test:
	pytest --disable-warnings

doctest:
	pytest mikeio/dfs*.py mikeio/dfsu/*.py mikeio/eum.py mikeio/pfs/*.py mikeio/spatial/grid_geometry.py --doctest-modules
	rm -f *.dfs* # remove temporary files, created from doctests

perftest:
	pytest tests/performance/ --durations=0

typecheck:
	mypy mikeio/dataarray.py mikeio/dataset.py

coverage: 
	pytest --cov-report html --cov=mikeio tests/

docs: FORCE
	cd docs; make html ;cd -

FORCE:
