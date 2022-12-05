
build: test
	python setup.py sdist bdist_wheel

test:
	pytest --disable-warnings

doctest:
	pytest mikeio/dfsu.py --doctest-modules

coverage: 
	pytest --cov-report html --cov=mikeio tests/

docs: FORCE
	cd docs; make html ;cd -

FORCE:
