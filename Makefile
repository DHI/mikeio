
build: test
	python setup.py sdist bdist_wheel

test:
	pytest --disable-warnings

coverage: 
	pytest --cov-report html --cov=mikeio tests/

docs: FORCE
	cd docs; make html ;cd -

FORCE:
