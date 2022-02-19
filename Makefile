all: help

help:
	@echo "To install required packages, run 'make install' from a clean 'python:3.6' (or higher) conda environment."

install:
	pip install -r requirements.txt

lint:
	pylint dogwood
	pylint tests

test:
	pytest --cov=dogwood tests
	coverage xml

package_prod:
	rm -rf dist
	python3 -m build
	python3 -m twine upload dist/*

package_test:
	rm -rf dist
	python3 -m build
	python3 -m twine upload --repository testpypi dist/*
