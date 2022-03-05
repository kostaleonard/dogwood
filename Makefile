all: help

help:
	@echo "To install required packages, run 'make install' from a clean 'python:3.9' (or higher) conda environment."

install:
	pip install -r requirements.txt

lint:
	pylint dogwood
	pylint tests

test:
	pytest -m "not slowtest" --cov=dogwood tests
	coverage xml

test_slow:
	pytest -m "slowtest" --cov=dogwood tests
	coverage xml

test_full:
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
