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
	pytest -m "slowtest and not veryslowtest" --cov=dogwood tests
	coverage xml

test_veryslow:
	pytest -m "veryslowtest" --cov=dogwood tests
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

hello_opencl:
	clang -framework OpenCL -o bin/hello_opencl dogwood/acceleration/hello_opencl.c
	./bin/hello_opencl

clean:
	rm -f bin/hello_opencl
