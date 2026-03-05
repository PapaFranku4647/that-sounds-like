.PHONY: install install-ml format lint test

install:
	python -m pip install --upgrade pip
	pip install -e ".[dev]"

install-ml:
	python -m pip install --upgrade pip
	pip install -e ".[dev,ml,analytics]"

format:
	ruff format src tests

lint:
	ruff check src tests

test:
	python -m pytest
