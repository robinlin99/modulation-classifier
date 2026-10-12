.PHONY: help install install-dev test lint format clean docs build upload tensorboard

# Default target
help:
	@echo "Available commands:"
	@echo "  install     Install the package"
	@echo "  install-dev Install in development mode with dev dependencies"
	@echo "  test        Run tests"
	@echo "  lint        Run linting (flake8 + mypy)"
	@echo "  format      Format code with black"
	@echo "  clean       Remove build artifacts"
	@echo "  docs        Build documentation"
	@echo "  build       Build distribution packages"
	@echo "  upload      Upload to PyPI (requires credentials)"
	@echo "  tensorboard Launch TensorBoard on ./runs (live training curves)"

# Live training curves (open http://localhost:6006)
tensorboard:
	tensorboard --logdir runs

# Installation
install:
	pip install -e .

install-dev:
	pip install -e ".[dev,docs]"

# Testing
test:
	pytest tests/ -v --cov=modulation_classifier --cov-report=html --cov-report=term-missing

test-fast:
	pytest tests/ -v -x

# Code quality
lint:
	flake8 src tests examples
	mypy src/modulation_classifier

format:
	black src tests examples
	isort src tests examples

format-check:
	black --check src tests examples
	isort --check-only src tests examples

# Cleaning
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	find . -type d -name __pycache__ -delete
	find . -type f -name "*.pyc" -delete

# Documentation
docs:
	cd docs && make html

docs-clean:
	cd docs && make clean

# Building and uploading
build: clean
	python -m build

upload-test: build
	python -m twine upload --repository testpypi dist/*

upload: build
	python -m twine upload dist/*

# Development setup
setup-dev: install-dev
	pre-commit install

# Example usage
example:
	cd examples && python example_usage.py --minimal

example-full:
	cd examples && python example_usage.py

# Training shortcuts (adjust paths as needed)
train-demo:
	modulation-train --data data.hdf5 --labels train_labels.csv --epochs 10 --batch-size 32

predict-demo:
	modulation-predict --model best_model.pt --data test_data.hdf5 --output predictions.csv