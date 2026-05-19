.PHONY: install dev test lint run clean

install:
	pip install -e .

dev:
	pip install -e ".[dev,notebook]"

test:
	pytest tests/ -v --tb=short

lint:
	ruff check src/ tests/
	mypy src/

run:
	python -m quantmind.cli --ticker SPY --years 3

run-ai:
	python -m quantmind.cli --ticker SPY --years 3 --use-ai

clean:
	rm -rf output/*.png build/ dist/ *.egg-info src/*.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
