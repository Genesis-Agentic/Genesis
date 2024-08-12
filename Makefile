all: lint mypy test

lint:
	pylint genesis_agentic || true
	flake8 genesis_agentic tests || true
mypy:
	mypy genesis_agentic || true

test:
	python -m unittest discover -s tests


.PHONY: all lint mypy test
