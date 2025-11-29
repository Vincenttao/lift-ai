PYTHON ?= python

.PHONY: install-dev test

install-dev:
	$(PYTHON) -m pip install -e .[dev]

test:
	$(PYTHON) -m pytest
