PYTHON ?= python
SOURCE_PATHS := src tests scripts

.PHONY: lint format format-check test prepare-data train evaluate predict tune

lint:
	$(PYTHON) -m ruff check $(SOURCE_PATHS)

format:
	$(PYTHON) -m black $(SOURCE_PATHS)

format-check:
	$(PYTHON) -m black --check $(SOURCE_PATHS)

test:
	$(PYTHON) -m pytest

prepare-data:
	$(PYTHON) scripts/build_dataset_manifest.py

train evaluate predict tune:
	$(PYTHON) -m hccr.cli $@
