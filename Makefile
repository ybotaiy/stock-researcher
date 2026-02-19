PYTHON := .venv/bin/python

.PHONY: eval test

eval:
	$(PYTHON) -m src.evals --strategy momentum

test:
	$(PYTHON) -m pytest tests/ -v
