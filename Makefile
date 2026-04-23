VENV   := .venv
PYTHON := $(VENV)/bin/python
PIP    := $(VENV)/bin/pip
ST     := $(VENV)/bin/streamlit

.PHONY: run install setup

run: $(ST)
	cd app && ../$(ST) run streamlit_app.py --server.headless true

install: $(VENV)
	$(PIP) install -r app/requirements.txt --quiet

setup:
	python3 -m venv $(VENV)
	$(MAKE) install

$(VENV):
	python3 -m venv $(VENV)

$(ST): $(VENV)
	$(MAKE) install
