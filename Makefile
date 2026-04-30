VENV   := .venv
PYTHON := $(VENV)/Scripts/python
PIP    := $(VENV)/Scripts/pip
ST     := $(VENV)/Scripts/streamlit

.PHONY: run install setup

run: $(ST)
	$(ST) run app/streamlit_app.py --server.headless true

install: $(VENV)
	$(PIP) install -r app/requirements.txt --quiet

setup:
	py -m venv $(VENV)
	$(MAKE) install

$(VENV):
	py -m venv $(VENV)

$(ST): $(VENV)
	$(MAKE) install
