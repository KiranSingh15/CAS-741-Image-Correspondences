# src/Makefile

VENV = .venv_ifcs
REQS = Requirements.txt

PYTHON := $(shell command -v python || command -v python3.13 || command -v python3)
PYTHON_VERSION_OK := $(shell $(PYTHON) -c "import sys; v=sys.version_info; print(v >= (3, 13))" 2>/dev/null || echo "False")

ifeq ($(OS),Windows_NT)
  PIP = $(VENV)/Scripts/pip.exe
  PY = $(VENV)/Scripts/python.exe
  RM = rmdir /s /q
else
  PIP = $(VENV)/bin/pip
  PY = $(VENV)/bin/python
  RM = rm -rf
endif

install:
	@echo "Using Python: $(PYTHON)"
	@if [ "$(PYTHON_VERSION_OK)" != "True" ]; then \
		echo "ERROR: Python 3.13+ is required, but '$(PYTHON)' is not sufficient."; \
		exit 1; \
	fi
	@if [ ! -d "$(VENV)" ]; then \
		$(PYTHON) -m venv $(VENV); \
	else \
		echo "Virtual environment already exists."; \
	fi
	@$(PY) -m pip install --upgrade pip
	@$(PIP) install -r $(REQS)

clean:
	@echo "Removing virtual environment..."
	@$(RM) $(VENV)

test:
	@echo "Running unit tests..."
	@$(PY) test/run_unit_checks.py
