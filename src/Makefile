REQUIREMENTS = Requirements.txt
VENV_NAME := .venv_ifcs
PYTHON_VER_LINUX := python3.13
PYTHON_VER_MAC := python@3.13
PYTHON := $(VENV_NAME)/bin/python3.13

venv:
	@echo "Creating virtual environment..."
	@if [ ! -d "$(VENV_NAME)" ]; then \
		UNAME=$$(uname); \
		if [ "$$UNAME" = "Darwin" ]; then \
			echo "Detected macOS. Installing dependencies with brew (if needed)..."; \
			brew install $(PYTHON_VER_MAC) || true; \
			$(PYTHON_VER_MAC) -m venv $(VENV_NAME); \
		elif [ "$$UNAME" = "Linux" ]; then \
			echo "Detected Linux. Installing dependencies with apt..."; \
			sudo apt update -y && \
			sudo apt install -y python3.10-venv python3.10-tk && \
			$(PYTHON_VER_LINUX) -m venv $(VENV_NAME); \
		elif [ "$$OS" = "Windows_NT" ]; then \
			echo "Detected Windows. Creating virtual environment with system Python..."; \
			python -m venv $(VENV_NAME); \
		else \
			echo "Unsupported OS: $$UNAME"; exit 1; \
		fi \
	else \
		echo "Virtual environment already exists."; \
	fi

install: venv
	@echo "Installing dependencies..."
	@if [ "$$OS" = "Windows_NT" ]; then \
		$(VENV_NAME)/Scripts/python.exe -m pip install --upgrade pip && \
		$(VENV_NAME)/Scripts/python.exe -m pip install -r $(REQUIREMENTS); \
	else \
		$(PYTHON) -m pip install --upgrade pip && \
		$(PYTHON) -m pip install -r $(REQUIREMENTS); \
	fi

clean:
	@echo "Cleaning virtual environment and cache files..."
	rm -rf $(VENV_NAME) __pycache__ *.pyc