# Detect Python version (prefer 3.12, fallback to 3.11, then 3.10, then python3)
PYTHON := $(shell \
	if command -v python3.12 >/dev/null 2>&1; then \
		echo python3.12; \
	elif command -v python3.11 >/dev/null 2>&1; then \
		echo python3.11; \
	elif command -v python3.10 >/dev/null 2>&1; then \
		echo python3.10; \
	elif command -v python3 >/dev/null 2>&1; then \
		echo python3; \
	else \
		echo python3.12; \
	fi)

# Use venv Python if it exists, otherwise use detected Python
VENV_PYTHON := $(shell if [ -f venv/bin/python ]; then echo venv/bin/python; else echo $(PYTHON); fi)

venv:
	$(PYTHON) -m venv venv
	@echo "Run '. venv/bin/activate' to activate the virtual environment."
	@echo "Using Python: $(PYTHON)"
	
install:
	@if [ ! -f venv/bin/python ]; then \
		echo "Error: Virtual environment not found. Please run 'make venv' first."; \
		exit 1; \
	fi
	venv/bin/pip install -r requirements.txt
	venv/bin/python -c "import cmdstanpy; cmdstanpy.install_cmdstan()"

forecast:
	$(VENV_PYTHON) src/forecast_model.py

run_dashboard:
	streamlit run src/app.py

update-data:
	@echo "Updating crime data from Boston.gov API (incremental)..."
	@if [ ! -f venv/bin/python ]; then \
		echo "Error: Virtual environment not found. Please run 'make venv' first."; \
		exit 1; \
	fi
	venv/bin/pip install -q requests pyyaml 2>/dev/null || true
	$(VENV_PYTHON) scripts/update_data.py --incremental

update-data-full:
	@echo "Performing full data refresh from Boston.gov API..."
	@if [ ! -f venv/bin/python ]; then \
		echo "Error: Virtual environment not found. Please run 'make venv' first."; \
		exit 1; \
	fi
	venv/bin/pip install -q requests pyyaml 2>/dev/null || true
	$(VENV_PYTHON) scripts/update_data.py --full

check-data:
	@echo "Checking data freshness..."
	@if [ ! -f venv/bin/python ]; then \
		echo "Error: Virtual environment not found. Please run 'make venv' first."; \
		exit 1; \
	fi
	$(VENV_PYTHON) -c "from src.data_prep import check_data_freshness; check_data_freshness()"

retrain:
	@echo "Updating data and retraining models..."
	$(MAKE) update-data
	$(MAKE) forecast
