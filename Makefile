# ===========================================
# Jira AI Co-pilot - Makefile
# ===========================================

.PHONY: install install-dev test lint format sync sync-full train predict risk alerts chat dashboard demo clean help

# -------------------------------------------
# Installation
# -------------------------------------------

install: ## Install production dependencies
	pip install -e .

install-dev: ## Install development dependencies
	pip install -e ".[dev]"
	pre-commit install

# -------------------------------------------
# Code Quality
# -------------------------------------------

test: ## Run tests with coverage
	pytest tests/ -v --cov=src --cov-report=term-missing --cov-report=html

test-fast: ## Run tests without coverage
	pytest tests/ -v

lint: ## Run linter
	ruff check src/ tests/

format: ## Format code
	ruff format src/ tests/
	ruff check --fix src/ tests/

typecheck: ## Run type checker
	mypy src/

quality: lint typecheck test ## Run all quality checks

# -------------------------------------------
# Data Operations
# -------------------------------------------

sync: ## Incremental sync from Jira
	python -m src.jira_client.sync --mode incremental

sync-full: ## Full sync from Jira (initial or reset)
	python -m src.jira_client.sync --mode full

# -------------------------------------------
# Model Operations
# -------------------------------------------

train: ## Train all ML models
	python -m src.models.trainer --all

train-estimator: ## Train ticket estimator model only
	python -m src.models.trainer --model ticket_estimator

train-risk: ## Train sprint risk model only
	python -m src.models.trainer --model sprint_risk

# -------------------------------------------
# Predictions & Analysis
# -------------------------------------------

predict: ## Predict duration for a ticket (usage: make predict TICKET=PROJ-123)
	python -m src.models.ticket_estimator predict $(TICKET)

risk: ## Show current sprint risk score
	python -m src.interface.cli risk sprint

risk-team: ## Show team workload
	python -m src.interface.cli risk team

alerts: ## Show active alerts
	python -m src.interface.cli alerts

# -------------------------------------------
# Interactive Interfaces
# -------------------------------------------

chat: ## Start interactive chat with AI
	python -m src.interface.cli chat

dashboard: ## Launch Streamlit dashboard
	streamlit run src/interface/dashboard.py --server.headless true

# -------------------------------------------
# Release Planning
# -------------------------------------------

plan-release: ## Generate release plan
	python -m src.interface.cli plan-release

# -------------------------------------------
# Demo & Development
# -------------------------------------------

demo: ## Generate demo data and launch dashboard
	python scripts/generate_demo_data.py
	$(MAKE) dashboard

notebook: ## Start Jupyter notebook
	jupyter notebook notebooks/

# -------------------------------------------
# Maintenance
# -------------------------------------------

clean: ## Clean generated files
	rm -rf __pycache__ .pytest_cache .mypy_cache .ruff_cache
	rm -rf htmlcov .coverage coverage.xml
	rm -rf build dist *.egg-info
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true

clean-data: ## Clean data files (careful!)
	rm -rf data/*.duckdb data/*.duckdb.wal
	rm -rf data/raw/*.json
	rm -rf data/exports/*.csv

clean-models: ## Clean trained models
	rm -rf models/*.pkl models/*.joblib models/metadata.json

reset: clean clean-data clean-models ## Full reset (careful!)

# -------------------------------------------
# Help
# -------------------------------------------

help: ## Show this help
	@echo "Jira AI Co-pilot - Available commands:"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

.DEFAULT_GOAL := help
