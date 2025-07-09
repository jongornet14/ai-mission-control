.PHONY: help install test lint format clean dev build deploy

help: ## Show this help message
	@echo 'Usage: make [target]'
	@echo ''
	@echo 'Targets:'
	@egrep '^(.+)\s*:.*?##\s*(.+)' $(MAKEFILE_LIST) | column -t -c 2 -s ':#'

install: ## Install dependencies
	pip install -r requirements.txt
	pip install -r requirements-dev.txt

test: ## Run tests
	pytest tests/ -v --cov=apps --cov=shared

test-unit: ## Run unit tests only
	pytest tests/unit/ -v

test-integration: ## Run integration tests only
	pytest tests/integration/ -v

test-e2e: ## Run end-to-end tests only
	pytest tests/e2e/ -v

lint: ## Run linting
	flake8 apps/ shared/ tests/
	mypy apps/ shared/

format: ## Format code
	black apps/ shared/ tests/
	isort apps/ shared/ tests/

clean: ## Clean up temporary files
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/

dev: ## Start development environment
	docker-compose -f docker-compose.yml -f docker-compose.dev.yml up

build: ## Build all services
	docker-compose build

deploy: ## Deploy to production
	./scripts/deployment/deploy.sh

logs: ## Show logs from all services
	docker-compose logs -f
