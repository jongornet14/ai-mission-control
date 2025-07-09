.PHONY: help build up down logs clean test status

help:		## Show available commands
	@echo "AI Mission Control - Container Commands"
	@echo "======================================"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

build:		## Build all containers
	@echo "Building all containers..."
	docker-compose build

up:		## Start all services
	@echo "Starting all services..."
	docker-compose up -d
	@echo "Services started! Check status with 'make status'"

down:		## Stop all services
	@echo "Stopping all services..."
	docker-compose down

logs:		## View service logs
	@echo "Service logs (Ctrl+C to exit):"
	docker-compose logs -f

status:		## Check service status
	@echo "Service Status:"
	@docker-compose ps

clean:		## Clean up containers and volumes
	@echo "Cleaning up..."
	docker-compose down -v
	docker system prune -f

test:		## Run tests
	@echo "Running tests..."
	python examples/basic_usage.py

health:		## Check service health
	@echo "Checking service health..."
	@for port in 50051 50052 50053 50054; do \
		echo -n "Port $$port: "; \
		curl -s -o /dev/null -w "%{http_code}" http://localhost:$$port/health || echo "No response"; \
		echo ""; \
	done

restart:	## Restart all services
	@echo "Restarting all services..."
	docker-compose restart

rebuild:	## Rebuild and restart all services
	@echo "Rebuilding and restarting..."
	docker-compose down
	docker-compose build
	docker-compose up -d
