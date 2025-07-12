# FILENAME: Makefile
# AI Mission Control - Docker Management

# Configuration
IMAGE_NAME = ai-mission-control:latest
CONTAINER_NAME = ai-mission-control
PROJECT_DIR = $(shell pwd)

# Default target
.PHONY: help
help: ## Show this help message
	@echo "AI Mission Control - Docker Management"
	@echo "======================================"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# Build commands
.PHONY: build
build: ## Build the Docker image
	./build.sh

.PHONY: rebuild
rebuild: ## Rebuild the Docker image from scratch
	docker system prune -f
	./build.sh

# Run commands
.PHONY: shell
shell: ## Start interactive shell in container
	docker run -it --rm \
		--gpus all \
		--name $(CONTAINER_NAME)-shell \
		-v $(PROJECT_DIR):/workspace/project \
		-v ai-mc-experiments:/workspace/experiments \
		-v ai-mc-logs:/workspace/logs \
		-v ai-mc-models:/workspace/models \
		$(IMAGE_NAME) /bin/bash

.PHONY: jupyter
jupyter: ## Start Jupyter Lab (port 8080)
	@echo "Starting Jupyter Lab on http://localhost:8080"
	docker run -it --rm \
		--gpus all \
		--name $(CONTAINER_NAME)-jupyter \
		-p 8080:8080 \
		-v $(PROJECT_DIR):/workspace/project \
		-v ai-mc-experiments:/workspace/experiments \
		-v ai-mc-logs:/workspace/logs \
		-v ai-mc-models:/workspace/models \
		$(IMAGE_NAME) jupyter lab --ip=0.0.0.0 --port=8080 --no-browser --allow-root

.PHONY: tensorboard
tensorboard: ## Start TensorBoard (port 6006)
	@echo "Starting TensorBoard on http://localhost:6006"
	docker run -it --rm \
		--name $(CONTAINER_NAME)-tensorboard \
		-p 6006:6006 \
		-v ai-mc-logs:/workspace/logs \
		$(IMAGE_NAME) tensorboard --logdir=/workspace/logs --host=0.0.0.0 --port=6006

# Development commands
.PHONY: dev
dev: ## Start development container (detached with all services)
	docker run -d \
		--gpus all \
		--name $(CONTAINER_NAME)-dev \
		-p 8080:8080 \
		-p 6006:6006 \
		-v $(PROJECT_DIR):/workspace/project \
		-v ai-mc-experiments:/workspace/experiments \
		-v ai-mc-logs:/workspace/logs \
		-v ai-mc-models:/workspace/models \
		$(IMAGE_NAME) sleep infinity
	@echo "Development container started. Use 'make exec' to enter."

.PHONY: exec
exec: ## Execute shell in running dev container
	docker exec -it $(CONTAINER_NAME)-dev /bin/bash

.PHONY: stop-dev
stop-dev: ## Stop development container
	docker stop $(CONTAINER_NAME)-dev || true
	docker rm $(CONTAINER_NAME)-dev || true

# Experiment commands
.PHONY: run-experiment
run-experiment: ## Run an experiment script (usage: make run-experiment SCRIPT=rl_tests.py)
ifndef SCRIPT
	@echo "Usage: make run-experiment SCRIPT=your_script.py"
	@exit 1
endif
	docker run -it --rm \
		--gpus all \
		--name $(CONTAINER_NAME)-experiment \
		-v $(PROJECT_DIR):/workspace/project \
		-v ai-mc-experiments:/workspace/experiments \
		-v ai-mc-logs:/workspace/logs \
		-v ai-mc-models:/workspace/models \
		$(IMAGE_NAME) python3 /workspace/project/$(SCRIPT)

.PHONY: run-rl-tests
run-rl-tests: ## Run rl_tests.py with default parameters
	$(MAKE) run-experiment SCRIPT=rl_tests.py

# Utility commands
.PHONY: test
test: ## Test the installation
	docker run --rm --gpus all $(IMAGE_NAME) python3 /workspace/scripts/verify_installation.py

.PHONY: logs
logs: ## Show logs from running containers
	@echo "=== Development Container Logs ==="
	docker logs $(CONTAINER_NAME)-dev 2>/dev/null || echo "No dev container running"
	@echo "=== Jupyter Container Logs ==="
	docker logs $(CONTAINER_NAME)-jupyter 2>/dev/null || echo "No jupyter container running"

.PHONY: ps
ps: ## Show running AI Mission Control containers
	@echo "AI Mission Control Containers:"
	@docker ps --filter "name=$(CONTAINER_NAME)" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"

# Cleanup commands
.PHONY: clean
clean: ## Stop and remove all containers
	docker stop $(CONTAINER_NAME)-dev $(CONTAINER_NAME)-jupyter $(CONTAINER_NAME)-tensorboard $(CONTAINER_NAME)-shell 2>/dev/null || true
	docker rm $(CONTAINER_NAME)-dev $(CONTAINER_NAME)-jupyter $(CONTAINER_NAME)-tensorboard $(CONTAINER_NAME)-shell 2>/dev/null || true

.PHONY: clean-volumes
clean-volumes: ## Remove all volumes (WARNING: deletes all experiment data)
	@echo "WARNING: This will delete all experiment data!"
	@read -p "Are you sure? [y/N] " -n 1 -r; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		docker volume rm ai-mc-experiments ai-mc-logs ai-mc-models 2>/dev/null || true; \
		echo "Volumes removed."; \
	else \
		echo "Cancelled."; \
	fi

.PHONY: clean-all
clean-all: clean clean-volumes ## Stop containers and remove all data

# Docker system commands
.PHONY: docker-info
docker-info: ## Show Docker system information
	@echo "=== Docker Images ==="
	docker images | grep -E "(ai-mission-control|REPOSITORY)"
	@echo ""
	@echo "=== Docker Volumes ==="
	docker volume ls | grep -E "(ai-mc|DRIVER)"
	@echo ""
	@echo "=== Docker System Usage ==="
	docker system df

.PHONY: docker-prune
docker-prune: ## Clean up Docker system (removes unused containers, networks, images)
	docker system prune -f

# Setup commands
.PHONY: setup-volumes
setup-volumes: ## Create Docker volumes for persistent storage
	docker volume create ai-mc-experiments
	docker volume create ai-mc-logs
	docker volume create ai-mc-models
	@echo "Volumes created for persistent storage"

.PHONY: copy-scripts
copy-scripts: ## Copy your RL scripts to container workspace
	@if [ ! -f "rl_tests.py" ]; then echo "rl_tests.py not found in current directory"; exit 1; fi
	docker run --rm \
		-v $(PROJECT_DIR):/workspace/project \
		-v ai-mc-experiments:/workspace/experiments \
		$(IMAGE_NAME) bash -c "\
			cp /workspace/project/rl_tests.py /workspace/experiments/ && \
			cp /workspace/project/hyper_framework.py /workspace/experiments/ 2>/dev/null || true && \
			cp /workspace/project/hyper_optimizers.py /workspace/experiments/ 2>/dev/null || true && \
			ls -la /workspace/experiments/"

# Installation commands
.PHONY: install-package
install-package: ## Install additional package (usage: make install-package PACKAGE=finrl)
ifndef PACKAGE
	@echo "Usage: make install-package PACKAGE=package_name"
	@exit 1
endif
	docker run -it --rm \
		--gpus all \
		-v ai-mc-experiments:/workspace/experiments \
		$(IMAGE_NAME) bash -c "pip install $(PACKAGE) && pip list | grep -i $(PACKAGE)"

# Quick start
.PHONY: start
start: setup-volumes shell ## Quick start: create volumes and open shell

.PHONY: quick-test
quick-test: ## Quick test with CartPole
	docker run --rm \
		--gpus all \
		-v $(PROJECT_DIR):/workspace/project \
		$(IMAGE_NAME) python3 -c "\
import gym; \
env = gym.make('CartPole-v1'); \
print('✅ CartPole environment created successfully!'); \
print(f'Observation space: {env.observation_space}'); \
print(f'Action space: {env.action_space}');"