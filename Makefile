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
		-v ai-mc-experiments:/workspace/experiments \
		$(IMAGE_NAME) tensorboard --logdir=/workspace/experiments --host=0.0.0.0 --port=6006
		
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
run-rl-tests: ## Run universal_rl.py with CartPole PPO config and logging
	docker run -it --rm \
		--gpus all \
		--name $(CONTAINER_NAME)-experiment \
		-v $(PROJECT_DIR):/workspace/project \
		-v ai-mc-experiments:/workspace/experiments \
		-v ai-mc-logs:/workspace/logs \
		-v ai-mc-models:/workspace/models \
		$(IMAGE_NAME) python3 /workspace/project/scripts/universal_rl.py \
			--config /workspace/configs/cartpole_ppo.yaml \
			--environment.name CartPole-v1 \
			--training.total_episodes 500 \
			--logging.save_videos \
			--logging.log_activations

.PHONY: run-mujoco
run-mujoco: ## Run MuJoCo experiment (usage: make run-mujoco CONFIG=halfcheetah)
ifndef CONFIG
	$(MAKE) run-mujoco CONFIG=halfcheetah
else
	docker run -it --rm \
		--gpus all \
		--name $(CONTAINER_NAME)-experiment \
		-v $(PROJECT_DIR):/workspace/project \
		-v ai-mc-experiments:/workspace/experiments \
		-v ai-mc-logs:/workspace/logs \
		-v ai-mc-models:/workspace/models \
		$(IMAGE_NAME) python3 /workspace/project/scripts/universal_rl.py \
			--config /workspace/project/scripts/configs/mujoco/mujoco_$(CONFIG).yaml \
			--logging.save_videos \
			--logging.log_activations
endif

# Distributed Training Commands
.PHONY: distributed-up
distributed-up: ## Start distributed RL training (4 workers + coordinator)
	@echo "🚀 Starting distributed RL training..."
	@mkdir -p distributed_shared logs
	docker-compose -f docker-compose.distributed.yml up -d
	@echo "✅ Distributed training started!"
	@echo "📊 Monitor with: make distributed-logs"
	@echo "📈 TensorBoard (all workers): make distributed-tensorboard"
	@echo "📈 TensorBoard (worker-0): make distributed-tensorboard-worker-0"
	@echo "🎯 Dashboard at: http://localhost:8081"

.PHONY: distributed-down
distributed-down: ## Stop distributed RL training
	@echo "🛑 Stopping distributed training..."
	docker-compose -f docker-compose.distributed.yml down
	@echo "✅ Distributed training stopped"

.PHONY: distributed-tensorboard
distributed-tensorboard: ## Start TensorBoard for ALL distributed workers (port 6007)
	@echo "📊 Starting TensorBoard for all distributed workers on http://localhost:6007"
	@echo "📁 Monitoring: distributed_shared/worker_logs/"
	@if lsof -i :6007 >/dev/null 2>&1; then \
		echo "⚠️  Port 6007 already in use. Stopping existing TensorBoard..."; \
		pkill -f "tensorboard.*6007" 2>/dev/null || true; \
		sleep 2; \
	fi
	@# Check multiple possible log locations
	@LOG_DIR=""; \
	if [ -d "distributed_shared/worker_logs" ]; then \
		LOG_DIR="distributed_shared/worker_logs"; \
	elif [ -d "distributed_shared" ] && find distributed_shared -name "tensorboard" -type d | head -1 >/dev/null 2>&1; then \
		LOG_DIR="distributed_shared"; \
		echo "📁 Found tensorboard logs in distributed_shared subdirectories"; \
	else \
		echo "❌ No tensorboard logs found. Checking what exists:"; \
		ls -la distributed_shared/ 2>/dev/null || echo "distributed_shared directory not found"; \
		find distributed_shared -name "*tensorboard*" -o -name "*log*" 2>/dev/null || echo "No log files found"; \
		echo "💡 Try running training longer to generate logs, or check: make distributed-logs"; \
		exit 1; \
	fi; \
	echo "📊 Using log directory: $$LOG_DIR"; \
	nohup tensorboard --logdir=$$LOG_DIR --host=0.0.0.0 --port=6007 --reload_interval=30 > tensorboard_distributed.log 2>&1 &
	@echo "✅ TensorBoard started in background"
	@echo "📊 View at: http://localhost:6007"
	@echo "📋 Logs: tail -f tensorboard_distributed.log"

.PHONY: distributed-find-logs
distributed-find-logs: ## Find where the actual log files are located
	@echo "🔍 Searching for distributed training logs..."
	@echo "============================================="
	@if [ -d "distributed_shared" ]; then \
		echo "📁 distributed_shared contents:"; \
		ls -la distributed_shared/; \
		echo ""; \
		echo "🔍 Searching for tensorboard directories:"; \
		find distributed_shared -name "tensorboard" -type d 2>/dev/null || echo "No tensorboard directories found"; \
		echo ""; \
		echo "🔍 Searching for any log files:"; \
		find distributed_shared -name "*.log" 2>/dev/null | head -10 || echo "No .log files found"; \
		echo ""; \
		echo "🔍 Directory structure (2 levels):"; \
		find distributed_shared -maxdepth 2 -type d 2>/dev/null || echo "Cannot show directory structure"; \
	else \
		echo "❌ distributed_shared directory not found"; \
		echo "💡 Run: make distributed-up"; \
	fi

.PHONY: distributed-tensorboard-worker-0
distributed-tensorboard-worker-0: ## Start TensorBoard for worker-0 only (port 6008)
	@echo "📊 Starting TensorBoard for worker-0 on http://localhost:6008"
	@if lsof -i :6008 >/dev/null 2>&1; then \
		echo "⚠️  Port 6008 already in use. Stopping existing TensorBoard..."; \
		pkill -f "tensorboard.*6008" 2>/dev/null || true; \
		sleep 2; \
	fi
	@if [ ! -d "distributed_shared/worker_logs/worker_0/tensorboard" ]; then \
		echo "❌ Worker 0 logs not found. Make sure training is running."; \
		exit 1; \
	fi
	nohup tensorboard --logdir=distributed_shared/worker_logs/worker_0/tensorboard --host=0.0.0.0 --port=6008 > tensorboard_worker0.log 2>&1 &
	@echo "✅ Worker-0 TensorBoard started at http://localhost:6008"

.PHONY: distributed-tensorboard-worker-1
distributed-tensorboard-worker-1: ## Start TensorBoard for worker-1 only (port 6009)
	@echo "📊 Starting TensorBoard for worker-1 on http://localhost:6009"
	@if lsof -i :6009 >/dev/null 2>&1; then \
		echo "⚠️  Port 6009 already in use. Stopping existing TensorBoard..."; \
		pkill -f "tensorboard.*6009" 2>/dev/null || true; \
		sleep 2; \
	fi
	@if [ ! -d "distributed_shared/worker_logs/worker_1/tensorboard" ]; then \
		echo "❌ Worker 1 logs not found. Make sure training is running."; \
		exit 1; \
	fi
	nohup tensorboard --logdir=distributed_shared/worker_logs/worker_1/tensorboard --host=0.0.0.0 --port=6009 > tensorboard_worker1.log 2>&1 &
	@echo "✅ Worker-1 TensorBoard started at http://localhost:6009"

.PHONY: distributed-tensorboard-worker-2
distributed-tensorboard-worker-2: ## Start TensorBoard for worker-2 only (port 6010)
	@echo "📊 Starting TensorBoard for worker-2 on http://localhost:6010"
	@if lsof -i :6010 >/dev/null 2>&1; then \
		echo "⚠️  Port 6010 already in use. Stopping existing TensorBoard..."; \
		pkill -f "tensorboard.*6010" 2>/dev/null || true; \
		sleep 2; \
	fi
	@if [ ! -d "distributed_shared/worker_logs/worker_2/tensorboard" ]; then \
		echo "❌ Worker 2 logs not found. Make sure training is running."; \
		exit 1; \
	fi
	nohup tensorboard --logdir=distributed_shared/worker_logs/worker_2/tensorboard --host=0.0.0.0 --port=6010 > tensorboard_worker2.log 2>&1 &
	@echo "✅ Worker-2 TensorBoard started at http://localhost:6010"

.PHONY: distributed-tensorboard-worker-3
distributed-tensorboard-worker-3: ## Start TensorBoard for worker-3 only (port 6011)
	@echo "📊 Starting TensorBoard for worker-3 on http://localhost:6011"
	@if lsof -i :6011 >/dev/null 2>&1; then \
		echo "⚠️  Port 6011 already in use. Stopping existing TensorBoard..."; \
		pkill -f "tensorboard.*6011" 2>/dev/null || true; \
		sleep 2; \
	fi
	@if [ ! -d "distributed_shared/worker_logs/worker_3/tensorboard" ]; then \
		echo "❌ Worker 3 logs not found. Make sure training is running."; \
		exit 1; \
	fi
	nohup tensorboard --logdir=distributed_shared/worker_logs/worker_3/tensorboard --host=0.0.0.0 --port=6011 > tensorboard_worker3.log 2>&1 &
	@echo "✅ Worker-3 TensorBoard started at http://localhost:6011"

.PHONY: distributed-tensorboard-stop
distributed-tensorboard-stop: ## Stop all distributed TensorBoard instances
	@echo "🛑 Stopping TensorBoard instances..."
	@pkill -f "tensorboard.*distributed_shared" 2>/dev/null || echo "No distributed TensorBoard processes found"
	@pkill -f "tensorboard.*600[789]" 2>/dev/null || echo "No worker TensorBoard processes found"
	@pkill -f "tensorboard.*601[01]" 2>/dev/null || echo "No worker TensorBoard processes found"
	@rm -f tensorboard_*.log 2>/dev/null || true
	@echo "✅ All TensorBoard instances stopped"

.PHONY: distributed-tensorboard-status
distributed-tensorboard-status: ## Check status of TensorBoard instances
	@echo "📊 TensorBoard Status:"
	@echo "====================="
	@echo "🔍 Checking ports 6007-6011..."
	@for port in 6007 6008 6009 6010 6011; do \
		if lsof -i :$port >/dev/null 2>&1; then \
			echo "✅ Port $port: ACTIVE - http://localhost:$port"; \
		else \
			echo "❌ Port $port: FREE"; \
		fi; \
	done
	@echo ""
	@echo "📋 TensorBoard Processes:"
	@ps aux | grep tensorboard | grep -v grep || echo "No TensorBoard processes running"
	@echo ""
	@echo "📁 Log Files:"
	@ls -la tensorboard_*.log 2>/dev/null || echo "No TensorBoard log files"

.PHONY: distributed-tensorboard-docker
distributed-tensorboard-docker: ## Start TensorBoard in Docker container (port 6007)
	@echo "📊 Starting TensorBoard in Docker on http://localhost:6007"
	docker run -it --rm \
		--name $(CONTAINER_NAME)-tensorboard-distributed \
		-p 6007:6007 \
		-v $(PROJECT_DIR)/distributed_shared:/workspace/distributed_shared \
		$(IMAGE_NAME) tensorboard \
		--logdir=/workspace/distributed_shared/worker_logs \
		--host=0.0.0.0 --port=6007 --reload_interval=30

.PHONY: distributed-logs
distributed-logs: ## Show logs from all distributed services
	docker-compose -f docker-compose.distributed.yml logs -f

.PHONY: distributed-logs-coordinator
distributed-logs-coordinator: ## Show coordinator logs only
	docker-compose -f docker-compose.distributed.yml logs -f coordinator

.PHONY: distributed-logs-worker-0
distributed-logs-worker-0: ## Show worker-0 logs only
	docker-compose -f docker-compose.distributed.yml logs -f worker-0

.PHONY: distributed-logs-worker-1
distributed-logs-worker-1: ## Show worker-1 logs only
	docker-compose -f docker-compose.distributed.yml logs -f worker-1

.PHONY: distributed-logs-worker-2
distributed-logs-worker-2: ## Show worker-2 logs only
	docker-compose -f docker-compose.distributed.yml logs -f worker-2

.PHONY: distributed-logs-worker-3
distributed-logs-worker-3: ## Show worker-3 logs only
	docker-compose -f docker-compose.distributed.yml logs -f worker-3

.PHONY: distributed-status
distributed-status: ## Show status of distributed training
	@echo "📊 Distributed Training Status:"
	@echo "================================"
	docker-compose -f docker-compose.distributed.yml ps
	@echo ""
	@echo "🖥️  GPU Usage:"
	@docker exec rl-worker-0 nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv 2>/dev/null || echo "GPU info not available"
	@echo ""
	@echo "📁 Shared Directory:"
	@ls -la distributed_shared/ 2>/dev/null || echo "Shared directory not found"

.PHONY: distributed-restart-all
distributed-restart-all: ## Restart all distributed services
	docker-compose -f docker-compose.distributed.yml restart

.PHONY: distributed-restart-coordinator
distributed-restart-coordinator: ## Restart coordinator only
	docker-compose -f docker-compose.distributed.yml restart coordinator

.PHONY: distributed-restart-worker-0
distributed-restart-worker-0: ## Restart worker-0 only
	docker-compose -f docker-compose.distributed.yml restart worker-0

.PHONY: distributed-restart-worker-1
distributed-restart-worker-1: ## Restart worker-1 only
	docker-compose -f docker-compose.distributed.yml restart worker-1

.PHONY: distributed-restart-worker-2
distributed-restart-worker-2: ## Restart worker-2 only
	docker-compose -f docker-compose.distributed.yml restart worker-2

.PHONY: distributed-restart-worker-3
distributed-restart-worker-3: ## Restart worker-3 only
	docker-compose -f docker-compose.distributed.yml restart worker-3

.PHONY: distributed-shell-coordinator
distributed-shell-coordinator: ## Open shell in coordinator container
	docker exec -it rl-coordinator /bin/bash

.PHONY: distributed-shell-worker-0
distributed-shell-worker-0: ## Open shell in worker-0 container
	docker exec -it rl-worker-0 /bin/bash

.PHONY: distributed-shell-worker-1
distributed-shell-worker-1: ## Open shell in worker-1 container
	docker exec -it rl-worker-1 /bin/bash

.PHONY: distributed-shell-worker-2
distributed-shell-worker-2: ## Open shell in worker-2 container
	docker exec -it rl-worker-2 /bin/bash

.PHONY: distributed-shell-worker-3
distributed-shell-worker-3: ## Open shell in worker-3 container
	docker exec -it rl-worker-3 /bin/bash

.PHONY: distributed-watch-gpu
distributed-watch-gpu: ## Monitor GPU usage in real-time
	@echo "👀 Watching GPU usage (Ctrl+C to stop)..."
	@while true; do \
		clear; \
		echo "🖥️  GPU Usage - $(shell date)"; \
		echo "================================"; \
		docker exec rl-worker-0 nvidia-smi 2>/dev/null || echo "GPU not available"; \
		echo ""; \
		echo "📊 Container Stats:"; \
		docker stats --no-stream rl-coordinator rl-worker-0 rl-worker-1 rl-worker-2 rl-worker-3 2>/dev/null || true; \
		sleep 5; \
	done

.PHONY: distributed-clean
distributed-clean: ## Clean up distributed training containers and volumes
	@echo "🧹 Cleaning up distributed training..."
	docker-compose -f docker-compose.distributed.yml down -v
	rm -rf distributed_shared/ 2>/dev/null || true
	@echo "✅ Distributed training cleanup complete"

.PHONY: distributed-ports
distributed-ports: ## Check what's using TensorBoard ports
	@echo "🔍 Checking TensorBoard ports (6007-6011):"
	@echo "==========================================="
	@for port in 6007 6008 6009 6010 6011; do \
		echo -n "Port $port: "; \
		if lsof -i :$port 2>/dev/null; then \
			echo ""; \
		else \
			echo "FREE"; \
		fi; \
	done
	@echo ""
	@echo "💡 To stop existing TensorBoard: make distributed-tensorboard-stop"

.PHONY: distributed-check-crazylogger
distributed-check-crazylogger: ## Check if CrazyLogger is working in workers
	@echo "🔍 Checking CrazyLogger status in all workers..."
	@for worker in 0 1 2 3; do \
		echo "=== Worker $$worker ==="; \
		docker exec rl-worker-$$worker bash -c "\
			echo 'Python packages:'; \
			python -c 'import sys; print(sys.path)' 2>/dev/null || echo 'Python path issue'; \
			echo 'CrazyLogger import test:'; \
			python -c 'from crazylogging.crazy_logger import CrazyLogger; print(\"CrazyLogger OK\")' 2>/dev/null || echo 'CrazyLogger import failed'; \
			echo 'Log directories:'; \
			find /workspace -path '*/tensorboard*' -o -path '*crazy*' -o -path '*experiment*' 2>/dev/null || echo 'No experiment logs'; \
		"; \
	done

.PHONY: distributed-debug-logs
distributed-debug-logs: ## Debug why no tensorboard logs exist
	@echo "🐛 Debugging distributed training logs..."
	@echo "1. Checking if training is actually running RL (not dummy):"
	@make distributed-logs-worker-0 | tail -20
	@echo ""
	@echo "2. Checking for any experiment files:"
	@docker exec rl-worker-0 find /workspace -name "*experiment*" -o -name "*crazy*" 2>/dev/null || echo "No experiment files"
	@echo ""
	@echo "3. Checking Python imports in worker:"
	@docker exec rl-worker-0 python -c "from crazylogging.crazy_logger import CrazyLogger; print('✅ CrazyLogger works')" 2>/dev/null || echo "❌ CrazyLogger import failed"
	
.PHONY: distributed-analysis
distributed-analysis: ## Show distributed training analysis and file structure
	@echo "📊 Distributed Training Analysis"
	@echo "================================"
	@echo ""
	@echo "📁 Log Directory Structure:"
	@ls -la distributed_shared/ 2>/dev/null || echo "Distributed training not started yet"
	@echo ""
	@echo "🧠 Worker Logs:"
	@ls -la distributed_shared/worker_logs/ 2>/dev/null || echo "No worker logs yet"
	@echo ""
	@echo "📈 TensorBoard Options:"
	@echo "  All workers:     make distributed-tensorboard          (port 6007)"
	@echo "  Worker 0 only:   make distributed-tensorboard-worker-0 (port 6008)"
	@echo "  Worker 1 only:   make distributed-tensorboard-worker-1 (port 6009)"
	@echo "  Worker 2 only:   make distributed-tensorboard-worker-2 (port 6010)"
	@echo "  Worker 3 only:   make distributed-tensorboard-worker-3 (port 6011)"
	@echo ""
	@echo "📋 Latest Metrics (if available):"
	@find distributed_shared/worker_logs -name "experiment_summary.json" -exec echo "  {}" \; -exec cat {} \; 2>/dev/null || echo "No experiment summaries yet"

.PHONY: distributed-open-tensorboard
distributed-open-tensorboard: ## Open TensorBoard URLs in browser (macOS/Linux)
	@echo "🌐 Opening TensorBoard in browser..."
	@if command -v open >/dev/null 2>&1; then \
		open "http://localhost:6007" && echo "📊 Opened all workers TensorBoard"; \
	elif command -v xdg-open >/dev/null 2>&1; then \
		xdg-open "http://localhost:6007" && echo "📊 Opened all workers TensorBoard"; \
	else \
		echo "📊 TensorBoard URL: http://localhost:6007"; \
	fi

.PHONY: distributed-quick-start
distributed-quick-start: distributed-up ## Quick start distributed training
	@echo "⚡ Distributed training quick start complete!"
	@echo "🎯 Training started with 4 workers"
	@echo "📊 Monitor: make distributed-logs"
	@echo "🖥️  GPU usage: make distributed-watch-gpu"

.PHONY: distributed-quick-stop
distributed-quick-stop: distributed-down ## Quick stop distributed training
	@echo "⚡ Distributed training quick stop complete!"

# Distributed experiment commands
.PHONY: run-distributed-halfcheetah
run-distributed-halfcheetah: ## Run distributed HalfCheetah experiment
	@echo "🏃 Starting distributed HalfCheetah training..."
	@mkdir -p distributed_shared
	ENV=HalfCheetah-v4 docker-compose -f docker-compose.distributed.yml up -d
	@echo "✅ Distributed HalfCheetah training started!"

.PHONY: run-distributed-cartpole
run-distributed-cartpole: ## Run distributed CartPole experiment  
	@echo "🛒 Starting distributed CartPole training..."
	@mkdir -p distributed_shared
	ENV=CartPole-v1 docker-compose -f docker-compose.distributed.yml up -d
	@echo "✅ Distributed CartPole training started!"

.PHONY: run-distributed-custom
run-distributed-custom: ## Run distributed training with custom environment (usage: make run-distributed-custom ENV=Ant-v4)
ifndef ENV
	@echo "Usage: make run-distributed-custom ENV=your_environment"
	@exit 1
endif
	@echo "🎮 Starting distributed $(ENV) training..."
	@mkdir -p distributed_shared
	ENV=$(ENV) docker-compose -f docker-compose.distributed.yml up -d
	@echo "✅ Distributed $(ENV) training started!"
			
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
	@echo "=== Distributed Training Logs ==="
	docker-compose -f docker-compose.distributed.yml logs --tail=20 2>/dev/null || echo "No distributed training running"

.PHONY: ps
ps: ## Show running AI Mission Control containers
	@echo "AI Mission Control Containers:"
	@docker ps --filter "name=$(CONTAINER_NAME)" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
	@echo ""
	@echo "Distributed Training Containers:"
	@docker ps --filter "name=rl-" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" 2>/dev/null || echo "No distributed containers running"

# Cleanup commands
.PHONY: clean
clean: ## Stop and remove all containers
	docker stop $(CONTAINER_NAME)-dev $(CONTAINER_NAME)-jupyter $(CONTAINER_NAME)-tensorboard $(CONTAINER_NAME)-shell 2>/dev/null || true
	docker rm $(CONTAINER_NAME)-dev $(CONTAINER_NAME)-jupyter $(CONTAINER_NAME)-tensorboard $(CONTAINER_NAME)-shell 2>/dev/null || true
	docker-compose -f docker-compose.distributed.yml down 2>/dev/null || true

.PHONY: clean-volumes
clean-volumes: ## Remove all volumes (WARNING: deletes all experiment data)
	@echo "WARNING: This will delete all experiment data!"
	@echo -n "Are you sure? [y/N] "; \
	read REPLY; \
	if [ "$$REPLY" = "y" ] || [ "$$REPLY" = "Y" ]; then \
		docker volume rm ai-mc-experiments ai-mc-logs ai-mc-models 2>/dev/null || true; \
		rm -rf distributed_shared/ 2>/dev/null || true; \
		echo "Volumes removed."; \
	else \
		echo "Cancelled."; \
	fi

.PHONY: clean-all
clean-all: clean clean-volumes distributed-clean ## Stop containers and remove all data

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