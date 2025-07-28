# FILENAME: Makefile
# AI Mission Control - Docker Management

# Configuration
IMAGE_NAME = ai-mission-control:latest
CONTAINER_NAME = ai-mission-control
PROJECT_DIR = $(shell pwd)

# Default number of workers for distributed training (can be overridden)
NUM_WORKERS ?= 12

# Default target
.PHONY: help
help: ## Show this help message
	@echo "AI Mission Control - Docker Management"
	@echo "======================================"
	@echo "General Commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-40s\033[0m %s\n", $$1, $$2}' | grep -v 'distributed-' | grep -v 'legacy-' | grep -v 'quick-' | grep -v 'run-distributed-' | grep -v 'test' | grep -v 'logs' | grep -v 'ps' | grep -v 'clean' | grep -v 'docker-' | grep -v 'setup-' | grep -v 'copy-' | grep -v 'install-' | grep -v 'start' | grep -v 'quick-test'

	@echo ""
	@echo "Distributed Training Commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-40s\033[0m %s\n", $$1, $$2}' | grep 'distributed-' | grep -v 'legacy-'

	@echo ""
	@echo "Quick Start Distributed Commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-40s\033[0m %s\n", $$1, $$2}' | grep 'quick-'

	@echo ""
	@echo "Experiment Commands (Distributed):"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-40s\033[0m %s\n", $$1, $$2}' | grep 'run-distributed-'

	@echo ""
	@echo "Utility & Cleanup Commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-40s\033[0m %s\n", $$1, $$2}' | grep -E '(test|logs|ps|clean|docker-|setup-|copy-|install-|start|quick-test)'

	@echo ""
	@echo "For more detailed usage, refer to the Makefile itself."


# Build commands
.PHONY: build
build: ## Build the Docker image
	@echo "📦 Building Docker image: $(IMAGE_NAME)..."
	./build.sh
	@echo "✅ Docker image built successfully."

.PHONY: rebuild
rebuild: ## Rebuild the Docker image from scratch
	@echo "🧹 Pruning Docker system before rebuild..."
	docker system prune -f
	@echo "📦 Rebuilding Docker image: $(IMAGE_NAME) from scratch..."
	./build.sh
	@echo "✅ Docker image rebuilt successfully."

# Run commands
.PHONY: shell
shell: ## Start interactive shell in container
	@echo "🚀 Starting interactive shell in container..."
	docker run -it --rm \
		--gpus all \
		--name $(CONTAINER_NAME)-shell \
		-v $(PROJECT_DIR):/workspace/project \
		-v ai-mc-experiments:/workspace/experiments \
		-v ai-mc-logs:/workspace/logs \
		-v ai-mc-models:/workspace/models \
		$(IMAGE_NAME) /bin/bash
	@echo "✅ Shell session ended."

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
	@echo "✅ Jupyter Lab session ended."

.PHONY: tensorboard
tensorboard: ## Start TensorBoard (port 6006)
	@echo "Starting TensorBoard on http://localhost:6006"
	docker run -it --rm \
		--name $(CONTAINER_NAME)-tensorboard \
		-p 6006:6006 \
		-v ai-mc-experiments:/workspace/experiments \
		$(IMAGE_NAME) tensorboard --logdir=/workspace/experiments --host=0.0.0.0 --port=6006
	@echo "✅ TensorBoard session ended."

# Development commands
.PHONY: dev
dev: ## Start development container (detached with all services)
	@echo "🚀 Starting development container ($(CONTAINER_NAME)-dev) in detached mode..."
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
	@echo "✅ Development container started. Use 'make exec' to enter."
	@echo "💡 Remember to stop it with 'make stop-dev' when done."

.PHONY: exec
exec: ## Execute shell in running dev container
	@echo "➡️ Entering development container ($(CONTAINER_NAME)-dev)..."
	docker exec -it $(CONTAINER_NAME)-dev /bin/bash
	@echo "✅ Exited development container shell."

.PHONY: stop-dev
stop-dev: ## Stop and remove development container
	@echo "🛑 Stopping development container ($(CONTAINER_NAME)-dev)..."
	docker stop $(CONTAINER_NAME)-dev > /dev/null 2>&1 || true
	docker rm $(CONTAINER_NAME)-dev > /dev/null 2>&1 || true
	@echo "✅ Development container stopped and removed."

# Experiment commands
.PHONY: run-experiment
run-experiment: ## Run an experiment script (usage: make run-experiment SCRIPT=rl_tests.py)
ifndef SCRIPT
	@echo "Usage: make run-experiment SCRIPT=your_script.py"
	@exit 1
endif
	@echo "🔬 Running experiment script: $(SCRIPT)..."
	docker run -it --rm \
		--gpus all \
		--name $(CONTAINER_NAME)-experiment \
		-v $(PROJECT_DIR):/workspace/project \
		-v ai-mc-experiments:/workspace/experiments \
		-v ai-mc-logs:/workspace/logs \
		-v ai-mc-models:/workspace/models \
		$(IMAGE_NAME) python3 /workspace/project/$(SCRIPT)
	@echo "✅ Experiment $(SCRIPT) finished."

.PHONY: run-rl-tests
run-rl-tests: ## Run universal_rl.py with CartPole PPO config and logging
	@echo "🧪 Running RL tests for CartPole PPO..."
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
	@echo "✅ CartPole RL tests completed."

.PHONY: run-mujoco
run-mujoco: ## Run MuJoCo experiment (usage: make run-mujoco CONFIG=halfcheetah)
ifndef CONFIG
	$(MAKE) run-mujoco CONFIG=halfcheetah
else
	@echo "🤖 Running MuJoCo experiment with config: $(CONFIG)..."
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
	@echo "✅ MuJoCo experiment for $(CONFIG) finished."
endif

.PHONY: record-best-agent-video
record-best-agent-video: ## Record videos of best agent for TensorBoard
	@echo "🎥 Recording best agent videos for TensorBoard..."
	docker run --rm --gpus all \
		-v $(PROJECT_DIR):/workspace/project \
		-v $(PROJECT_DIR)/distributed_shared:/workspace/distributed_shared \
		-v $(PROJECT_DIR)/tensorboard_videos:/workspace/tensorboard_videos \
		$(IMAGE_NAME) python /workspace/project/scripts/record_tensorboard_video.py \
			--shared_dir /workspace/distributed_shared \
			--tensorboard_dir /workspace/tensorboard_videos \
			--env_name ${ENV:-CartPole-v1} \
			--num_episodes 3
	@echo "✅ Videos recorded! Start TensorBoard: tensorboard --logdir ./tensorboard_videos"

## Scalable Distributed Training Commands


.PHONY: distributed-up
distributed-up: ## Start distributed RL training with $(NUM_WORKERS) workers
	@echo "🚀 Starting distributed RL training with $(NUM_WORKERS) workers (ENV: ${ENV:-CartPole-v1})..."
	@# Validation checks
	@if [ ! -f "docker-compose.distributed.yml" ]; then \
		echo "❌ Error: docker-compose.distributed.yml not found in current directory"; \
		exit 1; \
	fi
	@if [ $(NUM_WORKERS) -lt 1 ] || [ $(NUM_WORKERS) -gt 12 ]; then \
		echo "❌ Error: NUM_WORKERS must be between 1 and 12 (requested: $(NUM_WORKERS))"; \
		exit 1; \
	fi
	@# Check if containers are already running
	@if docker ps --format "{{.Names}}" | grep -q "^distributed_"; then \
		echo "⚠️  Warning: Distributed containers already running:"; \
		docker ps --filter "name=distributed_" --format "  {{.Names}} ({{.Status}})"; \
		echo "💡 Use 'make distributed-down' first, or 'make distributed-scale WORKERS=$(NUM_WORKERS)' to change worker count"; \
		exit 1; \
	fi
	@# Create shared directory
	@mkdir -p distributed_shared logs
	@# Build services list dynamically
	@echo "📋 Building services list for $(NUM_WORKERS) workers..."
	@SERVICES="coordinator tensorboard-distributed"; \
	for i in $$(seq 0 $$(($(NUM_WORKERS) - 1))); do \
		SERVICES="$$SERVICES worker-$$i"; \
	done; \
	echo "🎯 Starting services: $$SERVICES"; \
	if ENV=${ENV:-CartPole-v1} docker-compose -f docker-compose.distributed.yml up -d $$SERVICES; then \
		echo "✅ Distributed training started successfully!"; \
		sleep 3; \
		echo "📊 Container status:"; \
		docker ps --filter "name=distributed_" --format "table {{.Names}}\t{{.Status}}"; \
		echo ""; \
		echo "🔗 Quick access:"; \
		echo "  📊 Monitor: make distributed-status"; \
		echo "  📋 Logs: make distributed-logs"; \
		echo "  📈 TensorBoard: make distributed-tensorboard"; \
		echo "  🐚 Shell: make distributed-shell-any-worker"; \
		echo "  🏥 Health: make distributed-health-check"; \
	else \
		echo "❌ Failed to start distributed training"; \
		echo "🔍 Checking what went wrong..."; \
		docker-compose -f docker-compose.distributed.yml logs --tail=10; \
		exit 1; \
	fi

.PHONY: distributed-scale
distributed-scale: ## Scale distributed workers (usage: make distributed-scale WORKERS=6)
ifndef WORKERS
	@echo "Usage: make distributed-scale WORKERS=<number_of_workers>"
	@echo "e.g., make distributed-scale WORKERS=6 to scale to 6 workers."
	@exit 1
endif
	@echo "📈 Scaling workers to $(WORKERS)..."
	ENV=${ENV:-CartPole-v1} docker-compose -f docker-compose.distributed.yml up -d --scale worker=$(WORKERS)
	@echo "✅ Workers scaled to $(WORKERS)."

.PHONY: distributed-down
distributed-down: ## Stop distributed RL training and remove containers
	@echo "🛑 Stopping distributed training..."
	@echo "🔄 Stopping TensorBoard instances first..."
	@$(MAKE) distributed-tensorboard-stop > /dev/null 2>&1 || true
	@echo "🐳 Stopping Docker containers..."
	docker-compose -f docker-compose.distributed.yml down
	@echo ""
	@echo "✅ Distributed training stopped."
	@echo ""
	@echo "🧹 Would you like to clean up files and data?"
	@echo "  📁 make distributed-clean    - Remove shared directories and volumes"
	@echo "  🗑️  make clean-all           - Complete cleanup (containers + data + volumes)"
	@echo ""
	@echo "💡 Example: make distributed-clean"
	@echo "💡 To restart: make distributed-up"

.PHONY: distributed-status
distributed-status: ## Show status of distributed training services
	@echo "📊 Distributed Training Status:"
	@echo "================================"
	@echo "Expected workers based on last 'distributed-up' or 'distributed-scale' call."
	docker-compose -f docker-compose.distributed.yml ps
	@echo ""
	@echo "🐳 Active Worker Containers:"
	@docker ps --filter "name=distributed_worker" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" || echo "No distributed workers running."
	@echo ""
	@echo "🖥️ GPU Usage (first worker if available):"
	@if docker ps -q --filter "name=distributed_worker" | head -1 > /dev/null; then \
		docker exec $$(docker ps -q --filter "name=distributed_worker" | head -1) nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv 2>/dev/null; \
	else \
		echo "No workers running to check GPU info."; \
	fi
	@echo ""
	@echo "📁 Shared Directory Contents (distributed_shared):"
	@ls -la distributed_shared/ 2>/dev/null || echo "Shared directory 'distributed_shared/' not found. Run 'make distributed-up' first."

.PHONY: distributed-tensorboard
distributed-tensorboard: ## Start TensorBoard for ALL distributed workers (port 6007)
	@echo "📊 Starting TensorBoard for all distributed workers on http://localhost:6007"
	@echo "📁 Monitoring: distributed_shared/worker_logs/ and subdirectories"
	@if lsof -i :6007 >/dev/null 2>&1; then \
		echo "⚠️ Port 6007 already in use. Attempting to stop existing TensorBoard..."; \
		pkill -f "tensorboard.*6007" 2>/dev/null || true; \
		sleep 2; \
	fi
	@# Attempt to find the most appropriate log directory
	@LOG_DIR=""; \
	if [ -d "distributed_shared/worker_logs" ]; then \
		LOG_DIR="distributed_shared/worker_logs"; \
	elif [ -d "distributed_shared" ]; then \
		LOG_DIR="distributed_shared"; \
	fi; \
	if [ -z "$$LOG_DIR" ]; then \
		echo "❌ No tensorboard logs directory found under distributed_shared. Make sure training is running and logs are being generated."; \
		echo "💡 Try running training longer to generate logs, or check: make distributed-find-logs"; \
		exit 1; \
	fi; \
	echo "📊 Using log directory: $$LOG_DIR"; \
	nohup tensorboard --logdir=$$LOG_DIR --host=0.0.0.0 --port=6007 --reload_interval=30 > tensorboard_distributed.log 2>&1 &
	@echo "✅ TensorBoard started in background."
	@echo "📊 View at: http://localhost:6007"
	@echo "📋 Logs for this TensorBoard instance: tail -f tensorboard_distributed.log"

.PHONY: distributed-find-logs
distributed-find-logs: ## Find where the actual log files are located for distributed training
	@echo "🔍 Searching for distributed training logs..."
	@echo "============================================="
	@if [ -d "distributed_shared" ]; then \
		echo "📁 Contents of 'distributed_shared':"; \
		ls -la distributed_shared/; \
		echo ""; \
		echo "🔍 Searching for 'tensorboard' directories (max depth 3):"; \
		find distributed_shared -maxdepth 3 -name "tensorboard" -type d 2>/dev/null || echo "No 'tensorboard' directories found."; \
		echo ""; \
		echo "🔍 Searching for worker-specific log directories (max depth 3):"; \
		find distributed_shared -maxdepth 3 -type d -name "worker_*" 2>/dev/null || echo "No worker-specific log directories found."; \
		echo ""; \
		echo "🔍 Full directory structure (up to 4 levels deep):"; \
		find distributed_shared -maxdepth 4 -type d 2>/dev/null | sed 's|^[^/]*|./&|' | sort || echo "Cannot show directory structure."; \
	else \
		echo "❌ 'distributed_shared' directory not found. Run 'make distributed-up' first to create it."; \
	fi

.PHONY: distributed-logs
distributed-logs: ## Show live logs from all active distributed services
	@echo "📋 Showing live logs for all distributed services (Ctrl+C to stop)..."
	docker-compose -f docker-compose.distributed.yml logs -f

.PHONY: distributed-logs-coordinator
distributed-logs-coordinator: ## Show live coordinator logs only
	@echo "📋 Showing live logs for coordinator (Ctrl+C to stop)..."
	docker-compose -f docker-compose.distributed.yml logs -f coordinator

.PHONY: distributed-logs-workers
distributed-logs-workers: ## Show a snippet of logs from all workers
	@echo "📋 Latest logs from all active workers (last 10 lines):"
	@docker ps --filter "name=distributed_worker" --format "{{.Names}}" | xargs -I {} sh -c 'echo "--- {} ---"; docker logs {} --tail=10 2>/dev/null || echo "No logs or worker not found."'

.PHONY: distributed-logs-worker
distributed-logs-worker: ## Show live logs for a specific worker (usage: WORKER=1 make distributed-logs-worker)
ifndef WORKER
	@echo "Usage: WORKER=<worker_number> make distributed-logs-worker"
	@echo "Example: WORKER=0 make distributed-logs-worker"
	@echo "Available workers (by index, starting from 0):"
	@docker ps --filter "name=distributed_worker" --format "{{.Names}}" | nl -v 0
	@exit 1
endif
	@WORKER_NAME=$$(docker ps --filter "name=distributed_worker" --format "{{.Names}}" | sed -n "$$((WORKER+1))p"); \
	if [ -z "$$WORKER_NAME" ]; then \
		echo "❌ Worker with index $(WORKER) not found or not running. Use 'make distributed-status' to see running workers."; \
		exit 1; \
	fi; \
	echo "📋 Showing live logs for $(WORKER_NAME) (Ctrl+C to stop)..."
	docker logs -f "$$WORKER_NAME"

.PHONY: distributed-shell-coordinator
distributed-shell-coordinator: ## Open shell in coordinator container
	@if ! docker ps --format "{{.Names}}" | grep -q "^distributed_coordinator$$"; then \
		echo "❌ Coordinator container not running. Start with 'make distributed-up'"; \
		exit 1; \
	fi
	@echo "🐚 Entering coordinator container..."
	docker exec -it distributed_coordinator /bin/bash

.PHONY: distributed-shell-worker
distributed-shell-worker: ## Open shell in a specific worker container (usage: WORKER=1 make distributed-shell-worker)
ifndef WORKER
	@echo "Usage: WORKER=<worker_number> make distributed-shell-worker (worker number 0-based)"
	@echo "Example: WORKER=0 make distributed-shell-worker"
	@echo "Available workers (by index, starting from 0):"
	@docker ps --filter "name=distributed_worker" --format "{{.Names}}" | nl -v 0
	@exit 1
endif
	@WORKER_NAME=$$(docker ps --filter "name=distributed_worker" --format "{{.Names}}" | sed -n "$$((WORKER+1))p"); \
	if [ -z "$$WORKER_NAME" ]; then \
		echo "❌ Worker with index $(WORKER) not found or not running. Use 'make distributed-status' to see running workers."; \
		exit 1; \
	fi; \
	echo "🐚 Opening shell in $(WORKER_NAME)..."
	docker exec -it "$$WORKER_NAME" /bin/bash
	@echo "✅ Exited worker shell."

.PHONY: distributed-shell-any-worker
distributed-shell-any-worker: ## Open shell in the first available worker container
	@echo "🐚 Opening shell in first available worker..."
	@WORKER_NAME=$$(docker ps -q --filter "name=distributed_worker" --format "{{.Names}}" | head -1); \
	if [ -z "$$WORKER_NAME" ]; then \
		echo "❌ No workers found or running. Run 'make distributed-up' first."; \
		exit 1; \
	fi; \
	docker exec -it "$$WORKER_NAME" /bin/bash
	@echo "✅ Exited worker shell."

.PHONY: distributed-restart-all
distributed-restart-all: ## Restart all distributed services (coordinator and workers)
	@echo "🔄 Restarting all distributed services..."
	docker-compose -f docker-compose.distributed.yml restart
	@echo "✅ All distributed services restarted."

.PHONY: distributed-restart-coordinator
distributed-restart-coordinator: ## Restart coordinator service only
	@echo "🔄 Restarting coordinator service..."
	docker-compose -f docker-compose.distributed.yml restart coordinator
	@echo "✅ Coordinator restarted."

.PHONY: distributed-restart-workers
distributed-restart-workers: ## Restart all worker services
	@echo "🔄 Restarting all workers..."
	@docker ps --filter "name=distributed_worker" --format "{{.Names}}" | xargs -I {} sh -c 'echo "Restarting {}..."; docker restart {} > /dev/null'
	@echo "✅ All workers restarted."

.PHONY: distributed-watch-gpu
distributed-watch-gpu: ## Monitor GPU usage and container stats in real-time
	@echo "👀 Watching GPU usage and container stats (Ctrl+C to stop)..."
	@if ! command -v nvidia-smi &> /dev/null; then \
		echo "⚠️ nvidia-smi not found on host. GPU monitoring might be limited."; \
		echo "Attempting to check GPU usage inside a worker container if available."; \
	fi
	@while true; do \
		clear; \
		echo "🖥️ GPU Usage - $$(date)"; \
		echo "================================"; \
		if docker ps -q --filter "name=distributed_worker" | head -1 > /dev/null; then \
			docker exec $$(docker ps -q --filter "name=distributed_worker" | head -1) nvidia-smi 2>/dev/null || echo "Could not retrieve GPU info from worker."; \
		else \
			echo "No worker containers running to check GPU."; \
		fi; \
		echo ""; \
		echo "📊 Active Distributed Container Stats:"; \
		docker stats --no-stream $$(docker ps --filter "name=distributed_" --format "{{.Names}}") 2>/dev/null || echo "No distributed containers to show stats for."; \
		echo ""; \
		echo "🔢 Active Workers: $$(docker ps --filter "name=distributed_worker" --format "{{.Names}}" | wc -l)"; \
		sleep 5; \
	done

.PHONY: distributed-worker-count
distributed-worker-count: ## Show current number of active workers and list them
	@echo "🔢 Active workers: $$(docker ps --filter "name=distributed_worker" --format "{{.Names}}" | wc -l)"
	@echo "📋 Worker list:"
	@docker ps --filter "name=distributed_worker" --format "table {{.Names}}\t{{.Status}}\t{{.RunningFor}}" || echo "No distributed workers running."

.PHONY: distributed-check-workers
distributed-check-workers: ## Check what workers are actually doing (top processes, log directory check)
	@echo "🔍 Checking worker activity..."
	@echo "=============================="
	@docker ps --filter "name=distributed_worker" --format "{{.Names}}" | while read worker; do \
		echo "=== $$worker ==="; \
		docker exec $$worker bash -c "\
			echo 'Top processes:'; \
			ps aux | head -5; \
			echo 'Python processes:'; \
			ps aux | grep python | grep -v grep | head -2 || echo 'No Python process found.'; \
			echo 'Log directory check:'; \
			find /workspace -maxdepth 4 -name '*tensorboard*' -o -name '*experiment*' -o -name '*log*' 2>/dev/null | head -5 || echo 'No relevant log directories found.'; \
		" 2>/dev/null || echo "Cannot access $$worker or command failed."; \
		echo ""; \
	done


## Experiment Commands with Scalable Workers


.PHONY: run-distributed-halfcheetah
run-distributed-halfcheetah: ## Run distributed HalfCheetah experiment (default 4 workers)
	@echo "🏃 Starting distributed HalfCheetah training with $(NUM_WORKERS) workers..."
	@mkdir -p distributed_shared
	ENV=HalfCheetah-v4 NUM_WORKERS=$(NUM_WORKERS) $(MAKE) distributed-up
	@echo "✅ Distributed HalfCheetah training started!"

.PHONY: run-distributed-cartpole
run-distributed-cartpole: ## Run distributed CartPole experiment (default 4 workers)
	@echo "🛒 Starting distributed CartPole training with $(NUM_WORKERS) workers..."
	@mkdir -p distributed_shared
	ENV=CartPole-v1 NUM_WORKERS=$(NUM_WORKERS) $(MAKE) distributed-up
	@echo "✅ Distributed CartPole training started!"

.PHONY: run-distributed-custom
run-distributed-custom: ## Run distributed training with custom environment (usage: make run-distributed-custom ENV=Ant-v4 NUM_WORKERS=8)
ifndef ENV
	@echo "Usage: make run-distributed-custom ENV=your_environment [NUM_WORKERS=N]"
	@echo "Example: make run-distributed-custom ENV=Ant-v4 NUM_WORKERS=8"
	@exit 1
endif
	@echo "🎮 Starting distributed $(ENV) training with $(NUM_WORKERS) workers..."
	@mkdir -p distributed_shared
	ENV=$(ENV) NUM_WORKERS=$(NUM_WORKERS) $(MAKE) distributed-up
	@echo "✅ Distributed $(ENV) training started with $(NUM_WORKERS) workers!"


## Quick Start Distributed Commands


.PHONY: distributed-quick-4
distributed-quick-4: ## Quick start with 4 workers
	$(MAKE) distributed-up NUM_WORKERS=4

.PHONY: distributed-quick-8
distributed-quick-8: ## Quick start with 8 workers
	$(MAKE) distributed-up NUM_WORKERS=8

.PHONY: distributed-quick-12
distributed-quick-12: ## Quick start with 12 workers
	$(MAKE) distributed-up NUM_WORKERS=12

.PHONY: distributed-quick-16
distributed-quick-16: ## Quick-start with 16 workers
	$(MAKE) distributed-up NUM_WORKERS=16


## Legacy Compatibility Commands (Individual Worker)

# These commands directly reference specific worker names (e.g., worker-0).
# It's recommended to use the more general 'distributed-logs-worker WORKER=0' format instead.

.PHONY: legacy-distributed-tensorboard-worker-0
legacy-distributed-tensorboard-worker-0: ## Start TensorBoard for worker-0 only (port 6008)
	@echo "📊 Starting TensorBoard for worker-0 on http://localhost:6008 (Legacy)"
	@if lsof -i :6008 >/dev/null 2>&1; then \
		echo "⚠️ Port 6008 already in use. Stopping existing TensorBoard..."; \
		pkill -f "tensorboard.*6008" 2>/dev/null || true; \
		sleep 2; \
	fi
	@WORKER_LOG_DIR=$(find distributed_shared -name "worker_0" -type d | head -1); \
	if [ -z "$WORKER_LOG_DIR" ]; then \
		echo "❌ Worker 0 logs not found. Make sure training is running. Try 'make distributed-find-logs'."; \
		exit 1; \
	fi; \
	TENSORBOARD_DIR=$(find "$WORKER_LOG_DIR" -name "tensorboard" -type d | head -1); \
	if [ -z "$TENSORBOARD_DIR" ]; then \
		echo "❌ Worker 0 tensorboard logs not found within $WORKER_LOG_DIR. Ensure logs are being generated."; \
		exit 1; \
	fi; \
	nohup tensorboard --logdir=$TENSORBOARD_DIR --host=0.0.0.0 --port=6008 > tensorboard_worker0.log 2>&1 &
	@echo "✅ Worker-0 TensorBoard started at http://localhost:6008"
	@echo "📋 Logs for this TensorBoard instance: tail -f tensorboard_worker0.log"

.PHONY: legacy-distributed-logs-worker-0
legacy-distributed-logs-worker-0: ## Show worker-0 logs only (legacy compatibility)
	@echo "📋 Showing live logs for worker-0 (legacy, Ctrl+C to stop)..."
	@docker logs distributed_worker_0 --tail=50 -f 2>/dev/null || echo "Worker 'distributed_worker_0' not found or not running."

.PHONY: legacy-distributed-logs-worker-1
legacy-distributed-logs-worker-1: ## Show worker-1 logs only (legacy compatibility)
	@echo "📋 Showing live logs for worker-1 (legacy, Ctrl+C to stop)..."
	@docker logs distributed_worker_1 --tail=50 -f 2>/dev/null || echo "Worker 'distributed_worker_1' not found or not running."

.PHONY: legacy-distributed-shell-worker-0
legacy-distributed-shell-worker-0: ## Open shell in worker-0 container (legacy compatibility)
	@echo "🐚 Opening shell in worker-0 (legacy)..."
	@docker exec -it distributed_worker_0 /bin/bash 2>/dev/null || echo "Worker 'distributed_worker_0' not found or not running."
	@echo "✅ Exited worker shell."


## Debugging and Diagnostics


.PHONY: distributed-tensorboard-stop
distributed-tensorboard-stop: ## Stop all distributed TensorBoard instances started by Makefile
	@echo "🛑 Stopping TensorBoard instances started by Makefile..."
	@pkill -f "tensorboard.*6007" 2>/dev/null || true # Main distributed TB
	@pkill -f "tensorboard.*6008" 2>/dev/null || true # Worker 0 legacy TB
	@pkill -f "tensorboard.*distributed_shared" 2>/dev/null || true
	@pkill -f "tensorboard_*.log" 2>/dev/null || true # Remove log files
	@rm -f tensorboard_*.log 2>/dev/null || true
	@echo "✅ All known TensorBoard instances stopped and log files removed."

.PHONY: distributed-tensorboard-status
distributed-tensorboard-status: ## Check status of TensorBoard instances
	@echo "📊 TensorBoard Status:"
	@echo "====================="
	@echo "🔍 Checking common TensorBoard ports (6006-6011)..."
	@for port in 6006 6007 6008 6009 6010 6011; do \
		if lsof -i :$port >/dev/null 2>&1; then \
			echo "✅ Port $port: ACTIVE - http://localhost:$port (PID: $(lsof -i :$port -t))"; \
		else \
			echo "❌ Port $port: FREE"; \
		fi; \
	done
	@echo ""
	@echo "📋 All TensorBoard Processes:"
	@ps aux | grep tensorboard | grep -v grep || echo "No TensorBoard processes running on host."

.PHONY: distributed-analysis
distributed-analysis: ## Show distributed training analysis and file structure overview
	@echo "📊 Distributed Training Analysis"
	@echo "================================"
	@echo ""
	@echo "📁 'distributed_shared' Log Directory Structure (top 3 levels):"
	@if [ -d "distributed_shared" ]; then \
		find distributed_shared -maxdepth 3 -type d 2>/dev/null | sort | sed 's|^[^/]*|./&|'; \
	else \
		echo "Distributed training not started yet. Run 'make distributed-up'."; \
	fi
	@echo ""
	@echo "📈 TensorBoard access points:"
	@echo "  All workers:        make distributed-tensorboard (port 6007)"
	@echo "  Worker 0 only:      make legacy-distributed-tensorboard-worker-0 (port 6008)"
	@echo ""
	@echo "📋 Latest Metrics (if 'experiment_summary.json' exists in worker logs):"
	@find distributed_shared/worker_logs -name "experiment_summary.json" -exec echo "  File: {}" \; -exec cat {} \; 2>/dev/null || echo "No 'experiment_summary.json' found yet."

.PHONY: distributed-clean
distributed-clean: ## Clean up distributed training containers, networks, and volumes
	@echo "🧹 Cleaning up distributed training (stopping containers, removing networks and volumes)..."
	@echo "⚠️  WARNING: This will delete training data in distributed_shared/"
	@echo -n "Are you sure you want to proceed? Type 'yes' to confirm: "; \
	read REPLY; \
	if [ "$REPLY" = "yes" ]; then \
		echo "🛑 Stopping TensorBoard instances..."; \
		$(MAKE) distributed-tensorboard-stop > /dev/null 2>&1 || true; \
		echo "🐳 Stopping containers..."; \
		docker-compose -f docker-compose.distributed.yml down -v --rmi all; \
		echo "📁 Removing local 'distributed_shared/' directory..."; \
		rm -rf distributed_shared/ 2>/dev/null || true; \
		echo "✅ Distributed training cleanup complete."; \
	else \
		echo "❌ Operation cancelled. No cleanup performed."; \
	fi

.PHONY: distributed-quick-restart
distributed-quick-restart: ## Quick restart: stop everything and start with last settings
	@echo "🔄 Quick restart of distributed training..."
	@$(MAKE) distributed-down > /dev/null 2>&1 || true
	@sleep 3
	@$(MAKE) distributed-up

.PHONY: distributed-save-results
distributed-save-results: ## Save current training results to timestamped backup
	@echo "💾 Saving current training results..."
	@TIMESTAMP=$(date +%Y%m%d_%H%M%S); \
	if [ -d "distributed_shared" ]; then \
		mkdir -p backups; \
		cp -r distributed_shared "backups/distributed_shared_$TIMESTAMP"; \
		echo "✅ Results saved to: backups/distributed_shared_$TIMESTAMP"; \
		echo "📊 Backup size: $(du -sh backups/distributed_shared_$TIMESTAMP | cut -f1)"; \
	else \
		echo "❌ No distributed_shared directory found to backup"; \
	fi


## Utility Commands


.PHONY: test
test: ## Test the installation by running a verification script in the container
	@echo "🧪 Running installation verification test..."
	docker run --rm --gpus all $(IMAGE_NAME) python3 /workspace/scripts/verify_installation.py
	@echo "✅ Installation test completed."

.PHONY: logs
logs: ## Show recent logs from main running AI Mission Control containers
	@echo "=== Development Container Logs (Last 20 lines) ==="
	docker logs $(CONTAINER_NAME)-dev --tail=20 2>/dev/null || echo "No dev container running or no logs available."
	@echo "=== Jupyter Container Logs (Last 20 lines) ==="
	docker logs $(CONTAINER_NAME)-jupyter --tail=20 2>/dev/null || echo "No jupyter container running or no logs available."
	@echo "=== Distributed Training Logs (Last 20 lines from all services) ==="
	docker-compose -f docker-compose.distributed.yml logs --tail=20 2>/dev/null || echo "No distributed training running or no logs available."

.PHONY: ps
ps: ## Show running AI Mission Control and Distributed Training containers
	@echo "AI Mission Control Containers:"
	@docker ps --filter "name=$(CONTAINER_NAME)" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" || echo "No AI Mission Control containers running."
	@echo ""
	@echo "Distributed Training Containers (managed by docker-compose):"
	@docker ps --filter "name=distributed_" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" 2>/dev/null || echo "No distributed containers running."

# Cleanup commands
.PHONY: clean
clean: ## Stop and remove all general AI Mission Control containers
	@echo "🧹 Stopping and removing general AI Mission Control containers..."
	docker stop $(CONTAINER_NAME)-dev $(CONTAINER_NAME)-jupyter $(CONTAINER_NAME)-tensorboard $(CONTAINER_NAME)-shell > /dev/null 2>&1 || true
	docker rm $(CONTAINER_NAME)-dev $(CONTAINER_NAME)-jupyter $(CONTAINER_NAME)-tensorboard $(CONTAINER_NAME)-shell > /dev/null 2>&1 || true
	@echo "✅ General AI Mission Control containers stopped and removed."

.PHONY: clean-volumes
clean-volumes: ## Remove all AI Mission Control volumes (WARNING: deletes all experiment data)
	@echo "⚠️ WARNING: This will delete ALL experiment data, logs, and models stored in Docker volumes!"
	@echo -n "Are you absolutely sure you want to proceed? Type 'yes' to confirm: "; \
	read REPLY; \
	if [ "$REPLY" = "yes" ]; then \
		echo "Deleting volumes: ai-mc-experiments, ai-mc-logs, ai-mc-models..."; \
		docker volume rm ai-mc-experiments ai-mc-logs ai-mc-models 2>/dev/null || true; \
		echo "Also removing local 'distributed_shared/' directory..."; \
		rm -rf distributed_shared/ 2>/dev/null || true; \
		echo "✅ Volumes and shared directories removed."; \
	else \
		echo "❌ Operation cancelled. Volumes were NOT removed."; \
	fi

.PHONY: clean-all
clean-all: clean distributed-clean clean-volumes ## Stop all containers and remove all data (including volumes)
	@echo "Cleaning up all containers, distributed services, and volumes..."
	@echo "This is the most comprehensive cleanup."
	$(MAKE) clean
	$(MAKE) distributed-clean
	$(MAKE) clean-volumes REPLY=yes # Force confirmation for clean-volumes
	@echo "✅ Full cleanup complete."

# Docker system commands
.PHONY: docker-info
docker-info: ## Show Docker system information for AI Mission Control components
	@echo "=== Docker Images (filtered for AI Mission Control) ==="
	docker images | grep -E "(ai-mission-control|REPOSITORY)" || echo "No 'ai-mission-control' images found."
	@echo ""
	@echo "=== Docker Volumes (filtered for AI Mission Control) ==="
	docker volume ls | grep -E "(ai-mc|DRIVER)" || echo "No 'ai-mc' volumes found."
	@echo ""
	@echo "=== Docker System Disk Usage ==="
	docker system df

.PHONY: docker-prune
docker-prune: ## Clean up Docker system (removes unused containers, networks, images, build cache)
	@echo "🧹 Pruning Docker system (containers, networks, images, build cache)..."
	docker system prune -a -f
	@echo "✅ Docker system pruned."

# Setup commands
.PHONY: setup-volumes
setup-volumes: ## Create Docker volumes for persistent storage
	@echo "📦 Creating Docker volumes for persistent storage (if they don't exist)..."
	docker volume create ai-mc-experiments > /dev/null 2>&1 || true
	docker volume create ai-mc-logs > /dev/null 2>&1 || true
	docker volume create ai-mc-models > /dev/null 2>&1 || true
	@echo "✅ Volumes created: ai-mc-experiments, ai-mc-logs, ai-mc-models."

.PHONY: copy-scripts
copy-scripts: ## Copy your RL scripts to container workspace for easy access
	@echo "📂 Copying specified RL scripts to /workspace/experiments in container..."
	@if [ ! -f "rl_tests.py" ]; then \
		echo "❌ Error: 'rl_tests.py' not found in your current directory ($(PROJECT_DIR))."; \
		echo "Please ensure the script exists or adjust the Makefile target."; \
		exit 1; \
	fi
	docker run --rm \
		-v $(PROJECT_DIR):/workspace/project \
		-v ai-mc-experiments:/workspace/experiments \
		$(IMAGE_NAME) bash -c "\
			cp /workspace/project/rl_tests.py /workspace/experiments/ && \
			cp /workspace/project/hyper_framework.py /workspace/experiments/ 2>/dev/null || true && \
			cp /workspace/project/hyper_optimizers.py /workspace/experiments/ 2>/dev/null || true && \
			echo 'Copied files:'; \
			ls -la /workspace/experiments/"
	@echo "✅ Scripts copied."

# Installation commands
.PHONY: install-package
install-package: ## Install an additional Python package into the Docker image (usage: make install-package PACKAGE=finrl)
ifndef PACKAGE
	@echo "Usage: make install-package PACKAGE=<package_name>"
	@echo "Example: make install-package PACKAGE=finrl"
	@exit 1
endif
	@echo "➕ Installing Python package: $(PACKAGE) into $(IMAGE_NAME)..."
	docker run -it --rm \
		--gpus all \
		-v ai-mc-experiments:/workspace/experiments \
		$(IMAGE_NAME) bash -c "pip install $(PACKAGE) && echo 'Installation status:' && pip list | grep -i $(PACKAGE)"
	@echo "✅ Package installation attempt completed for $(PACKAGE)."

# Quick start
.PHONY: start
start: setup-volumes shell ## Quick start: create volumes and open an interactive shell in the container
	@echo "🚀 Quick start completed: volumes set up, and you are in the container shell."

.PHONY: quick-test
quick-test: ## Quick test: verify basic Python and Gym environment setup with CartPole
	@echo "⚡ Running quick test for Python and CartPole Gym environment..."
	docker run --rm \
		--gpus all \
		-v $(PROJECT_DIR):/workspace/project \
		$(IMAGE_NAME) python3 -c "\
import gym; \
try: \
    env = gym.make('CartPole-v1'); \
    print('✅ CartPole-v1 environment created successfully!'); \
    print(f'Observation space: {env.observation_space}'); \
    print(f'Action space: {env.action_space}'); \
    env.close(); \
except Exception as e: \
    print(f'❌ Failed to create CartPole-v1 environment: {e}'); \
    import sys; sys.exit(1);"
	@echo "✅ Quick test completed."

.PHONY: distributed-health-check
distributed-health-check: ## Run comprehensive worker health check
	@echo "🏥 Running distributed worker health check..."
	@chmod +x worker_health_check.sh
	@./worker_health_check.sh