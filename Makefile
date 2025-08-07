# Distributed RL Training Makefile
# Enhanced with JSON Config Support and Bayesian Optimization

.PHONY: dist-start
dist-start: ## Start distributed training (2 workers by default)
	@printf "\033[1;32mStarting distributed training with JSON configs...\033[0m\n"
	@printf "\033[1;36mArchitecture: Bayesian Optimization + DistributedWorker(BaseWorker)\033[0m\n"
	@printf "\033[1;33mUsing config: ${CONFIG:-cartpole_distributed.json}\033[0m\n"
	CONFIG=${CONFIG:-cartpole_distributed.json} docker-compose up -d

.PHONY: dist-start-fg
dist-start-fg: ## Start distributed training in foreground (see logs)
	@printf "\033[1;32mStarting distributed training (foreground)...\033[0m\n"
	@printf "\033[1;33mUsing config: ${CONFIG:-cartpole_distributed.json}\033[0m\n"
	CONFIG=${CONFIG:-cartpole_distributed.json} docker-compose up

.PHONY: dist-stop
dist-stop: ## Stop distributed training
	@printf "\033[1;31mStopping distributed training...\033[0m\n"
	docker-compose down

.PHONY: dist-restart
dist-restart: ## Restart distributed training
	@printf "\033[1;34mRestarting distributed training...\033[0m\n"
	$(MAKE) dist-stop
	sleep 2
	$(MAKE) dist-start

.PHONY: dist-status
dist-status: ## Show distributed training status
	@printf "\033[1;35mDistributed Training Status:\033[0m\n"
	@echo "==============================="
	@docker-compose ps
	@echo ""
	@printf "\033[1;36mContainer Health:\033[0m\n"
	@docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" | grep -E "(rl-|ai-mc-)"
	@echo ""
	@if [ -f "./distributed_shared/coordinator_status.json" ]; then \
		printf "\033[1;34mCoordinator Status:\033[0m\n"; \
		cat ./distributed_shared/coordinator_status.json | python3 -m json.tool 2>/dev/null || echo "Status file exists but invalid JSON"; \
	else \
		printf "\033[1;31mNo coordinator status file found\033[0m\n"; \
	fi

.PHONY: dist-logs
dist-logs: ## Show recent logs from all services
	@printf "\033[1;36mRecent logs from all services:\033[0m\n"
	docker-compose logs --tail=50

.PHONY: dist-follow
dist-follow: ## Follow logs in real-time
	@printf "\033[1;36mFollowing distributed training logs (Ctrl+C to stop)...\033[0m\n"
	docker-compose logs -f

.PHONY: dist-logs-coordinator
dist-logs-coordinator: ## Show coordinator logs only
	@printf "\033[1;36mCoordinator logs:\033[0m\n"
	docker-compose logs --tail=100 coordinator

.PHONY: dist-logs-worker
dist-logs-worker: ## Show specific worker logs (usage: WORKER=0 make dist-logs-worker)
ifndef WORKER
	@printf "\033[1;31mUsage: WORKER=<worker_number> make dist-logs-worker\033[0m\n"
	@printf "\033[1;33mExample: WORKER=0 make dist-logs-worker\033[0m\n"
	@exit 1
endif
	@printf "\033[1;36mWorker-$(WORKER) logs:\033[0m\n"
	docker-compose logs --tail=100 worker-$(WORKER)

.PHONY: dist-clean
dist-clean: ## Clean up all distributed training resources
	@printf "\033[1;31mCleaning up distributed training...\033[0m\n"
	docker-compose down -v --remove-orphans
	@printf "\033[1;31mRemoving shared data...\033[0m\n"
	sudo rm -rf ./distributed_shared/*
	sudo rm -rf ./tensorboard_videos/*
	@printf "\033[1;32mCleanup complete\033[0m\n"

.PHONY: dist-clean-soft
dist-clean-soft: ## Clean up containers but keep data
	@printf "\033[1;33mSoft cleanup (keeping data)...\033[0m\n"
	docker-compose down --remove-orphans

# Build commands
.PHONY: dist-build
dist-build: ## Build all Docker images
	@printf "\033[1;34mBuilding Docker images...\033[0m\n"
	docker-compose build

.PHONY: dist-rebuild
dist-rebuild: ## Force rebuild all Docker images
	@printf "\033[1;34mForce rebuilding Docker images...\033[0m\n"
	docker-compose build --no-cache

# Environment shortcuts with JSON configs
.PHONY: dist-cartpole
dist-cartpole: ## Start CartPole distributed training
	@printf "\033[1;32mStarting CartPole distributed training...\033[0m\n"
	CONFIG=cartpole_distributed.json $(MAKE) dist-start

.PHONY: dist-halfcheetah
dist-halfcheetah: ## Start HalfCheetah distributed training
	@printf "\033[1;32mStarting HalfCheetah distributed training...\033[0m\n"
	CONFIG=halfcheetah_distributed.json $(MAKE) dist-start

.PHONY: dist-cartpole-fg
dist-cartpole-fg: ## Start CartPole in foreground
	@printf "\033[1;32mStarting CartPole (foreground)...\033[0m\n"
	CONFIG=cartpole_distributed.json $(MAKE) dist-start-fg

.PHONY: dist-halfcheetah-fg
dist-halfcheetah-fg: ## Start HalfCheetah in foreground
	@printf "\033[1;32mStarting HalfCheetah (foreground)...\033[0m\n"
	CONFIG=halfcheetah_distributed.json $(MAKE) dist-start-fg

# Config management
.PHONY: dist-show-config
dist-show-config: ## Show current config file contents
	@if [ -f "./configs/${CONFIG:-cartpole_distributed.json}" ]; then \
		printf "\033[1;33mConfig: ${CONFIG:-cartpole_distributed.json}\033[0m\n"; \
		echo "=========================================="; \
		cat ./configs/${CONFIG:-cartpole_distributed.json} | python3 -m json.tool; \
	else \
		printf "\033[1;31mConfig file not found: ./configs/${CONFIG:-cartpole_distributed.json}\033[0m\n"; \
	fi

.PHONY: dist-list-configs
dist-list-configs: ## List available config files
	@printf "\033[1;33mAvailable config files:\033[0m\n"
	@echo "=========================="
	@ls -la ./configs/*.json 2>/dev/null || echo "No config files found in ./configs/"

.PHONY: dist-validate-config
dist-validate-config: ## Validate config file syntax
	@printf "\033[1;36mValidating config: ${CONFIG:-cartpole_distributed.json}\033[0m\n"
	@python3 -c "import json; json.load(open('./configs/${CONFIG:-cartpole_distributed.json}'))" && printf "\033[1;32mConfig file is valid JSON\033[0m\n" || printf "\033[1;31mConfig file has JSON syntax errors\033[0m\n"

# Development and debugging
.PHONY: dist-debug
dist-debug: ## Show detailed debug info
	@printf "\033[1;35mDistributed Training Debug Info\033[0m\n"
	@echo "=================================="
	@$(MAKE) dist-status
	@echo ""
	@printf "\033[1;33mConfig Files:\033[0m\n"
	@$(MAKE) dist-list-configs
	@echo ""
	@printf "\033[1;33mShared Directory Contents:\033[0m\n"
	@ls -la ./distributed_shared/ 2>/dev/null || echo "Shared directory doesn't exist"
	@echo ""
	@printf "\033[1;36mRecent Coordinator Logs:\033[0m\n"
	@$(MAKE) dist-logs-coordinator | tail -20
	@echo ""
	@printf "\033[1;36mRecent Worker-0 Logs:\033[0m\n"
	@WORKER=0 $(MAKE) dist-logs-worker | tail -20 2>/dev/null || echo "Worker-0 not available"

.PHONY: dist-shell
dist-shell: ## Open shell in development container
	@printf "\033[1;36mOpening shell in development container...\033[0m\n"
	docker-compose exec dev bash

.PHONY: dist-shell-coordinator
dist-shell-coordinator: ## Open shell in coordinator container
	@printf "\033[1;36mOpening shell in coordinator container...\033[0m\n"
	docker-compose exec coordinator bash

.PHONY: dist-monitor
dist-monitor: ## Monitor training progress (auto-refresh every 30s)
	@printf "\033[1;36mMonitoring distributed training (Ctrl+C to stop)...\033[0m\n"
	@while true; do \
		clear; \
		echo "=== Distributed Training Monitor - $$(date) ==="; \
		$(MAKE) dist-status; \
		echo ""; \
		echo "Next update in 30 seconds... (Ctrl+C to stop)"; \
		sleep 30; \
	done

# Service management
.PHONY: dist-restart-coordinator
dist-restart-coordinator: ## Restart only the coordinator
	@printf "\033[1;34mRestarting coordinator...\033[0m\n"
	docker-compose restart coordinator

.PHONY: dist-restart-workers
dist-restart-workers: ## Restart all workers
	@printf "\033[1;34mRestarting all workers...\033[0m\n"
	docker-compose restart worker-0 worker-1

.PHONY: dist-scale-workers
dist-scale-workers: ## Scale workers (usage: WORKERS=4 make dist-scale-workers)
ifndef WORKERS
	@printf "\033[1;31mUsage: WORKERS=<number> make dist-scale-workers\033[0m\n"
	@printf "\033[1;33mExample: WORKERS=4 make dist-scale-workers\033[0m\n"
	@exit 1
endif
	@printf "\033[1;32mScaling to $(WORKERS) workers...\033[0m\n"
	@printf "\033[1;31mNote: You may need to update docker-compose.yml for more than 2 workers\033[0m\n"
	docker-compose up -d --scale worker-0=$(WORKERS)

# TensorBoard and monitoring
.PHONY: dist-tensorboard
dist-tensorboard: ## Start TensorBoard (if not already running)
	@printf "\033[1;36mStarting TensorBoard...\033[0m\n"
	docker-compose up -d tensorboard
	@printf "\033[1;36mTensorBoard available at: http://localhost:6006\033[0m\n"

.PHONY: dist-jupyter
dist-jupyter: ## Start Jupyter Lab (if not already running)
	@printf "\033[1;36mStarting Jupyter Lab...\033[0m\n"
	docker-compose up -d jupyter
	@printf "\033[1;36mJupyter Lab available at: http://localhost:8080\033[0m\n"

.PHONY: dist-services
dist-services: ## Show all available service URLs
	@printf "\033[1;36mAvailable Services:\033[0m\n"
	@echo "====================="
	@echo "TensorBoard:  http://localhost:6006"
	@echo "Jupyter Lab:  http://localhost:8080"
	@echo "Alternative:  http://localhost:8888"

# Video recording
.PHONY: dist-record-video
dist-record-video: ## Record videos of best agent
	@printf "\033[1;36mRecording videos of best agent...\033[0m\n"
	@if [ ! -f "./scripts/video_recorder.py" ]; then \
		printf "\033[1;31mvideo_recorder.py not found in scripts/\033[0m\n"; \
		exit 1; \
	fi
	$(eval ENV_NAME := $(if $(ENV),$(ENV),CartPole-v1))
	docker-compose exec -T dev bash -c "\
		source /opt/miniforge/etc/profile.d/conda.sh && \
		conda activate automl && \
		python scripts/video_recorder.py \
		--shared_dir /workspace/distributed_shared \
		--env_name $(ENV_NAME) \
		--num_episodes 3"

# Add these after the existing dist-halfcheetah-fg command:

# Multi-worker configurations
.PHONY: dist-cartpole-4
dist-cartpole-4: ## Start CartPole with 4 workers
	@printf "\033[1;32mStarting CartPole distributed training (4 workers)...\033[0m\n"
	CONFIG_FILE=cartpole_distributed.json NUM_WORKERS=4 docker-compose up -d coordinator worker-0 worker-1 worker-2 worker-3 tensorboard jupyter dev

.PHONY: dist-cartpole-8
dist-cartpole-8: ## Start CartPole with 8 workers
	@printf "\033[1;32mStarting CartPole distributed training (8 workers)...\033[0m\n"
	CONFIG_FILE=cartpole_distributed.json NUM_WORKERS=8 docker-compose up -d coordinator worker-0 worker-1 worker-2 worker-3 worker-4 worker-5 worker-6 worker-7 tensorboard jupyter dev

.PHONY: dist-cartpole-12
dist-cartpole-12: ## Start CartPole with 12 workers
	@printf "\033[1;32mStarting CartPole distributed training (12 workers)...\033[0m\n"
	CONFIG_FILE=cartpole_distributed.json NUM_WORKERS=12 docker-compose up -d coordinator worker-0 worker-1 worker-2 worker-3 worker-4 worker-5 worker-6 worker-7 worker-8 worker-9 worker-10 worker-11 tensorboard jupyter dev

# Generic multi-worker commands
.PHONY: dist-start-4
dist-start-4: ## Start distributed training with 4 workers
	@printf "\033[1;32mStarting distributed training (4 workers)...\033[0m\n"
	@printf "\033[1;33mUsing config: ${CONFIG:-cartpole_distributed.json}\033[0m\n"
	CONFIG=${CONFIG:-cartpole_distributed.json} NUM_WORKERS=4 docker-compose up -d coordinator worker-0 worker-1 worker-2 worker-3 tensorboard jupyter dev

.PHONY: dist-start-8
dist-start-8: ## Start distributed training with 8 workers
	@printf "\033[1;32mStarting distributed training (8 workers)...\033[0m\n"
	@printf "\033[1;33mUsing config: ${CONFIG:-cartpole_distributed.json}\033[0m\n"
	CONFIG=${CONFIG:-cartpole_distributed.json} NUM_WORKERS=8 docker-compose up -d coordinator worker-0 worker-1 worker-2 worker-3 worker-4 worker-5 worker-6 worker-7 tensorboard jupyter dev

.PHONY: dist-start-12
dist-start-12: ## Start distributed training with 12 workers
	@printf "\033[1;32mStarting distributed training (12 workers)...\033[0m\n"
	@printf "\033[1;33mUsing config: ${CONFIG:-cartpole_distributed.json}\033[0m\n"
	CONFIG=${CONFIG:-cartpole_distributed.json} NUM_WORKERS=12 docker-compose up -d coordinator worker-0 worker-1 worker-2 worker-3 worker-4 worker-5 worker-6 worker-7 worker-8 worker-9 worker-10 worker-11 tensorboard jupyter dev

.PHONY: dist-errors
dist-errors: ## Show all ERROR lines from all service logs
	@printf "\033[1;31mSearching for errors in all logs...\033[0m\n"
	@docker-compose logs | grep -i "error" || echo "No errors found."

.PHONY: dist-warning
dist-warning: ## Show all WARNING lines from all service logs
	@printf "\033[1;33mSearching for warnings in all logs...\033[0m\n"
	@docker-compose logs | grep -i "warning" || echo "No warnings found."

.PHONY: dist-hard-stop
dist-hard-stop: ## Stop all containers and erase all distributed data
	@printf "\033[1;31mHard stopping distributed training and erasing ALL shared data...\033[0m\n"
	docker-compose down -v --remove-orphans
	@printf "\033[1;31mRemoving ./distributed_shared/* ...\033[0m\n"
	sudo rm -rf ./distributed_shared/*
	@printf "\033[1;31mRemoving ./tensorboard_videos/* ...\033[0m\n"
	sudo rm -rf ./tensorboard_videos/*
	@printf "\033[1;32mAll containers stopped and data erased.\033[0m\n"

# Help
.PHONY: dist-help
dist-help: ## Show help for distributed commands
	@printf "\n\033[1;44;97m Distributed Training Commands (Enhanced with JSON Configs) \033[0m\n"
	@printf "\033[1;34m==========================================================\033[0m\n"
	@echo ""
	@printf "\033[1;42;30m Quick Start: \033[0m\n"
	@printf "  \033[1;32mmake dist-cartpole\033[0m                 # Start CartPole with optimized configs\n"
	@printf "  \033[1;32mmake dist-halfcheetah\033[0m              # Start HalfCheetah with optimized configs\n"
	@printf "  \033[1;36mmake dist-status\033[0m                   # Check training status\n"
	@printf "  \033[1;36mmake dist-logs\033[0m                     # See recent activity\n"
	@printf "  \033[1;31mmake dist-stop\033[0m                     # Stop everything\n"
	@echo ""
	@printf "\033[1;42;30m Multi-Worker Examples: \033[0m\n"
	@printf "  \033[1;32mmake dist-cartpole-4\033[0m               # CartPole with 4 workers\n"
	@printf "  \033[1;32mmake dist-cartpole-8\033[0m               # CartPole with 8 workers\n"
	@printf "  \033[1;32mmake dist-cartpole-12\033[0m              # CartPole with 12 workers\n"
	@printf "  \033[1;32mmake dist-halfcheetah-4\033[0m            # HalfCheetah with 4 workers\n"
	@printf "  \033[1;32mmake dist-halfcheetah-8\033[0m            # HalfCheetah with 8 workers\n"
	@printf "  \033[1;32mmake dist-halfcheetah-12\033[0m           # HalfCheetah with 12 workers\n"
	@printf "  \033[1;32mmake dist-start-4\033[0m                  # Generic 4-worker start (configurable)\n"
	@printf "  \033[1;32mmake dist-start-8\033[0m                  # Generic 8-worker start (configurable)\n"
	@printf "  \033[1;32mmake dist-start-12\033[0m                 # Generic 12-worker start (configurable)\n"
	@echo ""
	@printf "\033[1;42;30m Environment Options: \033[0m\n"
	@printf "  \033[1;32mmake dist-cartpole\033[0m                 # CartPole-v1 with Bayesian optimization\n"
	@printf "  \033[1;32mmake dist-halfcheetah\033[0m              # HalfCheetah-v4 with Bayesian optimization\n"
	@printf "  \033[1;32mmake dist-cartpole-fg\033[0m              # CartPole in foreground (see logs)\n"
	@printf "  \033[1;33mCONFIG=custom.json make dist-start\033[0m # Use custom config file\n"
	@echo ""
	@printf "\033[1;42;30m Config Management: \033[0m\n"
	@printf "  \033[1;33mmake dist-list-configs\033[0m             # Show available config files\n"
	@printf "  \033[1;33mmake dist-show-config\033[0m              # Show current config contents\n"
	@printf "  \033[1;33mmake dist-validate-config\033[0m          # Check config syntax\n"
	@printf "  \033[1;33mCONFIG=halfcheetah.json make dist-show-config\033[0m  # Show specific config\n"
	@echo ""
	@printf "\033[1;42;30m Monitoring & Debugging: \033[0m\n"
	@printf "  \033[1;36mmake dist-follow\033[0m                   # Follow logs in real-time\n"
	@printf "  \033[1;36mmake dist-debug\033[0m                    # Show detailed debug info\n"
	@printf "  \033[1;36mmake dist-monitor\033[0m                  # Auto-refresh status every 30s\n"
	@printf "  \033[1;36mWORKER=0 make dist-logs-worker\033[0m     # Specific worker logs\n"
	@printf "  \033[1;36mmake dist-shell\033[0m                    # Open development shell\n"
	@echo ""
	@printf "\033[1;42;30m Web Services: \033[0m\n"
	@printf "  \033[1;36mmake dist-tensorboard\033[0m              # Start TensorBoard (port 6006)\n"
	@printf "  \033[1;36mmake dist-jupyter\033[0m                  # Start Jupyter Lab (port 8080)\n"
	@printf "  \033[1;36mmake dist-services\033[0m                 # Show all service URLs\n"
	@echo ""
	@printf "\033[1;42;30m Video Recording: \033[0m\n"
	@printf "  \033[1;36mmake dist-record-video\033[0m             # Record videos of best agent\n"
	@printf "  \033[1;36mENV=HalfCheetah-v4 make dist-record-video\033[0m  # Record specific environment\n"
	@echo ""
	@printf "\033[1;42;30m Management: \033[0m\n"
	@printf "  \033[1;34mmake dist-restart\033[0m                  # Restart everything\n"
	@printf "  \033[1;34mmake dist-restart-coordinator\033[0m      # Restart only coordinator\n"
	@printf "  \033[1;31mmake dist-clean\033[0m                    # Remove everything (including data)\n"
	@printf "  \033[1;33mmake dist-clean-soft\033[0m               # Remove containers (keep data)\n"
	@printf "  \033[1;34mmake dist-build\033[0m                    # Build Docker images\n"
	@echo ""
	@printf "\033[1;44;97m Architecture: \033[0m\n"
	@printf "  \033[1;37mEnhanced DistributedWorker(BaseWorker) with JSON configs\033[0m\n"
	@printf "  \033[1;37mBayesian Optimization for hyperparameter tuning\033[0m\n"
	@printf "  \033[1;37m1 Coordinator + N Workers + Development tools\033[0m\n"
	@printf "  \033[1;37mFile-based communication via shared Docker volume\033[0m\n"
	@echo ""