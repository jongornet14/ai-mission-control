# AI Mission Control - Test Makefile
# Provides easy commands to run all testing suites

.PHONY: help test-all test-quick test-integration test-performance test-system test-health install-deps clean

# Default target
help:
	@echo "🎯 AI Mission Control - Test Commands"
	@echo "======================================"
	@echo ""
	@echo "Setup:"
	@echo "  make install-deps     - Create venv and install dependencies"
	@echo "  make setup           - Alias for install-deps"
	@echo ""
	@echo "Testing:"
	@echo "  make test-all         - Run comprehensive test suite"
	@echo "  make test-quick       - Run quick validation tests"
	@echo "  make test-integration - Run integration tests only"
	@echo "  make test-performance - Run performance tests only"
	@echo "  make test-system      - Run system tests only"
	@echo "  make test-health      - Run basic health checks"
	@echo ""
	@echo "Utilities:"
	@echo "  make clean           - Clean up test artifacts"
	@echo "  make clean-all       - Clean everything including venv"
	@echo "  make activate        - Show how to activate venv"
	@echo ""
	@echo "Prerequisites:"
	@echo "  • Docker and docker-compose installed"
	@echo "  • Python 3.7+ installed"
	@echo "  • AI Mission Control services running"
	@echo ""
	@echo "Quick start:"
	@echo "  1. make install-deps  (creates isolated ai-mission-control/)"
	@echo "  2. make test-health"
	@echo "  3. make test-quick"

# Create virtual environment and install dependencies
install-deps:
	@echo "📦 Setting up testing environment..."
	@if [ ! -d "ai-mission-control" ]; then \
		echo "Creating virtual environment..."; \
		python3 -m venv ai-mission-control; \
	fi
	@echo "Activating virtual environment and installing dependencies..."
	@. ai-mission-control/bin/activate && pip install --upgrade pip
	@. ai-mission-control/bin/activate && pip install pytest requests numpy matplotlib aiohttp
	@echo "✅ Testing environment ready!"
	@echo ""
	@echo "To activate the environment manually:"
	@echo "  source ai-mission-control/bin/activate"

# Setup virtual environment (alias for install-deps)
setup: install-deps

# Activate virtual environment (helper target)
activate:
	@echo "To activate the virtual environment, run:"
	@echo "  source ai-mission-control/bin/activate"

# Run comprehensive test suite
test-all:
	@echo "🚀 Running comprehensive test suite..."
	@echo "This will take several minutes..."
	@if [ ! -d "ai-mission-control" ]; then echo "❌ Virtual environment not found. Run 'make install-deps' first."; exit 1; fi
	@. ai-mission-control/bin/activate && python3 test_orchestrator.py

# Run quick validation tests
test-quick:
	@echo "⚡ Running quick validation tests..."
	@if [ ! -d "ai-mission-control" ]; then echo "❌ Virtual environment not found. Run 'make install-deps' first."; exit 1; fi
	@. ai-mission-control/bin/activate && python3 test_orchestrator.py --quick

# Run integration tests only
test-integration:
	@echo "🔧 Running integration tests..."
	@if [ -f "integration_test.sh" ]; then \
		chmod +x integration_test.sh && ./integration_test.sh; \
	else \
		echo "❌ integration_test.sh not found"; \
		exit 1; \
	fi

# Run performance tests only
test-performance:
	@echo "🚀 Running performance tests..."
	@if [ ! -d "ai-mission-control" ]; then echo "❌ Virtual environment not found. Run 'make install-deps' first."; exit 1; fi
	@if [ -f "performance_test.py" ]; then \
		. ai-mission-control/bin/activate && python3 performance_test.py; \
	else \
		echo "❌ performance_test.py not found"; \
		exit 1; \
	fi

# Run system tests only
test-system:
	@echo "🏗️  Running system tests..."
	@if [ -f "test-systems.sh" ]; then \
		chmod +x test-systems.sh && ./test-systems.sh --all; \
	else \
		echo "❌ test-systems.sh not found"; \
		exit 1; \
	fi

# Run basic health checks
test-health:
	@echo "🏥 Running basic health checks..."
	@echo "Checking if services are running..."
	@for port in 50051 50052 50053 50054 8080; do \
		echo -n "Port $$port: "; \
		if curl -s --max-time 3 http://localhost:$$port/health >/dev/null 2>&1; then \
			echo "✅ Healthy"; \
		else \
			echo "❌ Not responding"; \
		fi; \
	done
	@echo ""
	@echo "Checking Docker containers..."
	@for container in ai-mc-trading ai-mc-unity ai-mc-gym ai-mc-modern-rl ai-mc-gateway; do \
		echo -n "$$container: "; \
		if docker ps --format "{{.Names}}" | grep -q "^$$container$$"; then \
			echo "✅ Running"; \
		else \
			echo "❌ Not running"; \
		fi; \
	done

# Run pytest-based tests
test-pytest:
	@echo "🧪 Running pytest tests..."
	@if [ ! -d "ai-mission-control" ]; then echo "❌ Virtual environment not found. Run 'make install-deps' first."; exit 1; fi
	@if [ -f "test_ai_mission_control.py" ]; then \
		. ai-mission-control/bin/activate && python3 -m pytest test_ai_mission_control.py -v; \
	elif [ -f "enhanced_test_suite.py" ]; then \
		. ai-mission-control/bin/activate && python3 -m pytest enhanced_test_suite.py -v; \
	else \
		echo "❌ No pytest files found"; \
		exit 1; \
	fi

# Test environment creation capability
test-env-creation:
	@echo "🎮 Testing environment creation..."
	@echo "Creating CartPole environment..."
	@response=$$(curl -s -X POST http://localhost:50053/create/CartPole-v1 2>/dev/null); \
	if echo "$$response" | grep -q "session_id"; then \
		echo "✅ Environment creation successful"; \
		session_id=$$(echo "$$response" | python3 -c "import sys, json; print(json.load(sys.stdin)['session_id'])" 2>/dev/null); \
		echo "Session ID: $$session_id"; \
		echo "Testing environment reset..."; \
		reset_response=$$(curl -s -X POST http://localhost:50053/reset/$$session_id 2>/dev/null); \
		if echo "$$reset_response" | grep -q "observation"; then \
			echo "✅ Environment reset successful"; \
		else \
			echo "❌ Environment reset failed"; \
		fi; \
	else \
		echo "❌ Environment creation failed"; \
	fi

# Test computation capabilities
test-computation:
	@echo "🧮 Testing computation capabilities..."
	@echo "Running 10-step RL simulation..."
	@response=$$(curl -s -X POST http://localhost:50053/create/CartPole-v1 2>/dev/null); \
	if echo "$$response" | grep -q "session_id"; then \
		session_id=$$(echo "$$response" | python3 -c "import sys, json; print(json.load(sys.stdin)['session_id'])" 2>/dev/null); \
		curl -s -X POST http://localhost:50053/reset/$$session_id >/dev/null 2>&1; \
		total_reward=0; \
		for step in $$(seq 1 10); do \
			action=$$(($$step % 2)); \
			step_response=$$(curl -s -X POST -H "Content-Type: application/json" -d "{\"action\": $$action}" http://localhost:50053/step/$$session_id 2>/dev/null); \
			if echo "$$step_response" | grep -q "reward"; then \
				reward=$$(echo "$$step_response" | python3 -c "import sys, json; print(json.load(sys.stdin)['reward'])" 2>/dev/null); \
				total_reward=$$(echo "$$total_reward + $$reward" | bc -l 2>/dev/null || echo "$$total_reward"); \
			fi; \
		done; \
		echo "✅ Computation test completed (Total reward: $$total_reward)"; \
	else \
		echo "❌ Computation test failed"; \
	fi

# Clean up test artifacts
clean:
	@echo "🧹 Cleaning up test artifacts..."
	@rm -f *.pyc
	@rm -rf __pycache__
	@rm -rf .pytest_cache
	@rm -f ai_mission_control_performance.png
	@rm -f test_results.json
	@echo "✅ Cleanup completed!"

# Clean everything including virtual environment
clean-all: clean
	@echo "🧹 Removing virtual environment..."
	@rm -rf ai-mission-control
	@echo "✅ Complete cleanup finished!"

# Start services (if docker-compose.yml exists)
start-services:
	@echo "🚀 Starting AI Mission Control services..."
	@if [ -f "docker-compose.yml" ]; then \
		docker-compose up -d; \
		echo "⏳ Waiting for services to start..."; \
		sleep 10; \
		make test-health; \
	else \
		echo "❌ docker-compose.yml not found"; \
		echo "Please ensure services are running manually"; \
		exit 1; \
	fi

# Stop services (if docker-compose.yml exists)
stop-services:
	@echo "🛑 Stopping AI Mission Control services..."
	@if [ -f "docker-compose.yml" ]; then \
		docker-compose down; \
	else \
		echo "❌ docker-compose.yml not found"; \
		echo "Please stop services manually"; \
		exit 1; \
	fi

# Show service logs
logs:
	@echo "📋 Showing service logs..."
	@if [ -f "docker-compose.yml" ]; then \
		docker-compose logs --tail=50; \
	else \
		echo "Individual container logs:"; \
		for container in ai-mc-trading ai-mc-unity ai-mc-gym ai-mc-modern-rl ai-mc-gateway; do \
			echo "=== $$container ==="; \
			docker logs --tail=10 $$container 2>/dev/null || echo "Container not found"; \
			echo ""; \
		done; \
	fi

# Development workflow
dev-test: test-health test-env-creation test-computation
	@echo "✅ Development tests completed!"

# CI/CD workflow
ci-test: install-deps test-health test-integration test-pytest
	@echo "✅ CI/CD tests completed!"

# Full validation workflow
validate: install-deps start-services test-all
	@echo "✅ Full validation completed!"

# Show system status
status:
	@echo "📊 AI Mission Control System Status"
	@echo "=================================="
	@echo ""
	@echo "Docker containers:"
	@docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" | grep ai-mc || echo "No AI Mission Control containers running"
	@echo ""
	@echo "Service health:"
	@make test-health
	@echo ""
	@echo "Resource usage:"
	@docker stats --no-stream --format "table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}" | grep ai-mc || echo "No resource data available"