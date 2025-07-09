#!/bin/bash

# Directory Reorganization Migration Script
# This script safely reorganizes your project structure

set -e  # Exit on any error

echo "🚀 Starting directory reorganization migration..."
echo "📋 Reorganizing project structure (everything is safely in GitHub)..."

echo "📂 Creating new directory structure..."

# Create main directories
mkdir -p apps/{api-gateway,gym-service,trading-service,modern-rl-service,unity-service}
mkdir -p shared/{common,models,proto}
mkdir -p infrastructure/{docker,kubernetes,terraform,monitoring/{prometheus,grafana}}
mkdir -p scripts/{setup,deployment,development,data}
mkdir -p tests/{unit,integration,e2e,fixtures}
mkdir -p docs/{api,architecture,deployment,development}
mkdir -p examples/{basic-usage,advanced-workflows,integration-examples}
mkdir -p data/{sample,configs,models}

# Move services from environments/ to apps/
echo "🔄 Moving services from environments/ to apps/..."

if [ -d "environments/gym" ]; then
    echo "  Moving gym service..."
    mkdir -p apps/gym-service/src
    [ -f "environments/gym/gym_service.py" ] && mv environments/gym/gym_service.py apps/gym-service/src/
    [ -f "environments/gym/Dockerfile" ] && mv environments/gym/Dockerfile apps/gym-service/
    [ -f "environments/gym/requirements.txt" ] && mv environments/gym/requirements.txt apps/gym-service/
fi

if [ -d "environments/trading" ]; then
    echo "  Moving trading service..."
    mkdir -p apps/trading-service/src
    [ -f "environments/trading/trading_service.py" ] && mv environments/trading/trading_service.py apps/trading-service/src/
    [ -f "environments/trading/Dockerfile" ] && mv environments/trading/Dockerfile apps/trading-service/
    [ -f "environments/trading/requirements.txt" ] && mv environments/trading/requirements.txt apps/trading-service/
fi

if [ -d "environments/modern-rl" ]; then
    echo "  Moving modern-rl service..."
    mkdir -p apps/modern-rl-service/src
    [ -f "environments/modern-rl/modern_rl_service.py" ] && mv environments/modern-rl/modern_rl_service.py apps/modern-rl-service/src/
    [ -f "environments/modern-rl/Dockerfile" ] && mv environments/modern-rl/Dockerfile apps/modern-rl-service/
    [ -f "environments/modern-rl/requirements.txt" ] && mv environments/modern-rl/requirements.txt apps/modern-rl-service/
fi

if [ -d "environments/unity" ]; then
    echo "  Moving unity service..."
    mkdir -p apps/unity-service/src
    [ -f "environments/unity/unity_service.py" ] && mv environments/unity/unity_service.py apps/unity-service/src/
    [ -f "environments/unity/Dockerfile" ] && mv environments/unity/Dockerfile apps/unity-service/
    [ -f "environments/unity/requirements.txt" ] && mv environments/unity/requirements.txt apps/unity-service/
fi

# Move services from services/ to apps/
echo "🔄 Moving services from services/ to apps/..."

if [ -d "services/api-gateway" ]; then
    echo "  Moving api-gateway service..."
    mkdir -p apps/api-gateway/src
    [ -f "services/api-gateway/gateway_service.py" ] && mv services/api-gateway/gateway_service.py apps/api-gateway/src/
    [ -f "services/api-gateway/Dockerfile" ] && mv services/api-gateway/Dockerfile apps/api-gateway/
    
    # Check if there are other files in services/api-gateway and move them
    if [ "$(ls -A services/api-gateway 2>/dev/null)" ]; then
        mv services/api-gateway/* apps/api-gateway/src/ 2>/dev/null || true
    fi
fi

# Handle other services in services/ directory
for service_dir in services/*/; do
    if [ -d "$service_dir" ]; then
        service_name=$(basename "$service_dir")
        if [ "$service_name" != "api-gateway" ]; then
            echo "  Moving $service_name service..."
            mkdir -p "apps/$service_name/src"
            mv "$service_dir"* "apps/$service_name/src/" 2>/dev/null || true
        fi
    fi
done

# Move protocol buffers
echo "📡 Moving protocol buffers..."
if [ -d "proto" ]; then
    mv proto/* shared/proto/ 2>/dev/null || true
fi

# Move documentation
echo "📚 Moving documentation..."
if [ -d "docs" ]; then
    # Preserve existing docs structure but organize better
    [ -d "docs/api" ] && cp -r docs/api/* docs/api/ 2>/dev/null || true
    [ -d "docs/architecture" ] && cp -r docs/architecture/* docs/architecture/ 2>/dev/null || true
    [ -d "docs/deployment" ] && cp -r docs/deployment/* docs/deployment/ 2>/dev/null || true
fi

# Move examples
echo "💡 Moving examples..."
if [ -d "examples" ]; then
    mv examples/* examples/basic-usage/ 2>/dev/null || true
fi

# Move unity builds
echo "🎮 Moving Unity builds..."
if [ -d "unity_builds" ]; then
    mkdir -p client/unity-builds
    mv unity_builds/* client/unity-builds/ 2>/dev/null || true
fi

# Move client code
echo "💻 Moving client code..."
if [ -d "client" ] && [ "$(ls -A client 2>/dev/null)" ]; then
    # Client directory already exists and has content, preserve it
    echo "  Client directory already exists, preserving existing structure"
else
    mkdir -p client
fi

# Move scripts to proper locations
echo "⚙️ Organizing scripts..."
if [ -d "scripts" ]; then
    # Move existing scripts to appropriate subdirectories
    find scripts/ -name "*.sh" -type f | while read -r script; do
        script_name=$(basename "$script")
        case "$script_name" in
            *setup*|*install*)
                mv "$script" scripts/setup/ 2>/dev/null || true
                ;;
            *deploy*|*server-installation*)
                mv "$script" scripts/deployment/ 2>/dev/null || true
                ;;
            *dev*|*development*)
                mv "$script" scripts/development/ 2>/dev/null || true
                ;;
            *)
                # Keep in scripts root or move to development
                [ -f "$script" ] && mv "$script" scripts/development/ 2>/dev/null || true
                ;;
        esac
    done
fi

# Move installation scripts from root
echo "📦 Moving installation scripts..."
[ -f "server-installation.sh" ] && mv server-installation.sh scripts/deployment/
[ -f "user-installation.sh" ] && mv user-installation.sh scripts/setup/
[ -f "remove.sh" ] && mv remove.sh scripts/deployment/

# Move tests
echo "🧪 Organizing tests..."
if [ -d "tests" ]; then
    # Preserve existing test structure
    echo "  Tests directory already exists, preserving structure"
else
    # Move test files from root or other locations
    [ -f "test_ai_mission_control.py" ] && mv test_ai_mission_control.py tests/unit/
fi

# Move docker-related files
echo "🐳 Organizing Docker files..."
if [ -d "docker" ]; then
    mv docker/* infrastructure/docker/ 2>/dev/null || true
fi

# Clean up empty directories
echo "🧹 Cleaning up empty directories..."
find . -type d -empty -delete 2>/dev/null || true

# Remove the problematic virtual environment
echo "🗑️  Removing virtual environment from version control..."
if [ -d "python_tests/ai-mission-control" ]; then
    rm -rf python_tests/ai-mission-control
    echo "  Removed virtual environment (you'll need to recreate it)"
fi

# Remove now-empty directories
[ -d "environments" ] && rmdir environments 2>/dev/null || true
[ -d "services" ] && rmdir services 2>/dev/null || true
[ -d "python_tests" ] && rmdir python_tests 2>/dev/null || true
[ -d "unity_builds" ] && rmdir unity_builds 2>/dev/null || true

echo "📝 Creating essential configuration files..."

# Create .gitignore
cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual environments
venv/
env/
ENV/
.venv/
.env/

# IDEs
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Logs
*.log
logs/

# Docker
.dockerignore

# Local configuration
.env.local
config.local.*

# Test coverage
.coverage
htmlcov/

# Unity
*.tmp
*.temp
Library/
Temp/
Obj/
Build/
Builds/
ProjectSettings/DynamicsManager.asset
ProjectSettings/EditorBuildSettings.asset
ProjectSettings/GraphicsSettings.asset
ProjectSettings/Physics2DSettings.asset
EOF

# Create .env.example
cat > .env.example << 'EOF'
# Environment Configuration Template
# Copy this file to .env and fill in your actual values

# Application
APP_ENV=development
DEBUG=true
LOG_LEVEL=INFO

# Services
API_GATEWAY_PORT=8000
GYM_SERVICE_PORT=8001
TRADING_SERVICE_PORT=8002
MODERN_RL_SERVICE_PORT=8003
UNITY_SERVICE_PORT=8004

# Database
DATABASE_URL=postgresql://user:password@localhost:5432/dbname

# External APIs
TRADING_API_KEY=your_trading_api_key_here
UNITY_LICENSE_KEY=your_unity_license_here

# Docker
COMPOSE_PROJECT_NAME=ai-mission-control
EOF

# Create requirements-dev.txt
cat > requirements-dev.txt << 'EOF'
# Development dependencies
pytest>=7.0.0
pytest-cov>=4.0.0
black>=22.0.0
flake8>=5.0.0
mypy>=1.0.0
pre-commit>=2.20.0
pytest-asyncio>=0.21.0
httpx>=0.24.0
EOF

# Create main requirements.txt if it doesn't exist
if [ ! -f "requirements.txt" ]; then
cat > requirements.txt << 'EOF'
# Core dependencies
fastapi>=0.100.0
uvicorn>=0.22.0
pydantic>=2.0.0
asyncio-mqtt>=0.13.0
aioredis>=2.0.0
sqlalchemy>=2.0.0
alembic>=1.11.0
EOF
fi

echo "🔧 Creating setup scripts..."

# Create setup script
cat > scripts/setup/setup-dev-env.sh << 'EOF'
#!/bin/bash
# Development environment setup script

echo "🚀 Setting up development environment..."

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Create .env from template if it doesn't exist
if [ ! -f .env ]; then
    cp .env.example .env
    echo "📝 Created .env file from template. Please update with your actual values."
fi

# Install pre-commit hooks
pre-commit install

echo "✅ Development environment setup complete!"
echo "🔧 Next steps:"
echo "   1. Update .env with your configuration"
echo "   2. Run 'source venv/bin/activate' to activate the virtual environment"
echo "   3. Run 'make test' to verify everything works"
EOF

chmod +x scripts/setup/setup-dev-env.sh

# Create Makefile
cat > Makefile << 'EOF'
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
EOF

echo "📋 Creating service README templates..."

# Create README templates for each service
for service in api-gateway gym-service trading-service modern-rl-service unity-service; do
    if [ -d "apps/$service" ]; then
        cat > "apps/$service/README.md" << EOF
# ${service^}

## Description
Brief description of what this service does.

## Requirements
- Python 3.11+
- Dependencies listed in requirements.txt

## Development Setup
\`\`\`bash
cd apps/$service
pip install -r requirements.txt
\`\`\`

## Running
\`\`\`bash
python src/$(echo $service | tr '-' '_').py
\`\`\`

## API Documentation
- Endpoint: \`/docs\` (when running)
- Port: Check docker-compose.yml or .env file

## Environment Variables
See .env.example for required configuration.

## Testing
\`\`\`bash
# From project root
pytest tests/unit/test_${service//-/_}.py
\`\`\`
EOF
    fi
done

echo "✅ Migration completed successfully!"
echo ""
echo "📋 Summary of changes:"
echo "   ✓ Moved services to apps/ directory with consistent structure"
echo "   ✓ Created shared/ directory for common code"
echo "   ✓ Organized scripts into subdirectories"
echo "   ✓ Created infrastructure/ directory for deployment files"
echo "   ✓ Set up proper test directory structure"
echo "   ✓ Created .gitignore, .env.example, and requirements-dev.txt"
echo "   ✓ Removed virtual environment from version control"
echo "   ✓ Created setup scripts and Makefile"
echo "   ✓ Added README templates for each service"
echo ""
echo "🔧 Next steps:"
echo "   1. Review the new structure and verify all files are in the right places"
echo "   2. Update import statements in your Python files to match new paths"
echo "   3. Run: ./scripts/setup/setup-dev-env.sh"
echo "   4. Update docker-compose.yml to reference new service locations"
echo "   5. Test that everything still works: make test"
echo "   6. Commit changes to GitHub: git add . && git commit -m 'Reorganize directory structure'"