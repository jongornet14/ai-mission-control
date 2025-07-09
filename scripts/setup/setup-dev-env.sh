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
