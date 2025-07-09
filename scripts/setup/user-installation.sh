#!/bin/bash

# AI Mission Control - User Installation Script
# For users who want to test an existing AI Mission Control system

echo "Installing AI Mission Control test tools..."

# Install test dependencies
if command -v pip3 >/dev/null 2>&1; then
    pip3 install --user pytest requests docker
elif command -v pip >/dev/null 2>&1; then
    pip install --user pytest requests docker
else
    echo "Error: pip not found. Please install pip first."
    exit 1
fi

echo "Test dependencies installed!"
echo ""
echo "Usage:"
echo "  python test_ai_mission_control.py     # Run tests"
echo "  pytest test_ai_mission_control.py -v  # Verbose output"