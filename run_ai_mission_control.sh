#!/bin/bash

# AI Mission Control - Run Script
# Usage examples for running the container

IMAGE_NAME="ai-mission-control:latest"

case "${1:-}" in
    "shell"|"bash")
        echo "Starting AI Mission Control shell..."
        docker run -it --rm \
            --gpus all \
            -v "$(pwd)":/workspace/project \
            ${IMAGE_NAME} /bin/bash
        ;;
    "jupyter")
        echo "Starting Jupyter Lab on port 8080..."
        docker run -it --rm \
            --gpus all \
            -p 8080:8080 \
            -v "$(pwd)":/workspace/project \
            ${IMAGE_NAME} jupyter lab --ip=0.0.0.0 --port=8080 --no-browser --allow-root
        ;;
    "experiment")
        echo "Running RL experiment..."
        if [ -z "$2" ]; then
            echo "Usage: $0 experiment <script_name>"
            echo "Example: $0 experiment rl_tests.py"
            exit 1
        fi
        docker run -it --rm \
            --gpus all \
            -v "$(pwd)":/workspace/project \
            ${IMAGE_NAME} python3 /workspace/project/$2
        ;;
    "test")
        echo "Running installation test..."
        docker run --rm \
            --gpus all \
            ${IMAGE_NAME} python3 /workspace/scripts/verify_installation.py
        ;;
    *)
        echo "AI Mission Control - Usage:"
        echo "  $0 shell       - Start interactive shell"
        echo "  $0 jupyter     - Start Jupyter Lab on port 8080"
        echo "  $0 experiment <script> - Run an experiment script"
        echo "  $0 test        - Test installation"
        echo ""
        echo "Examples:"
        echo "  $0 shell"
        echo "  $0 jupyter"
        echo "  $0 experiment rl_tests.py"
        echo "  $0 test"
        ;;
esac
