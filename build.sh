#!/bin/bash

# FILENAME: build.sh
# AI Mission Control - Build and Installation Script
# Version: 1.0.0
# Architecture: x86_64
# Base: Ubuntu 22.04.1 with CUDA

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
CONTAINER_NAME="ai-mission-control"
IMAGE_NAME="ai-mission-control:latest"
DOCKERFILE_NAME="Dockerfile"
BUILD_ARGS=""

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo -e "${BLUE}"
    echo "=========================================="
    echo "   AI Mission Control - Build Script"
    echo "   Hyperparameter Optimization for RL"
    echo "=========================================="
    echo -e "${NC}"
}

# Check requirements
check_requirements() {
    print_status "Checking system requirements..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    # Check Docker daemon
    if ! docker info &> /dev/null; then
        print_error "Docker daemon is not running. Please start Docker."
        exit 1
    fi
    
    # Check NVIDIA Docker (optional but recommended)
    if command -v nvidia-docker &> /dev/null; then
        print_success "NVIDIA Docker runtime detected"
    elif docker info | grep -q nvidia; then
        print_success "NVIDIA Docker runtime detected via Docker info"
    else
        print_warning "NVIDIA Docker runtime not detected. GPU support may be limited."
        print_warning "Install nvidia-docker2 for optimal GPU performance."
    fi
    
    # Check required files
    if [ ! -f "hypercontroller-exact.yml" ]; then
        print_error "hypercontroller-exact.yml not found in current directory"
        exit 1
    fi
    
    print_success "All requirements satisfied"
}

# Build Docker image
build_image() {
    print_status "Building AI Mission Control Docker image..."
    print_status "This may take 15-30 minutes depending on your internet connection..."
    
    # Build the image
    docker build \
        --tag ${IMAGE_NAME} \
        --file ${DOCKERFILE_NAME} \
        --build-arg BUILDKIT_INLINE_CACHE=1 \
        ${BUILD_ARGS} \
        .
    
    if [ $? -eq 0 ]; then
        print_success "Docker image built successfully: ${IMAGE_NAME}"
    else
        print_error "Failed to build Docker image"
        exit 1
    fi
}

# Verify installation
verify_installation() {
    print_status "Verifying installation..."
    
    # Run verification script inside container
    docker run --rm ${IMAGE_NAME} python3 /workspace/scripts/verify_installation.py
    
    if [ $? -eq 0 ]; then
        print_success "Installation verification completed successfully"
    else
        print_error "Installation verification failed"
        exit 1
    fi
}

# Test GPU support
test_gpu_support() {
    print_status "Testing GPU support..."
    
    # Try to run with GPU support
    if docker run --rm --gpus all ${IMAGE_NAME} python3 -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'GPU Count: {torch.cuda.device_count()}' if torch.cuda.is_available() else 'No GPU detected')" 2>/dev/null; then
        print_success "GPU support test completed"
    else
        print_warning "GPU support test failed or no GPU available"
        print_warning "Container will work with CPU-only mode"
    fi
}

# Create usage examples
create_examples() {
    print_status "Creating usage examples..."
    
    # Create run script
    cat > run_ai_mission_control.sh << 'EOF'
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
EOF
    
    chmod +x run_ai_mission_control.sh
    
    # Create Docker Compose file
    cat > docker-compose.yml << EOF
version: '3.8'

services:
  ai-mission-control:
    image: ${IMAGE_NAME}
    container_name: ${CONTAINER_NAME}
    volumes:
      - ./:/workspace/project
      - ai_mc_experiments:/workspace/experiments
      - ai_mc_logs:/workspace/logs
      - ai_mc_models:/workspace/models
    ports:
      - "8080:8080"  # Jupyter Lab
      - "8888:8888"  # Alternative Jupyter port
      - "6006:6006"  # TensorBoard
    environment:
      - CUDA_VISIBLE_DEVICES=0
      - PYTHONPATH=/workspace/project
    stdin_open: true
    tty: true
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=all

volumes:
  ai_mc_experiments:
  ai_mc_logs:
  ai_mc_models:
EOF

    print_success "Created run_ai_mission_control.sh and docker-compose.yml"
}

# Display usage information
show_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --build-only       Build image without verification"
    echo "  --verify-only      Only run verification (requires pre-built image)"
    echo "  --no-gpu-test      Skip GPU support testing"
    echo "  --clean            Remove existing image before building"
    echo "  --help             Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                 # Full build and verification"
    echo "  $0 --build-only    # Build image only"
    echo "  $0 --clean         # Clean build"
}

# Clean existing images
clean_images() {
    print_status "Cleaning existing images..."
    
    if docker images | grep -q ${CONTAINER_NAME}; then
        docker rmi ${IMAGE_NAME} || true
        print_success "Cleaned existing images"
    else
        print_status "No existing images to clean"
    fi
}

# Main execution
main() {
    print_header
    
    # Parse command line arguments
    BUILD_ONLY=false
    VERIFY_ONLY=false
    SKIP_GPU_TEST=false
    CLEAN_BUILD=false
    
    for arg in "$@"; do
        case $arg in
            --build-only)
                BUILD_ONLY=true
                ;;
            --verify-only)
                VERIFY_ONLY=true
                ;;
            --no-gpu-test)
                SKIP_GPU_TEST=true
                ;;
            --clean)
                CLEAN_BUILD=true
                ;;
            --help)
                show_usage
                exit 0
                ;;
            *)
                print_error "Unknown option: $arg"
                show_usage
                exit 1
                ;;
        esac
    done
    
    # Execute based on options
    if [ "$VERIFY_ONLY" = true ]; then
        verify_installation
        if [ "$SKIP_GPU_TEST" = false ]; then
            test_gpu_support
        fi
    else
        check_requirements
        
        if [ "$CLEAN_BUILD" = true ]; then
            clean_images
        fi
        
        build_image
        
        if [ "$BUILD_ONLY" = false ]; then
            verify_installation
            if [ "$SKIP_GPU_TEST" = false ]; then
                test_gpu_support
            fi
        fi
    fi
    
    create_examples
    
    print_success "AI Mission Control setup completed!"
    echo ""
    print_status "Quick start:"
    echo "  ./run_ai_mission_control.sh shell    # Interactive shell"
    echo "  ./run_ai_mission_control.sh jupyter  # Jupyter Lab"
    echo "  ./run_ai_mission_control.sh test     # Verify installation"
    echo ""
    print_status "For more advanced usage, see docker-compose.yml"
}

# Run main function
main "$@"