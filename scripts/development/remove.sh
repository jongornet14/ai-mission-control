#!/bin/bash

# AI Mission Control - Removal Script
# Completely removes all containers, images, volumes, and project files for clean reinstall

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Project info
PROJECT_NAME="AI Mission Control"
VERSION="1.0.0-removal"

echo -e "${RED}"
echo "=========================================================="
echo "  AI Mission Control - Removal Script v$VERSION"
echo "=========================================================="
echo -e "${NC}"
echo "This script will completely remove all AI Mission Control components"
echo "for a clean reinstall."
echo ""

# Function to print status
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_section() {
    echo -e "${PURPLE}[SECTION]${NC} $1"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check if we're in the right directory
check_project_directory() {
    if [[ ! -f "docker-compose.yml" ]] && [[ ! -d "environments" ]]; then
        print_error "This doesn't appear to be an AI Mission Control project directory."
        print_error "Please run this script from the project root directory."
        exit 1
    fi
}

# Function to stop and remove containers
stop_and_remove_containers() {
    print_section "Stopping and removing containers..."
    
    if command_exists docker-compose && [[ -f "docker-compose.yml" ]]; then
        print_status "Stopping all services..."
        docker-compose down -v --remove-orphans || true
        
        print_status "Removing any remaining containers..."
        # Remove containers by name pattern
        docker ps -a --filter "name=ai-mc-" --format "{{.Names}}" | xargs -r docker rm -f || true
        
        # Remove containers by network
        docker ps -a --filter "network=ai-mission-control-network" --format "{{.Names}}" | xargs -r docker rm -f || true
    else
        print_warning "docker-compose.yml not found, attempting manual container cleanup..."
        
        # Stop and remove containers by name pattern
        docker ps -a --filter "name=ai-mc-" --format "{{.Names}}" | xargs -r docker stop || true
        docker ps -a --filter "name=ai-mc-" --format "{{.Names}}" | xargs -r docker rm -f || true
    fi
    
    print_status "Containers stopped and removed"
}

# Function to remove Docker images
remove_docker_images() {
    print_section "Removing Docker images..."
    
    # Remove images by repository pattern
    print_status "Removing AI Mission Control images..."
    docker images --filter "reference=ai-mission-control*" --format "{{.Repository}}:{{.Tag}}" | xargs -r docker rmi -f || true
    docker images --filter "reference=*ai-mc-*" --format "{{.Repository}}:{{.Tag}}" | xargs -r docker rmi -f || true
    
    # Remove images by directory name pattern
    local current_dir=$(basename "$PWD")
    docker images --filter "reference=${current_dir}*" --format "{{.Repository}}:{{.Tag}}" | xargs -r docker rmi -f || true
    
    print_status "Docker images removed"
}

# Function to remove Docker volumes
remove_docker_volumes() {
    print_section "Removing Docker volumes..."
    
    # Remove named volumes
    print_status "Removing named volumes..."
    docker volume ls --filter "name=ai-mc-" --format "{{.Name}}" | xargs -r docker volume rm -f || true
    docker volume ls --filter "name=ai-mission-control" --format "{{.Name}}" | xargs -r docker volume rm -f || true
    
    # Remove orphaned volumes
    print_status "Removing orphaned volumes..."
    docker volume prune -f || true
    
    print_status "Docker volumes removed"
}

# Function to remove Docker networks
remove_docker_networks() {
    print_section "Removing Docker networks..."
    
    # Remove custom networks
    print_status "Removing AI Mission Control networks..."
    docker network ls --filter "name=ai-mission-control" --format "{{.Name}}" | xargs -r docker network rm || true
    
    print_status "Docker networks removed"
}

# Function to remove project files
remove_project_files() {
    print_section "Removing project files and directories..."
    
    print_status "Removing environment containers..."
    rm -rf environments/
    
    print_status "Removing services..."
    rm -rf services/
    
    print_status "Removing client library..."
    rm -rf client/
    
    print_status "Removing protocol definitions..."
    rm -rf proto/
    
    print_status "Removing Docker configurations..."
    rm -rf docker/
    
    print_status "Removing examples..."
    rm -rf examples/
    
    print_status "Removing scripts..."
    rm -rf scripts/
    
    print_status "Removing documentation..."
    rm -rf docs/
    
    print_status "Removing tests..."
    rm -rf tests/
    
    print_status "Removing logs..."
    rm -rf logs/
    
    print_status "Removing configuration files..."
    rm -f docker-compose.yml
    rm -f Makefile
    rm -f .gitignore
    rm -f .env.template
    rm -f README.md
    
    print_status "Removing build artifacts..."
    rm -rf unity_builds/
    rm -rf __pycache__/
    rm -rf *.egg-info/
    rm -f *.log
    
    print_status "Project files removed"
}

# Function to clean up Python artifacts
cleanup_python_artifacts() {
    print_section "Cleaning up Python artifacts..."
    
    # Remove Python cache files
    find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    find . -type f -name "*.pyc" -delete 2>/dev/null || true
    find . -type f -name "*.pyo" -delete 2>/dev/null || true
    find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
    
    print_status "Python artifacts cleaned"
}

# Function to perform comprehensive Docker cleanup
comprehensive_docker_cleanup() {
    print_section "Performing comprehensive Docker cleanup..."
    
    print_status "Removing unused Docker resources..."
    docker system prune -af --volumes || true
    
    print_status "Removing dangling images..."
    docker image prune -af || true
    
    print_status "Removing unused networks..."
    docker network prune -f || true
    
    print_status "Docker cleanup complete"
}

# Function to show removal summary
show_removal_summary() {
    print_section "Removal Summary"
    
    echo -e "${CYAN}The following components have been removed:${NC}"
    echo "  All AI Mission Control containers"
    echo "  All AI Mission Control Docker images"
    echo "  All AI Mission Control Docker volumes"
    echo "  All AI Mission Control Docker networks"
    echo "  All project files and directories"
    echo "  All configuration files"
    echo "  All build artifacts and logs"
    echo "  All Python cache files"
    echo ""
    
    # Check what's left
    local remaining_containers=$(docker ps -a --filter "name=ai-mc-" --format "{{.Names}}" 2>/dev/null | wc -l)
    local remaining_images=$(docker images --filter "reference=*ai-mc-*" --format "{{.Repository}}" 2>/dev/null | wc -l)
    local remaining_volumes=$(docker volume ls --filter "name=ai-mc-" --format "{{.Name}}" 2>/dev/null | wc -l)
    
    if [[ $remaining_containers -eq 0 && $remaining_images -eq 0 && $remaining_volumes -eq 0 ]]; then
        echo -e "${GREEN}Complete removal successful!${NC}"
    else
        print_warning "Some resources may still remain:"
        [[ $remaining_containers -gt 0 ]] && echo "  Containers: $remaining_containers"
        [[ $remaining_images -gt 0 ]] && echo "  Images: $remaining_images"
        [[ $remaining_volumes -gt 0 ]] && echo "  Volumes: $remaining_volumes"
    fi
}

# Main function
main() {
    # Parse command line arguments
    case "${1:-}" in
        -h|--help)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  -h, --help        Show this help message"
            echo "  --force           Skip confirmation prompts"
            echo "  --keep-images     Keep Docker images (only remove containers and files)"
            echo "  --docker-only     Only clean Docker resources (keep project files)"
            echo "  --files-only      Only remove project files (keep Docker resources)"
            exit 0
            ;;
        --force)
            FORCE_REMOVAL=true
            ;;
        --keep-images)
            KEEP_IMAGES=true
            ;;
        --docker-only)
            DOCKER_ONLY=true
            ;;
        --files-only)
            FILES_ONLY=true
            ;;
    esac
    
    echo -e "${YELLOW}WARNING: This will completely remove all AI Mission Control components!${NC}"
    echo ""
    echo "This includes:"
    echo "  - All running containers and services"
    echo "  - All Docker images, volumes, and networks"
    echo "  - All project files and directories"
    echo "  - All configuration and log files"
    echo ""
    
    if [[ "${FORCE_REMOVAL:-}" != "true" ]]; then
        read -p "Are you absolutely sure you want to proceed? (type 'yes' to confirm): " -r
        echo
        if [[ ! $REPLY == "yes" ]]; then
            echo "Removal cancelled."
            exit 0
        fi
    fi
    
    echo ""
    print_status "Starting complete removal of AI Mission Control..."
    
    # Check if we're in the right directory
    check_project_directory
    
    # Execute removal steps based on options
    if [[ "${FILES_ONLY:-}" != "true" ]]; then
        stop_and_remove_containers
        remove_docker_networks
        remove_docker_volumes
        
        if [[ "${KEEP_IMAGES:-}" != "true" ]]; then
            remove_docker_images
        fi
        
        if [[ "${FORCE_REMOVAL:-}" == "true" ]]; then
            comprehensive_docker_cleanup
        fi
    fi
    
    if [[ "${DOCKER_ONLY:-}" != "true" ]]; then
        remove_project_files
        cleanup_python_artifacts
    fi
    
    show_removal_summary
    
    echo ""
    echo -e "${GREEN}AI Mission Control removal complete!${NC}"
    echo ""
    echo -e "${CYAN}To reinstall:${NC}"
    echo "  1. Run: ./setup.sh"
    echo "  2. Run: make build"
    echo "  3. Run: make up"
    echo ""
}

# Run main function if script is executed directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi