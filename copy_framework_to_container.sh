#!/bin/bash
# FILENAME: copy_framework_to_container.sh
# Script to copy all Universal RL Framework files into the AI Mission Control container

set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}Copying Universal RL Framework to container...${NC}"

# Start a temporary container to copy files
CONTAINER_NAME="ai-mc-copy-temp"

echo "Starting temporary container..."
docker run -d --name $CONTAINER_NAME \
    --gpus all \
    -v $(pwd):/workspace/host \
    ai-mission-control:latest sleep 300

# Function to copy file and create directory if needed
copy_file() {
    local src=$1
    local dest=$2
    local desc=$3
    
    if [ -f "$src" ]; then
        echo "Copying $desc..."
        docker exec $CONTAINER_NAME mkdir -p $(dirname $dest)
        docker exec $CONTAINER_NAME cp /workspace/host/$src $dest
        echo -e "${GREEN}✓${NC} $desc copied to $dest"
    else
        echo "⚠️  $src not found, skipping..."
    fi
}

# Copy main framework files
echo -e "\n${BLUE}Main Framework Files:${NC}"
copy_file "scripts/universal_rl.py" "/workspace/scripts/universal_rl.py" "Universal RL Script"

# Copy algorithm files
echo -e "\n${BLUE}Algorithm Files:${NC}"
copy_file "scripts/algorithms/__init__.py" "/workspace/scripts/algorithms/__init__.py" "Algorithms __init__"
copy_file "scripts/algorithms/ppo.py" "/workspace/scripts/algorithms/ppo.py" "PPO Algorithm"

# Copy environment files
echo -e "\n${BLUE}Environment Files:${NC}"
copy_file "scripts/environments/__init__.py" "/workspace/scripts/environments/__init__.py" "Environments __init__"
copy_file "scripts/environments/gym_wrapper.py" "/workspace/scripts/environments/gym_wrapper.py" "Gym Environment Wrapper"

# Copy configuration files
echo -e "\n${BLUE}Configuration Files:${NC}"
copy_file "scripts/configs/cartpole_ppo.yaml" "/workspace/configs/cartpole_ppo.yaml" "CartPole PPO Config"
copy_file "scripts/configs/pendulum_ppo.yaml" "/workspace/configs/pendulum_ppo.yaml" "Pendulum PPO Config"
copy_file "scripts/configs/hyperopt_example.yaml" "/workspace/configs/hyperopt_example.yaml" "Hyperopt Example Config"
copy_file "scripts/configs/mujoco_example.yaml" "/workspace/configs/mujoco_example.yaml" "MuJoCo Example Config"

# Copy run examples
echo -e "\n${BLUE}Example Scripts:${NC}"
copy_file "scripts/run_examples.sh" "/workspace/scripts/run_examples.sh" "Run Examples Script"

# Make scripts executable
echo -e "\n${BLUE}Setting permissions...${NC}"
docker exec $CONTAINER_NAME chmod +x /workspace/scripts/universal_rl.py
docker exec $CONTAINER_NAME chmod +x /workspace/scripts/run_examples.sh

# Create a snapshot of this container with all files
echo -e "\n${BLUE}Creating snapshot with framework files...${NC}"
docker commit $CONTAINER_NAME ai-mission-control:with-framework

# Clean up temporary container
echo "Cleaning up..."
docker stop $CONTAINER_NAME
docker rm $CONTAINER_NAME

echo -e "\n${GREEN}✅ Universal RL Framework copied successfully!${NC}"
echo ""
echo "Your framework is now available in the container:"
echo "  Main script: /workspace/scripts/universal_rl.py"
echo "  Algorithms:  /workspace/scripts/algorithms/"
echo "  Environments: /workspace/scripts/environments/"
echo "  Configs:     /workspace/configs/"
echo ""
echo "To use the updated container:"
echo "  docker run -it --rm --gpus all ai-mission-control:with-framework /bin/bash"
echo ""
echo "Or update your Makefile to use 'ai-mission-control:with-framework' as the image name."