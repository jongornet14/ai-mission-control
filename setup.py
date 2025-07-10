#!/usr/bin/env python3
"""
FILENAME: setup.py
AI Mission Control - Setup and Installation Script
Hyperparameter Optimization for RL Neural Networks
"""

import os
import sys
import subprocess
import platform
import shutil
from pathlib import Path
import argparse

class Colors:
    """ANSI color codes for terminal output"""
    BLUE = '\033[0;34m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    RED = '\033[0;31m'
    NC = '\033[0m'  # No Color

class AIMissionControlSetup:
    """Main setup class for AI Mission Control"""
    
    def __init__(self):
        self.name = "ai-mission-control"
        self.version = "1.0.0"
        self.description = "Hyperparameter Optimization for RL Neural Networks"
        
    def print_status(self, message):
        print(f"{Colors.BLUE}[INFO]{Colors.NC} {message}")
        
    def print_success(self, message):
        print(f"{Colors.GREEN}[SUCCESS]{Colors.NC} {message}")
        
    def print_warning(self, message):
        print(f"{Colors.YELLOW}[WARNING]{Colors.NC} {message}")
        
    def print_error(self, message):
        print(f"{Colors.RED}[ERROR]{Colors.NC} {message}")
        
    def print_header(self):
        print(f"{Colors.BLUE}")
        print("=" * 50)
        print("   AI Mission Control - Setup Script")
        print("   Hyperparameter Optimization for RL")
        print(f"   Version: {self.version}")
        print("=" * 50)
        print(f"{Colors.NC}")
        
    def check_system_requirements(self):
        """Check if system meets requirements"""
        self.print_status("Checking system requirements...")
        
        # Check Python version
        if sys.version_info < (3, 8):
            self.print_error("Python 3.8 or higher is required")
            return False
            
        # Check platform
        if platform.system() not in ['Linux', 'Darwin', 'Windows']:
            self.print_error("Unsupported platform")
            return False
            
        # Check Docker
        if not shutil.which('docker'):
            self.print_error("Docker is not installed")
            self.print_status("Please install Docker: https://docs.docker.com/get-docker/")
            return False
            
        # Check Docker daemon
        try:
            subprocess.run(['docker', 'info'], 
                         capture_output=True, check=True)
        except subprocess.CalledProcessError:
            self.print_error("Docker daemon is not running")
            return False
            
        self.print_success("System requirements satisfied")
        return True
        
    def check_gpu_support(self):
        """Check for GPU support"""
        self.print_status("Checking GPU support...")
        
        # Check NVIDIA Docker
        try:
            result = subprocess.run(['docker', 'info'], 
                                  capture_output=True, text=True)
            if 'nvidia' in result.stdout.lower():
                self.print_success("NVIDIA Docker runtime detected")
                return True
        except:
            pass
            
        # Check nvidia-docker command
        if shutil.which('nvidia-docker'):
            self.print_success("NVIDIA Docker command found")
            return True
            
        self.print_warning("GPU support not detected")
        self.print_warning("Install nvidia-docker2 for GPU acceleration")
        return False
        
    def create_project_structure(self, project_dir):
        """Create project directory structure"""
        self.print_status(f"Creating project structure in {project_dir}")
        
        directories = [
            'experiments',
            'logs', 
            'models',
            'data',
            'configs',
            'scripts',
            'notebooks',
            'results'
        ]
        
        for directory in directories:
            dir_path = Path(project_dir) / directory
            dir_path.mkdir(parents=True, exist_ok=True)
            
        self.print_success("Project structure created")
        
    def create_example_scripts(self, project_dir):
        """Create example scripts and configurations"""
        self.print_status("Creating example scripts...")
        
        scripts_dir = Path(project_dir) / 'scripts'
        
        # Example hyperparameter optimization script
        example_script = scripts_dir / 'example_hyperopt.py'
        with open(example_script, 'w') as f:
            f.write('''#!/usr/bin/env python3
"""
Example Hyperparameter Optimization Script for AI Mission Control
"""

import torch
import numpy as np
import argparse
from collections import defaultdict

def main():
    parser = argparse.ArgumentParser(description='Example Hyperparameter Optimization')
    parser.add_argument('--env', type=str, default='CartPole-v1')
    parser.add_argument('--method', type=str, default='Random')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--episodes', type=int, default=1000)
    
    args = parser.parse_args()
    
    print(f"Running hyperparameter optimization for {args.env}")
    print(f"Method: {args.method}")
    print(f"Seed: {args.seed}")
    print(f"Episodes: {args.episodes}")
    
    # Your hyperparameter optimization code here
    # This is where you would integrate with your hyper_framework
    
    print("Example completed successfully!")

if __name__ == "__main__":
    main()
''')
        
        # Example configuration file
        config_dir = Path(project_dir) / 'configs'
        example_config = config_dir / 'default_config.yaml'
        with open(example_config, 'w') as f:
            f.write('''# AI Mission Control - Default Configuration

environment:
  name: "CartPole-v1"
  max_episode_steps: 500

hyperparameter_bounds:
  learning_rate: [1e-5, 1e-2]
  batch_size: [32, 512] 
  gamma: [0.9, 0.999]
  epsilon: [0.01, 0.3]

optimization:
  method: "GP-UCB"
  total_frames: 100000
  t_ready: 5000

logging:
  log_interval: 100
  save_model: true
  tensorboard: true
''')
        
        # README file
        readme_file = Path(project_dir) / 'README.md'
        with open(readme_file, 'w') as f:
            f.write(f'''# AI Mission Control - Hyperparameter Optimization

Version: {self.version}

## Overview
{self.description}

## Quick Start

### Using Docker (Recommended)
```bash
# Start interactive shell
./run_ai_mission_control.sh shell

# Start Jupyter Lab
./run_ai_mission_control.sh jupyter

# Run experiment
./run_ai_mission_control.sh experiment scripts/example_hyperopt.py
```

### Using Docker Compose
```bash
# Start all services
docker-compose up -d

# Access Jupyter Lab
# Open http://localhost:8080 in your browser
```

## Directory Structure
- `experiments/` - Experiment results and logs
- `scripts/` - Python scripts and experiments  
- `configs/` - Configuration files
- `models/` - Saved model checkpoints
- `data/` - Training data
- `notebooks/` - Jupyter notebooks
- `logs/` - Training logs
- `results/` - Analysis and results

## Available Hyperparameter Optimization Methods
- Random Search
- Grid Search
- Gaussian Process UCB (GP-UCB)
- Population Based Training (PB2)
- HyperBand
- UCB variants
- Kalman Filter methods
- Custom HyperController

## Environment Support
- OpenAI Gym environments
- Unity ML-Agents environments
- MuJoCo physics environments
- Custom RL environments
- Financial trading environments (FinRL)

## GPU Support
This container supports CUDA for GPU acceleration. Ensure you have:
- NVIDIA GPU drivers
- Docker with NVIDIA runtime
- nvidia-docker2 installed

## Examples
See the `scripts/` directory for example implementations.

## Troubleshooting
If you encounter issues:
1. Verify Docker installation: `docker --version`
2. Check GPU support: `nvidia-smi`
3. Test container: `./run_ai_mission_control.sh test`

For more help, check the documentation or open an issue.
''')
        
        self.print_success("Example scripts and documentation created")
        
    def create_run_script(self, project_dir):
        """Create the run script"""
        run_script_content = '''#!/bin/bash

# FILENAME: run_ai_mission_control.sh
# AI Mission Control - Run Script
# Usage examples for running the container

IMAGE_NAME="ai-mission-control:latest"

case "${1:-}" in
    "shell"|"bash")
        echo "Starting AI Mission Control shell..."
        docker run -it --rm \\
            --gpus all \\
            -v "$(pwd)":/workspace/project \\
            ${IMAGE_NAME} /bin/bash
        ;;
    "jupyter")
        echo "Starting Jupyter Lab on port 8080..."
        docker run -it --rm \\
            --gpus all \\
            -p 8080:8080 \\
            -v "$(pwd)":/workspace/project \\
            ${IMAGE_NAME} jupyter lab --ip=0.0.0.0 --port=8080 --no-browser --allow-root
        ;;
    "experiment")
        echo "Running RL experiment..."
        if [ -z "$2" ]; then
            echo "Usage: $0 experiment <script_name>"
            echo "Example: $0 experiment rl_tests.py"
            exit 1
        fi
        docker run -it --rm \\
            --gpus all \\
            -v "$(pwd)":/workspace/project \\
            ${IMAGE_NAME} python3 /workspace/project/$2
        ;;
    "test")
        echo "Running installation test..."
        docker run --rm \\
            --gpus all \\
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
'''
        
        run_script_path = Path(project_dir) / 'run_ai_mission_control.sh'
        with open(run_script_path, 'w') as f:
            f.write(run_script_content)
        os.chmod(run_script_path, 0o755)
        
    def create_docker_compose(self, project_dir):
        """Create Docker Compose file"""
        compose_content = f'''# FILENAME: docker-compose.yml
version: '3.8'

services:
  ai-mission-control:
    image: {self.name}:latest
    container_name: {self.name}
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
'''
        
        compose_path = Path(project_dir) / 'docker-compose.yml'
        with open(compose_path, 'w') as f:
            f.write(compose_content)
            
    def create_devcontainer_config(self, project_dir):
        """Create VSCode dev container configuration"""
        devcontainer_dir = Path(project_dir) / '.devcontainer'
        devcontainer_dir.mkdir(exist_ok=True)
        
        devcontainer_content = '''{
    "name": "AI Mission Control",
    "image": "ai-mission-control:latest",
    "runArgs": [
        "--gpus=all",
        "--shm-size=8gb"
    ],
    "workspaceFolder": "/workspace/project",
    "mounts": [
        "source=${localWorkspaceFolder},target=/workspace/project,type=bind",
        "source=ai-mc-experiments,target=/workspace/experiments,type=volume",
        "source=ai-mc-models,target=/workspace/models,type=volume",
        "source=ai-mc-logs,target=/workspace/logs,type=volume"
    ],
    "forwardPorts": [8080, 8888, 6006],
    "portsAttributes": {
        "8080": {
            "label": "Jupyter Lab",
            "onAutoForward": "notify"
        },
        "8888": {
            "label": "Jupyter Alt",
            "onAutoForward": "ignore"
        },
        "6006": {
            "label": "TensorBoard",
            "onAutoForward": "notify"
        }
    },
    "customizations": {
        "vscode": {
            "extensions": [
                "ms-python.python",
                "ms-toolsai.jupyter",
                "SBSnippets.pytorch-snippets",
                "njpwerner.autodocstring",
                "ms-python.flake8",
                "ms-python.black-formatter"
            ],
            "settings": {
                "python.defaultInterpreterPath": "/opt/miniforge/envs/automl/bin/python",
                "python.terminal.activateEnvironment": true,
                "jupyter.jupyterServerType": "local",
                "files.watcherExclude": {
                    "**/experiments/**": true,
                    "**/logs/**": true,
                    "**/models/**": true
                }
            }
        }
    },
    "postCreateCommand": "echo 'AI Mission Control Dev Container Ready!'",
    "remoteUser": "root"
}'''
        
        devcontainer_path = devcontainer_dir / 'devcontainer.json'
        with open(devcontainer_path, 'w') as f:
            f.write(devcontainer_content)
            
    def create_vscode_config(self, project_dir):
        """Create VSCode workspace configuration"""
        vscode_dir = Path(project_dir) / '.vscode'
        vscode_dir.mkdir(exist_ok=True)
        
        # Launch configuration
        launch_content = '''{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Python: RL Tests",
            "type": "python",
            "request": "launch",
            "program": "${workspaceFolder}/scripts/rl_tests.py",
            "args": [
                "--env", "CartPole-v1",
                "--method", "GP-UCB",
                "--seed", "42",
                "--total_frames", "100000"
            ],
            "console": "integratedTerminal",
            "cwd": "${workspaceFolder}",
            "env": {
                "CUDA_VISIBLE_DEVICES": "0",
                "PYTHONPATH": "${workspaceFolder}"
            }
        },
        {
            "name": "Python: Current File",
            "type": "python",
            "request": "launch",
            "program": "${file}",
            "console": "integratedTerminal",
            "cwd": "${workspaceFolder}"
        }
    ]
}'''
        
        launch_path = vscode_dir / 'launch.json'
        with open(launch_path, 'w') as f:
            f.write(launch_content)
            
        # Tasks configuration
        tasks_content = '''{
    "version": "2.0.0",
    "tasks": [
        {
            "label": "Start Jupyter Lab",
            "type": "shell",
            "command": "jupyter lab --ip=0.0.0.0 --port=8080 --no-browser --allow-root",
            "group": "build",
            "presentation": {
                "echo": true,
                "reveal": "always",
                "focus": false,
                "panel": "new"
            }
        },
        {
            "label": "Start TensorBoard",
            "type": "shell",
            "command": "tensorboard --logdir=/workspace/logs --host=0.0.0.0 --port=6006",
            "group": "build",
            "presentation": {
                "echo": true,
                "reveal": "always",
                "focus": false,
                "panel": "new"
            }
        }
    ]
}'''
        
        tasks_path = vscode_dir / 'tasks.json'
        with open(tasks_path, 'w') as f:
            f.write(tasks_content)
            
        # Settings configuration
        settings_content = '''{
    "python.defaultInterpreterPath": "/opt/miniforge/envs/automl/bin/python",
    "python.terminal.activateEnvironment": true,
    "jupyter.jupyterServerType": "local",
    "files.exclude": {
        "**/__pycache__": true,
        "**/.pytest_cache": true,
        "**/experiments/*/logs": true
    },
    "editor.formatOnSave": true,
    "python.formatting.provider": "black"
}'''
        
        settings_path = vscode_dir / 'settings.json'
        with open(settings_path, 'w') as f:
            f.write(settings_content)
            
    def build_docker_image(self, project_dir):
        """Build the Docker image"""
        self.print_status("Building Docker image...")
        self.print_status("This may take 15-30 minutes...")
        
        try:
            result = subprocess.run([
                'docker', 'build',
                '-t', f'{self.name}:latest',
                '-f', 'Dockerfile',
                '.'
            ], cwd=project_dir, check=True, capture_output=True, text=True)
            
            self.print_success("Docker image built successfully")
            return True
        except subprocess.CalledProcessError as e:
            self.print_error(f"Failed to build Docker image: {e}")
            self.print_error(f"stdout: {e.stdout}")
            self.print_error(f"stderr: {e.stderr}")
            return False
            
    def verify_installation(self, project_dir):
        """Verify the installation"""
        self.print_status("Verifying installation...")
        
        try:
            result = subprocess.run([
                'docker', 'run', '--rm',
                f'{self.name}:latest',
                'python3', '/workspace/scripts/verify_installation.py'
            ], check=True, capture_output=True, text=True)
            
            print(result.stdout)
            self.print_success("Installation verified successfully")
            return True
        except subprocess.CalledProcessError as e:
            self.print_error(f"Installation verification failed: {e}")
            self.print_error(f"stdout: {e.stdout}")
            self.print_error(f"stderr: {e.stderr}")
            return False
            
    def install(self, project_dir, build_image=True, verify=True):
        """Main installation process"""
        project_path = Path(project_dir).resolve()
        
        self.print_header()
        
        # Check system requirements
        if not self.check_system_requirements():
            return False
            
        # Check GPU support
        gpu_available = self.check_gpu_support()
        
        # Create project structure
        self.create_project_structure(project_path)
        
        # Create example scripts
        self.create_example_scripts(project_path)
        
        # Create run script
        self.create_run_script(project_path)
        
        # Create Docker Compose file
        self.create_docker_compose(project_path)
        
        # Create VSCode configurations
        self.create_devcontainer_config(project_path)
        self.create_vscode_config(project_path)
        
        # Copy required files to project directory
        current_dir = Path(__file__).parent
        required_files = [
            'Dockerfile',
            'hypercontroller-exact.yml',
            'build.sh',
            'run_ai_mission_control.sh',
            'docker-compose.yml'
        ]
        
        self.print_status("Copying installation files...")
        for file_name in required_files:
            src_file = current_dir / file_name
            if src_file.exists():
                dst_file = project_path / file_name
                shutil.copy2(src_file, dst_file)
                self.print_status(f"Copied {file_name}")
        
        # Build Docker image if requested
        if build_image:
            if not self.build_docker_image(project_path):
                return False
                
        # Verify installation if requested
        if verify and build_image:
            if not self.verify_installation(project_path):
                return False
                
        # Final success message
        self.print_success("AI Mission Control installation completed!")
        print()
        self.print_status("Quick start commands:")
        print(f"  cd {project_path}")
        print("  ./run_ai_mission_control.sh shell    # Interactive shell")
        print("  ./run_ai_mission_control.sh jupyter  # Jupyter Lab")
        print("  ./run_ai_mission_control.sh test     # Test installation")
        print()
        
        if gpu_available:
            self.print_success("GPU support is available!")
        else:
            self.print_warning("GPU support not detected - CPU mode only")
            
        return True
        
    def uninstall(self):
        """Uninstall AI Mission Control"""
        self.print_status("Uninstalling AI Mission Control...")
        
        try:
            # Remove Docker images
            subprocess.run([
                'docker', 'rmi', f'{self.name}:latest'
            ], check=True, capture_output=True)
            self.print_success("Docker images removed")
        except subprocess.CalledProcessError:
            self.print_warning("No Docker images to remove")
            
        self.print_success("Uninstallation completed")
        
    def update(self, project_dir):
        """Update AI Mission Control installation"""
        self.print_status("Updating AI Mission Control...")
        
        # Rebuild Docker image
        if self.build_docker_image(project_dir):
            self.print_success("Update completed successfully")
            return True
        else:
            self.print_error("Update failed")
            return False


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='AI Mission Control Setup Script',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python setup.py install ./my_project          # Full installation
  python setup.py install ./my_project --no-build  # Setup without building
  python setup.py uninstall                     # Remove installation
  python setup.py update ./my_project           # Update existing installation
        '''
    )
    
    parser.add_argument(
        'action',
        choices=['install', 'uninstall', 'update'],
        help='Action to perform'
    )
    
    parser.add_argument(
        'project_dir',
        nargs='?',
        default='./ai_mission_control_project',
        help='Project directory path (default: ./ai_mission_control_project)'
    )
    
    parser.add_argument(
        '--no-build',
        action='store_true',
        help='Skip Docker image building'
    )
    
    parser.add_argument(
        '--no-verify',
        action='store_true', 
        help='Skip installation verification'
    )
    
    parser.add_argument(
        '--version',
        action='version',
        version=f'AI Mission Control {AIMissionControlSetup().version}'
    )
    
    args = parser.parse_args()
    
    setup = AIMissionControlSetup()
    
    if args.action == 'install':
        success = setup.install(
            args.project_dir,
            build_image=not args.no_build,
            verify=not args.no_verify
        )
        sys.exit(0 if success else 1)
        
    elif args.action == 'uninstall':
        setup.uninstall()
        sys.exit(0)
        
    elif args.action == 'update':
        success = setup.update(args.project_dir)
        sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()