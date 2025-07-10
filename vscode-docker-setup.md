FILENAME: vscode-docker-setup.txt

===============================================================================
                    VSCode Docker Setup for AI Mission Control
===============================================================================

This guide shows you how to set up Visual Studio Code to work seamlessly with 
your AI Mission Control Docker container for RL hyperparameter optimization.

===============================================================================
PREREQUISITES
===============================================================================

- Docker Desktop installed and running
- Visual Studio Code installed  
- AI Mission Control Docker image built (follow main README.md)

===============================================================================
REQUIRED VSCODE EXTENSIONS
===============================================================================

Install these extensions in VSCode:

ESSENTIAL EXTENSIONS:
1. Docker (ms-azuretools.vscode-docker)
   - Manage Docker containers, images, and compose files
   - Right-click container management

2. Dev Containers (ms-vscode-remote.remote-containers)
   - Develop inside Docker containers
   - Full IDE experience in container

3. Remote Development (ms-vscode-remote.vscode-remote-extensionpack)
   - Includes Dev Containers, SSH, WSL support

RECOMMENDED EXTENSIONS FOR RL DEVELOPMENT:
4. Python (ms-python.python)
   - Python language support, debugging, IntelliSense

5. Jupyter (ms-toolsai.jupyter)
   - Jupyter notebook support in VSCode

6. PyTorch Snippets (SBSnippets.pytorch-snippets)
   - PyTorch code snippets

7. autoDocstring (njpwerner.autodocstring)
   - Auto-generate Python docstrings

8. GitLens (eamodio.gitlens)
   - Enhanced Git capabilities

===============================================================================
METHOD 1: DEV CONTAINER SETUP (RECOMMENDED)
===============================================================================

This method provides the best development experience by running VSCode inside 
the container.

STEP 1: Create Dev Container Configuration

Create .devcontainer/devcontainer.json in your project root:

{
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
}

STEP 2: Create Launch Configuration

Create .vscode/launch.json for debugging:

{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Python: RL Tests",
            "type": "python",
            "request": "launch",
            "program": "${workspaceFolder}/rl_tests.py",
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
        },
        {
            "name": "Python: Hyperparameter Optimization",
            "type": "python",
            "request": "launch",
            "program": "${workspaceFolder}/scripts/hyperopt_experiment.py",
            "args": ["--config", "configs/default_config.yaml"],
            "console": "integratedTerminal",
            "cwd": "${workspaceFolder}"
        }
    ]
}

STEP 3: Create Tasks

Create .vscode/tasks.json for common tasks:

{
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
            },
            "options": {
                "cwd": "/workspace"
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
        },
        {
            "label": "Install Package in Editable Mode",
            "type": "shell",
            "command": "pip install -e .",
            "group": "build",
            "presentation": {
                "echo": true,
                "reveal": "always",
                "focus": false,
                "panel": "new"
            }
        },
        {
            "label": "Run Tests",
            "type": "shell",
            "command": "python -m pytest tests/ -v",
            "group": "test",
            "presentation": {
                "echo": true,
                "reveal": "always",
                "focus": false,
                "panel": "new"
            }
        }
    ]
}

STEP 4: Open in Dev Container

1. Open your project folder in VSCode
2. Press Ctrl+Shift+P (or Cmd+Shift+P on Mac)
3. Type "Dev Containers: Reopen in Container"
4. Select it and wait for the container to start
5. VSCode will reload inside the container

===============================================================================
METHOD 2: DOCKER EXTENSION MANAGEMENT
===============================================================================

If you prefer to manage Docker containers manually through VSCode:

STEP 1: Start Container

Use the Docker extension sidebar:
1. Open Docker extension in VSCode
2. Find your ai-mission-control:latest image
3. Right-click → "Run Interactive"
4. Configure port mappings: 8080:8080, 8888:8888, 6006:6006
5. Add volume mounts for your code

STEP 2: Attach to Running Container

1. In Docker extension, find your running container
2. Right-click → "Attach Visual Studio Code"
3. New VSCode window opens inside container

===============================================================================
METHOD 3: DOCKER COMPOSE WITH VSCODE
===============================================================================

STEP 1: Create docker-compose.override.yml

Create this file for development overrides:

version: '3.8'

services:
  ai-mission-control:
    volumes:
      # Mount source code for live editing
      - ./:/workspace/project:cached
      # VS Code server extensions
      - vscode-server:/root/.vscode-server
    ports:
      - "8080:8080"   # Jupyter Lab
      - "8888:8888"   # Jupyter Alt
      - "6006:6006"   # TensorBoard
      - "2222:22"     # SSH (optional)
    environment:
      - DISPLAY=${DISPLAY:-:0}
    # Keep container running
    tty: true
    stdin_open: true

volumes:
  vscode-server:

STEP 2: Start with Compose

docker-compose up -d

STEP 3: Attach VSCode

Use Docker extension to attach to the running container.

===============================================================================
WORKSPACE SETTINGS
===============================================================================

Create .vscode/settings.json for project-specific settings:

{
    "python.defaultInterpreterPath": "/opt/miniforge/envs/automl/bin/python",
    "python.terminal.activateEnvironment": true,
    "python.analysis.extraPaths": [
        "/workspace/project"
    ],
    "jupyter.jupyterServerType": "local",
    "jupyter.notebookFileRoot": "${workspaceFolder}",
    "files.associations": {
        "*.yml": "yaml",
        "*.yaml": "yaml"
    },
    "files.exclude": {
        "**/__pycache__": true,
        "**/.pytest_cache": true,
        "**/experiments/*/logs": true,
        "**/models/checkpoints": true
    },
    "editor.formatOnSave": true,
    "python.formatting.provider": "black",
    "python.linting.enabled": true,
    "python.linting.flake8Enabled": true,
    "autoDocstring.docstringFormat": "google",
    "git.ignoreLimitWarning": true
}

===============================================================================
DEBUGGING SETUP
===============================================================================

GPU Debugging:
For CUDA debugging, add to your launch configuration:

{
    "env": {
        "CUDA_LAUNCH_BLOCKING": "1",
        "TORCH_USE_CUDA_DSA": "1"
    }
}

Memory Profiling:
Add this configuration for memory debugging:

{
    "name": "Python: Memory Profile",
    "type": "python",
    "request": "launch",
    "program": "${file}",
    "args": [],
    "console": "integratedTerminal",
    "env": {
        "PYTHONPATH": "${workspaceFolder}",
        "CUDA_VISIBLE_DEVICES": "0"
    },
    "justMyCode": false
}

===============================================================================
WORKING WITH JUPYTER IN VSCODE
===============================================================================

Native Jupyter Integration:
1. Install Jupyter extension
2. Open .ipynb files directly in VSCode
3. Select the Python interpreter from the container
4. Run cells directly in VSCode interface

External Jupyter Lab:
1. Start Jupyter Lab task (Ctrl+Shift+P → "Tasks: Run Task" → "Start Jupyter Lab")
2. Open browser to http://localhost:8080
3. Develop in external Jupyter Lab interface

===============================================================================
FILE SYNCHRONIZATION
===============================================================================

Real-time Code Sync:
With dev containers or volume mounts, your code changes are immediately 
available in the container.

Git Integration:
Git works normally inside the container. Your commits and branches are preserved.

===============================================================================
COMMON WORKFLOWS
===============================================================================

1. Starting a New Experiment:
   cd /workspace/project
   python scripts/create_experiment.py --name my_experiment
   cd experiments/my_experiment
   code config.yaml  # Edit in VSCode
   python train.py   # Run with debugger

2. Hyperparameter Tuning:
   # Use the launch configuration "Python: Hyperparameter Optimization"
   # Set breakpoints in your hyperparameter optimization code
   # Press F5 to start debugging

3. Monitoring Training:
   # Start TensorBoard task
   # Open http://localhost:6006 in browser
   # Or use VSCode TensorBoard extension

===============================================================================
TROUBLESHOOTING
===============================================================================

Container Won't Start:
- Check if Docker is running
- Verify image exists: docker images | grep ai-mission-control
- Check port conflicts: netstat -tulpn | grep 8080

GPU Not Available:
- Verify nvidia-docker2 is installed
- Check container starts with GPU: 
  docker run --rm --gpus all ai-mission-control:latest nvidia-smi

VSCode Extensions Not Working:
- Rebuild dev container: Ctrl+Shift+P → "Dev Containers: Rebuild Container"
- Check extension compatibility with remote development

Slow Performance:
- Allocate more memory to Docker Desktop
- Use SSD storage for Docker volumes
- Enable BuildKit: export DOCKER_BUILDKIT=1

===============================================================================
PRO TIPS
===============================================================================

1. Use Dev Container for the best experience
2. Mount volumes for persistent data
3. Use tasks for common operations
4. Set up debugging for effective development
5. Use Jupyter extension for notebook development
6. Configure Git inside the container for version control

===============================================================================
QUICK START COMMANDS
===============================================================================

# Install extensions (in VSCode extensions marketplace):
- Docker
- Dev Containers  
- Remote Development
- Python
- Jupyter

# Create project structure:
python setup.py install ./my_rl_project
cd my_rl_project

# Open in VSCode:
code .

# Open in dev container:
Ctrl+Shift+P → "Dev Containers: Reopen in Container"

# Start developing:
# - Set breakpoints in your code
# - Press F5 to debug
# - Use Ctrl+Shift+P → "Tasks: Run Task" for common operations
# - Open .ipynb files for Jupyter development

===============================================================================
END OF GUIDE
===============================================================================

Happy coding with AI Mission Control! 🚀