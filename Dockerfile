# FILENAME: Dockerfile
# AI Mission Control - Hyperparameter Optimization for RL Neural Networks
# Base: Ubuntu 22.04.1 with CUDA support
# Architecture: x86_64

FROM nvidia/cuda:12.1-devel-ubuntu22.04

# Prevent timezone prompts during installation
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# Set container metadata
LABEL name="ai-mission-control"
LABEL version="1.0.0"
LABEL description="Hyperparameter optimization for RL neural networks with Unity ML-Agents"
LABEL maintainer="AI Mission Control Team"

# Set Python environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV CUDA_VISIBLE_DEVICES=0

# Install system dependencies
RUN apt-get update && apt-get install -y \
    wget \
    curl \
    git \
    build-essential \
    pkg-config \
    libssl-dev \
    libffi-dev \
    libxml2-dev \
    libxslt1-dev \
    zlib1g-dev \
    libjpeg-dev \
    libpng-dev \
    libfreetype6-dev \
    libblas-dev \
    liblapack-dev \
    gfortran \
    libhdf5-dev \
    libopenmpi-dev \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libglu1-mesa \
    libxrandr2 \
    libxss1 \
    libxcursor1 \
    libxcomposite1 \
    libasound2 \
    libxi6 \
    libxtst6 \
    unzip \
    xvfb \
    x11vnc \
    fluxbox \
    wmctrl \
    && rm -rf /var/lib/apt/lists/*

# Install Miniforge (conda replacement with conda-forge default)
RUN wget -q https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -O miniforge.sh && \
    bash miniforge.sh -b -p /opt/miniforge && \
    rm miniforge.sh

# Add conda to PATH
ENV PATH="/opt/miniforge/bin:$PATH"

# Initialize conda
RUN conda init bash

# Copy environment specification
COPY hypercontroller-exact.yml /tmp/hypercontroller-exact.yml

# Create conda environment from exact specification
RUN conda env create -f /tmp/hypercontroller-exact.yml && \
    conda clean -afy

# Activate environment
ENV CONDA_DEFAULT_ENV=automl
ENV PATH="/opt/miniforge/envs/automl/bin:$PATH"

# Install additional packages not in conda
RUN /opt/miniforge/envs/automl/bin/pip install --no-cache-dir \
    # Financial RL
    finrl==3.0.8 \
    \
    # Unity ML-Agents suite
    mlagents==1.0.0 \
    mlagents-envs==1.0.0 \
    gym-unity==0.28.0 \
    unity-environment==1.0.0 \
    \
    # Additional hyperparameter optimization tools
    optuna==3.6.1 \
    hyperopt==0.2.7 \
    ax-platform==0.4.2 \
    bayesian-optimization==1.5.1 \
    \
    # Experiment tracking and monitoring
    wandb==0.18.7 \
    mlflow==2.21.1 \
    neptune==1.12.1 \
    clearml==1.18.2 \
    \
    # Advanced visualization
    plotly==5.24.1 \
    dash==2.18.2 \
    bokeh==3.6.2 \
    seaborn==0.13.2 \
    \
    # Additional RL environments
    pybullet==3.2.6 \
    dm-control==1.0.22 \
    atari-py==0.2.9 \
    ale-py==0.8.1 \
    \
    # Model interpretation and analysis
    shap==0.46.0 \
    lime==0.2.0.1 \
    captum==0.7.0 \
    \
    # Advanced optimization
    nevergrad==1.0.4 \
    gpytorch==1.13 \
    botorch==0.12.0 \
    \
    # Parallel computing
    joblib==1.4.2 \
    dask==2024.12.1 \
    \
    # Development and debugging tools
    ipdb==0.13.13 \
    line_profiler==4.1.3 \
    memory_profiler==0.61.0 \
    \
    # Additional utilities
    hydra-core==1.3.2 \
    rich==13.9.4 \
    typer==0.15.1 \
    click==8.1.8

# Set up workspace directory
WORKDIR /workspace

# Create directory structure for experiments
RUN mkdir -p /workspace/{experiments,logs,models,data,configs,scripts,notebooks}

# Copy project files (if available)
# COPY rl_tests.py /workspace/scripts/
# COPY hyper_framework.py /workspace/scripts/
# COPY hyper_optimizers.py /workspace/scripts/

# Set up display for Unity (headless mode)
ENV DISPLAY=:99

# Create startup script for X11 forwarding
RUN echo '#!/bin/bash\n\
Xvfb :99 -screen 0 1024x768x24 -ac +extension GLX +render -noreset &\n\
export DISPLAY=:99\n\
exec "$@"' > /usr/local/bin/entrypoint.sh && \
    chmod +x /usr/local/bin/entrypoint.sh

# Install additional CUDA libraries for optimal GPU performance
RUN /opt/miniforge/envs/automl/bin/pip install --no-cache-dir \
    nvidia-ml-py==12.560.30 \
    cupy-cuda12x==13.3.0

# Set up Jupyter configuration (optional, for development)
RUN mkdir -p /root/.jupyter && \
    echo "c.NotebookApp.ip = '0.0.0.0'" >> /root/.jupyter/jupyter_notebook_config.py && \
    echo "c.NotebookApp.port = 8080" >> /root/.jupyter/jupyter_notebook_config.py && \
    echo "c.NotebookApp.open_browser = False" >> /root/.jupyter/jupyter_notebook_config.py && \
    echo "c.NotebookApp.allow_root = True" >> /root/.jupyter/jupyter_notebook_config.py

# Create a simple test script to verify installation
RUN echo '#!/usr/bin/env python3\n\
import sys\n\
import torch\n\
import numpy as np\n\
import gym\n\
import mlagents_envs\n\
import finrl\n\
import torchrl\n\
import ray\n\
print("=" * 50)\n\
print("AI Mission Control - Installation Verification")\n\
print("=" * 50)\n\
print(f"Python: {sys.version}")\n\
print(f"PyTorch: {torch.__version__}")\n\
print(f"CUDA Available: {torch.cuda.is_available()}")\n\
if torch.cuda.is_available():\n\
    print(f"CUDA Version: {torch.version.cuda}")\n\
    print(f"GPU Count: {torch.cuda.device_count()}")\n\
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")\n\
print(f"NumPy: {np.__version__}")\n\
print(f"Gym: {gym.__version__}")\n\
try:\n\
    import optuna\n\
    print(f"Optuna: {optuna.__version__}")\n\
except ImportError:\n\
    print("Optuna: Not installed")\n\
try:\n\
    import wandb\n\
    print(f"WandB: {wandb.__version__}")\n\
except ImportError:\n\
    print("WandB: Not installed")\n\
print("=" * 50)\n\
print("✅ All core packages imported successfully!")\n\
print("🚀 AI Mission Control is ready for hyperparameter optimization!")\n\
print("=" * 50)' > /workspace/scripts/verify_installation.py && \
    chmod +x /workspace/scripts/verify_installation.py

# Expose ports for Jupyter and other services (optional)
EXPOSE 8080 8888 6006

# Set environment to activate conda by default
RUN echo "source /opt/miniforge/etc/profile.d/conda.sh && conda activate automl" >> /root/.bashrc

# Default entrypoint
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]

# Default command
CMD ["/bin/bash"]

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD python3 -c "import torch, numpy, gym, mlagents_envs, finrl; print('Health check passed')" || exit 1

# Final metadata
ENV CONTAINER_NAME="ai-mission-control"
ENV BUILD_DATE="2025-01-09"
ENV VERSION="1.0.0"