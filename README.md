# AI Mission Control 🚀

**Advanced Distributed Reinforcement Learning Framework with Bayesian Hyperparameter Optimization**

AI Mission Control is a sophisticated, config-driven distributed reinforcement learning system designed for scalable training across multiple GPUs and environments. It supports multiple RL algorithms, intelligent hyperparameter optimization, and seamless orchestration of training experiments.

## Features

- **Multi-Algorithm Support**: PPO, DDPG with proper Ornstein-Uhlenbeck noise
- **Distributed Training**: Scalable coordinator-worker architecture
- **Bayesian Optimization**: Intelligent hyperparameter tuning with Gaussian Process optimization
- **Config-Driven**: JSON-based configuration system for reproducible experiments
- **GPU Distribution**: Automatic GPU assignment and load balancing
- **Real-time Monitoring**: TensorBoard integration and Jupyter Lab development environment
- **Docker Integration**: Containerized deployment with proper resource management

## Table of Contents

- [Quick Start](#-quick-start)
- [Installation](#-installation)
- [Configuration](#-configuration)
- [Training](#-training)
- [Algorithms](#-algorithms)
- [Environment Support](#-environment-support)
- [Monitoring](#-monitoring)
- [Development](#-development)
- [API Reference](#-api-reference)
- [Troubleshooting](#-troubleshooting)

## Quick Start

### Prerequisites

- Docker with GPU support (NVIDIA Container Toolkit)
- NVIDIA GPUs with CUDA support
- At least 8GB GPU memory recommended

### 1. Clone and Setup

```bash
git clone https://github.com/jongornet14/ai-mission-control.git
cd ai-mission-control

# System uses Makefile for efficient build management
make help  # See all available commands
```

### 2. Start Training

```bash
# Basic 4-worker training on CartPole
make dist-cartpole

# Advanced DDPG training on HalfCheetah with 8 workers
make dist-start WORKERS=8 CONFIG=ddpg_halfcheetah.json GPUS=0,1,2,3

# Quick development environment
make dist-quick-dev
```

### 3. Monitor Progress

- **TensorBoard**: http://localhost:6006
- **Jupyter Lab**: http://localhost:8080
- **Logs**: `make dist-logs`

## Installation

### Docker Installation (Recommended)

The system is designed to run in Docker containers with all dependencies pre-configured.

```bash
# Build the Docker image (automatic on first run)
make build

# Or use the training commands (builds automatically)
make dist-cartpole
```

### Local Development Installation

For local development without Docker:

```bash
# Create conda environment
conda create -n automl python=3.10
conda activate automl

# Install dependencies
pip install -e .

# Install additional requirements
pip install torch torchvision gymnasium stable-baselines3
pip install scikit-optimize tensorboard jupyter
```

## Configuration

The system uses JSON configuration files for reproducible experiments. Configurations are stored in the `configs/` directory.

### Configuration Structure

```json
{
  "experiment_name": "ppo_cartpole_experiment",
  "algorithm": "ppo",
  "environment": {
    "name": "CartPole-v1",
    "max_episode_steps": 500
  },
  "optimizer": {
    "type": "adam",
    "learning_rate": 0.001
  },
  "hyperparameter_optimizer": {
    "type": "bayesian",
    "n_calls": 50,
    "search_space": {
      "learning_rate": [0.0001, 0.01],
      "gamma": [0.9, 0.999],
      "clip_range": [0.1, 0.3]
    }
  },
  "training": {
    "total_timesteps": 100000,
    "eval_freq": 5000,
    "save_freq": 10000
  }
}
```

### Available Configurations

| Config File | Algorithm | Environment | Description |
|-------------|-----------|-------------|-------------|
| `cartpole_distributed.json` | PPO | CartPole-v1 | Basic discrete control |
| `ddpg_halfcheetah.json` | DDPG | HalfCheetah-v4 | Continuous control |
| `ppo_lunarlander.json` | PPO | LunarLander-v2 | Discrete control with gravity |
| `ddpg_pendulum.json` | DDPG | Pendulum-v1 | Simple continuous control |

### Creating Custom Configurations

```bash
# Copy an existing config
cp configs/cartpole_distributed.json configs/my_experiment.json

# Edit the configuration
nano configs/my_experiment.json

# Run with your config  
make dist-start CONFIG=my_experiment.json WORKERS=6 GPUS=0,1
```

## Training

### Distributed Training with Make

The system uses an advanced Makefile for efficient dependency management and parallel execution:

```bash
# Start training with default settings (4 workers, CartPole)
make dist-cartpole

# Advanced configurations
make dist-start WORKERS=8 CONFIG=ddpg_halfcheetah.json GPUS=0,1,2,3

# Quick worker scaling
make dist-8 CONFIG=my_config.json

# Stop all training
make dist-stop

# Check status
make dist-status

# Show all available commands
make help
```

### Training Options

| Variable | Description | Default |
|----------|-------------|---------|
| `WORKERS` | Number of worker processes | 4 |
| `CONFIG` | Configuration file | cartpole_distributed.json |
| `GPUS` | Comma-separated GPU IDs | 0,1,2,3 |
| `COMPOSE_FILE` | Docker compose file | docker-compose.scalable.yml |

### Why Makefile > Bash Scripts

- **Dependency Management**: Only rebuilds what's changed
- **Parallel Execution**: Built-in `-j` support for parallel workers  
- **Standardized**: Universal build tool developers expect
- **Integration**: Better IDE and CI/CD integration
- **Caching**: Automatic detection of what needs rebuilding

### Worker ID Management

The system ensures deterministic worker IDs for distributed training:

- Workers are named `rl-worker-0`, `rl-worker-1`, etc.
- Each worker gets a unique configuration from the coordinator
- GPU assignment follows round-robin distribution
- Worker IDs are maintained across restarts

## 🤖 Algorithms

### Proximal Policy Optimization (PPO)

Suitable for both discrete and continuous action spaces.

**Key Features:**
- Clipped objective function
- Adaptive KL divergence penalty
- Value function clipping
- Generalized Advantage Estimation (GAE)

**Best For:** General-purpose RL, stable training, discrete control

### Deep Deterministic Policy Gradient (DDPG)

Optimized for continuous control tasks with proper noise implementation.

**Key Features:**
- Actor-critic architecture
- Experience replay buffer
- Target networks with soft updates
- Mathematically correct Ornstein-Uhlenbeck noise

**Best For:** Continuous control, robotics, precise action control

### Algorithm Selection

```json
{
  "algorithm": "ppo",  // or "ddpg"
  "algorithm_config": {
    // PPO specific
    "clip_range": 0.2,
    "ent_coef": 0.01,
    "vf_coef": 0.5,
    
    // DDPG specific  
    "tau": 0.005,
    "buffer_size": 1000000,
    "noise_std": 0.1
  }
}
```

## Environment Support

### Supported Environments

#### Gymnasium Environments
- **CartPole-v1**: Discrete control, balance pole
- **LunarLander-v2**: Discrete control, land spacecraft
- **MountainCar-v0**: Discrete control, reach goal
- **Acrobot-v1**: Discrete control, swing up

#### MuJoCo Environments  
- **HalfCheetah-v4**: Continuous control, locomotion
- **Ant-v4**: Continuous control, quadruped
- **Hopper-v3**: Continuous control, single-leg hopping
- **Walker2d-v3**: Continuous control, bipedal walking
- **Pendulum-v1**: Continuous control, inverted pendulum

### Environment Configuration

```json
{
  "environment": {
    "name": "HalfCheetah-v4",
    "max_episode_steps": 1000,
    "reward_threshold": 4800,
    "wrappers": [
      "normalize_observation",
      "normalize_reward"
    ]
  }
}
```

### Adding Custom Environments

```python
# In intellinaut/environments/gym_wrapper.py
def create_custom_env(env_name, **kwargs):
    """Create custom environment with preprocessing."""
    if env_name == "MyCustomEnv-v0":
        import my_custom_env
        env = my_custom_env.make(**kwargs)
    else:
        env = gymnasium.make(env_name, **kwargs)
    
    return env
```

## Monitoring

### TensorBoard

Access real-time training metrics at http://localhost:6006

**Available Metrics:**
- Episode rewards and lengths
- Policy loss and value loss  
- Learning rate schedules
- Hyperparameter optimization progress
- GPU utilization and memory usage

### Jupyter Lab

Development environment at http://localhost:8080

**Features:**
- Interactive experiment analysis
- Real-time plotting and visualization
- Model inspection and debugging
- Custom metric computation

### Logging

```bash
# Monitor coordinator
make dist-logs-coordinator

# Monitor specific worker  
make dist-logs-worker WORKER=0

# Check all services
make dist-status

# View shared training data
ls -la distributed_shared/
```

## Development

### Development Container

Access the development container for debugging and experimentation:

```bash
# Start development environment
make dist-quick-dev

# Access development container  
make dist-shell

# Run experiments manually
python scripts/worker_entry.py --config configs/my_config.json
```

### Code Structure

```
ai-mission-control/
├── intellinaut/              # Main package
│   ├── algorithms/           # RL algorithms (PPO, DDPG)
│   ├── workers/             # Distributed workers
│   ├── optimizers/          # Bayesian optimization
│   ├── environments/        # Environment wrappers
│   └── config/             # Configuration management
├── configs/                 # JSON configurations
├── scripts/                 # Entry points
├── distributed_shared/      # Shared training data
├── Makefile                 # Build system and task runner
└── docker-compose.scalable.yml  # Scalable container orchestration
```

### Adding New Algorithms

1. Create algorithm file in `intellinaut/algorithms/`
2. Implement base class interface
3. Add to algorithm registry in `base.py`
4. Create configuration template
5. Test with distributed training

### Custom Optimizers

```python
# In intellinaut/optimizers/
class CustomOptimizer:
    def __init__(self, search_space):
        self.search_space = search_space
    
    def suggest(self, n_suggestions=1):
        """Suggest next hyperparameters to try."""
        pass
    
    def update(self, suggestions, scores):
        """Update optimizer with results."""
        pass
```

## API Reference

### Configuration Loader

```python
from intellinaut.config.loader import ConfigLoader

# Load and validate configuration
loader = ConfigLoader()
config = loader.load_config("configs/my_config.json")

# Override parameters
overrides = {"training.total_timesteps": 50000}
config = loader.apply_overrides(config, overrides)
```

### BaseWorker

```python
from intellinaut.workers.base import BaseWorker

# Create worker with configuration
worker = BaseWorker(
    worker_id=0,
    config=config,
    shared_dir="/path/to/shared"
)

# Run training
worker.run()
```

### Bayesian Optimizer

```python
from intellinaut.optimizers.bayesian import BayesianOptimizer

# Create optimizer
optimizer = BayesianOptimizer(
    search_space={"lr": [0.001, 0.1]},
    n_calls=50
)

# Get suggestions
suggestions = optimizer.suggest(n_suggestions=4)
```

## Troubleshooting

### Common Issues

#### GPU Memory Issues
```bash
# Check GPU memory usage
nvidia-smi

# Reduce batch size in config
{
  "algorithm_config": {
    "batch_size": 32  // Reduce from 64
  }
}
```

#### Container Communication Issues
```bash
# Check network connectivity
docker network ls
docker network inspect ai-mission-control_rl-training

# Restart with clean state
./run_distributed.sh --clean
./run_distributed.sh
```

#### Configuration Errors
```bash
# Validate configuration
python -c "
from intellinaut.config.loader import ConfigLoader
loader = ConfigLoader()
config = loader.load_config('configs/my_config.json')
print('✓ Configuration valid')
"
```

#### Worker Synchronization Issues
```bash
# Check coordinator health
docker logs rl-coordinator

# Verify shared directory permissions
ls -la distributed_shared/
chmod 755 distributed_shared/

# Check worker logs
for i in {0..3}; do
    echo "=== Worker $i ==="
    docker logs rl-worker-$i --tail 20
done
```

### Performance Tuning

#### Optimize GPU Usage
- Use multiple smaller workers instead of few large ones
- Balance workers across available GPUs
- Monitor GPU utilization with `nvidia-smi`

#### Memory Management
- Adjust replay buffer size for DDPG
- Use gradient accumulation for large batch sizes
- Enable memory growth for TensorFlow models

#### Network Optimization
- Use SSD storage for shared directory
- Consider NFS for multi-machine deployment  
- Optimize checkpoint frequency

### Getting Help

1. **Check Logs**: Always start with container logs
2. **Configuration**: Validate JSON syntax and parameter ranges
3. **Resources**: Monitor GPU memory and CPU usage
4. **Issues**: Submit issues with logs and configuration files

## License

MIT License - see LICENSE file for details.

## Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## Citation

```bibtex
@software{ai_mission_control,
  title={AI Mission Control: Distributed Reinforcement Learning Framework},
  author={Your Name},
  year={2025},
  url={https://github.com/jongornet14/ai-mission-control}
}
```