# Config-Driven RL Training

This system is designed to be **config-file driven** rather than command-line argument heavy. All experiment parameters are defined in JSON config files, making experiments more reproducible and easier to manage.

## 🎯 **Core Philosophy**

**Preferred Approach**: Use config files for everything, minimal CLI args  
**Avoid**: Long command lines with dozens of arguments  
**Result**: Clean, reproducible, version-controllable experiments

## 📁 **Available Config Files**

```
configs/
├── sample_ppo_config.json          # Basic PPO on CartPole
├── sample_ddpg_config.json         # DDPG on Pendulum
├── lunarlander_ppo_config.json     # PPO on LunarLander
└── halfcheetah_ddpg_config.json    # DDPG on HalfCheetah
```

## 🚀 **Quick Start**

### **1. List Available Configs**
```bash
python config_runner.py --list
```

### **2. Validate a Config**
```bash
python config_runner.py --validate configs/sample_ppo_config.json
```

### **3. Run an Experiment**
```bash
python config_runner.py --config configs/sample_ppo_config.json --worker_id 1
```

### **4. Run with Override**
```bash
python config_runner.py --config configs/sample_ddpg_config.json --worker_id 1 --device cpu
```

## 📄 **Config File Structure**

```json
{
  "experiment": {
    "name": "my_experiment",
    "description": "Description of what this experiment does"
  },
  "environment": {
    "name": "cartpole",           // Environment preset or full name
    "preset": true,               // Whether to use preset mapping
    "normalize_observations": true
  },
  "algorithm": {
    "name": "ppo",                // ppo, ddpg
    "optimizer": "adam",          // adam, sgd, rmsprop, adamw
    "hyperparam_optimizer": "bayesian", // bayesian, random, grid, none
    "learning_rate": 0.0003,
    // Algorithm-specific parameters...
  },
  "training": {
    "max_episodes": 1000,
    "device": "cuda:0",
    "checkpoint_frequency": 50,
    "status_update_frequency": 10
  },
  "distributed": {
    "sync_frequency": 10,
    "shared_dir": "/workspace/shared",
    "num_workers": 4
  }
}
```

## 🛠 **Available Options**

### **Algorithms**
- `ppo`: Proximal Policy Optimization
- `ddpg`: Deep Deterministic Policy Gradient

### **Optimizers**  
- `adam`: Adaptive Moment Estimation
- `adamw`: Adam with Weight Decay
- `sgd`: Stochastic Gradient Descent
- `rmsprop`: Root Mean Square Propagation

### **Hyperparameter Optimizers**
- `bayesian`: Bayesian Optimization
- `random`: Random Search (coming soon)
- `grid`: Grid Search (coming soon)  
- `none`: No optimization

### **Environment Presets**
- `cartpole` → `CartPole-v1`
- `lunarlander` → `LunarLander-v2`
- `pendulum` → `Pendulum-v1`
- `halfcheetah` → `HalfCheetah-v4`
- `ant` → `Ant-v4`
- `walker2d` → `Walker2d-v4`
- `humanoid` → `Humanoid-v4`
- `bipedal` → `BipedalWalker-v3`

## 🎮 **Usage Examples**

### **PPO on CartPole**
```bash
python config_runner.py --config configs/sample_ppo_config.json --worker_id 1
```

### **DDPG on Pendulum**
```bash
python config_runner.py --config configs/sample_ddpg_config.json --worker_id 1
```

### **Override Device to CPU**
```bash
python config_runner.py --config configs/halfcheetah_ddpg_config.json --worker_id 1 --device cpu
```

### **Validate All Configs**
```bash
python config_runner.py --validate-all
```

## 🔧 **Direct Worker Script Usage**

If you prefer to use the worker script directly:

```bash
# Basic usage
python scripts/worker_entry.py --config configs/sample_ppo_config.json --worker_id 1

# With overrides
python scripts/worker_entry.py --config configs/sample_ppo_config.json --worker_id 1 --device cpu --max_episodes 500

# List available options
python scripts/worker_entry.py --list-options

# Validate config
python scripts/worker_entry.py --config configs/sample_ppo_config.json --validate-config
```

## 📊 **Config File Benefits**

1. **Reproducibility**: Exact experiment parameters saved with results
2. **Version Control**: Config files can be tracked in git
3. **Easy Sharing**: Send a config file instead of a long command
4. **Documentation**: Self-documenting experiment parameters
5. **Automation**: Easy to generate configs programmatically
6. **No Typos**: Less chance of command-line argument mistakes

## 🎯 **Best Practices**

1. **Use meaningful names**: `cartpole_ppo_baseline.json` not `config1.json`
2. **Document experiments**: Fill in `experiment.description`
3. **Version configs**: Keep old configs when experimenting
4. **Validate first**: Always validate configs before running
5. **Small overrides only**: Use CLI args sparingly for minor tweaks

## 🚦 **Validation**

The system automatically validates:
- Algorithm compatibility with environments
- Valid optimizer selections
- Required config sections
- Parameter ranges and types

Example validation:
```bash
$ python config_runner.py --validate configs/sample_ppo_config.json
✓ sample_ppo_config.json is valid
```

This config-driven approach makes your RL experiments much more manageable and reproducible! 🎉
