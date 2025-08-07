#!/usr/bin/env python3
"""
Base Worker Class for Intellinaut RL Training
Core training logic with JSON logging for frontend integration
"""

import torch
import numpy as np
import time
import json
import os
from pathlib import Path
from collections import defaultdict, deque
from datetime import datetime
import threading
from typing import Dict, List, Optional, Any
import sys

# Intellinaut imports (package structure)
from ..algorithms.ppo import PPOAlgorithm
from ..algorithms.ddpg import DDPGAlgorithm
from ..environments.gym_wrapper import GymEnvironmentWrapper
from ..logging.logging import CrazyLogger
from ..logging.enhanced_logger import get_logger, LogLevel, EventType
from ..environments.gym_wrapper import universal_gym_step
from ..optimizers.bayesian import BayesianOptimizationManager


class BaseWorker:
    """
    Base Worker for RL Training

    Features:
    - Core RL training loop
    - JSON logging for frontend
    - Pickle logging for detailed ML data
    - Docker status indicators
    - Configurable episode limits
    - Async logging for performance
    - Algorithm selection (PPO, DDPG, etc.)
    - Optimizer selection (Adam, SGD, RMSprop, etc.)
    - Hyperparameter optimization (Bayesian, Random, Grid)
    """

    # Available algorithms
    ALGORITHMS = {
        "ppo": PPOAlgorithm,
        "ddpg": DDPGAlgorithm,
    }

    # Available optimizers
    OPTIMIZERS = {
        "adam": torch.optim.Adam,
        "sgd": torch.optim.SGD,
        "rmsprop": torch.optim.RMSprop,
        "adamw": torch.optim.AdamW,
    }

    # Available hyperparameter optimizers
    HYPERPARAM_OPTIMIZERS = {
        "bayesian": "BayesianOptimizationManager",
        "random": "RandomSearch",
        "grid": "GridSearch",
        "none": None,
    }

    # Popular environment presets
    ENVIRONMENT_PRESETS = {
        "cartpole": "CartPole-v1",
        "lunarlander": "LunarLander-v2", 
        "mountaincar": "MountainCar-v0",
        "pendulum": "Pendulum-v1",
        "halfcheetah": "HalfCheetah-v4",
        "ant": "Ant-v4",
        "walker2d": "Walker2d-v4",
        "humanoid": "Humanoid-v4",
        "bipedal": "BipedalWalker-v3",
    }

    def __init__(
        self,
        worker_id: int,
        log_dir: str,
        max_episodes: int = 1000,
        checkpoint_frequency: int = 50,
        status_update_frequency: int = 10,
        buffer_size: int = 100,
        algorithm: str = "ppo",
        optimizer: str = "adam",
        hyperparam_optimizer: str = "none",
    ):
        """
        Initialize Base Worker

        Args:
            worker_id: Unique worker identifier
            log_dir: Directory for logging outputs
            max_episodes: Maximum episodes to train (from config)
            checkpoint_frequency: Episodes between model saves
            status_update_frequency: Episodes between status updates
            buffer_size: Size of logging buffer for async writes
            algorithm: RL algorithm to use ("ppo", "ddpg", etc.)
            optimizer: Optimizer to use ("adam", "sgd", "rmsprop", "adamw")
            hyperparam_optimizer: Hyperparameter optimizer ("bayesian", "random", "grid", "none")
        """
        self.worker_id = worker_id
        self.log_dir = Path(log_dir)
        self.max_episodes = max_episodes
        self.checkpoint_frequency = checkpoint_frequency
        self.status_update_frequency = status_update_frequency

        # Configuration
        self.algorithm_name = algorithm.lower()
        self.optimizer_name = optimizer.lower()
        self.hyperparam_optimizer_name = hyperparam_optimizer.lower()
        
        # Validate selections
        self._validate_configuration()

        # Create directory structure
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir = self.log_dir / "models"
        self.models_dir.mkdir(exist_ok=True)

        # Training state
        self.current_episode = 0
        self.start_time = time.time()
        self.total_episodes = 0
        self.best_reward = float("-inf")
        self.status = "INITIALIZING"

        # Episode tracking
        self.episode_rewards = deque(maxlen=1000)  # Memory efficient
        self.episode_lengths = deque(maxlen=1000)
        self.training_metrics = defaultdict(lambda: deque(maxlen=buffer_size))

        # Logging buffers for async writes
        self.json_buffer = deque(maxlen=buffer_size)
        self.pickle_buffer = deque(maxlen=buffer_size)
        self.logging_thread = None
        self.logging_active = True

        # File paths
        self.status_file = self.log_dir / f"worker_{worker_id}_status.json"
        self.metrics_file = self.log_dir / f"worker_{worker_id}_metrics.json"
        self.detailed_file = self.log_dir / f"worker_{worker_id}_detailed.pkl"

        # RL Components (initialized later)
        self.env = None
        self.algorithm = None
        self.device = None
        self.crazy_logger = None
        self.hyperparam_optimizer = None

        # Initialize logging
        self._init_logging()
        self._update_status("INITIALIZED")

    def _validate_configuration(self):
        """Validate algorithm, optimizer, and hyperparam optimizer selections"""
        if self.algorithm_name not in self.ALGORITHMS:
            available = ", ".join(self.ALGORITHMS.keys())
            raise ValueError(f"Unknown algorithm '{self.algorithm_name}'. Available: {available}")
            
        if self.optimizer_name not in self.OPTIMIZERS:
            available = ", ".join(self.OPTIMIZERS.keys())
            raise ValueError(f"Unknown optimizer '{self.optimizer_name}'. Available: {available}")
            
        if self.hyperparam_optimizer_name not in self.HYPERPARAM_OPTIMIZERS:
            available = ", ".join(self.HYPERPARAM_OPTIMIZERS.keys())
            raise ValueError(f"Unknown hyperparam optimizer '{self.hyperparam_optimizer_name}'. Available: {available}")

    @classmethod
    def get_available_algorithms(cls):
        """Get list of available algorithms"""
        return list(cls.ALGORITHMS.keys())
    
    @classmethod  
    def get_available_optimizers(cls):
        """Get list of available optimizers"""
        return list(cls.OPTIMIZERS.keys())
    
    @classmethod
    def get_available_hyperparam_optimizers(cls):
        """Get list of available hyperparameter optimizers"""
        return list(cls.HYPERPARAM_OPTIMIZERS.keys())
    
    @classmethod
    def get_environment_presets(cls):
        """Get available environment presets"""
        return cls.ENVIRONMENT_PRESETS.copy()

    def resolve_environment_name(self, env_input: str) -> str:
        """Resolve environment name from preset or direct name"""
        if env_input.lower() in self.ENVIRONMENT_PRESETS:
            return self.ENVIRONMENT_PRESETS[env_input.lower()]
        return env_input

    def _init_logging(self):
        """Initialize logging systems"""
        # Initialize enhanced logger for structured logging
        self.logger = get_logger(
            component_name="worker",
            worker_id=self.worker_id,
            shared_dir=str(self.log_dir.parent) if self.log_dir.parent else str(self.log_dir),
            console_level=LogLevel.INFO,
            file_level=LogLevel.DEBUG,
            enable_performance_monitoring=True
        )
        
        # Initialize CrazyLogger for detailed ML logging
        self.crazy_logger = CrazyLogger(
            log_dir=str(self.log_dir / "crazy_logs"),
            experiment_name=f"worker_{self.worker_id}",
            buffer_size=1000,
        )

        # Log initialization with enhanced logger
        self.logger.log(
            LogLevel.INFO,
            "Worker initialized successfully",
            event_type=EventType.SYSTEM,
            context={
                "algorithm": self.algorithm_name,
                "optimizer": self.optimizer_name,
                "hyperparam_optimizer": self.hyperparam_optimizer_name,
                "max_episodes": self.max_episodes,
                "log_directory": str(self.log_dir)
            }
        )

        # Start async logging thread
        self.logging_thread = threading.Thread(
            target=self._async_logging_worker, daemon=True
        )
        self.logging_thread.start()

        # Initialize status file
        self._write_status_immediate(
            {
                "worker_id": self.worker_id,
                "status": "INITIALIZING",
                "start_time": datetime.now().isoformat(),
                "max_episodes": self.max_episodes,
            }
        )

    def _async_logging_worker(self):
        """Async worker for handling file writes"""
        while self.logging_active:
            try:
                # Process JSON buffer
                if self.json_buffer:
                    json_data = list(self.json_buffer)
                    self.json_buffer.clear()
                    self._write_json_batch(json_data)

                # Process pickle buffer
                if self.pickle_buffer:
                    pickle_data = list(self.pickle_buffer)
                    self.pickle_buffer.clear()
                    self._write_pickle_batch(pickle_data)

                time.sleep(1)  # Write every second

            except Exception as e:
                if hasattr(self, 'logger'):
                    self.logger.log(
                        LogLevel.ERROR,
                        "Async logging worker error",
                        event_type=EventType.ERROR,
                        context={"error": str(e), "error_type": type(e).__name__}
                    )
                else:
                    print(f"Worker {self.worker_id}: Logging error: {e}")
                time.sleep(5)  # Backoff on error

    def _write_json_batch(self, data_list: List[Dict]):
        """Write batched JSON data"""
        try:
            # Append to metrics file
            existing_data = []
            if self.metrics_file.exists():
                with open(self.metrics_file, "r") as f:
                    existing_data = json.load(f)

            # Add new data
            existing_data.extend(data_list)

            # Keep only recent data (memory management)
            if len(existing_data) > 10000:
                existing_data = existing_data[-5000:]  # Keep last 5000 entries

            # Write back
            with open(self.metrics_file, "w") as f:
                json.dump(existing_data, f, indent=2)

        except Exception as e:
            self.logger.log(
                LogLevel.ERROR,
                "JSON metrics write error",
                event_type=EventType.ERROR,
                context={
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "file_path": str(self.metrics_file),
                    "data_count": len(data_list)
                }
            )

    def _write_pickle_batch(self, data_list: List[Dict]):
        """Write batched pickle data"""
        try:
            import pickle

            # Append to pickle file
            existing_data = []
            if self.detailed_file.exists():
                with open(self.detailed_file, "rb") as f:
                    existing_data = pickle.load(f)

            existing_data.extend(data_list)

            # Memory management for pickle data
            if len(existing_data) > 1000:
                existing_data = existing_data[-500:]

            with open(self.detailed_file, "wb") as f:
                pickle.dump(existing_data, f)

        except Exception as e:
            self.logger.log(
                LogLevel.ERROR,
                "Pickle data write error",
                event_type=EventType.ERROR,
                context={
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "file_path": str(self.detailed_file),
                    "data_count": len(data_list)
                }
            )

    def _update_status(self, status: str, **kwargs):
        """Update worker status for Docker monitoring"""
        self.status = status

        status_data = {
            "worker_id": self.worker_id,
            "status": status,
            "episode": self.current_episode,
            "total_episodes": self.total_episodes,
            "uptime_seconds": int(time.time() - self.start_time),
            "last_heartbeat": datetime.now().isoformat(),
            "best_reward": self.best_reward,
            **kwargs,
        }

        # Immediate write for status (critical for Docker)
        self._write_status_immediate(status_data)

        # Also log to structured logger and Docker logs
        self.logger.log(
            LogLevel.INFO,
            f"Worker status updated: {status}",
            event_type=EventType.SYSTEM,
            context={
                "status": status,
                "episode": self.current_episode,
                "total_episodes": self.total_episodes,
                "uptime_seconds": int(time.time() - self.start_time),
                "best_reward": self.best_reward
            }
        )
        print(
            f"WORKER_STATUS: {status} worker_id={self.worker_id} episode={self.current_episode}"
        )

    def _write_status_immediate(self, status_data: Dict):
        """Write status immediately (not buffered)"""
        try:
            with open(self.status_file, "w") as f:
                json.dump(status_data, f, indent=2)
        except Exception as e:
            self.logger.log(
                LogLevel.ERROR,
                "Status file write error",
                event_type=EventType.ERROR,
                context={
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "file_path": str(self.status_file)
                }
            )

    def setup_rl_components(
        self, env_name: str, device: str = "cuda:0", lr: float = 3e-4, **algorithm_config
    ):
        """Initialize RL environment and algorithm"""
        self._update_status("LOADING_COMPONENTS")

        try:
            self.device = torch.device(device if torch.cuda.is_available() else "cpu")

            # Resolve environment name (handle presets)
            resolved_env_name = self.resolve_environment_name(env_name)

            # Create environment
            env_wrapper = GymEnvironmentWrapper(
                env_name=resolved_env_name,
                device=self.device,
                normalize_observations=True,
                max_episode_steps=None,
            )
            self.env = env_wrapper.create()

            # Create algorithm based on selection
            algorithm_class = self.ALGORITHMS[self.algorithm_name]
            
            # Setup algorithm config with optimizer selection
            config = {
                "learning_rate": lr,
                **algorithm_config
            }
            
            # Create algorithm instance
            self.algorithm = algorithm_class(
                env=self.env, 
                device=self.device,
                **config
            )
            
            # Override optimizers if specified
            if hasattr(self.algorithm, 'policy_optimizer') or hasattr(self.algorithm, 'actor_optimizer'):
                self._setup_custom_optimizers(lr)

            # Setup hyperparameter optimizer if specified
            if self.hyperparam_optimizer_name != "none":
                self._setup_hyperparam_optimizer()

            # Log initial setup
            env_info = {
                "worker_id": self.worker_id,
                "environment": resolved_env_name,
                "environment_preset": env_name if env_name != resolved_env_name else None,
                "algorithm": self.algorithm_name,
                "optimizer": self.optimizer_name,
                "hyperparam_optimizer": self.hyperparam_optimizer_name,
                "device": str(self.device),
                "obs_space_shape": list(self.env.observation_space.shape),
                "action_space_type": str(type(self.env.action_space)),
                "learning_rate": lr,
            }

            if hasattr(self.env.action_space, "n"):
                env_info["action_space_size"] = self.env.action_space.n
            else:
                env_info["action_space_shape"] = list(self.env.action_space.shape)

            # Log to both systems
            self.crazy_logger.log_step(**env_info)

            # Log hyperparameters
            hyperparams = self.algorithm.get_hyperparameters()
            hyperparams.update(env_info)
            self.crazy_logger.log_hyperparameters(hyperparams)

            self._update_status("READY", environment=resolved_env_name, device=str(self.device))

            # Log initialization completion to enhanced logger
            self.logger.log(
                LogLevel.INFO,
                "RL components setup completed",
                event_type=EventType.SYSTEM,
                context={
                    "environment": resolved_env_name,
                    "environment_preset": env_name if env_name != resolved_env_name else None,
                    "algorithm": self.algorithm_name.upper(),
                    "optimizer": self.optimizer_name,
                    "device": str(self.device),
                    "observation_space_shape": list(self.env.observation_space.shape),
                    "action_space_info": env_info.get("action_space_size") or env_info.get("action_space_shape")
                }
            )

        except Exception as e:
            self._update_status("ERROR", error=str(e))
            raise e

    def _setup_custom_optimizers(self, lr: float):
        """Setup custom optimizers for the algorithm"""
        optimizer_class = self.OPTIMIZERS[self.optimizer_name]
        
        if self.algorithm_name == "ppo":
            # PPO has separate policy and value optimizers
            if hasattr(self.algorithm, 'policy_net') and hasattr(self.algorithm, 'value_net'):
                self.algorithm.policy_optimizer = optimizer_class(
                    self.algorithm.policy_net.parameters(), lr=lr
                )
                self.algorithm.value_optimizer = optimizer_class(
                    self.algorithm.value_net.parameters(), lr=lr
                )
        elif self.algorithm_name == "ddpg":
            # DDPG has separate actor and critic optimizers
            if hasattr(self.algorithm, 'actor') and hasattr(self.algorithm, 'critic'):
                self.algorithm.actor_optimizer = optimizer_class(
                    self.algorithm.actor.parameters(), lr=lr
                )
                self.algorithm.critic_optimizer = optimizer_class(
                    self.algorithm.critic.parameters(), lr=lr
                )
        
        self.logger.log(
            LogLevel.INFO,
            f"Custom optimizer configured: {self.optimizer_name}",
            event_type=EventType.SYSTEM,
            context={
                "optimizer_class": optimizer_class.__name__,
                "algorithm": self.algorithm_name,
                "learning_rate": lr
            }
        )

    def _setup_hyperparam_optimizer(self):
        """Setup hyperparameter optimizer"""
        if self.hyperparam_optimizer_name == "bayesian":
            # Initialize Bayesian optimizer with algorithm's hyperparameter space
            param_space = self._get_hyperparam_space()
            self.hyperparam_optimizer = BayesianOptimizationManager(
                parameter_space=param_space,
                n_initial_points=10,
            )
        elif self.hyperparam_optimizer_name == "random":
            # Random search implementation would go here
            self.logger.log(
                LogLevel.WARNING,
                "Random search hyperparameter optimizer not yet implemented",
                event_type=EventType.SYSTEM
            )
        elif self.hyperparam_optimizer_name == "grid":
            # Grid search implementation would go here  
            self.logger.log(
                LogLevel.WARNING,
                "Grid search hyperparameter optimizer not yet implemented",
                event_type=EventType.SYSTEM
            )
        
        if self.hyperparam_optimizer:
            self.logger.log(
                LogLevel.INFO,
                f"{self.hyperparam_optimizer_name.capitalize()} hyperparameter optimizer initialized",
                event_type=EventType.SYSTEM,
                context={"optimizer_type": self.hyperparam_optimizer_name}
            )

    def _get_hyperparam_space(self):
        """Get hyperparameter space for optimization based on algorithm"""
        if self.algorithm_name == "ppo":
            return {
                "learning_rate": (1e-5, 1e-2),
                "gamma": (0.9, 0.9999),
                "lambda_gae": (0.8, 0.99),
                "clip_epsilon": (0.1, 0.3),
                "entropy_coef": (1e-5, 1e-2),
                "critic_coef": (0.1, 2.0),
            }
        elif self.algorithm_name == "ddpg":
            return {
                "learning_rate_actor": (1e-5, 1e-2),
                "learning_rate_critic": (1e-4, 1e-2),
                "gamma": (0.9, 0.9999),
                "tau": (0.001, 0.01),
                "noise_sigma": (0.1, 0.5),
            }
        else:
            return {}

    def collect_episode(self) -> Dict[str, Any]:
        """Collect a complete episode with detailed logging"""
        episode_start_time = time.time()

        obs = self.env.reset()
        episode_data = {
            "observations": [obs],
            "actions": [],
            "rewards": [],
            "log_probs": [],
            "dones": [],
        }

        episode_reward = 0
        episode_length = 0

        for step in range(1000):  # Max steps per episode
            # Get action from policy
            action, log_prob = self.algorithm.get_action(obs)

            # Take step in environment
            next_obs, reward, done, truncated, info = universal_gym_step(
                self.env, action
            )

            # Store data
            episode_data["actions"].append(action)
            episode_data["rewards"].append(reward)
            episode_data["log_probs"].append(log_prob)
            episode_data["observations"].append(next_obs)
            episode_data["dones"].append(done)

            episode_reward += self._to_scalar(reward)
            episode_length += 1
            obs = next_obs

            if done or truncated:
                break

        episode_time = time.time() - episode_start_time

        return {
            "episode_data": episode_data,
            "episode_reward": episode_reward,
            "episode_length": episode_length,
            "episode_time": episode_time,
        }

    def train_on_episode(self, episode_data: Dict) -> Dict[str, Any]:
        """Train algorithm on episode data"""
        train_start_time = time.time()

        # Convert episode data to rollout format
        rollout = []
        for i in range(len(episode_data["rewards"])):
            rollout.append(
                (
                    episode_data["observations"][i],
                    episode_data["actions"][i],
                    episode_data["rewards"][i],
                    (
                        episode_data["observations"][i + 1]
                        if i + 1 < len(episode_data["observations"])
                        else episode_data["observations"][i]
                    ),
                    episode_data["dones"][i],
                )
            )

        # Train algorithm
        train_metrics = self.algorithm.train_step(rollout)
        train_time = time.time() - train_start_time

        train_metrics["training_time"] = train_time
        train_metrics["worker_id"] = self.worker_id

        return train_metrics

    def log_episode(
        self, episode_reward: float, episode_length: int, train_metrics: Dict
    ):
        """Log episode data to both JSON and pickle systems"""
        timestamp = datetime.now().isoformat()

        # JSON data for frontend (lightweight)
        json_entry = {
            "timestamp": timestamp,
            "episode": self.current_episode,
            "worker_id": self.worker_id,
            "reward": episode_reward,
            "length": episode_length,
            "avg_reward_last_100": self._get_avg_reward(100),
            "reward_trend": self._get_reward_trend(),
            "training_time": train_metrics.get("training_time", 0),
            "status": self.status,
        }

        # Add to JSON buffer (async write)
        self.json_buffer.append(json_entry)

        # Pickle data for detailed ML analysis (comprehensive)
        pickle_entry = {
            "timestamp": timestamp,
            "episode": self.current_episode,
            "worker_id": self.worker_id,
            "episode_reward": episode_reward,
            "episode_length": episode_length,
            "train_metrics": train_metrics,
            "recent_rewards": list(self.episode_rewards)[-50:],  # Last 50 episodes
            "algorithm_state": {
                "learning_rate": self.algorithm.learning_rate,
                # Add more algorithm-specific metrics
            },
        }

        # Add to pickle buffer (async write)
        self.pickle_buffer.append(pickle_entry)

        # Log to CrazyLogger as well
        all_metrics = {**json_entry, **train_metrics}
        self.crazy_logger.log_episode(**all_metrics)

    def save_checkpoint(self):
        """Save model checkpoint"""
        try:
            checkpoint = {
                "algorithm": self.algorithm_name,
                "optimizer": self.optimizer_name,
                "hyperparam_optimizer": self.hyperparam_optimizer_name,
                "worker_id": self.worker_id,
                "episode": self.current_episode,
                "total_episodes": self.total_episodes,
                "best_reward": self.best_reward,
                "timestamp": time.time(),
                "algorithm_state": self.algorithm.state_dict(),
            }

            # Algorithm-specific checkpoint data
            if self.algorithm_name == "ppo":
                checkpoint.update({
                    "policy_state_dict": self.algorithm.policy_net.state_dict(),
                    "value_state_dict": self.algorithm.value_net.state_dict(),
                    "policy_optimizer_state_dict": self.algorithm.policy_optimizer.state_dict(),
                    "value_optimizer_state_dict": self.algorithm.value_optimizer.state_dict(),
                })
            elif self.algorithm_name == "ddpg":
                checkpoint.update({
                    "actor_state_dict": self.algorithm.actor.state_dict(),
                    "critic_state_dict": self.algorithm.critic.state_dict(),
                    "target_actor_state_dict": self.algorithm.target_actor.state_dict(),
                    "target_critic_state_dict": self.algorithm.target_critic.state_dict(),
                    "actor_optimizer_state_dict": self.algorithm.actor_optimizer.state_dict(),
                    "critic_optimizer_state_dict": self.algorithm.critic_optimizer.state_dict(),
                })

            model_file = (
                self.models_dir
                / f"worker_{self.worker_id}_{self.algorithm_name}_episode_{self.current_episode}.pt"
            )
            torch.save(checkpoint, model_file)

            self.logger.log(
                LogLevel.INFO,
                f"Checkpoint saved successfully",
                event_type=EventType.CHECKPOINT,
                context={
                    "episode": self.current_episode,
                    "model_file": str(model_file),
                    "best_reward": self.best_reward,
                    "algorithm": self.algorithm_name
                }
            )
            return True

        except Exception as e:
            self.logger.log(
                LogLevel.ERROR,
                "Checkpoint save failed",
                event_type=EventType.ERROR,
                context={
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "episode": self.current_episode,
                    "algorithm": self.algorithm_name
                }
            )
            return False

    def should_terminate(self) -> bool:
        """Check if worker should terminate"""
        # Check episode limit
        if self.current_episode >= self.max_episodes:
            return True

        # Check for termination signals (can be overridden in subclasses)
        return False

    def run(self, env_name: str, device: str = "cuda:0", lr: float = 3e-4, **algorithm_config):
        """
        Main training loop - fixed number of episodes
        
        Args:
            env_name: Environment name or preset
            device: Training device
            lr: Learning rate
            **algorithm_config: Additional algorithm-specific configuration
        """
        self.logger.log(
            LogLevel.INFO,
            f"Starting training session",
            event_type=EventType.TRAINING_START,
            context={
                "max_episodes": self.max_episodes,
                "algorithm": self.algorithm_name,
                "environment": env_name,
                "device": device,
                "learning_rate": lr,
                "config": algorithm_config
            }
        )

        try:
            # Setup RL components
            self.setup_rl_components(env_name=env_name, device=device, lr=lr, **algorithm_config)

            self._update_status("TRAINING")

            # Training loop
            for episode in range(self.max_episodes):
                self.current_episode = episode + 1
                self.total_episodes += 1

                # Hyperparameter optimization step
                if self.hyperparam_optimizer and episode > 0 and episode % 50 == 0:
                    self._optimize_hyperparameters(episode)

                # Collect episode
                episode_result = self.collect_episode()
                episode_reward = episode_result["episode_reward"]
                episode_length = episode_result["episode_length"]

                # Train on episode
                train_metrics = self.train_on_episode(episode_result["episode_data"])

                # Store episode stats
                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(episode_length)

                # Update best reward
                if episode_reward > self.best_reward:
                    self.best_reward = episode_reward

                # Log episode info
                resolved_env_name = self.resolve_environment_name(env_name)
                env_info = {
                    "worker_id": self.worker_id,
                    "environment": resolved_env_name,
                    "algorithm": self.algorithm_name,
                    "optimizer": self.optimizer_name,
                    "device": str(self.device),
                    "obs_space_shape": list(self.env.observation_space.shape),
                    "action_space_type": str(type(self.env.action_space)),
                    "learning_rate": lr,
                }

                # Log episode
                self.log_episode(episode_reward, episode_length, train_metrics)

                # Update hyperparameters in logs
                hyperparams = self.algorithm.get_hyperparameters()
                hyperparams.update(env_info)
                self.crazy_logger.log_hyperparameters(hyperparams)

                # Progress updates
                if episode % self.status_update_frequency == 0:
                    avg_reward = self._get_avg_reward(100)
                    self._update_status(
                        "TRAINING",
                        avg_reward=avg_reward,
                        episodes_remaining=self.max_episodes - episode,
                    )

                    self.logger.log(
                        LogLevel.INFO,
                        f"Training progress update",
                        event_type=EventType.TRAINING_PROGRESS,
                        context={
                            "episode": episode,
                            "max_episodes": self.max_episodes,
                            "episode_reward": episode_reward,
                            "avg_reward_100": avg_reward,
                            "best_reward": self.best_reward,
                            "episodes_remaining": self.max_episodes - episode
                        }
                    )

                # Checkpointing
                if episode % self.checkpoint_frequency == 0:
                    self.save_checkpoint()

                # Check for early termination
                if self.should_terminate():
                    self.logger.log(
                        LogLevel.INFO,
                        "Training termination requested",
                        event_type=EventType.SYSTEM,
                        context={
                            "episode": episode,
                            "reason": "early_termination",
                            "best_reward": self.best_reward
                        }
                    )
                    break

            # Training completed
            self._update_status("COMPLETED")
            final_avg = self._get_avg_reward(100)
            self.logger.log(
                LogLevel.INFO,
                "Training session completed successfully",
                event_type=EventType.TRAINING_END,
                context={
                    "algorithm": self.algorithm_name,
                    "total_episodes": self.current_episode,
                    "best_reward": self.best_reward,
                    "final_average": final_avg,
                    "training_duration_seconds": int(time.time() - self.start_time)
                }
            )

        except KeyboardInterrupt:
            self._update_status("INTERRUPTED")
            self.logger.log(
                LogLevel.WARNING,
                "Training interrupted by user",
                event_type=EventType.SYSTEM,
                context={"episode": self.current_episode, "reason": "keyboard_interrupt"}
            )
        except Exception as e:
            self._update_status("ERROR", error=str(e))
            self.logger.log(
                LogLevel.ERROR,
                "Training failed with error",
                event_type=EventType.ERROR,
                context={
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "episode": self.current_episode,
                    "algorithm": self.algorithm_name
                }
            )
            raise
        finally:
            self.cleanup()

    def _optimize_hyperparameters(self, episode: int):
        """Perform hyperparameter optimization step"""
        if not self.hyperparam_optimizer:
            return
            
        try:
            # Get recent performance metric (average reward over last 50 episodes)
            recent_performance = self._get_avg_reward(50)
            
            if self.hyperparam_optimizer_name == "bayesian":
                # Get current hyperparameters
                current_hyperparams = self.algorithm.get_hyperparameters()
                
                # Register the observation with the optimizer
                # This would typically be implemented with the actual optimizer's API
                self.logger.log(
                    LogLevel.INFO,
                    "Hyperparameter optimization step",
                    event_type=EventType.HYPERPARAMETER_OPTIMIZATION,
                    context={
                        "episode": episode,
                        "performance": recent_performance,
                        "optimizer_type": self.hyperparam_optimizer_name,
                        "current_hyperparams": current_hyperparams
                    }
                )
                
                # For now, just log that optimization would happen here
                # In a full implementation, you would:
                # 1. Register current hyperparams and performance with optimizer
                # 2. Get suggested new hyperparams from optimizer
                # 3. Update algorithm with new hyperparams
                
        except Exception as e:
            self.logger.log(
                LogLevel.ERROR,
                "Hyperparameter optimization error",
                event_type=EventType.ERROR,
                context={
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "optimizer_type": self.hyperparam_optimizer_name
                }
            )

    def cleanup(self):
        """Cleanup resources"""
        self.logger.log(
            LogLevel.INFO,
            "Worker cleanup started",
            event_type=EventType.SYSTEM
        )

        # Final checkpoint
        self.save_checkpoint()

        # Stop async logging
        self.logging_active = False
        if self.logging_thread and self.logging_thread.is_alive():
            self.logging_thread.join(timeout=5)

        # Final log flush
        if self.json_buffer:
            self._write_json_batch(list(self.json_buffer))
        if self.pickle_buffer:
            self._write_pickle_batch(list(self.pickle_buffer))

        # Close CrazyLogger
        if self.crazy_logger:
            self.crazy_logger.close()

        # Final status
        self._update_status("SHUTDOWN")
        self.logger.log(
            LogLevel.INFO,
            "Worker cleanup completed successfully",
            event_type=EventType.SYSTEM,
            context={
                "total_episodes": self.current_episode,
                "best_reward": self.best_reward,
                "total_uptime_seconds": int(time.time() - self.start_time)
            }
        )

    # Utility methods
    def _to_scalar(self, x):
        """Convert tensor to scalar safely"""
        if torch.is_tensor(x):
            return x.item() if x.dim() == 0 else x.mean().item()
        else:
            return float(x)

    def _get_avg_reward(self, n: int) -> float:
        """Get average reward over last n episodes"""
        if not self.episode_rewards:
            return 0.0
        recent = list(self.episode_rewards)[-n:]
        return np.mean(recent)

    def _get_reward_trend(self) -> float:
        """Calculate reward trend (recent vs older episodes)"""
        if len(self.episode_rewards) < 20:
            return 0.0

        recent = list(self.episode_rewards)[-10:]
        older = list(self.episode_rewards)[-20:-10]

        return np.mean(recent) - np.mean(older)


# Example usage for testing
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="BaseWorker RL Training")
    parser.add_argument("--worker_id", type=int, default=0, help="Worker ID")
    parser.add_argument(
        "--log_dir", type=str, default="./worker_logs", help="Log directory"
    )
    parser.add_argument("--max_episodes", type=int, default=1000, help="Max episodes")
    parser.add_argument("--env", type=str, default="CartPole-v1", help="Environment name or preset")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    
    # Algorithm selection
    parser.add_argument(
        "--algorithm", type=str, default="ppo", 
        choices=BaseWorker.get_available_algorithms(),
        help="RL algorithm to use"
    )
    
    # Optimizer selection  
    parser.add_argument(
        "--optimizer", type=str, default="adam",
        choices=BaseWorker.get_available_optimizers(), 
        help="Optimizer to use"
    )
    
    # Hyperparameter optimizer selection
    parser.add_argument(
        "--hyperparam_optimizer", type=str, default="none",
        choices=BaseWorker.get_available_hyperparam_optimizers(),
        help="Hyperparameter optimizer to use"
    )
    
    # Algorithm-specific hyperparameters
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--clip_epsilon", type=float, default=0.2, help="PPO clipping parameter")
    parser.add_argument("--entropy_coef", type=float, default=1e-4, help="Entropy coefficient")
    parser.add_argument("--tau", type=float, default=0.005, help="DDPG soft update parameter")

    args = parser.parse_args()

    # Print available options
    print("Available algorithms:", BaseWorker.get_available_algorithms())
    print("Available optimizers:", BaseWorker.get_available_optimizers())
    print("Available hyperparam optimizers:", BaseWorker.get_available_hyperparam_optimizers())
    print("Environment presets:", list(BaseWorker.get_environment_presets().keys()))
    
    # Create algorithm-specific config
    algorithm_config = {
        "gamma": args.gamma,
    }
    
    if args.algorithm == "ppo":
        algorithm_config.update({
            "clip_epsilon": args.clip_epsilon,
            "entropy_coef": args.entropy_coef,
        })
    elif args.algorithm == "ddpg":
        algorithm_config.update({
            "tau": args.tau,
        })

    # Create and run worker
    worker = BaseWorker(
        worker_id=args.worker_id, 
        log_dir=args.log_dir, 
        max_episodes=args.max_episodes,
        algorithm=args.algorithm,
        optimizer=args.optimizer,
        hyperparam_optimizer=args.hyperparam_optimizer,
    )

    worker.run(
        env_name=args.env, 
        device=args.device, 
        lr=args.lr,
        **algorithm_config
    )
