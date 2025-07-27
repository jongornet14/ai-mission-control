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
from ..environments.gym_wrapper import GymEnvironmentWrapper
from ..logging.logging import CrazyLogger
from ..environments.gym_wrapper import universal_gym_step


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
    """

    def __init__(
        self,
        worker_id: int,
        log_dir: str,
        max_episodes: int = 1000,
        checkpoint_frequency: int = 50,
        status_update_frequency: int = 10,
        buffer_size: int = 100,
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
        """
        self.worker_id = worker_id
        self.log_dir = Path(log_dir)
        self.max_episodes = max_episodes
        self.checkpoint_frequency = checkpoint_frequency
        self.status_update_frequency = status_update_frequency

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

        # Initialize logging
        self._init_logging()
        self._update_status("INITIALIZED")

        print(f"BaseWorker {worker_id} initialized")
        print(f"Log directory: {log_dir}")
        print(f"Max episodes: {max_episodes}")

    def _init_logging(self):
        """Initialize logging systems"""
        # Initialize CrazyLogger for detailed ML logging
        self.crazy_logger = CrazyLogger(
            log_dir=str(self.log_dir / "crazy_logs"),
            experiment_name=f"worker_{self.worker_id}",
            buffer_size=1000,
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
            print(f"Worker {self.worker_id}: JSON write error: {e}")

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
            print(f"Worker {self.worker_id}: Pickle write error: {e}")

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

        # Also log to stdout for Docker logs
        print(
            f"WORKER_STATUS: {status} worker_id={self.worker_id} episode={self.current_episode}"
        )

    def _write_status_immediate(self, status_data: Dict):
        """Write status immediately (not buffered)"""
        try:
            with open(self.status_file, "w") as f:
                json.dump(status_data, f, indent=2)
        except Exception as e:
            print(f"Worker {self.worker_id}: Status write error: {e}")

    def setup_rl_components(
        self, env_name: str, device: str = "cuda:0", lr: float = 3e-4
    ):
        """Initialize RL environment and algorithm"""
        self._update_status("LOADING_COMPONENTS")

        try:
            self.device = torch.device(device if torch.cuda.is_available() else "cpu")

            # Create environment
            env_wrapper = GymEnvironmentWrapper(
                env_name=env_name,
                device=self.device,
                normalize_observations=True,
                max_episode_steps=None,
            )
            self.env = env_wrapper.create()

            # Create algorithm
            self.algorithm = PPOAlgorithm(
                env=self.env, device=self.device, learning_rate=lr
            )

            # Log initial setup
            env_info = {
                "worker_id": self.worker_id,
                "environment": env_name,
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

            self._update_status("READY", environment=env_name, device=str(self.device))

            print(f"Worker {self.worker_id}: Environment {env_name} created")
            print(f"Worker {self.worker_id}: PPO algorithm initialized")
            print(f"Worker {self.worker_id}: Using device {self.device}")

        except Exception as e:
            self._update_status("ERROR", error=str(e))
            raise e

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
                "policy_state_dict": self.algorithm.policy_net.state_dict(),
                "value_state_dict": self.algorithm.value_net.state_dict(),
                "policy_optimizer_state_dict": self.algorithm.policy_optimizer.state_dict(),
                "value_optimizer_state_dict": self.algorithm.value_optimizer.state_dict(),
                "worker_id": self.worker_id,
                "episode": self.current_episode,
                "total_episodes": self.total_episodes,
                "best_reward": self.best_reward,
                "timestamp": time.time(),
            }

            model_file = (
                self.models_dir
                / f"worker_{self.worker_id}_episode_{self.current_episode}.pt"
            )
            torch.save(checkpoint, model_file)

            print(
                f"Worker {self.worker_id}: Checkpoint saved at episode {self.current_episode}"
            )
            return True

        except Exception as e:
            print(f"Worker {self.worker_id}: Checkpoint save error: {e}")
            return False

    def should_terminate(self) -> bool:
        """Check if worker should terminate"""
        # Check episode limit
        if self.current_episode >= self.max_episodes:
            return True

        # Check for termination signals (can be overridden in subclasses)
        return False

    def run(self, env_name: str, device: str = "cuda:0", lr: float = 3e-4):
        """
        Main training loop - fixed number of episodes
        """
        print(
            f"Worker {self.worker_id}: Starting training for {self.max_episodes} episodes"
        )

        try:
            # Setup RL components
            self.setup_rl_components(env_name=env_name, device=device, lr=lr)

            self._update_status("TRAINING")

            # Training loop
            for episode in range(self.max_episodes):
                self.current_episode = episode + 1
                self.total_episodes += 1

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

                # Log episode
                self.log_episode(episode_reward, episode_length, train_metrics)

                # Progress updates
                if episode % self.status_update_frequency == 0:
                    avg_reward = self._get_avg_reward(100)
                    self._update_status(
                        "TRAINING",
                        avg_reward=avg_reward,
                        episodes_remaining=self.max_episodes - episode,
                    )

                    print(
                        f"Worker {self.worker_id}: Episode {episode}/{self.max_episodes} | "
                        f"Reward: {episode_reward:.2f} | Avg: {avg_reward:.2f} | Best: {self.best_reward:.2f}"
                    )

                # Checkpointing
                if episode % self.checkpoint_frequency == 0:
                    self.save_checkpoint()

                # Check for early termination
                if self.should_terminate():
                    print(f"Worker {self.worker_id}: Early termination requested")
                    break

            # Training completed
            self._update_status("COMPLETED")
            print(f"Worker {self.worker_id}: Training completed!")
            print(f"Best reward: {self.best_reward:.2f}")
            print(f"Final average: {self._get_avg_reward(100):.2f}")

        except KeyboardInterrupt:
            self._update_status("INTERRUPTED")
            print(f"Worker {self.worker_id}: Training interrupted by user")
        except Exception as e:
            self._update_status("ERROR", error=str(e))
            print(f"Worker {self.worker_id}: Training error: {e}")
            raise
        finally:
            self.cleanup()

    def cleanup(self):
        """Cleanup resources"""
        print(f"Worker {self.worker_id}: Cleaning up...")

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
        print(f"Worker {self.worker_id}: Cleanup completed")

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
    parser.add_argument("--env", type=str, default="CartPole-v1", help="Environment")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")

    args = parser.parse_args()

    # Create and run worker
    worker = BaseWorker(
        worker_id=args.worker_id, log_dir=args.log_dir, max_episodes=args.max_episodes
    )

    worker.run(env_name=args.env, device=args.device, lr=args.lr)
