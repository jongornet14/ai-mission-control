#!/usr/bin/env python3
"""
DistributedWorker - Simple inheritance from BaseWorker
Adds distributed training capabilities while keeping BaseWorker clean
"""

import torch
import json
import time
import shutil
from pathlib import Path
import sys
import logging

# Import your base worker
from intellinaut.workers.base import BaseWorker
from intellinaut.logging.debugging import create_worker_debugger


class DistributedWorker(BaseWorker):
    """
    Distributed Worker - inherits from BaseWorker and adds coordination capabilities

    Features:
    - All BaseWorker functionality (episode collection, training, logging)
    - Model synchronization with coordinator
    - Distributed checkpointing
    - Signal-based communication
    """

    def __init__(self, worker_id, shared_dir, **kwargs):
        """
        Initialize DistributedWorker

        Args:
            worker_id: Unique worker identifier
            shared_dir: Shared directory for coordination
            **kwargs: All other BaseWorker arguments
        """
        # Setup log directory within shared space
        log_dir = f"{shared_dir}/worker_logs/worker_{worker_id}"

        # Initialize parent BaseWorker
        super().__init__(worker_id=worker_id, log_dir=log_dir, **kwargs)

        # Add distributed-specific setup
        self.shared_dir = Path(shared_dir)
        self.setup_distributed_directories()

        # Initialize debugger for this worker
        self.debugger = create_worker_debugger(
            str(self.shared_dir), worker_id=worker_id
        )

        print(f"\033[92mDistributedWorker {worker_id} initialized\033[0m")
        print(f"\033[94mShared directory: {shared_dir}\033[0m")
        if self.debugger:
            self.debugger.log_text("INFO", f"DistributedWorker {worker_id} initialized")
            self.debugger.log_text("INFO", f"Shared directory: {shared_dir}")

        # Make sure BaseWorker does NOT overwrite self.debugger
        # If BaseWorker sets self.debugger, you may want to refactor BaseWorker to accept a debugger argument
        # and pass self.debugger to super().__init__ if needed.

    def perturb_model(self):
        """
        Apply perturbation to the model's weights.
        """
        try:
            # Apply small random noise to the policy and value networks
            for param in self.algorithm.policy_net.parameters():
                param.data += (
                    torch.randn_like(param) * 0.01 * (self.worker_id + 1)
                )  # Scale noise by worker ID

            for param in self.algorithm.value_net.parameters():
                param.data += (
                    torch.randn_like(param) * 0.01 * (self.worker_id + 1)
                )  # Scale noise by worker ID

            print(
                f"\033[92mWorker {self.worker_id}: Model perturbed successfully\033[0m"
            )
            if self.debugger:
                self.debugger.log_text(
                    "INFO", f"Worker {self.worker_id}: Model perturbed successfully"
                )
        except Exception as e:
            print(
                f"\033[91mWorker {self.worker_id}: Error perturbing model: {e}\033[0m"
            )
            if self.debugger:
                self.debugger.log_text(
                    "ERROR", f"Worker {self.worker_id}: Error perturbing model: {e}"
                )

    def setup_distributed_directories(self):
        """Create necessary directories for distributed coordination"""
        self.models_dir = self.shared_dir / "models"
        self.metrics_dir = self.shared_dir / "metrics"
        self.best_model_dir = self.shared_dir / "best_model"
        self.signals_dir = self.shared_dir / "signals"

        # Create all directories
        for directory in [
            self.models_dir,
            self.metrics_dir,
            self.best_model_dir,
            self.signals_dir,
        ]:
            directory.mkdir(parents=True, exist_ok=True)

        print(
            f"\033[92mWorker {self.worker_id}: Distributed directories created\033[0m"
        )
        if hasattr(self, "debugger") and self.debugger:
            self.debugger.log_text(
                "INFO", f"Worker {self.worker_id}: Distributed directories created"
            )

    def check_for_coordinator_updates(self):
        """Check if coordinator has signaled a model update"""
        signal_file = self.signals_dir / f"update_worker_{self.worker_id}.signal"
        if signal_file.exists():
            print(f"\033[93mWorker {self.worker_id}: Update signal received\033[0m")
            if self.debugger:
                self.debugger.log_text(
                    "INFO", f"Worker {self.worker_id}: Update signal received"
                )
            self.load_best_model_from_coordinator()
            signal_file.unlink()  # Remove the signal file after processing
            return True
        return False

    def load_model_from_coordinator(self):
        """Load the model assigned to this worker by the coordinator."""
        try:
            assigned_model_file = self.models_dir / f"worker_{self.worker_id}_latest.pt"
            if not assigned_model_file.exists():
                print(
                    f"\033[93mWorker {self.worker_id}: No assigned model available\033[0m"
                )
                return False

            # Load checkpoint
            checkpoint = torch.load(assigned_model_file, map_location=self.device)

            # Update model weights
            self.algorithm.policy_net.load_state_dict(checkpoint["policy_state_dict"])
            self.algorithm.value_net.load_state_dict(checkpoint["value_state_dict"])

            print(
                f"\033[92mWorker {self.worker_id}: Loaded assigned model from coordinator\033[0m"
            )
            if self.debugger:
                self.debugger.log_text(
                    "INFO",
                    f"Worker {self.worker_id}: Loaded assigned model from coordinator",
                )
            return True

        except Exception as e:
            print(
                f"\033[91mWorker {self.worker_id}: Error loading assigned model: {e}\033[0m"
            )
            if self.debugger:
                self.debugger.log_text(
                    "ERROR",
                    f"Worker {self.worker_id}: Error loading assigned model: {e}",
                )
            return False

    def save_checkpoint(self):
        """Enhanced checkpoint saving - calls parent + adds distributed features"""
        # Call parent's checkpoint method first
        parent_success = super().save_checkpoint()

        # Add distributed checkpoint
        distributed_success = self.save_distributed_checkpoint()

        return parent_success and distributed_success

    def save_distributed_checkpoint(self):
        """Save checkpoint in format expected by coordinator"""
        try:
            # Create checkpoint with distributed metadata
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

            # Save to shared models directory for coordinator
            model_file = (
                self.models_dir
                / f"worker_{self.worker_id}_episode_{self.current_episode}.pt"
            )
            torch.save(checkpoint, model_file)

            # Save performance metrics for coordinator evaluation
            self.save_performance_metrics()

            print(
                f"\033[92mWorker {self.worker_id}: Distributed checkpoint saved\033[0m"
            )
            if self.debugger:
                self.debugger.log_text(
                    "INFO", f"Worker {self.worker_id}: Distributed checkpoint saved"
                )
            return True

        except Exception as e:
            print(
                f"\033[91mWorker {self.worker_id}: Distributed checkpoint error: {e}\033[0m"
            )
            if self.debugger:
                self.debugger.log_text(
                    "ERROR",
                    f"Worker {self.worker_id}: Distributed checkpoint error: {e}",
                )
            return False

    def save_performance_metrics(self):
        """Save performance metrics for coordinator to evaluate"""
        try:
            metrics = {
                "worker_id": self.worker_id,
                "total_episodes": self.total_episodes,
                "avg_reward": self._get_avg_reward(100),
                "reward_change": self.calculate_reward_change(),
                "episode_rewards": list(self.episode_rewards)[
                    -50:
                ],  # Last 50 for analysis
                "timestamp": time.time(),
                "training_time": time.time() - self.start_time,
                "best_reward": self.best_reward,
                "current_status": self.status,
            }

            metrics_file = (
                self.metrics_dir / f"worker_{self.worker_id}_performance.json"
            )
            with open(metrics_file, "w") as f:
                json.dump(metrics, f, indent=2)

            return True

        except Exception as e:
            print(
                f"\033[91mWorker {self.worker_id}: Error saving performance metrics: {e}\033[0m"
            )
            if self.debugger:
                self.debugger.log_text(
                    "ERROR",
                    f"Worker {self.worker_id}: Error saving performance metrics: {e}",
                )
            return False

    def calculate_reward_change(self):
        """Calculate reward trend for coordinator evaluation"""
        if len(self.episode_rewards) < 20:
            return 0.0

        import numpy as np

        recent_rewards = list(self.episode_rewards)[-10:]
        older_rewards = list(self.episode_rewards)[-20:-10]

        recent_avg = np.mean(recent_rewards)
        older_avg = np.mean(older_rewards)

        return recent_avg - older_avg

    def should_terminate(self):
        """Check termination conditions - parent conditions + distributed signals"""
        # Check parent termination conditions first
        if super().should_terminate():
            return True

        # Check distributed termination signals
        worker_terminate = self.shared_dir / f"terminate_worker_{self.worker_id}.signal"
        if worker_terminate.exists():
            print(
                f"\033[91mWorker {self.worker_id}: Individual termination signal received\033[0m"
            )
            if self.debugger:
                self.debugger.log_text(
                    "WARNING",
                    f"Worker {self.worker_id}: Individual termination signal received",
                )
            return True

        global_terminate = self.shared_dir / "terminate_all.signal"
        if global_terminate.exists():
            print(
                f"\033[91mWorker {self.worker_id}: Global termination signal received\033[0m"
            )
            if self.debugger:
                self.debugger.log_text(
                    "WARNING",
                    f"Worker {self.worker_id}: Global termination signal received",
                )
            return True

        stop_all_signal = self.shared_dir / "stop_all_workers.signal"
        if stop_all_signal.exists():
            print(
                f"\033[91mWorker {self.worker_id}: Stop-all-workers signal received\033[0m"
            )
            if self.debugger:
                self.debugger.log_text(
                    "WARNING",
                    f"Worker {self.worker_id}: Stop-all-workers signal received",
                )
            return True

        return False

    def run_distributed(
        self, env_name, device="cuda:0", lr=3e-4, sync_check_frequency=10
    ):
        """
        Run distributed training with coordinator synchronization

        Args:
            env_name: Environment name (e.g., 'CartPole-v1')
            device: Device to use ('cuda:0' or 'cpu')
            lr: Learning rate
            sync_check_frequency: Episodes between checking for coordinator updates
        """
        print(f"\033[92mWorker {self.worker_id}: Starting distributed training\033[0m")
        print(f"\033[94mEnvironment: {env_name}\033[0m")
        print(
            f"\033[94mSync check frequency: every {sync_check_frequency} episodes\033[0m"
        )
        if self.debugger:
            self.debugger.log_text(
                "INFO", f"Worker {self.worker_id}: Starting distributed training"
            )
            self.debugger.log_text("INFO", f"Environment: {env_name}")
            self.debugger.log_text(
                "INFO", f"Sync check frequency: every {sync_check_frequency} episodes"
            )

        try:
            # Setup RL components (uses parent method)
            # Make sure to inject the debugger into the algorithm if possible
            self.setup_rl_components(env_name=env_name, device=device, lr=lr)
            # If your PPOAlgorithm supports a debugger argument, pass it there as well

            self._update_status("DISTRIBUTED_TRAINING")

            # Main distributed training loop
            for episode in range(self.max_episodes):
                self.current_episode = episode + 1
                self.total_episodes += 1

                config_path = (
                    self.shared_dir
                    / "metrics"
                    / f"worker_{self.worker_id}_hyperparameters.json"
                )
                if config_path.exists():
                    with open(config_path, "r") as f:
                        config = json.load(f)
                    # Accept both flat dicts and nested configs
                    if isinstance(config, dict) and all(
                        isinstance(v, (int, float)) for v in config.values()
                    ):
                        new_hyperparams = config
                    elif (
                        "worker" in config and "optimization_params" in config["worker"]
                    ):
                        new_hyperparams = config["worker"]["optimization_params"]
                    else:
                        new_hyperparams = {}
                    if new_hyperparams:
                        self.algorithm.update_hyperparameters(new_hyperparams)
                        print(
                            f"\033[92mWorker {self.worker_id}: Applied new hyperparameters from coordinator: {new_hyperparams}\033[0m"
                        )
                        if self.debugger:
                            self.debugger.log_text(
                                "INFO",
                                f"Worker {self.worker_id}: Applied new hyperparameters from coordinator: {new_hyperparams}",
                            )
                    else:
                        import warnings

                        warnings.warn(
                            f"No valid hyperparameters found in: {config_path}"
                        )
                        if self.debugger:
                            self.debugger.log_text(
                                "WARNING",
                                f"No valid hyperparameters found in: {config_path}",
                            )
                else:
                    import warnings

                    warnings.warn(f"Path not found: {config_path}")
                    if self.debugger:
                        self.debugger.log_text(
                            "WARNING", f"Path not found: {config_path}"
                        )

                # Check for coordinator updates periodically
                if episode % sync_check_frequency == 0 and episode > 0:
                    if self.check_for_coordinator_updates():
                        print(
                            f"\033[94mWorker {self.worker_id}: Coordinator update detected\033[0m"
                        )
                        if self.debugger:
                            self.debugger.log_text(
                                "INFO",
                                f"Worker {self.worker_id}: Coordinator update detected",
                            )
                        self.load_best_model_from_coordinator()

                # Collect episode (uses parent method)
                episode_result = self.collect_episode()
                episode_reward = episode_result["episode_reward"]
                episode_length = episode_result["episode_length"]

                # Train on episode (uses parent method)
                train_metrics = self.train_on_episode(episode_result["episode_data"])

                # Update tracking (same as parent)
                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(episode_length)

                if episode_reward > self.best_reward:
                    self.best_reward = episode_reward

                # Log episode (uses parent method)
                env_info = {
                    "worker_id": self.worker_id,
                    "environment": env_name,
                    "device": str(self.device),
                    "obs_space_shape": list(self.env.observation_space.shape),
                    "action_space_type": str(type(self.env.action_space)),
                    "learning_rate": lr,
                }

                # Log episode
                self.log_episode(episode_reward, episode_length, train_metrics)

                hyperparams = self.algorithm.get_hyperparameters()
                hyperparams.update(env_info)
                self.crazy_logger.log_hyperparameters(hyperparams)

                # Progress updates
                if episode % self.status_update_frequency == 0:
                    avg_reward = self._get_avg_reward(100)
                    reward_trend = self.calculate_reward_change()

                    self._update_status(
                        "DISTRIBUTED_TRAINING",
                        avg_reward=avg_reward,
                        reward_trend=reward_trend,
                        episodes_remaining=self.max_episodes - episode,
                        last_sync_check=episode // sync_check_frequency,
                    )

                    print(
                        f"\033[94mWorker {self.worker_id}: Episode {episode}/{self.max_episodes} | "
                        f"Reward: {episode_reward:.2f} | Avg: {avg_reward:.2f} | "
                        f"Trend: {reward_trend:.3f}\033[0m"
                    )
                    if self.debugger:
                        self.debugger.log_text(
                            "INFO",
                            f"Worker {self.worker_id}: Episode {episode}/{self.max_episodes} | "
                            f"Reward: {episode_reward:.2f} | Avg: {avg_reward:.2f} | "
                            f"Trend: {reward_trend:.3f}",
                        )

                # Distributed checkpointing
                if episode % self.checkpoint_frequency == 0:
                    self.save_checkpoint()  # This calls both parent and distributed methods

                # Check for termination
                if self.should_terminate():
                    print(
                        f"\033[91mWorker {self.worker_id}: Early termination requested\033[0m"
                    )
                    if self.debugger:
                        self.debugger.log_text(
                            "WARNING",
                            f"Worker {self.worker_id}: Early termination requested",
                        )
                    break

            # Training completed
            self._update_status("COMPLETED")
            print(
                f"\033[92mWorker {self.worker_id}: Distributed training completed!\033[0m"
            )
            print(f"\033[93mBest reward: {self.best_reward:.2f}\033[0m")
            print(f"\033[94mFinal average: {self._get_avg_reward(100):.2f}\033[0m")
            if self.debugger:
                self.debugger.log_text(
                    "INFO", f"Worker {self.worker_id}: Distributed training completed!"
                )
                self.debugger.log_text("INFO", f"Best reward: {self.best_reward:.2f}")
                self.debugger.log_text(
                    "INFO", f"Final average: {self._get_avg_reward(100):.2f}"
                )

        except KeyboardInterrupt:
            self._update_status("INTERRUPTED")
            print(
                f"\033[91mWorker {self.worker_id}: Training interrupted by user\033[0m"
            )
            if self.debugger:
                self.debugger.log_text(
                    "WARNING", f"Worker {self.worker_id}: Training interrupted by user"
                )
        except Exception as e:
            self._update_status("ERROR", error=str(e))
            print(f"\033[91mWorker {self.worker_id}: Training error: {e}\033[0m")
            if self.debugger:
                self.debugger.log_text(
                    "ERROR", f"Worker {self.worker_id}: Training error: {e}"
                )
            raise
        finally:
            # Cleanup (uses parent method)
            self.cleanup()


# Convenience function for easy usage
def create_distributed_worker(worker_id, shared_dir, max_episodes=1000):
    """
    Convenience function to create a DistributedWorker with sensible defaults

    Args:
        worker_id: Unique worker identifier
        shared_dir: Shared directory for coordination
        max_episodes: Maximum episodes to train

    Returns:
        DistributedWorker instance
    """
    return DistributedWorker(
        worker_id=worker_id,
        shared_dir=shared_dir,
        max_episodes=max_episodes,
        checkpoint_frequency=50,
        status_update_frequency=10,
    )


if __name__ == "__main__":
    # Example usage
    import argparse

    parser = argparse.ArgumentParser(description="DistributedWorker Example")
    parser.add_argument("--worker_id", type=int, default=0, help="Worker ID")
    parser.add_argument(
        "--shared_dir", type=str, default="./shared", help="Shared directory"
    )
    parser.add_argument("--env", type=str, default="CartPole-v1", help="Environment")
    parser.add_argument("--max_episodes", type=int, default=1000, help="Max episodes")

    args = parser.parse_args()

    # Create and run distributed worker
    worker = create_distributed_worker(
        worker_id=args.worker_id,
        shared_dir=args.shared_dir,
        max_episodes=args.max_episodes,
    )

    worker.run_distributed(env_name=args.env)
