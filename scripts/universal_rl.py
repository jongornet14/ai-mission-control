#!/usr/bin/env python3
"""
FILENAME: universal_rl.py
Universal RL Training Script with CrazyLogger integration
Simple version using stable packages + comprehensive logging
"""

import os
import sys
import yaml
import torch
import numpy as np
import argparse
from pathlib import Path
from datetime import datetime
from collections import defaultdict

# Simple imports - no TorchRL
import gym

# Local imports
from algorithms.ppo import PPOAlgorithm
from environments.gym_wrapper import GymEnvironmentWrapper
from crazylogging.crazy_logger import CrazyLogger


class UniversalRLTrainer:
    """Universal RL training class with crazy comprehensive logging"""

    def __init__(self, config_path=None, **kwargs):
        """Initialize trainer with config file or direct parameters"""
        self.config = self.load_config(config_path, **kwargs)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Setup experiment directory first
        self.setup_experiment_dir()

        # Initialize CRAZY LOGGER! 🚀
        # Ensure logger is initialized after config and exp_dir setup
        self.logger = CrazyLogger(
            log_dir=self.exp_dir, experiment_name=self.config["experiment"]["name"]
        )

        # Initialize components
        self.env = None
        self.algorithm = None

        # Training state
        self.current_episode = 0
        self.total_frames_collected = 0
        self.best_reward = float("-inf")

        # Performance tracking
        self.episode_rewards = []
        self.episode_lengths = []

    def load_config(self, config_path=None, **kwargs):
        """Load configuration from YAML file or default, then merge with kwargs."""
        config = {}  # Initialize config
        found_config_file = False

        # Try to load from provided config_path
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, "r") as f:
                    config = yaml.safe_load(f)
                found_config_file = True
                print(f"✅ Loaded configuration from: {config_path}")
            except Exception as e:
                print(
                    f"❌ Error loading config from {config_path}: {e}. Attempting to load default config."
                )
        elif config_path:  # config_path was provided but didn't exist
            print(
                f"⚠️ Provided config file does not exist: {config_path}. Attempting to load default config."
            )

        # If not loaded from provided path, try default config
        if not found_config_file:
            default_config_path = (
                Path(__file__).parent.parent
                / "scripts"
                / "configs"
                / "mujoco"
                / "mujoco_halfcheetah.yaml"
            )
            if default_config_path.exists():
                try:
                    with open(default_config_path, "r") as f:
                        config = yaml.safe_load(f)
                    found_config_file = True
                    print(
                        f"✅ Loaded default configuration from: {default_config_path}"
                    )
                except Exception as e:
                    print(
                        f"❌ Error loading default config from {default_config_path}: {e}."
                    )
            else:
                print(f"❌ Default configuration file not found: {default_config_path}")

        # If still no config, raise an error
        if not found_config_file:
            description = "Configuration file"
            # Decide which path to show in the error message
            if config_path:
                display_path_for_error = Path(config_path)
            else:
                display_path_for_error = (
                    Path(__file__).parent.parent
                    / "scripts"
                    / "configs"
                    / "mujoco"
                    / "mujoco_halfcheetah.yaml"
                )
                description = "Default configuration file"  # Be more specific

            error_msg = f"❌ {description} not found: {display_path_for_error}"

            current_dir = Path.cwd()
            print(f"❌ {error_msg}")
            print(f"📍 Current directory: {current_dir}")
            print(f"📁 Directory contents: {list(current_dir.iterdir())}")

            # Use 'display_path_for_error' here
            if display_path_for_error.parent != current_dir:
                print(
                    f"📁 Parent directory {display_path_for_error.parent} contents: {list(display_path_for_error.parent.iterdir()) if display_path_for_error.parent.exists() else 'Directory does not exist'}"
                )

            # Log to your logger if available (logger might not be initialized yet if this is the first error)
            if hasattr(self, "logger"):
                self.logger.log_step(
                    error="ConfigFileNotFound",
                    file_path=str(display_path_for_error),
                    current_dir=str(current_dir),
                    description=description,
                )
            raise FileNotFoundError(error_msg)

        # Override with command line arguments
        for key, value in kwargs.items():
            if value is not None:
                self.set_nested_config(config, key, value)

        return config

    def set_nested_config(self, config, key, value):
        """Set nested configuration value using dot notation"""
        keys = key.split(".")
        current = config
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]
        current[keys[-1]] = value

    def setup_experiment_dir(self):
        """Create experiment directory structure"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_name = f"{self.config['experiment']['name']}_{timestamp}"

        self.exp_dir = Path(self.config["experiment"]["save_dir"]) / exp_name
        self.exp_dir.mkdir(parents=True, exist_ok=True)

        print(f"📁 Experiment directory: {self.exp_dir}")

    def create_environment(self):
        """Create and configure the environment"""
        env_config = self.config["environment"]

        wrapper = GymEnvironmentWrapper(
            env_name=env_config["name"],
            device=self.device,  # Add this line
            normalize_observations=env_config.get("normalize_observations", True),
            max_episode_steps=env_config.get("max_episode_steps", None),
        )

        self.env = wrapper.create()

        # Log environment info
        env_info = {
            "env_name": env_config["name"],
            "obs_space_shape": self.env.observation_space.shape,
            "action_space_type": str(type(self.env.action_space)),
        }

        if hasattr(self.env.action_space, "n"):
            env_info["action_space_size"] = self.env.action_space.n
        else:
            env_info["action_space_shape"] = self.env.action_space.shape

        self.logger.log_step(**env_info)

        print(f"🎮 Environment: {env_config['name']}")
        print(f"   Observation space: {self.env.observation_space}")
        print(f"   Action space: {self.env.action_space}")

    def create_algorithm(self):
        """Create and configure the RL algorithm"""
        algo_config = self.config["algorithm"]

        self.algorithm = PPOAlgorithm(env=self.env, device=self.device, **algo_config)

        # Log initial hyperparameters
        self.logger.log_hyperparameters(self.algorithm.get_hyperparameters())

        print(f"🧠 Algorithm: {algo_config['name']}")

    def collect_episode(self):
        """Collect a complete episode with detailed logging"""
        self.logger.performance_tracker.start_timer("episode_collection")

        obs = self.env.reset()

        # REMOVE video collection from training episodes
        # frames = [] if self.config['logging']['save_videos'] else None

        episode_data = {
            "observations": [obs],
            "actions": [],
            "rewards": [],
            "log_probs": [],
            "values": [],
            "dones": [],
        }

        episode_reward = 0
        episode_length = 0

        for step in range(self.config["training"]["max_steps_per_episode"]):
            self.logger.performance_tracker.start_timer("action_selection")

            # Get action from policy
            action, log_prob = self.algorithm.get_action(obs)

            action_time = self.logger.performance_tracker.end_timer("action_selection")

            # Take step in environment
            self.logger.performance_tracker.start_timer("env_step")
            next_obs, reward, done, info = self.env.step(action)
            env_step_time = self.logger.performance_tracker.end_timer("env_step")

            # Store data
            episode_data["actions"].append(action)
            episode_data["rewards"].append(reward)
            episode_data["log_probs"].append(log_prob)
            episode_data["observations"].append(next_obs)
            episode_data["dones"].append(done)

            # Log step metrics
            obs_stats = tensor_stats(obs)
            step_metrics = {
                "step_reward": to_scalar(reward),
                "step_action": to_scalar(action),
                "step_log_prob": log_prob,
                "step_obs_mean": obs_stats["mean"],
                "step_obs_std": obs_stats["std"],
                "step_obs_max": obs_stats["max"],
                "step_obs_min": obs_stats["min"],
                "action_selection_time": action_time,
                "env_step_time": env_step_time,
                "cumulative_reward": episode_reward + to_scalar(reward),
            }

            self.logger.log_step(**step_metrics)

            episode_reward += to_scalar(reward)
            episode_length += 1
            obs = next_obs

            if done:
                break

        episode_collection_time = self.logger.performance_tracker.end_timer(
            "episode_collection"
        )

        return episode_data, episode_reward, episode_length, episode_collection_time

    def train_on_episode(self, episode_data):
        """Train algorithm on episode data with detailed logging"""
        self.logger.performance_tracker.start_timer("training")

        # Convert episode data to format for training
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

        training_time = self.logger.performance_tracker.end_timer("training")
        train_metrics["training_time"] = training_time

        # Log model state periodically
        if self.current_episode % self.config["training"]["save_frequency"] == 0:
            self.logger.log_model_state(
                self.algorithm.policy_net,
                self.algorithm.value_optimizer,
                train_metrics.get("loss_total", 0),
                save_weights=True,
            )

        # Log activations periodically
        if (
            self.config["logging"]["log_activations"]
            and self.current_episode % self.config["logging"]["activation_frequency"]
            == 0
        ):
            sample_obs = episode_data["observations"][0].unsqueeze(0).to(self.device)
            self.logger.log_activations(self.algorithm.policy_net, sample_obs)

        return train_metrics

    def evaluate_policy(self):
        """Evaluate current policy with detailed logging"""
        self.logger.performance_tracker.start_timer("evaluation")

        eval_rewards = []
        eval_lengths = []

        # Check if we should record videos this evaluation
        should_record = (
            self.config["logging"].get("save_eval_videos", False)
            and self.current_episode
            % self.config["logging"].get("eval_video_frequency", 100)
            == 0
        )

        eval_success_rate = 0  # Initialize for the success rate calculation

        for eval_ep in range(self.config["training"]["eval_episodes"]):
            obs = self.env.reset()
            eval_reward = 0
            eval_length = 0

            # Only record video for first eval episode when scheduled
            frames = [] if (should_record and eval_ep == 0) else None

            for step in range(self.config["training"]["max_steps_per_episode"]):
                action, _ = self.algorithm.get_action(obs)
                obs, reward, done, info = self.env.step(action)

                # Capture frame if recording
                if frames is not None:
                    try:
                        frame = self.env.render()
                        frames.append(frame)
                    except Exception as e:
                        print(f"❌ Eval frame capture failed: {e}")
                        frames = None

                eval_reward += to_scalar(reward)
                eval_length += 1

                if done:
                    # Check for success (environment specific)
                    if (
                        hasattr(self.env, "spec")
                        and self.env.spec
                        and hasattr(self.env.spec, "reward_threshold")
                        and self.env.spec.reward_threshold
                        and eval_reward >= self.env.spec.reward_threshold
                    ):
                        eval_success_rate += 1
                    break

            eval_rewards.append(eval_reward)
            eval_lengths.append(eval_length)

            # Save evaluation video
            if frames and len(frames) > 0:
                self.logger.log_video(frames, f"eval_episode_{self.current_episode}")
                # print(f"📹 Saved evaluation video for episode {self.current_episode}")

        evaluation_time = self.logger.performance_tracker.end_timer("evaluation")

        eval_metrics = {
            "eval_reward_mean": np.mean(eval_rewards),
            "eval_reward_std": np.std(eval_rewards),
            "eval_reward_min": np.min(eval_rewards),
            "eval_reward_max": np.max(eval_rewards),
            "eval_length_mean": np.mean(eval_lengths),
            "eval_length_std": np.std(eval_lengths),
            "evaluation_time": evaluation_time,
            "eval_success_rate": (
                eval_success_rate / self.config["training"]["eval_episodes"]
                if self.config["training"]["eval_episodes"] > 0
                else 0
            ),  # Add success rate
        }

        return eval_metrics

    def train(self):
        """Main training loop with CRAZY comprehensive logging"""
        from tqdm import tqdm  # Add this import

        print(f"\n🚀 Starting training with CrazyLogger...")
        print(f"Device: {self.device}")
        print("=" * 50)

        # Setup all components
        torch.manual_seed(self.config["experiment"]["seed"])
        self.create_environment()
        self.create_algorithm()

        training_config = self.config["training"]

        if training_config.get("total_frames"):
            max_steps = training_config.get("max_steps_per_episode", 1000)
            estimated_episodes = training_config["total_frames"] // max_steps
            training_config["total_episodes"] = estimated_episodes
            print(
                f"🔄 Using {estimated_episodes} episodes for {training_config['total_frames']} frames"
            )

        # Log initial configuration
        self.logger.log_step(
            **{
                "config_seed": self.config["experiment"]["seed"],
                "config_total_episodes": training_config["total_episodes"],
                "config_device": str(self.device),
            }
        )

        # Main training loop with tqdm progress bar
        pbar = tqdm(range(training_config["total_episodes"]), desc="Training Episodes")

        for episode in pbar:
            self.current_episode = episode

            # Collect episode with full logging
            episode_data, episode_reward, episode_length, collection_time = (
                self.collect_episode()
            )

            # Train on episode
            train_metrics = self.train_on_episode(episode_data)

            # Store episode stats
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_length)

            # Update progress bar with current reward
            pbar.set_postfix(
                {
                    "Reward": f"{to_scalar(episode_reward):.2f}",
                    "Length": episode_length,
                    "Best": f"{self.best_reward:.2f}",
                }
            )

            # Evaluation
            eval_metrics = {}
            if episode % training_config["eval_frequency"] == 0:
                eval_metrics = self.evaluate_policy()

                # print(f"\nEpisode {episode:4d} | "
                #     f"Reward: {eval_metrics['eval_reward_mean']:7.2f} ± {eval_metrics['eval_reward_std']:5.2f} | "
                #     f"Length: {eval_metrics['eval_length_mean']:6.1f} | "
                #     f"Success: {eval_metrics['eval_success_rate']:5.2f}")

                # Update best reward
                current_reward = eval_metrics["eval_reward_mean"]
                if current_reward > self.best_reward:
                    self.best_reward = current_reward
                    # print(f"🏆 New best reward: {current_reward:.2f}")

            # Replace the episode_metrics section with:
            episode_rewards_np = to_numpy(self.episode_rewards)
            episode_metrics = {
                "episode_reward": to_scalar(episode_reward),
                "episode_length": episode_length,
                "episode_collection_time": collection_time,
                "best_reward_so_far": self.best_reward,
                "avg_reward_last_100": (
                    np.mean(episode_rewards_np[-100:])
                    if len(episode_rewards_np) >= 100
                    else np.mean(episode_rewards_np)
                ),
                "reward_trend": (
                    to_scalar(episode_reward) - np.mean(episode_rewards_np[-10:])
                    if len(episode_rewards_np) >= 10
                    else 0
                ),
            }

            # Combine all metrics
            all_metrics = {**episode_metrics, **train_metrics, **eval_metrics}

            # Performance tracker stats
            perf_stats = self.logger.performance_tracker.get_stats()
            all_metrics.update(perf_stats)

            # Log complete episode
            self.logger.log_episode(**all_metrics)

            # Log hyperparameter updates if any
            current_hyperparams = self.algorithm.get_hyperparameters()
            self.logger.log_hyperparameters(current_hyperparams, episode_reward)

            # # Create custom plots periodically
            # if episode % 200 == 0 and episode > 0:
            #     self.create_custom_plots()

        pbar.close()  # Close the progress bar

        # Final evaluation and report
        final_eval = self.evaluate_policy()
        print(f"\n✅ Training completed!")
        print(f"🏆 Best reward: {self.best_reward:.2f}")
        print(
            f"🎯 Final reward: {final_eval['eval_reward_mean']:.2f} ± {final_eval['eval_reward_std']:.2f}"
        )

        # Generate final comprehensive report
        final_summary = self.logger.generate_final_report()

        # Close logger
        self.logger.close()

        return final_summary

    def create_custom_plots(self):
        """Create custom analysis plots"""
        import matplotlib.pyplot as plt

        # Convert to numpy once for all plotting
        rewards_np = to_numpy(self.episode_rewards)
        lengths_np = to_numpy(self.episode_lengths)

        if len(rewards_np) < 10:
            return

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))

        # Episode rewards
        axes[0, 0].plot(rewards_np, alpha=0.6, label="Episode Reward")
        if len(rewards_np) > 20:  # Use rewards_np consistently
            window = min(50, len(rewards_np) // 5)
            moving_avg = np.convolve(
                rewards_np, np.ones(window) / window, mode="valid"
            )  # Use rewards_np
            axes[0, 0].plot(
                range(window - 1, len(rewards_np)),
                moving_avg,
                "r-",
                linewidth=2,
                label=f"MA({window})",
            )
        axes[0, 0].set_xlabel("Episode")
        axes[0, 0].set_ylabel("Reward")
        axes[0, 0].set_title("Training Progress")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Episode lengths
        axes[0, 1].plot(lengths_np, alpha=0.6, color="green")  # Use lengths_np
        axes[0, 1].set_xlabel("Episode")
        axes[0, 1].set_ylabel("Episode Length")
        axes[0, 1].set_title("Episode Lengths")
        axes[0, 1].grid(True, alpha=0.3)

        # Reward distribution
        axes[1, 0].hist(
            rewards_np, bins=30, alpha=0.7, edgecolor="black"
        )  # Use rewards_np
        axes[1, 0].axvline(
            np.mean(rewards_np),
            color="red",
            linestyle="--",
            label=f"Mean: {np.mean(rewards_np):.2f}",
        )
        axes[1, 0].set_xlabel("Reward")
        axes[1, 0].set_ylabel("Frequency")
        axes[1, 0].set_title("Reward Distribution")
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # Recent performance (last 100 episodes)
        recent_rewards = (
            rewards_np[-100:] if len(rewards_np) > 100 else rewards_np
        )  # Use rewards_np
        axes[1, 1].plot(recent_rewards, alpha=0.7, color="orange")
        axes[1, 1].set_xlabel("Recent Episodes")
        axes[1, 1].set_ylabel("Reward")
        axes[1, 1].set_title(
            f"Recent Performance (Last {len(recent_rewards)} episodes)"
        )
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()

        # Log the plot
        self.logger.log_custom_plot("training_progress", fig, self.current_episode)
        plt.close()


def to_numpy(x):
    """Convert tensor to numpy safely"""
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    elif isinstance(x, (list, tuple)):
        return np.array([to_numpy(item) for item in x])
    else:
        return x


def to_scalar(x):
    """Convert tensor to scalar safely"""
    if torch.is_tensor(x):
        return x.item() if x.dim() == 0 else x.mean().item()
    else:
        return float(x)


def tensor_stats(x):
    """Get stats from tensor safely"""
    if torch.is_tensor(x):
        return {
            "mean": x.mean().item(),
            "std": x.std().item(),
            "max": x.max().item(),
            "min": x.min().item(),
        }
    else:
        x_np = np.array(x)
        return {
            "mean": np.mean(x_np),
            "std": np.std(x_np),
            "max": np.max(x_np),
            "min": np.min(x_np),
        }


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Universal RL Training with CrazyLogger"
    )

    # Configuration
    parser.add_argument("--config", type=str, help="Path to YAML config file")

    # Experiment settings
    parser.add_argument("--experiment.name", type=str, help="Experiment name")
    parser.add_argument("--experiment.seed", type=int, help="Random seed")

    # Environment settings
    parser.add_argument("--environment.name", type=str, help="Environment name")

    # Algorithm settings
    parser.add_argument("--algorithm.learning_rate", type=float, help="Learning rate")

    # Training settings
    parser.add_argument(
        "--training.total_episodes", type=int, help="Total training episodes"
    )

    # Logging settings
    parser.add_argument(
        "--logging.save_videos", action="store_true", help="Save episode videos"
    )
    parser.add_argument(
        "--logging.log_activations",
        action="store_true",
        help="Log neural network activations",
    )

    args = parser.parse_args()

    # Convert args to kwargs, filtering None values
    kwargs = {k: v for k, v in vars(args).items() if v is not None and k != "config"}
    # Create and run trainer
    trainer = UniversalRLTrainer(config_path=args.config, **kwargs)
    summary = trainer.train()

    print(f"\n🎉 Experiment completed!")
    print(f"📊 Full results: {summary}")


if __name__ == "__main__":
    main()
