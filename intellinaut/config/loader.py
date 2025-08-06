#!/usr/bin/env python3
"""
Configuration Loader for Hyperparameters
Loads hyperparameters from JSON files
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, Optional


class ConfigLoader:
    """Load and manage hyperparameter configurations from JSON files"""

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize config loader

        Args:
            config_path: Path to JSON config file
        """
        self.config_path = config_path
        self.config = {}

        if config_path:
            self.load_config(config_path)

    def load_config(self, config_path: str) -> Dict[str, Any]:
        """
        Load configuration from JSON file

        Args:
            config_path: Path to JSON configuration file

        Returns:
            Dictionary containing configuration
        """
        config_file = Path(config_path)

        if not config_file.exists():
            raise FileNotFoundError(
                f"\033[91mConfiguration file not found: {config_path}\033[0m"
            )

        try:
            with open(config_file, "r") as f:
                self.config = json.load(f)

            print(f"\033[92mLoaded configuration from: {config_path}\033[0m")
            return self.config

        except json.JSONDecodeError as e:
            raise ValueError(
                f"\033[91mInvalid JSON in config file {config_path}: {e}\033[0m"
            )
        except Exception as e:
            raise RuntimeError(
                f"\033[91mError loading config file {config_path}: {e}\033[0m"
            )

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value with dot notation support

        Args:
            key: Configuration key (supports dot notation like 'training.learning_rate')
            default: Default value if key not found

        Returns:
            Configuration value or default
        """
        keys = key.split(".")
        value = self.config

        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default

    def get_training_config(self) -> Dict[str, Any]:
        """Get training hyperparameters"""
        return self.get("training", {})

    def get_algorithm_config(self) -> Dict[str, Any]:
        """Get algorithm hyperparameters"""
        return self.get("algorithm", {})

    def get_environment_config(self) -> Dict[str, Any]:
        """Get environment configuration"""
        return self.get("environment", {})

    def get_distributed_config(self) -> Dict[str, Any]:
        """Get distributed training configuration"""
        return self.get("distributed", {})

    def override_with_args(self, **kwargs) -> None:
        """
        Override config values with command line arguments

        Args:
            **kwargs: Key-value pairs to override
        """
        for key, value in kwargs.items():
            if value is not None:  # Only override if value is provided
                self.set(key, value)

    def set(self, key: str, value: Any) -> None:
        """
        Set configuration value with dot notation support

        Args:
            key: Configuration key (supports dot notation)
            value: Value to set
        """
        keys = key.split(".")
        config = self.config

        # Navigate to the parent of the target key
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]

        # Set the final key
        config[keys[-1]] = value

    def save_config(self, output_path: str) -> None:
        """
        Save current configuration to JSON file

        Args:
            output_path: Path to save configuration
        """
        with open(output_path, "w") as f:
            json.dump(self.config, f, indent=2)
        print(f"\033[94mSaved configuration to: {output_path}\033[0m")

    def print_config(self) -> None:
        """Print current configuration in a readable format"""
        print("\033[93mCurrent Configuration:\033[0m")
        print("=" * 40)
        print(json.dumps(self.config, indent=2))
        print("=" * 40)

    def validate_config(self) -> bool:
        """
        Validate configuration has required fields

        Returns:
            True if valid, False otherwise
        """
        required_sections = ["training", "algorithm", "environment"]

        for section in required_sections:
            if section not in self.config:
                print(f"\033[91mMissing required section: {section}\033[0m")
                return False

        # Validate training section
        training = self.config.get("training", {})
        if "max_episodes" not in training:
            print("\033[91mMissing training.max_episodes\033[0m")
            return False

        # Validate algorithm section
        algorithm = self.config.get("algorithm", {})
        if "learning_rate" not in algorithm:
            print("\033[91mMissing algorithm.learning_rate\033[0m")
            return False

        print("\033[92mConfiguration validation passed\033[0m")
        return True


def create_default_config() -> Dict[str, Any]:
    """Create default configuration dictionary"""
    return {
        "experiment": {
            "name": "distributed_rl_experiment",
            "description": "Distributed RL training with hyperparameter optimization",
        },
        "environment": {
            "name": "CartPole-v1",
            "max_episode_steps": 500,
            "normalize_observations": True,
        },
        "algorithm": {
            "name": "PPO",
            "learning_rate": 3e-4,
            "batch_size": 64,
            "n_epochs": 4,
            "clip_range": 0.2,
            "entropy_coef": 0.01,
            "value_function_coef": 0.5,
            "max_grad_norm": 0.5,
            "gamma": 0.99,
            "gae_lambda": 0.95,
        },
        "training": {
            "max_episodes": 1000,
            "checkpoint_frequency": 50,
            "status_update_frequency": 10,
            "eval_frequency": 100,
            "device": "cuda:0",
        },
        "distributed": {
            "num_workers": 2,
            "sync_frequency": 10,
            "check_interval": 30,
            "min_episodes": 50,
        },
        "logging": {
            "log_level": "INFO",
            "save_videos": False,
            "video_frequency": 100,
            "log_activations": False,
            "activation_frequency": 200,
        },
    }


def save_default_config(output_path: str) -> None:
    """Save default configuration to file"""
    config = create_default_config()
    with open(output_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"\033[94mCreated default configuration: {output_path}\033[0m")


# Convenience function
def load_config(config_path: str) -> ConfigLoader:
    """Convenience function to load configuration"""
    return ConfigLoader(config_path)


if __name__ == "__main__":
    # Example usage
    import argparse

    parser = argparse.ArgumentParser(description="Configuration Loader")
    parser.add_argument("--create_default", type=str, help="Create default config file")
    parser.add_argument("--validate", type=str, help="Validate config file")
    parser.add_argument("--print", type=str, help="Print config file contents")

    args = parser.parse_args()

    if args.create_default:
        save_default_config(args.create_default)

    if args.validate:
        loader = ConfigLoader(args.validate)
        loader.validate_config()

    if args.print:
        loader = ConfigLoader(args.print)
        loader.print_config()
