#!/usr/bin/env python3
"""
Adaptive Worker Class for Intellinaut RL Training
Extends DistributedWorker with dynamic network architecture morphing
"""

import torch
import torch.nn as nn
import json
import numpy as np
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import copy
from collections import deque

# Intellinaut imports
from .distributed import DistributedWorker


class AdaptiveWorker(DistributedWorker):
    """
    Adaptive Worker with Dynamic Network Architecture Morphing

    Features:
    - Inherits all DistributedWorker functionality
    - Dynamic layer addition/removal based on performance
    - Discrete architecture jumps (standard in RL literature)
    - Weight transfer for compatible layers
    - Architecture evolution and sharing
    - Performance tracking per architecture
    - Configurable morphing constraints and triggers
    """

    def __init__(
        self,
        worker_id: int,
        shared_dir: str,
        morphing_frequency: int = 100,
        morphing_enabled: bool = True,
        **distributed_kwargs,
    ):
        """
        Initialize Adaptive Worker

        Args:
            worker_id: Unique worker identifier
            shared_dir: Shared directory for coordination
            morphing_frequency: Episodes between architecture evaluations
            morphing_enabled: Whether to enable architecture morphing
            **distributed_kwargs: Arguments passed to DistributedWorker
        """
        super().__init__(
            worker_id=worker_id, shared_dir=shared_dir, **distributed_kwargs
        )

        # Morphing settings (will be overridden by config)
        self.morphing_enabled = morphing_enabled
        self.morphing_frequency = morphing_frequency
        self.last_morphing_episode = 0
        self.morphing_count = 0

        # Architecture tracking
        self.current_architecture = None
        self.architecture_history = []
        self.architecture_performance = {}  # arch_string -> performance_metrics
        self.successful_architectures = deque(maxlen=20)  # Track top performers

        # Performance metrics for morphing decisions
        self.performance_window = 50  # Episodes to evaluate performance
        self.performance_metrics = {
            "reward_trend": deque(maxlen=self.performance_window),
            "loss_trend": deque(maxlen=self.performance_window),
            "convergence_speed": deque(maxlen=self.performance_window),
        }

        # Morphing configuration (loaded from coordinator config)
        self.morphing_config = {
            "enabled": True,
            "frequency_episodes": 100,
            "metrics": ["reward_trend", "loss_trend"],
            "thresholds": {
                "reward_improvement": 0.1,
                "reward_decline": -0.2,
                "loss_stagnation": 0.05,
            },
            "constraints": {
                "max_layers": 5,
                "max_neurons_per_layer": 512,
                "min_layers": 2,
                "min_neurons_per_layer": 16,
            },
            "architecture_options": [
                [32, 32],
                [64, 64],
                [128, 64],
                [256, 128],
                [128, 128],
                [64, 32],
                [256, 256],
                [512, 256],
                [256, 128, 64],
            ],
        }

        # Shared directory for architecture data
        self.arch_dir = self.shared_dir / "architectures"
        self.arch_dir.mkdir(exist_ok=True)
        self.arch_performance_file = (
            self.arch_dir / f"worker_{worker_id}_arch_performance.json"
        )
        self.successful_arch_file = self.arch_dir / "successful_architectures.json"

        self._update_status("ADAPTIVE_INIT", morphing_enabled=morphing_enabled)

        print(f"🧬 AdaptiveWorker {worker_id} initialized")
        print(f"🔄 Morphing frequency: {morphing_frequency} episodes")
        print(
            f"📐 Architecture morphing: {'ENABLED' if morphing_enabled else 'DISABLED'}"
        )

    def _apply_config_updates(self):
        """Apply configuration updates including morphing settings"""
        super()._apply_config_updates()

        # Apply morphing-specific config
        if "morphing" in self.current_config:
            morphing_config = self.current_config["morphing"]

            # Update morphing settings
            for key, value in morphing_config.items():
                if key in self.morphing_config:
                    old_value = self.morphing_config[key]
                    self.morphing_config[key] = value
                    if old_value != value:
                        print(
                            f"🧬 Worker {self.worker_id}: Morphing {key} updated: {old_value} → {value}"
                        )

            # Update instance variables
            self.morphing_enabled = self.morphing_config.get("enabled", True)
            self.morphing_frequency = self.morphing_config.get(
                "frequency_episodes", 100
            )

    def get_current_architecture(self) -> List[int]:
        """Get current network architecture as list of layer sizes"""
        if self.algorithm is None or self.algorithm.policy_net is None:
            return []

        architecture = []
        for module in self.algorithm.policy_net.modules():
            if isinstance(module, nn.Linear):
                architecture.append(module.in_features)

        # Add final layer output size
        if architecture:
            for module in reversed(list(self.algorithm.policy_net.modules())):
                if isinstance(module, nn.Linear):
                    architecture.append(module.out_features)
                    break

        return architecture

    def architecture_to_string(self, arch: List[int]) -> str:
        """Convert architecture list to string identifier"""
        return "_".join(map(str, arch))

    def calculate_performance_metrics(self) -> Dict[str, float]:
        """Calculate performance metrics for morphing decisions"""
        metrics = {}

        # Reward trend (recent vs older episodes)
        if len(self.episode_rewards) >= 20:
            recent_rewards = list(self.episode_rewards)[-10:]
            older_rewards = list(self.episode_rewards)[-20:-10]
            metrics["reward_trend"] = np.mean(recent_rewards) - np.mean(older_rewards)
        else:
            metrics["reward_trend"] = 0.0

        # Average reward over performance window
        if len(self.episode_rewards) >= 10:
            metrics["avg_reward"] = np.mean(
                list(self.episode_rewards)[-min(10, len(self.episode_rewards)) :]
            )
        else:
            metrics["avg_reward"] = 0.0

        # Reward variance (stability)
        if len(self.episode_rewards) >= 10:
            metrics["reward_variance"] = np.var(list(self.episode_rewards)[-10:])
        else:
            metrics["reward_variance"] = 0.0

        # Episodes since last improvement
        if self.episode_rewards:
            current_best = max(self.episode_rewards)
            episodes_since_best = 0
            for reward in reversed(list(self.episode_rewards)):
                if reward == current_best:
                    break
                episodes_since_best += 1
            metrics["episodes_since_best"] = episodes_since_best
        else:
            metrics["episodes_since_best"] = 0

        return metrics

    def should_morph_architecture(self) -> Tuple[bool, str]:
        """
        Determine if architecture should be morphed based on performance

        Returns:
            Tuple[bool, str]: (should_morph, reason)
        """
        if not self.morphing_enabled:
            return False, "morphing_disabled"

        # Check episode frequency
        episodes_since_morph = self.current_episode - self.last_morphing_episode
        if episodes_since_morph < self.morphing_frequency:
            return (
                False,
                f"frequency_not_met_{episodes_since_morph}/{self.morphing_frequency}",
            )

        # Need sufficient data for decision
        if len(self.episode_rewards) < 20:
            return False, "insufficient_data"

        # Calculate performance metrics
        metrics = self.calculate_performance_metrics()
        thresholds = self.morphing_config["thresholds"]

        # Decision logic based on performance
        reward_trend = metrics["reward_trend"]

        # Strong positive trend - try expanding network
        if reward_trend > thresholds["reward_improvement"]:
            return True, f"reward_improving_{reward_trend:.3f}"

        # Strong negative trend - try different architecture
        if reward_trend < thresholds["reward_decline"]:
            return True, f"reward_declining_{reward_trend:.3f}"

        # Stagnation - try architectural change
        episodes_since_best = metrics["episodes_since_best"]
        if episodes_since_best > self.morphing_frequency * 0.8:
            return True, f"stagnation_{episodes_since_best}_episodes"

        # High variance - try more stable architecture
        if metrics["reward_variance"] > np.mean(list(self.episode_rewards)[-20:]) * 0.5:
            return True, f"high_variance_{metrics['reward_variance']:.3f}"

        return False, "performance_acceptable"

    def select_new_architecture(
        self, current_arch: List[int], reason: str
    ) -> List[int]:
        """
        Select new architecture based on current performance and reason

        Args:
            current_arch: Current architecture
            reason: Reason for morphing

        Returns:
            List[int]: New architecture
        """
        options = self.morphing_config["architecture_options"]
        constraints = self.morphing_config["constraints"]

        # Filter valid options based on constraints
        valid_options = []
        for arch in options:
            if (
                len(arch) >= constraints["min_layers"]
                and len(arch) <= constraints["max_layers"]
                and all(
                    constraints["min_neurons_per_layer"]
                    <= n
                    <= constraints["max_neurons_per_layer"]
                    for n in arch
                )
            ):
                valid_options.append(arch)

        if not valid_options:
            print(
                f"⚠️ Worker {self.worker_id}: No valid architecture options, keeping current"
            )
            return current_arch

        # Remove current architecture from options
        current_arch_clean = (
            current_arch[1:-1] if len(current_arch) > 2 else current_arch
        )  # Remove input/output layers
        valid_options = [arch for arch in valid_options if arch != current_arch_clean]

        if not valid_options:
            print(f"⚠️ Worker {self.worker_id}: All options exhausted, keeping current")
            return current_arch

        # Selection strategy based on reason
        if "improving" in reason:
            # Reward improving - try larger architectures
            larger_options = [
                arch for arch in valid_options if sum(arch) > sum(current_arch_clean)
            ]
            if larger_options:
                valid_options = larger_options

        elif "declining" in reason or "stagnation" in reason:
            # Performance issues - try successful architectures from other workers or different sizes
            if self.successful_architectures:
                successful_options = [
                    arch
                    for arch in valid_options
                    if arch in list(self.successful_architectures)
                ]
                if successful_options:
                    valid_options = successful_options

        elif "variance" in reason:
            # High variance - try simpler, more stable architectures
            simpler_options = [
                arch for arch in valid_options if sum(arch) < sum(current_arch_clean)
            ]
            if simpler_options:
                valid_options = simpler_options

        # Select based on architecture performance history
        if hasattr(self, "architecture_performance") and self.architecture_performance:
            # Prefer architectures with good historical performance
            scored_options = []
            for arch in valid_options:
                arch_str = self.architecture_to_string(arch)
                if arch_str in self.architecture_performance:
                    perf = self.architecture_performance[arch_str]
                    score = (
                        perf.get("avg_reward", 0) + perf.get("reward_trend", 0) * 0.5
                    )
                    scored_options.append((score, arch))
                else:
                    # Unexplored architectures get neutral score
                    scored_options.append((0, arch))

            # Sort by score and add some randomness
            scored_options.sort(reverse=True)
            # Select from top 3 options with some randomness
            top_options = scored_options[: min(3, len(scored_options))]
            return np.random.choice([arch for _, arch in top_options])

        # Fallback: random selection
        return np.random.choice(valid_options)

    def transfer_weights(
        self, old_net: nn.Module, new_net: nn.Module
    ) -> Dict[str, Any]:
        """
        Transfer compatible weights from old network to new network

        Returns:
            Dict with transfer statistics
        """
        transfer_stats = {
            "layers_transferred": 0,
            "layers_initialized": 0,
            "parameters_transferred": 0,
            "parameters_initialized": 0,
        }

        old_layers = [m for m in old_net.modules() if isinstance(m, nn.Linear)]
        new_layers = [m for m in new_net.modules() if isinstance(m, nn.Linear)]

        # Transfer compatible layers
        for i, (old_layer, new_layer) in enumerate(zip(old_layers, new_layers)):
            old_weight = old_layer.weight.data
            old_bias = old_layer.bias.data if old_layer.bias is not None else None

            new_weight = new_layer.weight.data
            new_bias = new_layer.bias.data if new_layer.bias is not None else None

            # Determine transfer dimensions
            min_out = min(old_weight.shape[0], new_weight.shape[0])
            min_in = min(old_weight.shape[1], new_weight.shape[1])

            if min_out > 0 and min_in > 0:
                # Transfer compatible portion
                new_weight[:min_out, :min_in] = old_weight[:min_out, :min_in]
                if old_bias is not None and new_bias is not None:
                    new_bias[:min_out] = old_bias[:min_out]

                transfer_stats["layers_transferred"] += 1
                transfer_stats["parameters_transferred"] += min_out * min_in
                if old_bias is not None:
                    transfer_stats["parameters_transferred"] += min_out

                # Initialize new parameters (if any)
                if new_weight.shape[0] > min_out:
                    nn.init.xavier_uniform_(new_weight[min_out:, :])
                    transfer_stats["parameters_initialized"] += (
                        new_weight.shape[0] - min_out
                    ) * new_weight.shape[1]

                if new_weight.shape[1] > min_in:
                    nn.init.xavier_uniform_(new_weight[:, min_in:])
                    transfer_stats["parameters_initialized"] += new_weight.shape[0] * (
                        new_weight.shape[1] - min_in
                    )

                if new_bias is not None and new_bias.shape[0] > min_out:
                    nn.init.zeros_(new_bias[min_out:])
                    transfer_stats["parameters_initialized"] += (
                        new_bias.shape[0] - min_out
                    )
            else:
                # No compatible dimensions - initialize from scratch
                nn.init.xavier_uniform_(new_weight)
                if new_bias is not None:
                    nn.init.zeros_(new_bias)
                transfer_stats["layers_initialized"] += 1
                transfer_stats["parameters_initialized"] += new_weight.numel()
                if new_bias is not None:
                    transfer_stats["parameters_initialized"] += new_bias.numel()

        return transfer_stats

    def morph_architecture(self, new_arch: List[int]) -> bool:
        """
        Morph the current architecture to new architecture

        Args:
            new_arch: New architecture specification

        Returns:
            bool: Success of morphing operation
        """
        try:
            if self.algorithm is None:
                print(
                    f"Worker {self.worker_id}: Cannot morph - algorithm not initialized"
                )
                return False

            old_arch = self.get_current_architecture()
            old_arch_str = self.architecture_to_string(old_arch)
            new_arch_str = self.architecture_to_string(new_arch)

            print(f"AdaptiveWorker {self.worker_id}: Morphing architecture")
            print(f"   Old: {old_arch_str}")
            print(f"   New: {new_arch_str}")

            # Save current architecture performance
            current_metrics = self.calculate_performance_metrics()
            if old_arch_str not in self.architecture_performance:
                self.architecture_performance[old_arch_str] = {}

            self.architecture_performance[old_arch_str].update(
                {
                    "episodes_used": len(self.episode_rewards),
                    "avg_reward": current_metrics["avg_reward"],
                    "reward_trend": current_metrics["reward_trend"],
                    "last_updated": time.time(),
                }
            )

            # Create new networks with new architecture
            obs_space = self.env.observation_space
            action_space = self.env.action_space

            # Save old networks
            old_policy_net = copy.deepcopy(self.algorithm.policy_net)
            old_value_net = copy.deepcopy(self.algorithm.value_net)

            # Create new networks (this would need to be implemented in your PPOAlgorithm)
            # For now, we'll assume a method exists to recreate networks with new architecture
            if hasattr(self.algorithm, "recreate_networks"):
                new_policy_net, new_value_net = self.algorithm.recreate_networks(
                    new_arch
                )
            else:
                print(
                    f"Worker {self.worker_id}: recreate_networks not implemented, keeping current architecture"
                )
                return False

            # Transfer weights
            policy_transfer_stats = self.transfer_weights(
                old_policy_net, new_policy_net
            )
            value_transfer_stats = self.transfer_weights(old_value_net, new_value_net)

            # Update algorithm networks
            self.algorithm.policy_net = new_policy_net
            self.algorithm.value_net = new_value_net

            # Recreate optimizers for new networks
            self.algorithm._create_optimizers()

            # Update tracking
            self.current_architecture = new_arch
            self.architecture_history.append(
                {
                    "episode": self.current_episode,
                    "old_arch": old_arch,
                    "new_arch": new_arch,
                    "reason": getattr(self, "_last_morph_reason", "manual"),
                    "transfer_stats": {
                        "policy": policy_transfer_stats,
                        "value": value_transfer_stats,
                    },
                }
            )

            self.last_morphing_episode = self.current_episode
            self.morphing_count += 1

            # Log morphing event
            morph_log = {
                "timestamp": datetime.now().isoformat(),
                "event_type": "architecture_morph",
                "worker_id": self.worker_id,
                "episode": self.current_episode,
                "old_architecture": old_arch_str,
                "new_architecture": new_arch_str,
                "reason": getattr(self, "_last_morph_reason", "manual"),
                "morphing_count": self.morphing_count,
                "transfer_stats": {
                    "policy": policy_transfer_stats,
                    "value": value_transfer_stats,
                },
            }
            self.json_buffer.append(morph_log)

            self._update_status(
                "ARCHITECTURE_MORPHED",
                new_architecture=new_arch_str,
                morphing_count=self.morphing_count,
            )

            print(f"Worker {self.worker_id}: Architecture morphed successfully!")
            print(
                f"   Policy layers transferred: {policy_transfer_stats['layers_transferred']}"
            )
            print(
                f"   Value layers transferred: {value_transfer_stats['layers_transferred']}"
            )

            return True

        except Exception as e:
            print(f"Worker {self.worker_id}: Architecture morphing failed: {e}")
            self._update_status("MORPH_ERROR", error=str(e))
            return False

    def load_successful_architectures(self):
        """Load successful architectures shared by other workers"""
        try:
            if self.successful_arch_file.exists():
                with open(self.successful_arch_file, "r") as f:
                    data = json.load(f)
                    successful_archs = data.get("architectures", [])

                    # Update local successful architectures
                    for arch in successful_archs:
                        if arch not in self.successful_architectures:
                            self.successful_architectures.append(arch)

                    print(
                        f"Worker {self.worker_id}: Loaded {len(successful_archs)} successful architectures"
                    )

        except Exception as e:
            print(
                f"Worker {self.worker_id}: Error loading successful architectures: {e}"
            )

    def save_architecture_performance(self):
        """Save architecture performance data to shared directory"""
        try:
            # Save individual worker's architecture performance
            data = {
                "worker_id": self.worker_id,
                "current_architecture": self.architecture_to_string(
                    self.get_current_architecture()
                ),
                "morphing_count": self.morphing_count,
                "architecture_performance": self.architecture_performance,
                "architecture_history": self.architecture_history[
                    -10:
                ],  # Last 10 morphs
                "timestamp": time.time(),
            }

            with open(self.arch_performance_file, "w") as f:
                json.dump(data, f, indent=2)

            # Update successful architectures if current one is performing well
            current_metrics = self.calculate_performance_metrics()
            if (
                current_metrics["avg_reward"] > 0
                and current_metrics["reward_trend"] > 0
                and len(self.episode_rewards) > 50
            ):

                current_arch = self.get_current_architecture()[
                    1:-1
                ]  # Remove input/output layers
                if current_arch not in self.successful_architectures:
                    self.successful_architectures.append(current_arch)

                    # Save to shared successful architectures
                    shared_data = {"architectures": list(self.successful_architectures)}
                    with open(self.successful_arch_file, "w") as f:
                        json.dump(shared_data, f, indent=2)

        except Exception as e:
            print(
                f"Worker {self.worker_id}: Error saving architecture performance: {e}"
            )

    def adaptive_episode_hook(self):
        """Hook called after each episode for adaptive operations"""
        # Check for architecture morphing
        if self.morphing_enabled and self.algorithm is not None:
            should_morph, reason = self.should_morph_architecture()

            if should_morph:
                self._last_morph_reason = reason
                current_arch = self.get_current_architecture()
                new_arch = self.select_new_architecture(current_arch, reason)

                print(f"🧬 Worker {self.worker_id}: Morphing triggered - {reason}")
                self.morph_architecture(new_arch)

        # Load successful architectures from other workers
        if self.current_episode % 200 == 0:  # Every 200 episodes
            self.load_successful_architectures()

        # Save architecture performance data
        if self.current_episode % 100 == 0:  # Every 100 episodes
            self.save_architecture_performance()

    def distributed_episode_hook(self):
        """Override to include adaptive operations"""
        # Call parent distributed operations
        super().distributed_episode_hook()

        # Add adaptive operations
        self.adaptive_episode_hook()

    def setup_rl_components(
        self, env_name: str, device: str = "cuda:0", lr: float = 3e-4
    ):
        """Setup RL components with initial architecture tracking"""
        super().setup_rl_components(env_name, device, lr)

        # Initialize architecture tracking
        self.current_architecture = self.get_current_architecture()
        arch_str = self.architecture_to_string(self.current_architecture)

        print(f"📐 Worker {self.worker_id}: Initial architecture: {arch_str}")

        # Log initial architecture
        arch_log = {
            "timestamp": datetime.now().isoformat(),
            "event_type": "initial_architecture",
            "worker_id": self.worker_id,
            "architecture": arch_str,
            "total_parameters": sum(
                p.numel() for p in self.algorithm.policy_net.parameters()
            ),
        }
        self.json_buffer.append(arch_log)

    def run(self, env_name: str, device: str = "cuda:0", lr: float = 3e-4):
        """
        Main training loop with adaptive architecture morphing
        """
        print(
            f"🧬 AdaptiveWorker {self.worker_id}: Starting adaptive distributed training"
        )

        # Load any existing successful architectures
        self.load_successful_architectures()

        # Call parent run method (which includes all distributed functionality)
        try:
            super().run(env_name=env_name, device=device, lr=lr)

            # Additional completion logging for adaptive features
            print(f"AdaptiveWorker {self.worker_id}: Adaptive training completed!")
            print(f"Total architecture morphs: {self.morphing_count}")
            print(
                f"Final architecture: {self.architecture_to_string(self.get_current_architecture())}"
            )
            print(f"Architectures tested: {len(self.architecture_performance)}")

        finally:
            # Save final architecture performance data
            self.save_architecture_performance()


# Example usage and CLI interface
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="AdaptiveWorker RL Training with Architecture Morphing"
    )

    # Adaptive-specific arguments
    parser.add_argument("--worker_id", type=int, required=True, help="Worker ID")
    parser.add_argument(
        "--shared_dir", type=str, required=True, help="Shared directory"
    )
    parser.add_argument(
        "--morphing_frequency",
        type=int,
        default=100,
        help="Episodes between morphing evaluations",
    )
    parser.add_argument(
        "--morphing_enabled",
        action="store_true",
        default=True,
        help="Enable architecture morphing",
    )

    # Distributed arguments
    parser.add_argument(
        "--sync_interval", type=int, default=10, help="Episodes between syncs"
    )
    parser.add_argument(
        "--timeout_minutes", type=int, default=30, help="Sync timeout in minutes"
    )

    # Training arguments
    parser.add_argument("--max_episodes", type=int, default=1000, help="Max episodes")
    parser.add_argument("--env", type=str, default="CartPole-v1", help="Environment")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")

    # Optional overrides
    parser.add_argument(
        "--log_dir", type=str, default=None, help="Log directory override"
    )

    args = parser.parse_args()

    # Create and run adaptive worker
    worker = AdaptiveWorker(
        worker_id=args.worker_id,
        shared_dir=args.shared_dir,
        log_dir=args.log_dir,
        max_episodes=args.max_episodes,
        sync_interval=args.sync_interval,
        timeout_minutes=args.timeout_minutes,
        morphing_frequency=args.morphing_frequency,
        morphing_enabled=args.morphing_enabled,
    )

    worker.run(env_name=args.env, device=args.device, lr=args.lr)
