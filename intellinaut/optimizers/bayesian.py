#!/usr/bin/env python3
"""
Bayesian Optimization Manager for Hyperparameter and Architecture Optimization
Uses PyTorch with proper typing and data dimensions
"""

import torch
import torch.nn as nn
from torch import Tensor
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
from enum import Enum
import math
import os

# Add import for debugging
from intellinaut.logging.debugging import (
    create_hyperparameter_debugger,
    DistributedDebugger,
)


class ParameterType(Enum):
    """Types of parameters for optimization"""

    CONTINUOUS = "continuous"
    DISCRETE = "discrete"
    CATEGORICAL = "categorical"


@dataclass
class ParameterSpace:
    """Definition of a parameter space"""

    name: str
    param_type: ParameterType
    bounds: Union[
        Tuple[float, float], List[Any]
    ]  # (min, max) for continuous, list for discrete/categorical
    log_scale: bool = False  # For learning rates, etc.

    def sample(self, n_samples: int) -> Tensor:
        """Sample n values from this parameter space"""
        if self.param_type == ParameterType.CONTINUOUS:
            if self.log_scale:
                log_min, log_max = np.log(self.bounds[0]), np.log(self.bounds[1])
                samples = torch.exp(
                    (log_max - log_min) * torch.rand(size=(n_samples,)) + log_min
                )
            else:
                samples = (self.bounds[1] - self.bounds[0]) * torch.rand(
                    size=(n_samples,)
                ) + self.bounds[0]
        elif self.param_type == ParameterType.DISCRETE:
            # FIXED: Sample indices (0, 1, 2, ...) not values
            indices = torch.randint(0, len(self.bounds), (n_samples,))
            samples = indices.float()  # Store as float indices
        else:  # CATEGORICAL
            # FIXED: Sample indices (0, 1, 2, ...) not values
            indices = torch.randint(0, len(self.bounds), (n_samples,))
            samples = indices.float()  # Store as float indices

        return samples


@dataclass
class OptimizationConfig:
    """Configuration for the Bayesian optimizer"""

    acquisition_function: str = "EI"  # Expected Improvement
    n_candidates: int = 1000  # Candidates to evaluate acquisition on
    n_initial_random: int = 5  # Random evaluations before GP
    kernel_lengthscale: float = 1.0
    kernel_variance: float = 1.0
    noise_variance: float = 0.01


class SimpleGaussianProcess(nn.Module):
    """Simple Gaussian Process for Bayesian Optimization"""

    def __init__(
        self,
        input_dim: int,
        kernel_lengthscale: float = 1.0,
        kernel_variance: float = 1.0,
        noise_variance: float = 0.01,
    ):
        super().__init__()
        self.input_dim = input_dim

        # GP hyperparameters (learnable)
        self.log_lengthscale = nn.Parameter(torch.log(torch.tensor(kernel_lengthscale)))
        self.log_variance = nn.Parameter(torch.log(torch.tensor(kernel_variance)))
        self.log_noise = nn.Parameter(torch.log(torch.tensor(noise_variance)))

        # Training data
        self.X_train: Optional[Tensor] = None
        self.y_train: Optional[Tensor] = None
        self.K_inv: Optional[Tensor] = None

    def rbf_kernel(self, X1: Tensor, X2: Tensor) -> Tensor:
        """RBF (Gaussian) kernel"""
        lengthscale = torch.exp(self.log_lengthscale)
        variance = torch.exp(self.log_variance)

        # Squared distance matrix
        X1_norm = (X1**2).sum(dim=1, keepdim=True)
        X2_norm = (X2**2).sum(dim=1, keepdim=True)
        dist_sq = X1_norm + X2_norm.T - 2 * X1 @ X2.T

        # RBF kernel
        K = variance * torch.exp(-0.5 * dist_sq / (lengthscale**2))
        return K

    def fit(self, X: Tensor, y: Tensor) -> None:
        """Fit GP to training data"""
        self.X_train = X.clone()
        self.y_train = y.clone()

        # Compute kernel matrix and inverse
        K = self.rbf_kernel(X, X)
        K += torch.exp(self.log_noise) * torch.eye(K.shape[0])  # Add noise
        K += 1e-6 * torch.eye(K.shape[0])  # Add small regularization term

        # Stable matrix inversion
        try:
            self.K_inv = torch.linalg.inv(K)
        except:
            # Fallback to pseudo-inverse if singular
            self.K_inv = torch.linalg.pinv(K)

    def predict(self, X_new: Tensor) -> Tuple[Tensor, Tensor]:
        """Predict mean and variance at new points"""
        if self.X_train is None:
            raise ValueError("GP not fitted yet")

        # Kernel between new points and training points
        K_star = self.rbf_kernel(X_new, self.X_train)

        # Predictive mean
        mean = K_star @ self.K_inv @ self.y_train

        # Predictive variance
        K_star_star = self.rbf_kernel(X_new, X_new)
        variance = torch.diag(K_star_star - K_star @ self.K_inv @ K_star.T)

        return mean, variance


class BayesianOptimizationManager:
    """
    Bayesian Optimization Manager for hyperparameter and architecture optimization
    """

    def __init__(
        self,
        shared_dir: str,
        config: Optional[OptimizationConfig] = None,
        debugger: Optional[DistributedDebugger] = None,
        obs_dim: Optional[int] = None,
        action_dim: Optional[int] = None,
    ):
        """
        Initialize Bayesian Optimizer

        Args:
            shared_dir: Directory for saving optimization data
            obs_dim: Environment observation dimension (optional, for future use)
            action_dim: Environment action dimension (optional, for future use)
            config: Optimization configuration
            debugger: Optional DistributedDebugger instance
        """
        self.shared_dir = Path(shared_dir)
        self.config = config or OptimizationConfig()
        self.debugger = debugger or create_hyperparameter_debugger(str(self.shared_dir))

        # Defensive: print where logs will go and what entity this is
        if self.debugger:
            print(f"\033[1;36m[DEBUGGER] BayesianOptimizationManager entity '{self.debugger.entity_name}' logs to {self.debugger.logs_dir}\033[0m")
            self.debugger.log_text(
                "INFO", f"Initializing BayesianOptimizationManager"
            )
        else:
            print("WARNING: No debugger instance for BayesianOptimizationManager!")

        # Create optimization directory
        self.opt_dir = self.shared_dir / "optimization"
        self.opt_dir.mkdir(parents=True, exist_ok=True)

        # Define parameter space based on data dimensions
        self.parameter_spaces = self._create_parameter_spaces()
        self.param_dim = len(self.parameter_spaces)  # <-- Ensure this is always set

        # Initialize Gaussian Process
        self.gp = SimpleGaussianProcess(
            input_dim=self.param_dim,
            kernel_lengthscale=self.config.kernel_lengthscale,
            kernel_variance=self.config.kernel_variance,
            noise_variance=self.config.noise_variance,
        )

        # Optimization data
        self.evaluated_params: List[Tensor] = []
        self.evaluated_performance: List[float] = []
        self.best_performance = float("-inf")
        self.best_params: Optional[Tensor] = None

        # Track Docker container ID if running in Docker
        self.docker_container_id = self._get_docker_container_id()
        if self.debugger:
            self.debugger.log_text("INFO", f"Docker container id: {self.docker_container_id}")

        print(f"\033[92mParameter space dimension: {self.param_dim}\033[0m")
        if self.debugger:
            self.debugger.log_text(
                "INFO", f"Parameter space dimension: {self.param_dim}"
            )
        else:
            print("WARNING: No debugger instance for BayesianOptimizationManager!")

    def suggest_next_configuration(self, worker_id: int) -> Dict[str, Any]:
        """
        Suggest the next configuration to evaluate.
        Combines discrete and continuous parameter optimization.
        """
        # Step 1: Optimize discrete parameters
        best_discrete_config = self._optimize_discrete_parameters()

        # Step 2: Optimize continuous parameters for the best discrete configuration
        next_config = self.optimize_continuous_parameters(best_discrete_config)

        msg1 = f"Suggested configuration for worker {worker_id}: {next_config}"
        msg2 = f"[DEBUG] Suggested learning rate: {next_config.get('learning_rate')}"
        print(f"\033[94m{msg1}\033[0m")
        print(f"\033[91m{msg2}\033[0m")
        if self.debugger:
            # Log the current evaluated_params and their learning rates for debugging
            learning_rates = []
            for p in self.evaluated_params:
                idx = 0  # learning_rate is first param
                if len(p) > idx:
                    lr_val = p[idx].item()
                    # If log scale, convert back to real value
                    lr_val = math.exp(lr_val) if self.parameter_spaces[idx].log_scale else lr_val
                    learning_rates.append(lr_val)
            self.debugger.log_text(
                "DEBUG",
                f"Evaluated learning rates so far: {learning_rates}"
            )
            self.debugger.log_text("INFO", msg1)
            self.debugger.log_text("DEBUG", msg2)
        return next_config

    def _create_parameter_spaces(self) -> List[ParameterSpace]:
        spaces = []

        spaces.append(
            ParameterSpace(
                name="learning_rate",
                param_type=ParameterType.CONTINUOUS,
                bounds=(1e-5, 1),
                log_scale=True,
            )
        )
        spaces.append(
            ParameterSpace(
                name="batch_size",
                param_type=ParameterType.DISCRETE,
                bounds=[32, 64, 128, 256, 512],
            )
        )
        spaces.append(
            ParameterSpace(
                name="gamma",
                param_type=ParameterType.CONTINUOUS,
                bounds=(0.9, 0.999),
            )
        )
        spaces.append(
            ParameterSpace(
                name="entropy_coef",
                param_type=ParameterType.CONTINUOUS,
                bounds=(0.0, 0.1),
            )
        )
        spaces.append(
            ParameterSpace(
                name="clip_range",  # This will map to clip_epsilon in PPO
                param_type=ParameterType.CONTINUOUS,
                bounds=(0.1, 0.3),
            )
        )
        spaces.append(
            ParameterSpace(
                name="value_function_coef",  # This will map to critic_coef in PPO
                param_type=ParameterType.CONTINUOUS,
                bounds=(0.3, 1.0),
            )
        )
        # Add more as needed...

        return spaces

    def sample_parameter_space(self, n_samples: int) -> Tensor:
        """Sample n parameter configurations"""
        samples = []
        for space in self.parameter_spaces:
            param_samples = space.sample(n_samples)
            samples.append(param_samples.unsqueeze(1))

        return torch.cat(samples, dim=1)  # Shape: (n_samples, param_dim)

    def decode_parameters(self, params: Tensor) -> Dict[str, Any]:
        """Convert parameter tensor to readable dictionary"""
        config = {}
        for i, space in enumerate(self.parameter_spaces):
            value = params[i].item()
            if space.param_type == ParameterType.CATEGORICAL:
                idx = max(0, min(len(space.bounds) - 1, int(value)))
                config[space.name] = space.bounds[idx]
            elif space.param_type == ParameterType.DISCRETE:
                idx = max(0, min(len(space.bounds) - 1, int(value)))
                config[space.name] = space.bounds[idx]
            else:  # CONTINUOUS
                min_val, max_val = space.bounds
                config[space.name] = max(min_val, min(value, max_val))

        return config

    def thompson_sampling(self, X: Tensor) -> Tensor:
        try:
            mean, variance = self.gp.predict(X)
        except ValueError:  # or the specific error your GP raises
            warn_msg = (
                "[WARNING] GP not fitted yet: Thompson sampling is using the prior."
            )
            print(f"\033[93m{warn_msg}\033[0m")
            if self.debugger:
                self.debugger.log_text("WARNING", warn_msg)
            mean = torch.zeros(X.shape[0])
            variance = torch.ones(X.shape[0])  # or use kernel(X, X).diagonal()
        std = torch.sqrt(variance + 1e-8)
        samples = torch.normal(mean, std)
        # Log max, argmax, and parameter info for debugging
        if self.debugger:
            max_val = samples.max().item()
            argmax_idx = samples.argmax().item()
            param_names = [space.name for space in self.parameter_spaces]
            if X.shape[0] > argmax_idx:
                argmax_params = {param_names[i]: X[argmax_idx, i].item() for i in range(X.shape[1])}
            else:
                argmax_params = {}
            self.debugger.log_text(
                "INFO",
                f"Thompson sampling: max={max_val:.4f}, argmax={argmax_idx}, params_at_argmax={argmax_params}"
            )
        return samples

    def optimize_continuous_parameters(
        self, discrete_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Optimize continuous parameters for a given discrete configuration
        Uses only global sampling (no local refinement)
        """
        # Get continuous parameter indices
        continuous_indices = []
        for i, space in enumerate(self.parameter_spaces):
            if space.param_type == ParameterType.CONTINUOUS:
                continuous_indices.append(i)

        if not continuous_indices:
            return discrete_config

        # Sample densely around promising regions
        n_dense_samples = 5000  # Much denser for continuous params

        # Global sampling
        global_samples = self.sample_parameter_space(n_dense_samples)

        # Log the range of learning_rate values in the sampled candidates for debugging
        if self.debugger:
            lr_idx = 0  # learning_rate is first param
            lrs = global_samples[:, lr_idx].detach().cpu().numpy()
            if self.parameter_spaces[lr_idx].log_scale:
                lrs = np.exp(lrs)
            self.debugger.log_text(
                "DEBUG",
                f"Sampled learning_rate min={lrs.min():.6g}, max={lrs.max():.6g}, mean={lrs.mean():.6g}, std={lrs.std():.6g}"
            )

        global_scores = self.thompson_sampling(global_samples)
        best_idx = torch.argmax(global_scores)
        return self.decode_parameters(global_samples[best_idx])

    def _optimize_discrete_parameters(self) -> Dict[str, Any]:
        """Optimize discrete parameters by sampling full parameter space and picking best discrete config"""
        n_samples = 5000
        samples = self.sample_parameter_space(n_samples)
        scores = self.thompson_sampling(samples)
        best_idx = torch.argmax(scores)
        best_sample = samples[best_idx]
        decoded = self.decode_parameters(best_sample)
        # Extract only the discrete/categorical part for the next step
        discrete_config = {
            space.name: decoded[space.name]
            for space in self.parameter_spaces
            if space.param_type in [ParameterType.DISCRETE, ParameterType.CATEGORICAL]
        }
        return discrete_config

    def _get_discrete_combinations(self) -> List[Dict[str, Any]]:
        """Generate all combinations of discrete parameters"""
        import itertools

        discrete_spaces = []
        discrete_names = []

        for space in self.parameter_spaces:
            if space.param_type in [ParameterType.DISCRETE, ParameterType.CATEGORICAL]:
                discrete_spaces.append(space.bounds)
                discrete_names.append(space.name)

        # Generate all combinations
        combinations = []
        for combo in itertools.product(*discrete_spaces):
            config = dict(zip(discrete_names, combo))
            combinations.append(config)

        return combinations

    def _config_to_tensor(self, config: Dict[str, Any]) -> Tensor:
        param_tensor = torch.zeros(self.param_dim)
        for i, space in enumerate(self.parameter_spaces):
            value = config[space.name]
            if space.param_type == ParameterType.CATEGORICAL:
                param_tensor[i] = space.bounds.index(value)
            elif space.param_type == ParameterType.DISCRETE:
                param_tensor[i] = space.bounds.index(value)
            else:  # CONTINUOUS
                if space.log_scale:
                    param_tensor[i] = math.log(value)
                else:
                    param_tensor[i] = value
        return param_tensor

    def update_performance(
        self, worker_id: int, params: Dict[str, Any], performance: float
    ) -> None:
        """Update optimizer with new performance observation"""
        # Encode parameters back to tensor
        param_tensor = torch.zeros(self.param_dim)
        for i, space in enumerate(self.parameter_spaces):
            value = params[space.name]
            if space.param_type == ParameterType.CATEGORICAL:
                param_tensor[i] = space.bounds.index(value)
            elif space.param_type == ParameterType.DISCRETE:
                param_tensor[i] = space.bounds.index(value)
            else:
                if space.log_scale:
                    param_tensor[i] = math.log(value)
                else:
                    param_tensor[i] = value

        # Store observation
        self.evaluated_params.append(param_tensor)
        self.evaluated_performance.append(performance)

        # Update best performance
        if performance > self.best_performance:
            self.best_performance = performance
            self.best_params = param_tensor.clone()
            msg = f"New best performance: {performance:.4f} from worker {worker_id}"
            print(f"\033[92m{msg}\033[0m")
            if self.debugger:
                self.debugger.log_text("INFO", msg)

        msg2 = f"Updated performance for worker {worker_id}: {performance:.4f}"
        print(f"\033[93m{msg2}\033[0m")
        if self.debugger:
            self.debugger.log_text("INFO", msg2)

        msg3 = f"[DEBUG] Updated with learning rate: {params.get('learning_rate')}"
        print(f"\033[91m{msg3}\033[0m")
        if self.debugger:
            self.debugger.log_text("DEBUG", msg3)

        # Refit GP with new data
        if len(self.evaluated_params) > 1:
            X = torch.stack(self.evaluated_params)
            y = torch.tensor(self.evaluated_performance, dtype=torch.float32)
            self.gp.fit(X, y)

    def save_optimization_state(self) -> None:
        """Save optimization state to disk"""
        state = {
            "evaluated_params": [p.tolist() for p in self.evaluated_params],
            "evaluated_performance": self.evaluated_performance,
            "best_performance": self.best_performance,
            "best_params": (
                self.best_params.tolist() if self.best_params is not None else None
            ),
            "parameter_spaces": [
                {
                    "name": space.name,
                    "param_type": space.param_type.value,
                    "bounds": space.bounds,
                    "log_scale": space.log_scale,
                }
                for space in self.parameter_spaces
            ],
        }

        state_file = self.opt_dir / "optimization_state.json"
        with open(state_file, "w") as f:
            json.dump(state, f, indent=2)

    def load_optimization_state(self) -> bool:
        """Load optimization state from disk"""
        state_file = self.opt_dir / "optimization_state.json"

        if not state_file.exists():
            return False

        try:
            with open(state_file, "r") as f:
                state = json.load(f)

            self.evaluated_params = [torch.tensor(p) for p in state["evaluated_params"]]
            self.evaluated_performance = state["evaluated_performance"]
            self.best_performance = state["best_performance"]
            self.best_params = (
                torch.tensor(state["best_params"]) if state["best_params"] else None
            )

            # Refit GP if we have data
            if len(self.evaluated_params) > 1:
                X = torch.stack(self.evaluated_params)
                y = torch.tensor(self.evaluated_performance, dtype=torch.float32)
                self.gp.fit(X, y)

            msg = f"Loaded optimization state: {len(self.evaluated_params)} evaluations"
            print(f"\033[94m{msg}\033[0m")
            if self.debugger:
                self.debugger.log_text("INFO", msg)
            return True

        except Exception as e:
            err_msg = f"Error loading optimization state: {e}"
            print(f"\033[91m{err_msg}\033[0m")
            if self.debugger:
                self.debugger.log_text("ERROR", err_msg)
            return False

    def get_optimization_summary(self) -> Dict[str, Any]:
        """Get summary of optimization progress"""
        return {
            "total_evaluations": len(self.evaluated_params),
            "best_performance": self.best_performance,
            "best_config": (
                self.decode_parameters(self.best_params)
                if self.best_params is not None
                else None
            ),
            "recent_performance": (
                self.evaluated_performance[-10:] if self.evaluated_performance else []
            ),
        }

    def _get_docker_container_id(self) -> Optional[str]:
        """
        Try to get the Docker container ID if running inside a container.
        Returns the first 12 chars of the container ID, or None if not found.
        """
        cgroup_path = "/proc/self/cgroup"
        try:
            if os.path.exists(cgroup_path):
                with open(cgroup_path, "r") as f:
                    for line in f:
                        if "docker" in line:
                            parts = line.strip().split("/")
                            for part in parts:
                                if len(part) == 64:  # Docker container ID length
                                    return part[:12]
            # Fallback: check for hostname as container id
            hostname = os.environ.get("HOSTNAME")
            if hostname and len(hostname) >= 12:
                return hostname[:12]
        except Exception:
            pass
        return None


# Example usage
if __name__ == "__main__":
    # Initialize optimizer with environment dimensions
    optimizer = BayesianOptimizationManager(
        shared_dir="./test_optimization",
        obs_dim=17,  # HalfCheetah observation dimension
        action_dim=6,  # HalfCheetah action dimension
    )

    # Suggest configuration for worker
    config = optimizer.suggest_next_configuration(worker_id=0)
    print("Suggested config:", config)
    if optimizer.debugger:
        optimizer.debugger.log_text("INFO", f"Suggested config: {config}")

    # Simulate performance update
    optimizer.update_performance(worker_id=0, params=config, performance=1250.5)

    # Get summary
    summary = optimizer.get_optimization_summary()
    print("Optimization summary:", summary)
    if optimizer.debugger:
        optimizer.debugger.log_text("INFO", f"Optimization summary: {summary}")
