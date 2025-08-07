#!/usr/bin/env python3
"""
Enhanced Coordinator Functions for Config Management and Bayesian Optimization
Add these functions to your coordinator_entry.py
"""

import json
import sys
from pathlib import Path
import argparse
import time

# Add these imports to your coordinator_entry.py
sys.path.insert(0, "/workspace/project")
from intellinaut.config.loader import ConfigLoader
from intellinaut.optimizers.bayesian import BayesianOptimizationManager
from intellinaut.workers.coordinator import MinimalCoordinator


def load_base_config(config_path: str) -> ConfigLoader:
    """
    Load the base configuration file

    Args:
        config_path: Path to the base config JSON file

    Returns:
        ConfigLoader instance with loaded config

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If config file is invalid
    """
    try:
        config_loader = ConfigLoader(config_path)

        # Validate the config has required sections
        if not config_loader.validate_config():
            raise ValueError(f"Invalid configuration in {config_path}")

        print(f"Loaded base configuration from: {config_path}")
        return config_loader

    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in config file {config_path}: {e}")
    except Exception as e:
        raise RuntimeError(f"Error loading config file {config_path}: {e}")


def initialize_bayesian_optimizer(
    shared_dir: str, config_loader: ConfigLoader
) -> BayesianOptimizationManager:
    """
    Initialize Bayesian Optimization Manager

    Args:
        shared_dir: Shared directory path
        config_loader: Loaded configuration

    Returns:
        BayesianOptimizationManager instance

    Raises:
        ValueError: If environment config is missing or invalid
    """
    try:
        env_config = config_loader.get_environment_config()
        env_name = env_config.get("name")

        if not env_name:
            raise ValueError("Environment name not specified in config")

        # Get environment dimensions based on environment name
        # This is a simplified mapping - you might want to expand this
        env_dimensions = {
            "CartPole-v1": {"obs_dim": 4, "action_dim": 2},
            "HalfCheetah-v4": {"obs_dim": 17, "action_dim": 6},
            "Pendulum-v1": {"obs_dim": 3, "action_dim": 1},
            # Add more environments as needed
        }

        if env_name not in env_dimensions:
            # Default dimensions for unknown environments
            print(f"Unknown environment {env_name}, using default dimensions")
            obs_dim, action_dim = 8, 4
        else:
            dims = env_dimensions[env_name]
            obs_dim, action_dim = dims["obs_dim"], dims["action_dim"]

        # Initialize Bayesian Optimizer
        optimizer = BayesianOptimizationManager(
            shared_dir=shared_dir  # , obs_dim=obs_dim, action_dim=action_dim
        )

        # Try to load existing optimization state
        optimizer.load_optimization_state()

        print(f"Bayesian Optimizer initialized for {env_name}")
        print(f"Dimensions: {obs_dim} obs, {action_dim} actions")

        return optimizer

    except Exception as e:
        raise RuntimeError(f"Error initializing Bayesian Optimizer: {e}")


def generate_worker_configs(
    base_config: ConfigLoader,
    optimizer: BayesianOptimizationManager,
    num_workers: int,
    shared_dir: str,
) -> list:
    """
    Generate optimized configs for each worker using Bayesian Optimization

    Args:
        base_config: Base configuration loader
        optimizer: Bayesian optimization manager
        num_workers: Number of workers
        shared_dir: Shared directory path

    Returns:
        List of generated config file paths

    Raises:
        RuntimeError: If config generation fails
    """
    try:
        config_dir = Path(shared_dir) / "worker_configs"
        config_dir.mkdir(parents=True, exist_ok=True)

        generated_configs = []

        for worker_id in range(num_workers):
            print(f"Generating config for worker {worker_id}...")

            # Get optimized parameters from Bayesian Optimizer
            optimized_params = optimizer.suggest_next_configuration(worker_id)

            # Create worker-specific config by copying base config
            worker_config = ConfigLoader()
            worker_config.config = base_config.config.copy()

            # Update algorithm parameters with optimized values
            if "learning_rate" in optimized_params:
                worker_config.set(
                    "algorithm.learning_rate", optimized_params["learning_rate"]
                )
            if "batch_size" in optimized_params:
                worker_config.set(
                    "algorithm.batch_size", int(optimized_params["batch_size"])
                )
            if "gamma" in optimized_params:
                worker_config.set("algorithm.gamma", optimized_params["gamma"])
            if "entropy_coef" in optimized_params:
                worker_config.set(
                    "algorithm.entropy_coef", optimized_params["entropy_coef"]
                )

            # Update architecture parameters if they exist
            if "hidden_layer_1" in optimized_params:
                worker_config.set(
                    "algorithm.hidden_layer_1", int(optimized_params["hidden_layer_1"])
                )
            if "hidden_layer_2" in optimized_params:
                worker_config.set(
                    "algorithm.hidden_layer_2", int(optimized_params["hidden_layer_2"])
                )
            if "num_layers" in optimized_params:
                worker_config.set(
                    "algorithm.num_layers", int(optimized_params["num_layers"])
                )

            # Add worker-specific metadata
            worker_config.set("worker.worker_id", worker_id)
            worker_config.set(
                "worker.config_generated_at", optimized_params.get("suggested_at")
            )
            worker_config.set("worker.optimization_params", optimized_params)

            # Save worker config
            config_file = config_dir / f"worker_{worker_id}_config.json"
            worker_config.save_config(str(config_file))

            # Save flat hyperparameters for worker in metrics dir
            metrics_dir = Path(shared_dir) / "metrics"
            metrics_dir.mkdir(parents=True, exist_ok=True)
            hyper_file = metrics_dir / f"worker_{worker_id}_hyperparameters.json"
            with open(hyper_file, "w") as f:
                json.dump(optimized_params, f, indent=2)
            print(f"Saved hyperparameters for worker {worker_id}: {hyper_file}")

            generated_configs.append(str(config_file))
            print(f"Saved config for worker {worker_id}: {config_file}")

        print(f"Generated {len(generated_configs)} worker configs")
        return generated_configs

    except Exception as e:
        raise RuntimeError(f"Error generating worker configs: {e}")


def update_worker_performance(
    optimizer: BayesianOptimizationManager, shared_dir: str
) -> bool:
    """
    Update Bayesian Optimizer with worker performance data

    Args:
        optimizer: Bayesian optimization manager
        shared_dir: Shared directory path

    Returns:
        True if any updates were made, False otherwise
    """
    try:
        metrics_dir = Path(shared_dir) / "metrics"

        if not metrics_dir.exists():
            return False

        updated_any = False

        # Look for worker performance files
        for metrics_file in metrics_dir.glob("worker_*_performance.json"):
            try:
                with open(metrics_file, "r") as f:
                    metrics = json.load(f)

                worker_id = metrics.get("worker_id")
                avg_reward = metrics.get("avg_reward", 0)
                total_episodes = metrics.get("total_episodes", 0)

                # Only update if worker has enough episodes
                if total_episodes >= 20:  # Minimum episodes for reliable performance

                    # Load worker's optimization parameters
                    config_file = (
                        Path(shared_dir)
                        / "worker_configs"
                        / f"worker_{worker_id}_config.json"
                    )
                    if config_file.exists():
                        with open(config_file, "r") as f:
                            worker_config = json.load(f)

                        optimization_params = worker_config.get("worker", {}).get(
                            "optimization_params", {}
                        )

                        if optimization_params:
                            # Update optimizer with performance
                            optimizer.update_performance(
                                worker_id=worker_id,
                                params=optimization_params,
                                performance=avg_reward,
                            )
                            updated_any = True
                            print(
                                f"Updated performance for worker {worker_id}: {avg_reward:.2f}"
                            )

            except Exception as e:
                print(f"Error reading metrics for {metrics_file}: {e}")
                continue

        if updated_any:
            # Save optimization state
            optimizer.save_optimization_state()
            print(f"Saved optimization state")

        return updated_any

    except Exception as e:
        print(f"Error updating worker performance: {e}")
        return False


def handle_coordinator_error(error: Exception, shared_dir: str) -> None:
    """
    Handle coordinator errors by creating error files and signaling workers to stop.

    Args:
        error: Exception that occurred
        shared_dir: Shared directory path
    """
    try:
        error_dir = Path(shared_dir) / "errors"
        error_dir.mkdir(parents=True, exist_ok=True)

        error_data = {
            "error_type": type(error).__name__,
            "error_message": str(error),
            "timestamp": time.time(),
            "coordinator_status": "ERROR",
        }

        # Log the error to a file
        error_file = error_dir / f"coordinator_error_{int(time.time())}.json"
        with open(error_file, "w") as f:
            json.dump(error_data, f, indent=2)

        print(f"Coordinator error logged to: {error_file}")

        # Create a signal to stop all workers
        stop_signal_file = Path(shared_dir) / "stop_all_workers.signal"
        stop_signal_file.touch()
        print(f"Stop signal created for all workers: {stop_signal_file}")

    except Exception as log_error:
        print(f"Failed to log coordinator error or create stop signal: {log_error}")


# Enhanced coordinator class to integrate all functions
class EnhancedBayesianCoordinator:
    """
    Enhanced coordinator with Bayesian Optimization and config management
    """

    def __init__(
        self,
        shared_dir: str,
        config_path: str,
        num_workers: int = None,
        check_interval: int = 30,
    ):
        """
        Initialize enhanced coordinator

        Args:
            shared_dir: Shared directory path
            config_path: Path to base configuration file
            num_workers: Number of workers (if None, read from config)
            check_interval: Seconds between coordination checks
        """
        self.shared_dir = Path(shared_dir)
        self.config_path = config_path
        self.check_interval = check_interval

        try:
            # Load base configuration
            self.base_config = load_base_config(config_path)

            # Get number of workers from config if not specified
            if num_workers is None:
                self.num_workers = self.base_config.get("distributed.num_workers", 4)
            else:
                self.num_workers = num_workers

            # Initialize Bayesian Optimizer
            self.optimizer = initialize_bayesian_optimizer(
                str(shared_dir), self.base_config
            )

            # Generate initial worker configs
            self.worker_configs = generate_worker_configs(
                self.base_config, self.optimizer, self.num_workers, str(shared_dir)
            )

            # Initialize MinimalCoordinator for worker management
            self.worker_coordinator = MinimalCoordinator(
                shared_dir=shared_dir,
                num_workers=self.num_workers,
                check_interval=self.check_interval,
            )

            # Initialize status tracking
            self.start_time = time.time()
            self.sync_count = 0
            self.status_file = self.shared_dir / "coordinator_status.json"

            self._update_status("INITIALIZED")
            print(f"Enhanced Bayesian Coordinator initialized")

        except Exception as e:
            handle_coordinator_error(e, str(shared_dir))
            raise e

    def _update_status(self, status: str, **kwargs):
        """Update coordinator status"""
        import time
        from datetime import datetime

        status_data = {
            "status": status,
            "sync_count": self.sync_count,
            "uptime_seconds": int(time.time() - self.start_time),
            "last_heartbeat": datetime.now().isoformat(),
            "num_workers": self.num_workers,
            "config_path": self.config_path,
            "optimization_summary": self.optimizer.get_optimization_summary(),
            **kwargs,
        }

        try:
            with open(self.status_file, "w") as f:
                json.dump(status_data, f, indent=2)
        except Exception as e:
            print(f"Status write error: {e}")

        print(f"COORDINATOR_STATUS: {status} sync_count={self.sync_count}")

    def run(self):
        """Enhanced coordination loop with Bayesian Optimization"""
        print(f"Starting enhanced coordination with Bayesian Optimization")
        self._update_status("RUNNING")

        try:
            while not self.should_terminate():
                self.sync_count += 1
                self._update_status("SYNCING", current_sync=self.sync_count)

                print(f"\nSync #{self.sync_count}")

                # Step 1: Update Bayesian Optimizer with worker performance
                updated = update_worker_performance(
                    self.optimizer, str(self.shared_dir)
                )

                if updated:
                    print(f"Updated Bayesian Optimizer with new performance data")

                    # Regenerate configs if we have new optimization data
                    if self.sync_count % 5 == 0:  # Every 5 syncs
                        print(
                            f"Regenerating worker configs with updated optimization..."
                        )
                        self.worker_configs = generate_worker_configs(
                            self.base_config,
                            self.optimizer,
                            self.num_workers,
                            str(self.shared_dir),
                        )

                # Step 2: Delegate worker coordination to MinimalCoordinator
                best_worker = self.worker_coordinator.find_best_worker()

                if best_worker is not None:
                    print(f"Best worker identified: {best_worker}")

                    # Specify the target worker (e.g., worker 0 as a placeholder)
                    target_worker = (
                        0  # Replace with logic to determine the target worker
                    )

                    # Call copy_best_model with both source_worker and target_worker
                    if self.worker_coordinator.copy_best_model(
                        source_worker=best_worker, target_worker=target_worker
                    ):
                        print(
                            f"Copied best model from worker {best_worker} to worker {target_worker}"
                        )
                        self.worker_coordinator.signal_workers()
                    else:
                        print(f"Failed to copy model from worker {best_worker}")

                self._update_status("SLEEPING")
                time.sleep(self.check_interval)

        except KeyboardInterrupt:
            self._update_status("INTERRUPTED")
            print("\nCoordinator stopped by user")
        except Exception as e:
            handle_coordinator_error(e, str(self.shared_dir))
            self._update_status("ERROR", error=str(e))
            raise
        finally:
            self._update_status("SHUTDOWN")
            print(f"Enhanced coordinator finished after {self.sync_count} syncs")

    def should_terminate(self) -> bool:
        """Check for termination signals"""
        terminate_file = self.shared_dir / "terminate_coordinator.signal"
        return terminate_file.exists()


def main():
    """Main entry point for coordinator"""
    parser = argparse.ArgumentParser(description="Enhanced Distributed Coordinator")

    parser.add_argument("--shared_dir", required=True, help="Shared directory path")
    parser.add_argument("--config", required=True, help="Base configuration file path")
    parser.add_argument(
        "--num_workers", type=int, help="Number of workers (overrides config)"
    )
    parser.add_argument(
        "--check_interval", type=int, default=30, help="Seconds between checks"
    )

    args = parser.parse_args()

    print(f"Starting Enhanced Coordinator")
    print(f"Shared directory: {args.shared_dir}")
    print(f"Config file: {args.config}")
    print(f"Workers: {args.num_workers or 'from config'}")
    print(f"Check interval: {args.check_interval}s")

    # Create enhanced coordinator - ADD config_path parameter!
    coordinator = EnhancedBayesianCoordinator(
        shared_dir=args.shared_dir,
        config_path=args.config,  # <-- This was missing!
        num_workers=args.num_workers,
        check_interval=args.check_interval,
    )

    # Run coordination
    coordinator.run()


if __name__ == "__main__":
    main()
