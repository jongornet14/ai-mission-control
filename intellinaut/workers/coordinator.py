#!/usr/bin/env python3
"""
Minimal Distributed Coordinator
Does the essentials: find best worker, copy model, signal updates
"""

import json
import time
import shutil
import argparse
from pathlib import Path
import sys
import logging

import torch

from intellinaut.optimizers.bayesian import BayesianOptimizationManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


class MinimalCoordinator:
    def __init__(self, shared_dir, num_workers=4, check_interval=30, top_fraction=0.5):
        self.shared_dir = Path(shared_dir)
        self.num_workers = num_workers
        self.check_interval = check_interval
        self.top_fraction = top_fraction

        # Essential directories
        self.models_dir = self.shared_dir / "models"
        self.metrics_dir = self.shared_dir / "metrics"
        self.best_model_dir = self.shared_dir / "best_model"
        self.signals_dir = self.shared_dir / "signals"

        # Create directories
        for d in [
            self.models_dir,
            self.metrics_dir,
            self.best_model_dir,
            self.signals_dir,
        ]:
            d.mkdir(parents=True, exist_ok=True)

        # Initialize Bayesian optimizer
        self.optimizer = BayesianOptimizationManager(shared_dir=self.shared_dir)

        logging.info(f"Coordinator managing {num_workers} workers")

    def prepare_model_for_worker(self, worker_id: int):
        """
        Ensure the best model is available for a specific worker.

        Args:
            worker_id: ID of the worker to prepare the model for.
        """
        try:
            # Path to the best model
            best_model_path = self.best_model_dir / "current_best.pt"
            if not best_model_path.exists():
                logging.warning(f"Best model not found at {best_model_path}")
                return False

            # Signal file for the worker
            signal_file = self.signals_dir / f"update_worker_{worker_id}.signal"
            signal_file.touch()

            logging.info(f"Model prepared for worker {worker_id}: {signal_file}")
            return True

        except Exception as e:
            logging.error(f"Error preparing model for worker {worker_id}: {e}")
            return False

    def get_worker_metrics(self, worker_id):
        """
        Load metrics for one worker and update optimizer with performance.
        """
        metrics_file = self.metrics_dir / f"worker_{worker_id}_performance.json"
        try:
            if metrics_file.exists():
                with open(metrics_file, "r") as f:
                    metrics = json.load(f)

                # Update optimizer with performance
                hyperparams_file = (
                    self.metrics_dir / f"worker_{worker_id}_hyperparameters.json"
                )
                if hyperparams_file.exists():
                    with open(hyperparams_file, "r") as f:
                        hyperparams = json.load(f)
                    self.optimizer.update_performance(
                        worker_id=worker_id,
                        params=hyperparams,
                        performance=metrics["avg_reward"],
                    )

                logging.info(f"Metrics loaded for worker {worker_id}")
                return metrics
        except Exception as e:
            logging.error(f"Error loading worker {worker_id} metrics: {e}")
        return None

    def find_best_worker(self):
        """Find worker with best performance"""
        best_worker = None
        best_score = float("-inf")

        logging.info("Evaluating workers...")

        for worker_id in range(self.num_workers):
            metrics = self.get_worker_metrics(worker_id)
            if not metrics:
                continue

            # Check if reward is NaN
            avg_reward = metrics.get("avg_reward", 0)
            if avg_reward != avg_reward:  # NaN check
                logging.warning(f"Worker {worker_id}: Reward is NaN, skipping")
                continue

            # Simple scoring: reward_change + small bonus for avg_reward
            score = metrics.get("reward_change", 0) + avg_reward * 0.01
            episodes = metrics.get("total_episodes", 0)

            logging.info(f"Worker {worker_id}: {episodes} episodes, score: {score:.3f}")

            # Need at least 20 episodes and best score
            if episodes >= 20 and score > best_score:
                best_score = score
                best_worker = worker_id

        if best_worker is not None:
            logging.info(f"Best worker: {best_worker} (score: {best_score:.3f})")
        else:
            logging.warning("No eligible workers yet")

        return best_worker

    def copy_best_model(self, source_worker, target_worker):
        """
        Copy the model of the source worker to the target worker.

        Args:
            source_worker: ID of the worker providing the model.
            target_worker: ID of the worker receiving the model.
        """
        try:
            # Locate the source worker's latest model
            model_files = list(
                self.models_dir.glob(f"worker_{source_worker}_episode_*.pt")
            )
            if not model_files:
                logging.warning(f"No model found for worker {source_worker}")
                return False

            latest_model = max(model_files, key=lambda x: x.stat().st_mtime)

            # Copy to the target worker's directory
            target_model_path = self.models_dir / f"worker_{target_worker}_latest.pt"
            shutil.copy2(latest_model, target_model_path)

            logging.info(
                f"Copied model from worker {source_worker} to worker {target_worker}"
            )
            return True

        except Exception as e:
            logging.error(
                f"Error copying model from worker {source_worker} to worker {target_worker}: {e}"
            )
            return False

    def signal_workers(self):
        """
        Use quantiles to copy the top-performing workers' models to the lower-performing workers.
        """
        try:
            # Load metrics for all workers
            worker_metrics = []
            for worker_id in range(self.num_workers):
                metrics = self.get_worker_metrics(worker_id)
                if metrics:
                    worker_metrics.append((worker_id, metrics.get("avg_reward", 0)))

            # Sort workers by average reward (descending)
            worker_metrics.sort(key=lambda x: x[1], reverse=True)

            # Calculate the number of top workers dynamically
            num_top_workers = max(1, int(self.num_workers * self.top_fraction))
            top_workers = [
                worker_id for worker_id, _ in worker_metrics[:num_top_workers]
            ]
            bottom_workers = [
                worker_id for worker_id, _ in worker_metrics[num_top_workers:]
            ]

            logging.info(f"Top workers: {top_workers}")
            logging.info(f"Bottom workers: {bottom_workers}")

            # Copy models from top workers to bottom workers
            for i, bottom_worker in enumerate(bottom_workers):
                source_worker = top_workers[i % len(top_workers)]
                if not self.copy_best_model(source_worker, target_worker=bottom_worker):
                    continue

                suggested_config = None
                try:
                    suggested_config = self.optimizer.suggest_next_configuration(
                        worker_id=bottom_worker
                    )
                except Exception as e:
                    logging.error(
                        f"Error suggesting hyperparameters for worker {bottom_worker}: {e}"
                    )
                    continue

                # Check for NaN or inf in suggested_config
                if not all(
                    isinstance(v, (int, float))
                    and not (
                        isinstance(v, float)
                        and (v != v or v == float("inf") or v == float("-inf"))
                    )
                    for v in suggested_config.values()
                ):
                    logging.error(
                        f"Suggested hyperparameters for worker {bottom_worker} contain NaN or inf: {suggested_config}"
                    )
                    continue  # Skip saving/using these

                self.save_hyperparameters(bottom_worker, suggested_config)

        except Exception as e:
            logging.error(f"Error during worker synchronization: {e}")

    def save_hyperparameters(self, worker_id, config):
        """
        Save suggested hyperparameters for a worker.
        """
        try:
            config_file = self.metrics_dir / f"worker_{worker_id}_hyperparameters.json"
            with open(config_file, "w") as f:
                json.dump(config, f, indent=2)
            logging.info(f"Saved hyperparameters for worker {worker_id}: {config}")
        except Exception as e:
            logging.error(f"Error saving hyperparameters for worker {worker_id}: {e}")

    def should_terminate(self):
        """Check for termination signal"""
        terminate_file = self.shared_dir / "terminate_coordinator.signal"
        return terminate_file.exists()

    def run(self):
        """Main coordination loop"""
        logging.info(f"Starting coordination (check every {self.check_interval}s)")

        sync_count = 0

        try:
            while not self.should_terminate():
                sync_count += 1
                logging.info(f"\nSync #{sync_count}")

                # Find best worker
                best_worker = self.find_best_worker()

                if best_worker is not None:
                    # Debugging statement
                    logging.debug(f"Debug: Best worker for this sync is {best_worker}")

                    # Copy their model to another worker (e.g., worker 0 as a placeholder)
                    target_worker = (
                        0  # Replace with logic to determine the target worker
                    )
                    if self.copy_best_model(
                        source_worker=best_worker, target_worker=target_worker
                    ):
                        # Debugging statement
                        logging.debug(
                            f"Debug: Model from worker {best_worker} copied successfully to worker {target_worker}"
                        )

                        # Signal other workers to update
                        self.signal_workers()
                        logging.info(f"Sync #{sync_count} complete")
                    else:
                        logging.error(f"Sync #{sync_count} failed")

                # Wait before next check
                logging.info(f"Sleeping {self.check_interval}s...")
                time.sleep(self.check_interval)

        except KeyboardInterrupt:
            logging.warning("Coordinator stopped by user")
        except Exception as e:
            logging.error(f"Coordinator error: {e}")
        logging.info(f"Coordinator finished after {sync_count} syncs")


def main():
    parser = argparse.ArgumentParser(description="Minimal Distributed Coordinator")
    parser.add_argument("--shared_dir", required=True, help="Shared directory path")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers")
    parser.add_argument(
        "--check_interval", type=int, default=30, help="Seconds between checks"
    )
    parser.add_argument(
        "--top_fraction",
        type=float,
        default=0.25,
        help="Fraction of top workers to select",
    )

    args = parser.parse_args()

    coordinator = MinimalCoordinator(
        shared_dir=args.shared_dir,
        num_workers=args.num_workers,
        check_interval=args.check_interval,
        top_fraction=args.top_fraction,
    )

    coordinator.run()


if __name__ == "__main__":
    main()
