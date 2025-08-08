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

import torch

from ..optimizers.bayesian import BayesianOptimizationManager
from ..logging.debugging import create_coordinator_debugger


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

        # Initialize debugger for coordinator
        self.debugger = create_coordinator_debugger(str(self.shared_dir))

        # Initialize Bayesian optimizer
        self.optimizer = BayesianOptimizationManager(
            shared_dir=self.shared_dir, debugger=self.debugger
        )

        self.debugger.log_text("INFO", f"Coordinator managing {num_workers} workers")
        # Defensive: print where logs will go and what entity this is
        print(
            f"\033[1;36m[DEBUGGER] Coordinator entity '{self.debugger.entity_name}' logs to {self.debugger.logs_dir}\033[0m"
        )

    def prepare_model_for_worker(self, worker_id: int):
        try:
            best_model_path = self.best_model_dir / "current_best.pt"
            if not best_model_path.exists():
                self.debugger.log_text(
                    "WARNING", f"Best model not found at {best_model_path}"
                )
                return False

            signal_file = self.signals_dir / f"update_worker_{worker_id}.signal"
            signal_file.touch()

            self.debugger.log_text(
                "INFO", f"Model prepared for worker {worker_id}: {signal_file}"
            )
            return True

        except Exception as e:
            self.debugger.log_text(
                "ERROR", f"Error preparing model for worker {worker_id}: {e}"
            )
            return False

    def get_worker_metrics(self, worker_id):
        metrics_file = self.metrics_dir / f"worker_{worker_id}_performance.json"
        try:
            if metrics_file.exists():
                with open(metrics_file, "r") as f:
                    metrics = json.load(f)

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

                self.debugger.log_text("INFO", f"Metrics loaded for worker {worker_id}")
                return metrics
        except Exception as e:
            self.debugger.log_text(
                "ERROR", f"Error loading worker {worker_id} metrics: {e}"
            )
        return None

    def find_best_worker(self):
        best_worker = None
        best_score = float("-inf")

        self.debugger.log_text("INFO", "Evaluating workers...")

        for worker_id in range(self.num_workers):
            metrics = self.get_worker_metrics(worker_id)
            if not metrics:
                continue

            avg_reward = metrics.get("avg_reward", 0)
            if avg_reward != avg_reward:  # NaN check
                self.debugger.log_text(
                    "WARNING", f"Worker {worker_id}: Reward is NaN, skipping"
                )
                continue

            score = metrics.get("reward_change", 0) + avg_reward * 0.01
            episodes = metrics.get("total_episodes", 0)

            msg = f"Worker {worker_id}: {episodes} episodes, score: {score:.3f}"
            self.debugger.log_text("INFO", msg)

            if episodes >= 20 and score > best_score:
                best_score = score
                best_worker = worker_id

        if best_worker is not None:
            msg = f"Best worker: {best_worker} (score: {best_score:.3f})"
            self.debugger.log_text("INFO", msg)
        else:
            msg = "No eligible workers yet"
            self.debugger.log_text("WARNING", msg)

        return best_worker

    def copy_best_model(self, source_worker, target_worker):
        try:
            model_files = list(
                self.models_dir.glob(f"worker_{source_worker}_episode_*.pt")
            )
            if not model_files:
                msg = f"No model found for worker {source_worker}"
                self.debugger.log_text("WARNING", msg)
                return False

            latest_model = max(model_files, key=lambda x: x.stat().st_mtime)
            target_model_path = self.models_dir / f"worker_{target_worker}_latest.pt"
            shutil.copy2(latest_model, target_model_path)

            msg = f"Copied model from worker {source_worker} to worker {target_worker}"
            self.debugger.log_text("INFO", msg)
            return True
        except Exception as e:
            msg = f"Error copying model from worker {source_worker} to worker {target_worker}: {e}"
            self.debugger.log_text("ERROR", msg)
            return False

    def signal_workers(self):
        try:
            worker_metrics = []
            for worker_id in range(self.num_workers):
                metrics = self.get_worker_metrics(worker_id)
                if metrics:
                    worker_metrics.append((worker_id, metrics.get("avg_reward", 0)))

            worker_metrics.sort(key=lambda x: x[1], reverse=True)
            num_top_workers = max(1, int(self.num_workers * self.top_fraction))
            top_workers = [
                worker_id for worker_id, _ in worker_metrics[:num_top_workers]
            ]
            bottom_workers = [
                worker_id for worker_id, _ in worker_metrics[num_top_workers:]
            ]

            self.debugger.log_text("INFO", f"Top workers: {top_workers}")
            self.debugger.log_text("INFO", f"Bottom workers: {bottom_workers}")

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
                    msg = f"Error suggesting hyperparameters for worker {bottom_worker}: {e}"
                    self.debugger.log_text("ERROR", msg)
                    continue

                if not all(
                    isinstance(v, (int, float))
                    and not (
                        isinstance(v, float)
                        and (v != v or v == float("inf") or v == float("-inf"))
                    )
                    for v in suggested_config.values()
                ):
                    msg = f"Suggested hyperparameters for worker {bottom_worker} contain NaN or inf: {suggested_config}"
                    self.debugger.log_text("ERROR", msg)
                    continue

                self.save_hyperparameters(bottom_worker, suggested_config)
        except Exception as e:
            msg = f"Error during worker synchronization: {e}"
            self.debugger.log_text("ERROR", msg)

    def save_hyperparameters(self, worker_id, config):
        try:
            config_file = self.metrics_dir / f"worker_{worker_id}_hyperparameters.json"
            with open(config_file, "w") as f:
                json.dump(config, f, indent=2)
            msg = f"Saved hyperparameters for worker {worker_id}: {config}"
            self.debugger.log_text("INFO", msg)
        except Exception as e:
            msg = f"Error saving hyperparameters for worker {worker_id}: {e}"
            self.debugger.log_text("ERROR", msg)

    def should_terminate(self):
        terminate_file = self.shared_dir / "terminate_coordinator.signal"
        return terminate_file.exists()

    def run(self):
        self.debugger.log_text(
            "INFO", f"Starting coordination (check every {self.check_interval}s)"
        )

        sync_count = 0

        try:
            while not self.should_terminate():
                sync_count += 1
                msg = f"\nSync #{sync_count}"
                self.debugger.log_text("INFO", msg)

                best_worker = self.find_best_worker()

                if best_worker is not None:
                    self.debugger.log_text(
                        "DEBUG", f"Debug: Best worker for this sync is {best_worker}"
                    )

                    target_worker = (
                        0  # Replace with logic to determine the target worker
                    )
                    if self.copy_best_model(
                        source_worker=best_worker, target_worker=target_worker
                    ):
                        self.debugger.log_text(
                            "DEBUG",
                            f"Debug: Model from worker {best_worker} copied successfully to worker {target_worker}",
                        )

                        self.signal_workers()
                        msg = f"Sync #{sync_count} complete"
                        self.debugger.log_text("INFO", msg)
                    else:
                        msg = f"Sync #{sync_count} failed"
                        self.debugger.log_text("ERROR", msg)

                msg = f"Sleeping {self.check_interval}s..."
                self.debugger.log_text("INFO", msg)
                time.sleep(self.check_interval)
        except KeyboardInterrupt:
            msg = "Coordinator stopped by user"
            self.debugger.log_text("WARNING", msg)
        except Exception as e:
            msg = f"Coordinator error: {e}"
            self.debugger.log_text("ERROR", msg)
        msg = f"Coordinator finished after {sync_count} syncs"
        self.debugger.log_text("INFO", msg)


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
