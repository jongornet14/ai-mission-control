#!/usr/bin/env python3
"""
Distributed RL Coordinator Script
Manages 4 workers, selects best models, and coordinates training
"""

import torch
import json
import time
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import shutil
import pandas as pd
from typing import Dict, List, Optional


class DistributedCoordinator:
    def __init__(self, shared_dir, num_workers=4, check_interval=30, min_episodes=50):
        """
        Distributed RL Coordinator

        Args:
            shared_dir: Directory for shared model files
            num_workers: Number of workers to coordinate (default 4)
            check_interval: Seconds between worker checks
            min_episodes: Minimum episodes before considering worker for best model
        """
        self.shared_dir = Path(shared_dir)
        self.num_workers = num_workers
        self.check_interval = check_interval
        self.min_episodes = min_episodes

        # Create coordinator directories
        self.models_dir = self.shared_dir / "models"
        self.metrics_dir = self.shared_dir / "metrics"
        self.best_model_dir = self.shared_dir / "best_model"
        self.coordinator_dir = self.shared_dir / "coordinator"

        for dir_path in [
            self.models_dir,
            self.metrics_dir,
            self.best_model_dir,
            self.coordinator_dir,
        ]:
            dir_path.mkdir(parents=True, exist_ok=True)

        # Coordinator state
        self.start_time = time.time()
        self.last_sync_time = time.time()
        self.sync_count = 0
        self.worker_stats = {i: {} for i in range(num_workers)}
        self.best_worker_id = None
        self.best_performance = float("-inf")
        self.coordination_log = []

        print(f"Coordinator initialized")
        print(f"Managing {num_workers} workers")
        print(f"Shared directory: {shared_dir}")
        print(f"Check interval: {check_interval} seconds")

    def get_worker_metrics(self, worker_id: int) -> Optional[Dict]:
        """Load metrics for a specific worker"""
        try:
            metrics_file = self.metrics_dir / f"worker_{worker_id}_performance.json"
            if metrics_file.exists():
                with open(metrics_file, "r") as f:
                    metrics = json.load(f)
                return metrics
        except Exception as e:
            print(f"Error loading metrics for worker {worker_id}: {e}")
        return None

    def get_all_worker_metrics(self) -> Dict[int, Dict]:
        """Load metrics for all workers"""
        all_metrics = {}
        for worker_id in range(self.num_workers):
            metrics = self.get_worker_metrics(worker_id)
            if metrics:
                all_metrics[worker_id] = metrics
        return all_metrics

    def evaluate_worker_performance(self, metrics: Dict) -> float:
        """
        Evaluate worker performance using state-of-the-art metric
        Returns average change in reward
        """
        if not metrics or "reward_change" not in metrics:
            return float("-inf")

        # Primary metric: average change in reward
        reward_change = metrics.get("reward_change", 0)

        # Secondary factors (small weights)
        avg_reward = metrics.get("avg_reward", 0)
        total_episodes = metrics.get("total_episodes", 0)

        # Weight recent performance more heavily
        performance_score = reward_change

        # Small bonus for absolute performance and experience
        performance_score += 0.1 * avg_reward / 1000  # Normalize avg_reward
        performance_score += 0.01 * min(total_episodes / 1000, 1)  # Cap episode bonus

        return performance_score

    def find_best_worker(self) -> Optional[int]:
        """Find the best performing worker based on metrics"""
        all_metrics = self.get_all_worker_metrics()

        if not all_metrics:
            print("No worker metrics available")
            return None

        best_worker = None
        best_score = float("-inf")

        print("\n" + "=" * 60)
        print("WORKER PERFORMANCE EVALUATION")
        print("=" * 60)

        for worker_id, metrics in all_metrics.items():
            # Skip workers with insufficient episodes
            if metrics.get("total_episodes", 0) < self.min_episodes:
                print(
                    f"Worker {worker_id}: Insufficient episodes ({metrics.get('total_episodes', 0)}/{self.min_episodes})"
                )
                continue

            score = self.evaluate_worker_performance(metrics)

            print(f"Worker {worker_id}:")
            print(f"  Episodes: {metrics.get('total_episodes', 0)}")
            print(f"  Avg Reward: {metrics.get('avg_reward', 0):.4f}")
            print(f"  Reward Change: {metrics.get('reward_change', 0):.4f}")
            print(f"  Performance Score: {score:.4f}")
            print(f"  Training Time: {metrics.get('training_time', 0)/60:.1f} minutes")

            if score > best_score:
                best_score = score
                best_worker = worker_id

        print("=" * 60)
        if best_worker is not None:
            print(f"BEST WORKER: {best_worker} (Score: {best_score:.4f})")
        else:
            print("NO ELIGIBLE WORKERS FOUND")
        print("=" * 60)

        return best_worker

    def copy_best_model(self, best_worker_id: int) -> bool:
        """Copy the best worker's model to shared best_model directory"""
        try:
            # Find the latest model file for the best worker
            model_files = list(
                self.models_dir.glob(f"worker_{best_worker_id}_episode_*.pt")
            )
            if not model_files:
                print(f"No model files found for worker {best_worker_id}")
                return False

            # Get the most recent model
            latest_model = max(model_files, key=lambda x: x.stat().st_mtime)

            # Copy to best model directory
            best_model_path = self.best_model_dir / "current_best.pt"
            shutil.copy2(latest_model, best_model_path)

            # Save metadata about the best model
            best_metrics = self.get_worker_metrics(best_worker_id)
            metadata = {
                "best_worker_id": best_worker_id,
                "model_source": str(latest_model),
                "timestamp": time.time(),
                "sync_count": self.sync_count,
                "worker_metrics": best_metrics,
            }

            metadata_path = self.best_model_dir / "best_model_metadata.json"
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)

            print(f"✅ Copied best model from worker {best_worker_id}")
            print(f"   Source: {latest_model.name}")
            print(
                f"   Performance: {best_metrics.get('reward_change', 0):.4f} reward change"
            )

            return True

        except Exception as e:
            print(f"❌ Error copying best model: {e}")
            return False

    def signal_worker_updates(self, exclude_worker: Optional[int] = None):
        """Signal workers to update their models (except the best worker)"""
        updated_workers = []

        for worker_id in range(self.num_workers):
            if worker_id == exclude_worker:
                continue  # Don't update the worker that provided the best model

            try:
                signal_file = self.shared_dir / f"update_worker_{worker_id}.signal"
                signal_file.touch()
                updated_workers.append(worker_id)
            except Exception as e:
                print(f"Error signaling worker {worker_id}: {e}")

        if updated_workers:
            print(f"📡 Signaled model update to workers: {updated_workers}")

        return updated_workers

    def cleanup_old_files(self, keep_recent=5):
        """Clean up old model and metric files to save space"""
        try:
            # Clean old model files (keep most recent per worker)
            for worker_id in range(self.num_workers):
                model_files = sorted(
                    self.models_dir.glob(f"worker_{worker_id}_episode_*.pt"),
                    key=lambda x: x.stat().st_mtime,
                    reverse=True,
                )

                # Remove all but the most recent files
                for old_file in model_files[keep_recent:]:
                    old_file.unlink()

            print(f"🧹 Cleaned up old files (kept {keep_recent} recent per worker)")

        except Exception as e:
            print(f"Error during cleanup: {e}")

    def save_coordination_log(self):
        """Save coordination history and statistics"""
        try:
            log_data = {
                "coordination_history": self.coordination_log,
                "total_syncs": self.sync_count,
                "coordinator_runtime": time.time() - self.start_time,
                "worker_stats": self.worker_stats,
                "best_worker_history": [
                    entry for entry in self.coordination_log if "best_worker" in entry
                ],
            }

            log_file = self.coordinator_dir / "coordination_log.json"
            with open(log_file, "w") as f:
                json.dump(log_data, f, indent=2)

            # Create summary CSV for easy analysis
            if self.coordination_log:
                df = pd.DataFrame(self.coordination_log)
                summary_file = self.coordinator_dir / "coordination_summary.csv"
                df.to_csv(summary_file, index=False)

        except Exception as e:
            print(f"Error saving coordination log: {e}")

    def perform_sync(self) -> bool:
        """Perform one coordination cycle"""
        print(f"\n🔄 COORDINATION CYCLE {self.sync_count + 1}")
        print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # Find best worker
        best_worker = self.find_best_worker()

        if best_worker is None:
            print("⏳ No workers ready for sync yet")
            return False

        # Copy best model
        if not self.copy_best_model(best_worker):
            print("❌ Failed to copy best model")
            return False

        # Signal other workers to update
        updated_workers = self.signal_worker_updates(exclude_worker=best_worker)

        # Update coordinator state
        self.sync_count += 1
        self.last_sync_time = time.time()
        self.best_worker_id = best_worker

        # Log this coordination cycle
        best_metrics = self.get_worker_metrics(best_worker)
        log_entry = {
            "sync_count": self.sync_count,
            "timestamp": time.time(),
            "best_worker": best_worker,
            "best_performance": (
                best_metrics.get("reward_change", 0) if best_metrics else 0
            ),
            "updated_workers": updated_workers,
            "total_runtime": time.time() - self.start_time,
        }
        self.coordination_log.append(log_entry)

        # Periodic cleanup
        if self.sync_count % 10 == 0:
            self.cleanup_old_files()

        # Save progress
        self.save_coordination_log()

        print(f"✅ Sync {self.sync_count} completed successfully")
        return True

    def check_termination_conditions(self) -> bool:
        """Check if coordination should terminate"""
        # Check for global termination signal
        terminate_file = self.shared_dir / "terminate_all.signal"
        if terminate_file.exists():
            print("🛑 Global termination signal detected")
            return True

        # Check for coordinator-specific termination
        coord_terminate = self.shared_dir / "terminate_coordinator.signal"
        if coord_terminate.exists():
            print("🛑 Coordinator termination signal detected")
            return True

        return False

    def run(
        self, max_syncs: Optional[int] = None, max_runtime_hours: Optional[float] = None
    ):
        """
        Main coordination loop

        Args:
            max_syncs: Maximum number of sync cycles (None for unlimited)
            max_runtime_hours: Maximum runtime in hours (None for unlimited)
        """
        print(f"🚀 Starting coordination loop")
        print(f"Max syncs: {max_syncs if max_syncs else 'unlimited'}")
        print(
            f"Max runtime: {max_runtime_hours if max_runtime_hours else 'unlimited'} hours"
        )
        print(f"Check interval: {self.check_interval} seconds")

        try:
            while True:
                # Check termination conditions
                if self.check_termination_conditions():
                    break

                # Check runtime limit
                if max_runtime_hours:
                    runtime_hours = (time.time() - self.start_time) / 3600
                    if runtime_hours >= max_runtime_hours:
                        print(
                            f"⏰ Runtime limit reached ({runtime_hours:.1f}/{max_runtime_hours} hours)"
                        )
                        break

                # Check sync limit
                if max_syncs and self.sync_count >= max_syncs:
                    print(f"🎯 Sync limit reached ({self.sync_count}/{max_syncs})")
                    break

                # Perform coordination
                try:
                    self.perform_sync()
                except Exception as e:
                    print(f"❌ Error during sync: {e}")

                # Wait before next check
                print(f"😴 Sleeping for {self.check_interval} seconds...")
                time.sleep(self.check_interval)

        except KeyboardInterrupt:
            print("\n🛑 Coordination interrupted by user")
        except Exception as e:
            print(f"💥 Unexpected error: {e}")
        finally:
            self.cleanup()

    def cleanup(self):
        """Cleanup coordinator resources and save final state"""
        print(f"\n🧹 Coordinator cleanup")

        # Save final logs
        self.save_coordination_log()

        # Create final summary
        runtime = time.time() - self.start_time
        summary = {
            "total_runtime_hours": runtime / 3600,
            "total_syncs": self.sync_count,
            "syncs_per_hour": self.sync_count / (runtime / 3600) if runtime > 0 else 0,
            "best_worker_final": self.best_worker_id,
            "coordination_efficiency": len(self.coordination_log)
            / max(1, self.sync_count),
        }

        print(f"📊 FINAL COORDINATION SUMMARY:")
        print(f"   Runtime: {summary['total_runtime_hours']:.2f} hours")
        print(f"   Total syncs: {summary['total_syncs']}")
        print(f"   Syncs per hour: {summary['syncs_per_hour']:.1f}")
        print(f"   Final best worker: {summary['best_worker_final']}")

        # Save summary
        summary_file = self.coordinator_dir / "final_summary.json"
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)

        print("✅ Coordinator shutdown complete")


def main():
    parser = argparse.ArgumentParser(description="Distributed RL Coordinator")

    parser.add_argument(
        "--shared_dir",
        type=str,
        required=True,
        help="Shared directory for model exchange",
    )
    parser.add_argument(
        "--num_workers", type=int, default=4, help="Number of workers to coordinate"
    )
    parser.add_argument(
        "--check_interval", type=int, default=30, help="Seconds between worker checks"
    )
    parser.add_argument(
        "--min_episodes",
        type=int,
        default=50,
        help="Minimum episodes before considering worker",
    )
    parser.add_argument(
        "--max_syncs", type=int, default=None, help="Maximum number of sync cycles"
    )
    parser.add_argument(
        "--max_runtime_hours", type=float, default=None, help="Maximum runtime in hours"
    )

    args = parser.parse_args()

    # Create coordinator
    coordinator = DistributedCoordinator(
        shared_dir=args.shared_dir,
        num_workers=args.num_workers,
        check_interval=args.check_interval,
        min_episodes=args.min_episodes,
    )

    # Run coordination
    coordinator.run(max_syncs=args.max_syncs, max_runtime_hours=args.max_runtime_hours)


if __name__ == "__main__":
    main()
