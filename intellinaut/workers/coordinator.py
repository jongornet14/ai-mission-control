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


class MinimalCoordinator:
    def __init__(self, shared_dir, num_workers=4, check_interval=30):
        self.shared_dir = Path(shared_dir)
        self.num_workers = num_workers
        self.check_interval = check_interval
        
        # Essential directories
        self.models_dir = self.shared_dir / "models"
        self.metrics_dir = self.shared_dir / "metrics"
        self.best_model_dir = self.shared_dir / "best_model"
        self.signals_dir = self.shared_dir / "signals"
        
        # Create directories
        for d in [self.models_dir, self.metrics_dir, self.best_model_dir, self.signals_dir]:
            d.mkdir(parents=True, exist_ok=True)
        
        print(f" Coordinator managing {num_workers} workers")
        print(f" Shared directory: {shared_dir}")

    def get_worker_metrics(self, worker_id):
        """Load metrics for one worker"""
        metrics_file = self.metrics_dir / f"worker_{worker_id}_performance.json"
        try:
            if metrics_file.exists():
                with open(metrics_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            print(f" Error loading worker {worker_id} metrics: {e}")
        return None

    def find_best_worker(self):
        """Find worker with best performance"""
        best_worker = None
        best_score = float('-inf')
        
        print("\n Evaluating workers...")
        
        for worker_id in range(self.num_workers):
            metrics = self.get_worker_metrics(worker_id)
            if not metrics:
                continue
                
            # Simple scoring: reward_change + small bonus for avg_reward
            score = metrics.get('reward_change', 0) + metrics.get('avg_reward', 0) * 0.01
            episodes = metrics.get('total_episodes', 0)
            
            print(f"Worker {worker_id}: {episodes} episodes, score: {score:.3f}")
            
            # Need at least 20 episodes and best score
            if episodes >= 20 and score > best_score:
                best_score = score
                best_worker = worker_id
        
        if best_worker is not None:
            print(f" Best worker: {best_worker} (score: {best_score:.3f})")
        else:
            print(" No eligible workers yet")
            
        return best_worker

    def copy_best_model(self, worker_id):
        """Copy worker's latest model to best_model directory"""
        try:
            # Find latest model file for this worker
            model_files = list(self.models_dir.glob(f"worker_{worker_id}_episode_*.pt"))
            if not model_files:
                print(f" No models found for worker {worker_id}")
                return False
            
            # Get most recent
            latest_model = max(model_files, key=lambda x: x.stat().st_mtime)
            
            # Copy to best model location
            best_model_path = self.best_model_dir / "current_best.pt"
            shutil.copy2(latest_model, best_model_path)
            
            print(f" Copied model from worker {worker_id}: {latest_model.name}")
            return True
            
        except Exception as e:
            print(f" Error copying model: {e}")
            return False

    def signal_workers(self, exclude_worker=None):
        """Create signal files to tell workers to update"""
        signaled = []
        
        for worker_id in range(self.num_workers):
            if worker_id == exclude_worker:
                continue  # Don't signal the worker that provided the model
                
            try:
                signal_file = self.signals_dir / f"update_worker_{worker_id}.signal"
                signal_file.touch()
                signaled.append(worker_id)
            except Exception as e:
                print(f" Error signaling worker {worker_id}: {e}")
        
        if signaled:
            print(f" Signaled workers: {signaled}")
        
        return signaled

    def should_terminate(self):
        """Check for termination signal"""
        terminate_file = self.shared_dir / "terminate_coordinator.signal"
        return terminate_file.exists()

    def run(self):
        """Main coordination loop"""
        print(f" Starting coordination (check every {self.check_interval}s)")
        
        sync_count = 0
        
        try:
            while not self.should_terminate():
                sync_count += 1
                print(f"\n Sync #{sync_count}")
                
                # Find best worker
                best_worker = self.find_best_worker()
                
                if best_worker is not None:
                    # Copy their model
                    if self.copy_best_model(best_worker):
                        # Signal other workers to update
                        self.signal_workers(exclude_worker=best_worker)
                        print(f" Sync #{sync_count} complete")
                    else:
                        print(f" Sync #{sync_count} failed")
                
                # Wait before next check
                print(f" Sleeping {self.check_interval}s...")
                time.sleep(self.check_interval)
                
        except KeyboardInterrupt:
            print("\n Coordinator stopped by user")
        except Exception as e:
            print(f" Coordinator error: {e}")
        
        print(f" Coordinator finished after {sync_count} syncs")


def main():
    parser = argparse.ArgumentParser(description='Minimal Distributed Coordinator')
    parser.add_argument('--shared_dir', required=True, help='Shared directory path')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers')
    parser.add_argument('--check_interval', type=int, default=30, help='Seconds between checks')
    
    args = parser.parse_args()
    
    coordinator = MinimalCoordinator(
        shared_dir=args.shared_dir,
        num_workers=args.num_workers,
        check_interval=args.check_interval
    )
    
    coordinator.run()


if __name__ == "__main__":
    main()