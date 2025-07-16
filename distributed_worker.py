#!/usr/bin/env python3
"""
Distributed RL Worker Script
Clean version with CrazyLogger integration
"""

import torch
import numpy as np
import time
import argparse
import os
import json
import shutil
from pathlib import Path
from collections import defaultdict
import pandas as pd
from datetime import datetime

# Import your existing RL components and CrazyLogger
# from universal_rl import create_ppo_agent, create_environment
# from crazylogging.crazy_logger import CrazyLogger

class DistributedWorker:
    def __init__(self, worker_id, shared_dir, max_episodes=100, sync_interval=10, timeout_minutes=30):
        """
        Distributed RL Worker
        
        Args:
            worker_id: Unique worker identifier (0, 1, 2, 3)
            shared_dir: Directory for shared model files
            max_episodes: Episodes to train before checkpoint
            sync_interval: Episodes between checking for model updates
            timeout_minutes: Max time before forced sync
        """
        self.worker_id = worker_id
        self.shared_dir = Path(shared_dir)
        self.max_episodes = max_episodes
        self.sync_interval = sync_interval
        self.timeout_minutes = timeout_minutes
        
        # Create worker directories
        self.worker_dir = self.shared_dir / f"worker_{worker_id}"
        self.worker_dir.mkdir(parents=True, exist_ok=True)
        
        # Shared directories
        self.models_dir = self.shared_dir / "models"
        self.metrics_dir = self.shared_dir / "metrics"
        self.best_model_dir = self.shared_dir / "best_model"
        
        for dir_path in [self.models_dir, self.metrics_dir, self.best_model_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Training state
        self.total_episodes = 0
        self.start_time = time.time()
        self.last_sync_time = time.time()
        self.episode_rewards = []
        self.training_metrics = defaultdict(list)
        
        # Initialize CrazyLogger for comprehensive logging
        self.log_dir = self.shared_dir / "worker_logs" / f"worker_{worker_id}"
        try:
            from crazylogging.crazy_logger import CrazyLogger
            self.logger = CrazyLogger(
                log_dir=str(self.log_dir),
                experiment_name=f"distributed_worker_{worker_id}",
                buffer_size=5000  # Smaller buffer for distributed workers
            )
        except ImportError:
            print("Warning: CrazyLogger not available, using basic logging")
            self.logger = None
        
        print(f"Worker {worker_id} initialized")
        print(f"Shared directory: {shared_dir}")
        if self.logger:
            print(f"CrazyLogger directory: {self.log_dir}")
            print(f"TensorBoard: tensorboard --logdir {self.log_dir / 'tensorboard'}")
        print(f"Max episodes per checkpoint: {max_episodes}")

    def should_sync(self):
        """Check if worker should sync with coordinator"""
        time_elapsed = (time.time() - self.last_sync_time) / 60  # minutes
        episode_check = self.total_episodes % self.sync_interval == 0 and self.total_episodes > 0
        time_check = time_elapsed >= self.timeout_minutes
        
        return episode_check or time_check

    def check_for_model_update(self):
        """Check if coordinator has a new model for this worker"""
        try:
            # Check for coordinator signal
            update_file = self.shared_dir / f"update_worker_{self.worker_id}.signal"
            if update_file.exists():
                print(f"Worker {self.worker_id}: Model update signal received")
                return True
                
            # Check if best model is newer than our last sync
            best_model_file = self.best_model_dir / "current_best.pt"
            if best_model_file.exists():
                model_time = best_model_file.stat().st_mtime
                if model_time > self.last_sync_time:
                    print(f"Worker {self.worker_id}: New best model available")
                    return True
                    
        except Exception as e:
            print(f"Worker {self.worker_id}: Error checking for updates: {e}")
            
        return False

    def load_best_model(self, policy_module, value_module):
        """Load the best model from coordinator"""
        try:
            best_model_file = self.best_model_dir / "current_best.pt"
            if best_model_file.exists():
                checkpoint = torch.load(best_model_file, map_location='cpu')
                policy_module.load_state_dict(checkpoint['policy_state_dict'])
                value_module.load_state_dict(checkpoint['value_state_dict'])
                
                # Remove update signal
                update_file = self.shared_dir / f"update_worker_{self.worker_id}.signal"
                if update_file.exists():
                    update_file.unlink()
                
                self.last_sync_time = time.time()
                print(f"Worker {self.worker_id}: Loaded best model")
                
                # Log model update to CrazyLogger
                if self.logger:
                    self.logger.log_step(
                        model_update=True,
                        model_source="coordinator_best_model",
                        sync_time=self.last_sync_time
                    )
                
                return True
        except Exception as e:
            print(f"Worker {self.worker_id}: Error loading best model: {e}")
            
        return False

    def save_checkpoint(self, policy_module, value_module, optimizer):
        """Save current model and metrics"""
        try:
            # Save model
            checkpoint = {
                'policy_state_dict': policy_module.state_dict(),
                'value_state_dict': value_module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'worker_id': self.worker_id,
                'total_episodes': self.total_episodes,
                'timestamp': time.time()
            }
            
            model_file = self.models_dir / f"worker_{self.worker_id}_episode_{self.total_episodes}.pt"
            torch.save(checkpoint, model_file)
            
            # Save metrics
            avg_reward = np.mean(self.episode_rewards[-self.max_episodes:]) if self.episode_rewards else 0
            reward_change = self.calculate_reward_change()
            
            metrics = {
                'worker_id': self.worker_id,
                'total_episodes': self.total_episodes,
                'avg_reward': avg_reward,
                'reward_change': reward_change,
                'episode_rewards': self.episode_rewards[-self.max_episodes:],
                'timestamp': time.time(),
                'training_time': time.time() - self.start_time
            }
            
            metrics_file = self.metrics_dir / f"worker_{self.worker_id}_performance.json"
            with open(metrics_file, 'w') as f:
                json.dump(metrics, f, indent=2)
            
            print(f"Worker {self.worker_id}: Checkpoint saved - Episodes: {self.total_episodes}, Avg Reward: {avg_reward:.4f}, Change: {reward_change:.4f}")
            return True
            
        except Exception as e:
            print(f"Worker {self.worker_id}: Error saving checkpoint: {e}")
            return False

    def calculate_reward_change(self):
        """Calculate average change in reward (state-of-the-art metric)"""
        if len(self.episode_rewards) < 20:  # Need minimum episodes
            return 0.0
            
        recent_rewards = self.episode_rewards[-10:]
        older_rewards = self.episode_rewards[-20:-10]
        
        recent_avg = np.mean(recent_rewards)
        older_avg = np.mean(older_rewards)
        
        return recent_avg - older_avg

    def create_distributed_analysis_plots(self):
        """Create custom analysis plots for distributed training"""
        if len(self.episode_rewards) < 10:
            return
            
        try:
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            
            # Episode rewards with worker ID
            axes[0, 0].plot(self.episode_rewards, alpha=0.7, label=f'Worker {self.worker_id}')
            if len(self.episode_rewards) > 20:
                window = min(20, len(self.episode_rewards) // 4)
                moving_avg = np.convolve(self.episode_rewards, np.ones(window)/window, mode='valid')
                axes[0, 0].plot(range(window-1, len(self.episode_rewards)), moving_avg, 'r-', linewidth=2, label=f'MA({window})')
            axes[0, 0].set_xlabel('Episode')
            axes[0, 0].set_ylabel('Reward')
            axes[0, 0].set_title(f'Worker {self.worker_id} - Training Progress')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            # Reward change over time
            if len(self.episode_rewards) >= 20:
                reward_changes = []
                for i in range(10, len(self.episode_rewards)):
                    recent = np.mean(self.episode_rewards[i-9:i+1])
                    older = np.mean(self.episode_rewards[i-19:i-9])
                    reward_changes.append(recent - older)
                
                axes[0, 1].plot(reward_changes, alpha=0.7, color='green')
                axes[0, 1].axhline(y=0, color='red', linestyle='--', alpha=0.7)
                axes[0, 1].set_xlabel('Episode')
                axes[0, 1].set_ylabel('Reward Change')
                axes[0, 1].set_title(f'Worker {self.worker_id} - Performance Metric')
                axes[0, 1].grid(True, alpha=0.3)
            
            # Reward distribution
            axes[1, 0].hist(self.episode_rewards, bins=20, alpha=0.7, edgecolor='black')
            axes[1, 0].axvline(np.mean(self.episode_rewards), color='red', linestyle='--', 
                              label=f'Mean: {np.mean(self.episode_rewards):.2f}')
            axes[1, 0].set_xlabel('Reward')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].set_title(f'Worker {self.worker_id} - Reward Distribution')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
            
            # Training efficiency
            training_time_hours = (time.time() - self.start_time) / 3600
            episodes_per_hour = self.total_episodes / training_time_hours if training_time_hours > 0 else 0
            
            axes[1, 1].text(0.1, 0.8, f'Worker ID: {self.worker_id}', transform=axes[1, 1].transAxes, fontsize=12)
            axes[1, 1].text(0.1, 0.7, f'Total Episodes: {self.total_episodes}', transform=axes[1, 1].transAxes, fontsize=12)
            axes[1, 1].text(0.1, 0.6, f'Training Time: {training_time_hours:.2f}h', transform=axes[1, 1].transAxes, fontsize=12)
            axes[1, 1].text(0.1, 0.5, f'Episodes/Hour: {episodes_per_hour:.1f}', transform=axes[1, 1].transAxes, fontsize=12)
            axes[1, 1].text(0.1, 0.4, f'Best Reward: {max(self.episode_rewards):.2f}', transform=axes[1, 1].transAxes, fontsize=12)
            axes[1, 1].text(0.1, 0.3, f'Current Trend: {self.calculate_reward_change():.3f}', transform=axes[1, 1].transAxes, fontsize=12)
            axes[1, 1].set_title(f'Worker {self.worker_id} - Statistics')
            axes[1, 1].axis('off')
            
            plt.tight_layout()
            
            # Log the plot using CrazyLogger
            if self.logger:
                self.logger.log_custom_plot(f'distributed_worker_{self.worker_id}_analysis', fig, self.total_episodes)
            
            # Save plot to shared directory as well
            plot_path = self.shared_dir / f"worker_{self.worker_id}_analysis.png"
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"Worker {self.worker_id}: Error creating analysis plots: {e}")

    def should_terminate(self):
        """Check if worker should terminate (coordinator signal or timeout)"""
        terminate_file = self.shared_dir / f"terminate_worker_{self.worker_id}.signal"
        if terminate_file.exists():
            print(f"Worker {self.worker_id}: Termination signal received")
            return True
            
        # Global termination signal
        global_terminate = self.shared_dir / "terminate_all.signal"
        if global_terminate.exists():
            print(f"Worker {self.worker_id}: Global termination signal received")
            return True
            
        return False

def distributed_training_worker(args):
    """
    Main worker training function - simplified version for testing
    """
    # Initialize worker
    worker = DistributedWorker(
        worker_id=args.worker_id,
        shared_dir=args.shared_dir,
        max_episodes=args.max_episodes,
        sync_interval=args.sync_interval,
        timeout_minutes=args.timeout_minutes
    )
    
    # Log initial configuration
    if worker.logger:
        worker.logger.log_step(
            worker_id=args.worker_id,
            environment=args.env,
            device=str(args.device),
            learning_rate=args.lr,
            max_episodes=args.max_episodes,
            total_frames=args.total_frames,
            distributed_training=True
        )
    
    print(f"Worker {args.worker_id}: Starting training")
    
    # Dummy training loop for testing (replace with your actual training)
    try:
        for episode in range(args.max_episodes):
            if worker.should_terminate():
                break
            
            # Simulate episode
            time.sleep(1)  # Simulate training time
            episode_reward = np.random.normal(100, 20)  # Simulate reward
            worker.episode_rewards.append(episode_reward)
            worker.total_episodes += 1
            
            # Log episode data
            episode_metrics = {
                'episode_reward': episode_reward,
                'total_episodes': worker.total_episodes,
                'worker_id': worker.worker_id,
                'reward_change': worker.calculate_reward_change()
            }
            
            if worker.logger:
                worker.logger.log_episode(**episode_metrics)
            
            # Check for model updates periodically
            if worker.should_sync():
                if worker.check_for_model_update():
                    print(f"Worker {args.worker_id}: Would load best model here")
            
            # Save checkpoint periodically
            if worker.total_episodes % 10 == 0:
                # For testing, create dummy model states
                class DummyModule:
                    def state_dict(self):
                        return {'dummy': torch.tensor([1.0])}
                
                dummy_policy = DummyModule()
                dummy_value = DummyModule()
                dummy_optimizer = DummyModule()
                
                worker.save_checkpoint(dummy_policy, dummy_value, dummy_optimizer)
                worker.create_distributed_analysis_plots()
                
                print(f"Worker {args.worker_id}: Completed {worker.total_episodes} episodes")
            
            # Progress update
            if worker.total_episodes % 5 == 0:
                avg_reward = np.mean(worker.episode_rewards[-5:])
                print(f"Worker {args.worker_id}: Episode {worker.total_episodes}, Avg Reward: {avg_reward:.4f}")
                
    except KeyboardInterrupt:
        print(f"Worker {args.worker_id}: Training interrupted")
    except Exception as e:
        print(f"Worker {args.worker_id}: Error during training: {e}")
        if worker.logger:
            worker.logger.log_step(error=str(e), error_type="training_exception")
    finally:
        # Final analysis and cleanup
        worker.create_distributed_analysis_plots()
        
        if worker.logger:
            final_summary = worker.logger.generate_final_report()
            worker.logger.close()
            print(f"Worker {args.worker_id}: Final report: {final_summary}")
        
        print(f"Worker {args.worker_id}: Training completed - Total episodes: {worker.total_episodes}")

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Distributed RL Worker')
    
    # Worker-specific args
    parser.add_argument('--worker_id', type=int, required=True, help='Worker ID (0, 1, 2, 3)')
    parser.add_argument('--shared_dir', type=str, required=True, help='Shared directory for model exchange')
    parser.add_argument('--max_episodes', type=int, default=100, help='Episodes per checkpoint')
    parser.add_argument('--sync_interval', type=int, default=10, help='Episodes between sync checks')
    parser.add_argument('--timeout_minutes', type=int, default=30, help='Max minutes before forced sync')
    
    # Standard RL args
    parser.add_argument('--env', type=str, default='HalfCheetah-v4', help='Environment name')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--frames_per_batch', type=int, default=1000, help='Frames per batch')
    parser.add_argument('--total_frames', type=int, default=1000000, help='Total training frames')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--sub_batch_size', type=int, default=64, help='Sub-batch size')
    parser.add_argument('--max_grad_norm', type=float, default=1.0, help='Max gradient norm')
    
    args = parser.parse_args()
    
    # Validate worker_id
    if args.worker_id < 0 or args.worker_id > 3:
        raise ValueError("worker_id must be between 0 and 3")
    
    # Start distributed training
    distributed_training_worker(args)

if __name__ == "__main__":
    main()