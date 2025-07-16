#!/usr/bin/env python3
"""
Distributed RL Worker Script
Integrates with existing universal_rl.py for distributed training
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

# Import your existing RL components
from universal_rl import create_ppo_agent, create_environment

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
        
        print(f"Worker {worker_id} initialized")
        print(f"Shared directory: {shared_dir}")
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
    Main worker training function - integrates with your existing code
    """
    # Initialize worker
    worker = DistributedWorker(
        worker_id=args.worker_id,
        shared_dir=args.shared_dir,
        max_episodes=args.max_episodes,
        sync_interval=args.sync_interval,
        timeout_minutes=args.timeout_minutes
    )
    
    # Create environment and agent (using your existing functions)
    env = create_environment(args.env, device=args.device)
    policy_module, value_module, collector, loss_module, advantage_module, optimizer = create_ppo_agent(
        env, args.device, args.lr, args.frames_per_batch, args.total_frames
    )
    
    # Check for existing best model to start with
    worker.load_best_model(policy_module, value_module)
    
    print(f"Worker {args.worker_id}: Starting training")
    
    try:
        # Training loop
        for i, tensordict_data in enumerate(collector):
            if worker.should_terminate():
                break
                
            # Standard PPO training step (from your existing code)
            for _ in range(args.num_epochs):
                advantage_module(tensordict_data)
                data_view = tensordict_data.reshape(-1)
                
                # Mini-batch training
                for _ in range(args.frames_per_batch // args.sub_batch_size):
                    indices = torch.randperm(len(data_view))[:args.sub_batch_size]
                    subdata = data_view[indices]
                    
                    loss_vals = loss_module(subdata.to(args.device))
                    loss_value = (
                        loss_vals["loss_objective"] +
                        loss_vals["loss_critic"] +
                        loss_vals["loss_entropy"]
                    )
                    
                    loss_value.backward()
                    torch.nn.utils.clip_grad_norm_(loss_module.parameters(), args.max_grad_norm)
                    optimizer.step()
                    optimizer.zero_grad()
            
            # Record episode metrics
            episode_reward = tensordict_data["next", "reward"].mean().item()
            worker.episode_rewards.append(episode_reward)
            worker.total_episodes += 1
            
            # Check for model updates periodically
            if worker.should_sync():
                if worker.check_for_model_update():
                    worker.load_best_model(policy_module, value_module)
            
            # Save checkpoint every max_episodes
            if worker.total_episodes % worker.max_episodes == 0:
                worker.save_checkpoint(policy_module, value_module, optimizer)
                print(f"Worker {args.worker_id}: Completed {worker.total_episodes} episodes")
            
            # Progress update
            if worker.total_episodes % 10 == 0:
                avg_reward = np.mean(worker.episode_rewards[-10:])
                print(f"Worker {args.worker_id}: Episode {worker.total_episodes}, Avg Reward: {avg_reward:.4f}")
                
    except KeyboardInterrupt:
        print(f"Worker {args.worker_id}: Training interrupted")
    except Exception as e:
        print(f"Worker {args.worker_id}: Error during training: {e}")
    finally:
        # Final checkpoint
        worker.save_checkpoint(policy_module, value_module, optimizer)
        print(f"Worker {args.worker_id}: Training completed - Total episodes: {worker.total_episodes}")

def main():
    parser = argparse.ArgumentParser(description='Distributed RL Worker')
    
    # Worker-specific args
    parser.add_argument('--worker_id', type=int, required=True, help='Worker ID (0, 1, 2, 3)')
    parser.add_argument('--shared_dir', type=str, required=True, help='Shared directory for model exchange')
    parser.add_argument('--max_episodes', type=int, default=100, help='Episodes per checkpoint')
    parser.add_argument('--sync_interval', type=int, default=10, help='Episodes between sync checks')
    parser.add_argument('--timeout_minutes', type=int, default=30, help='Max minutes before forced sync')
    
    # Standard RL args (from your existing code)
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