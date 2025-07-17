#!/usr/bin/env python3
"""
Enhanced Distributed RL Worker Script
Clean OOP design with all training logic in the DistributedWorker class
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

# Import your RL components and CrazyLogger
from algorithms.ppo import PPOAlgorithm
from environments.gym_wrapper import GymEnvironmentWrapper
from crazylogging.crazy_logger import CrazyLogger


class DistributedWorker:
    def __init__(self, worker_id, shared_dir, max_episodes=100, sync_interval=10, timeout_minutes=30):
        """
        Enhanced Distributed RL Worker with CrazyLogger
        
        Args:
            worker_id: Unique worker identifier (0-19)
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
        self.episode_lengths = []
        self.training_metrics = defaultdict(list)
        
        # Initialize CrazyLogger for comprehensive logging
        self.log_dir = self.shared_dir / "worker_logs" / f"worker_{worker_id}"
        self.logger = CrazyLogger(
            log_dir=str(self.log_dir),
            experiment_name=f"distributed_worker_{worker_id}",
            buffer_size=5000  # Smaller buffer for distributed workers
        )
        
        # RL components (will be initialized in setup_rl_components)
        self.env = None
        self.algorithm = None
        self.device = None
        
        print(f"🚀 Enhanced Worker {worker_id} initialized")
        print(f"📁 Shared directory: {shared_dir}")
        print(f"📊 CrazyLogger directory: {self.log_dir}")
        print(f"📈 TensorBoard: tensorboard --logdir {self.log_dir / 'tensorboard'}")
        print(f"🎯 Max episodes per checkpoint: {max_episodes}")

    def setup_rl_components(self, env_name, device="cuda:0", lr=3e-4):
        """Initialize RL environment and algorithm"""
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        # Create environment
        env_wrapper = GymEnvironmentWrapper(
            env_name=env_name,
            device=self.device,
            normalize_observations=True,
            max_episode_steps=None
        )
        self.env = env_wrapper.create()
        
        # Create algorithm
        self.algorithm = PPOAlgorithm(
            env=self.env,
            device=self.device,
            learning_rate=lr,
            # Add other PPO hyperparameters as needed
        )
        
        # Log initial setup
        env_info = {
            'worker_id': self.worker_id,
            'environment': env_name,
            'device': str(self.device),
            'obs_space_shape': self.env.observation_space.shape,
            'action_space_type': str(type(self.env.action_space)),
        }
        
        if hasattr(self.env.action_space, 'n'):
            env_info['action_space_size'] = self.env.action_space.n
        else:
            env_info['action_space_shape'] = self.env.action_space.shape
        
        self.logger.log_step(**env_info)
        
        # Log initial hyperparameters
        hyperparams = self.algorithm.get_hyperparameters()
        hyperparams.update({
            'worker_id': self.worker_id,
            'sync_interval': self.sync_interval,
            'max_episodes': self.max_episodes,
            'distributed_training': True
        })
        self.logger.log_hyperparameters(hyperparams)
        
        print(f"🎮 Worker {self.worker_id}: Environment {env_name} created")
        print(f"🧠 Worker {self.worker_id}: PPO algorithm initialized")
        print(f"🖥️  Worker {self.worker_id}: Using device {self.device}")

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
                print(f"🔄 Worker {self.worker_id}: Model update signal received")
                self.logger.log_step(
                    model_update_signal=True,
                    signal_type="coordinator_update",
                    timestamp=time.time()
                )
                return True
                
            # Check if best model is newer than our last sync
            best_model_file = self.best_model_dir / "current_best.pt"
            if best_model_file.exists():
                model_time = best_model_file.stat().st_mtime
                if model_time > self.last_sync_time:
                    print(f"🔄 Worker {self.worker_id}: New best model available")
                    self.logger.log_step(
                        model_update_available=True,
                        model_timestamp=model_time,
                        last_sync=self.last_sync_time
                    )
                    return True
                    
        except Exception as e:
            print(f"❌ Worker {self.worker_id}: Error checking for updates: {e}")
            self.logger.log_step(error=str(e), error_type="model_update_check")
            
        return False

    def load_best_model(self):
        """Load the best model from coordinator"""
        try:
            best_model_file = self.best_model_dir / "current_best.pt"
            if best_model_file.exists():
                checkpoint = torch.load(best_model_file, map_location=self.device)
                self.algorithm.policy_net.load_state_dict(checkpoint['policy_state_dict'])
                self.algorithm.value_net.load_state_dict(checkpoint['value_state_dict'])
                
                # Remove update signal
                update_file = self.shared_dir / f"update_worker_{self.worker_id}.signal"
                if update_file.exists():
                    update_file.unlink()
                
                self.last_sync_time = time.time()
                print(f"✅ Worker {self.worker_id}: Loaded best model")
                
                # Log model update to CrazyLogger
                self.logger.log_step(
                    model_update_success=True,
                    model_source="coordinator_best_model",
                    sync_time=self.last_sync_time,
                    checkpoint_episode=checkpoint.get('total_episodes', 'unknown')
                )
                
                return True
        except Exception as e:
            print(f"❌ Worker {self.worker_id}: Error loading best model: {e}")
            self.logger.log_step(error=str(e), error_type="model_loading_error")
            
        return False

    def collect_episode(self):
        """Collect a complete episode with detailed logging"""
        self.logger.performance_tracker.start_timer('episode_collection')
        
        obs = self.env.reset()
        
        episode_data = {
            'observations': [obs],
            'actions': [],
            'rewards': [],
            'log_probs': [],
            'values': [],
            'dones': []
        }
        
        episode_reward = 0
        episode_length = 0
        step_times = []
        
        for step in range(1000):  # Max steps per episode
            self.logger.performance_tracker.start_timer('action_selection')
            
            # Get action from policy
            action, log_prob = self.algorithm.get_action(obs)
            
            action_time = self.logger.performance_tracker.end_timer('action_selection')
            
            # Take step in environment
            self.logger.performance_tracker.start_timer('env_step')
            next_obs, reward, done, info = self.env.step(action)
            env_step_time = self.logger.performance_tracker.end_timer('env_step')
            
            # Store data
            episode_data['actions'].append(action)
            episode_data['rewards'].append(reward)
            episode_data['log_probs'].append(log_prob)
            episode_data['observations'].append(next_obs)
            episode_data['dones'].append(done)

            # Log step metrics
            obs_stats = self.tensor_stats(obs)
            step_metrics = {
                'step_reward': self.to_scalar(reward),
                'step_action': self.to_scalar(action),
                'step_log_prob': log_prob,
                'step_obs_mean': obs_stats['mean'],
                'step_obs_std': obs_stats['std'],
                'step_obs_max': obs_stats['max'],
                'step_obs_min': obs_stats['min'],
                'action_selection_time': action_time,
                'env_step_time': env_step_time,
                'cumulative_reward': episode_reward + self.to_scalar(reward),
                'worker_id': self.worker_id
            }
            
            self.logger.log_step(**step_metrics)
            step_times.append(action_time + env_step_time)
            
            episode_reward += self.to_scalar(reward)
            episode_length += 1
            obs = next_obs
            
            if done:
                break
        
        episode_collection_time = self.logger.performance_tracker.end_timer('episode_collection')
        
        return episode_data, episode_reward, episode_length, episode_collection_time, step_times

    def train_on_episode(self, episode_data):
        """Train algorithm on episode data with detailed logging"""
        self.logger.performance_tracker.start_timer('training')
        
        # Convert episode data to format for training
        rollout = []
        for i in range(len(episode_data['rewards'])):
            rollout.append((
                episode_data['observations'][i],
                episode_data['actions'][i],
                episode_data['rewards'][i],
                episode_data['observations'][i+1] if i+1 < len(episode_data['observations']) else episode_data['observations'][i],
                episode_data['dones'][i]
            ))
        
        # Train algorithm
        train_metrics = self.algorithm.train_step(rollout)
        
        training_time = self.logger.performance_tracker.end_timer('training')
        train_metrics['training_time'] = training_time
        train_metrics['worker_id'] = self.worker_id
        
        # Log model state periodically
        if self.total_episodes % 20 == 0:
            self.logger.log_model_state(
                self.algorithm.policy_net, 
                self.algorithm.value_optimizer,
                train_metrics.get('loss_total', 0),
                save_weights=True
            )
        
        # Log activations periodically
        if self.total_episodes % 50 == 0:
            sample_obs = episode_data['observations'][0].unsqueeze(0).to(self.device)
            self.logger.log_activations(self.algorithm.policy_net, sample_obs)
        
        return train_metrics

    def save_checkpoint(self):
        """Save current model and metrics"""
        try:
            # Save model
            checkpoint = {
                'policy_state_dict': self.algorithm.policy_net.state_dict(),
                'value_state_dict': self.algorithm.value_net.state_dict(),
                'policy_optimizer_state_dict': self.algorithm.policy_optimizer.state_dict(),
                'value_optimizer_state_dict': self.algorithm.value_optimizer.state_dict(),
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
            
            print(f"💾 Worker {self.worker_id}: Checkpoint saved - Episodes: {self.total_episodes}, Avg Reward: {avg_reward:.4f}, Change: {reward_change:.4f}")
            
            # Log checkpoint event
            self.logger.log_step(
                checkpoint_saved=True,
                checkpoint_episode=self.total_episodes,
                avg_reward=avg_reward,
                reward_change=reward_change,
                model_file=str(model_file)
            )
            
            return True
            
        except Exception as e:
            print(f"❌ Worker {self.worker_id}: Error saving checkpoint: {e}")
            self.logger.log_step(error=str(e), error_type="checkpoint_save_error")
            return False

    def calculate_reward_change(self):
        """Calculate average change in reward (performance metric)"""
        if len(self.episode_rewards) < 20:
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
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # Episode rewards with distributed context
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
            
            # Reward change over time (distributed performance metric)
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
            
            # Episode length analysis
            if self.episode_lengths:
                axes[1, 0].plot(self.episode_lengths, alpha=0.7, color='purple')
                axes[1, 0].set_xlabel('Episode')
                axes[1, 0].set_ylabel('Episode Length')
                axes[1, 0].set_title(f'Worker {self.worker_id} - Episode Lengths')
                axes[1, 0].grid(True, alpha=0.3)
            
            # Distributed training statistics
            training_time_hours = (time.time() - self.start_time) / 3600
            episodes_per_hour = self.total_episodes / training_time_hours if training_time_hours > 0 else 0
            sync_count = self.total_episodes // self.sync_interval
            
            axes[1, 1].text(0.1, 0.9, f'Worker ID: {self.worker_id}', transform=axes[1, 1].transAxes, fontsize=14, weight='bold')
            axes[1, 1].text(0.1, 0.8, f'Total Episodes: {self.total_episodes}', transform=axes[1, 1].transAxes, fontsize=12)
            axes[1, 1].text(0.1, 0.7, f'Training Time: {training_time_hours:.2f}h', transform=axes[1, 1].transAxes, fontsize=12)
            axes[1, 1].text(0.1, 0.6, f'Episodes/Hour: {episodes_per_hour:.1f}', transform=axes[1, 1].transAxes, fontsize=12)
            axes[1, 1].text(0.1, 0.5, f'Best Reward: {max(self.episode_rewards):.2f}', transform=axes[1, 1].transAxes, fontsize=12)
            axes[1, 1].text(0.1, 0.4, f'Current Trend: {self.calculate_reward_change():.3f}', transform=axes[1, 1].transAxes, fontsize=12)
            axes[1, 1].text(0.1, 0.3, f'Model Syncs: ~{sync_count}', transform=axes[1, 1].transAxes, fontsize=12)
            axes[1, 1].text(0.1, 0.2, f'Last Sync: {(time.time() - self.last_sync_time)/60:.1f}m ago', transform=axes[1, 1].transAxes, fontsize=12)
            axes[1, 1].set_title(f'Worker {self.worker_id} - Distributed Stats')
            axes[1, 1].axis('off')
            
            plt.tight_layout()
            
            # Log the plot using CrazyLogger
            self.logger.log_custom_plot(f'distributed_worker_{self.worker_id}_analysis', fig, self.total_episodes)
            
            # Save plot to shared directory as well
            plot_path = self.shared_dir / f"worker_{self.worker_id}_analysis.png"
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"❌ Worker {self.worker_id}: Error creating analysis plots: {e}")
            self.logger.log_step(error=str(e), error_type="plot_creation_error")

    def should_terminate(self):
        """Check if worker should terminate"""
        terminate_file = self.shared_dir / f"terminate_worker_{self.worker_id}.signal"
        if terminate_file.exists():
            print(f"🛑 Worker {self.worker_id}: Termination signal received")
            return True
            
        # Global termination signal
        global_terminate = self.shared_dir / "terminate_all.signal"
        if global_terminate.exists():
            print(f"🛑 Worker {self.worker_id}: Global termination signal received")
            return True
            
        return False

    def to_scalar(self, x):
        """Convert tensor to scalar safely"""
        if torch.is_tensor(x):
            return x.item() if x.dim() == 0 else x.mean().item()
        else:
            return float(x)

    def tensor_stats(self, x):
        """Get stats from tensor safely"""
        if torch.is_tensor(x):
            return {
                'mean': x.mean().item(),
                'std': x.std().item(),
                'max': x.max().item(),
                'min': x.min().item()
            }
        else:
            x_np = np.array(x)
            return {
                'mean': np.mean(x_np),
                'std': np.std(x_np),
                'max': np.max(x_np),
                'min': np.min(x_np)
            }

    def run(self, args):
        """
        Main training loop - all distributed RL training logic
        This is the ONLY run method that does actual training
        """
        print(f"🚀 Worker {self.worker_id}: Starting REAL RL training with CrazyLogger")
        
        # Setup RL components
        self.setup_rl_components(
            env_name=args.env,
            device=args.device,
            lr=args.lr
        )
        
        try:
            from tqdm import tqdm
            pbar = tqdm(range(args.max_episodes), desc=f"Worker {self.worker_id}")
            
            for episode in pbar:
                if self.should_terminate():
                    break
                
                # Collect episode with full logging
                episode_data, episode_reward, episode_length, collection_time, step_times = self.collect_episode()
                
                # Train on episode
                train_metrics = self.train_on_episode(episode_data)
                
                # Store episode stats
                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(episode_length)
                self.total_episodes += 1
                
                # Update progress bar
                pbar.set_postfix({
                    'Reward': f'{episode_reward:.2f}',
                    'Length': episode_length,
                    'Trend': f'{self.calculate_reward_change():.3f}'
                })
                
                # Comprehensive episode logging
                episode_metrics = {
                    'episode_reward': episode_reward,
                    'episode_length': episode_length,
                    'episode_collection_time': collection_time,
                    'avg_step_time': np.mean(step_times) if step_times else 0,
                    'total_episodes': self.total_episodes,
                    'worker_id': self.worker_id,
                    'reward_change': self.calculate_reward_change(),
                    'avg_reward_last_100': np.mean(self.episode_rewards[-100:]) if len(self.episode_rewards) >= 100 else np.mean(self.episode_rewards),
                    'distributed_training': True,
                    'sync_due': self.should_sync()
                }
                
                # Combine episode and training metrics
                all_metrics = {**episode_metrics, **train_metrics}
                
                # Add performance tracker stats
                perf_stats = self.logger.performance_tracker.get_stats()
                all_metrics.update(perf_stats)
                
                # Log complete episode
                self.logger.log_episode(**all_metrics)
                
                # Check for model updates periodically
                if self.should_sync():
                    if self.check_for_model_update():
                        if self.load_best_model():
                            print(f"🔄 Worker {self.worker_id}: Loaded updated model from coordinator")
                
                # Save checkpoint periodically
                if self.total_episodes % 25 == 0:
                    self.save_checkpoint()
                    self.create_distributed_analysis_plots()
                    print(f"📊 Worker {self.worker_id}: Created analysis plots at episode {self.total_episodes}")
                
                # Progress update
                if self.total_episodes % 10 == 0:
                    avg_reward = np.mean(self.episode_rewards[-10:])
                    print(f"📈 Worker {self.worker_id}: Episode {self.total_episodes}, Avg Reward (last 10): {avg_reward:.4f}")
                    
            pbar.close()
            
        except KeyboardInterrupt:
            print(f"🛑 Worker {self.worker_id}: Training interrupted")
        except Exception as e:
            print(f"💥 Worker {self.worker_id}: Error during training: {e}")
            self.logger.log_step(error=str(e), error_type="training_exception")
            import traceback
            traceback.print_exc()
        finally:
            # Final analysis and cleanup
            self.create_distributed_analysis_plots()
            self.save_checkpoint()
            
            # Generate final report
            final_summary = self.logger.generate_final_report()
            print(f"📊 Worker {self.worker_id}: Final report generated")
            
            # Close logger
            self.logger.close()
            
            print(f"✅ Worker {self.worker_id}: Training completed - Total episodes: {self.total_episodes}")
            if self.episode_rewards:
                print(f"🏆 Worker {self.worker_id}: Best reward: {max(self.episode_rewards):.2f}")
                print(f"📈 Worker {self.worker_id}: Final average: {np.mean(self.episode_rewards[-10:]):.2f}")


def main():
    """Simplified main entry point - clean single responsibility"""
    parser = argparse.ArgumentParser(description='Enhanced Distributed RL Worker')
    
    # Worker-specific args
    parser.add_argument('--worker_id', type=int, required=True, help='Worker ID (0-19)')
    parser.add_argument('--shared_dir', type=str, required=True, help='Shared directory')
    parser.add_argument('--max_episodes', type=int, default=100, help='Episodes per checkpoint')
    parser.add_argument('--sync_interval', type=int, default=10, help='Episodes between syncs')
    parser.add_argument('--timeout_minutes', type=int, default=30, help='Max minutes before sync')
    
    # RL training args
    parser.add_argument('--env', type=str, default='CartPole-v1', help='Environment name')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    
    args = parser.parse_args()
    
    # Validate worker_id
    if args.worker_id < 0 or args.worker_id >= 20:
        raise ValueError("worker_id must be between 0 and 19")
    
    # Create worker and run training
    worker = DistributedWorker(
        worker_id=args.worker_id,
        shared_dir=args.shared_dir,
        max_episodes=args.max_episodes,
        sync_interval=args.sync_interval,
        timeout_minutes=args.timeout_minutes
    )
    
    # Single, clean call to run training
    worker.run(args)


if __name__ == "__main__":
    main()