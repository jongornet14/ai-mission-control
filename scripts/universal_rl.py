#!/usr/bin/env python3
"""
FILENAME: universal_rl.py
Universal RL Training Script with CrazyLogger integration
Simple version using stable packages + comprehensive logging
"""

import os
import sys
import yaml
import torch
import numpy as np
import argparse
from pathlib import Path
from datetime import datetime
from collections import defaultdict

# Simple imports - no TorchRL
import gym

# Local imports
from algorithms.ppo import PPOAlgorithm
from environments.gym_wrapper import GymEnvironmentWrapper
from crazylogging.crazy_logger import CrazyLogger


class UniversalRLTrainer:
    """Universal RL training class with crazy comprehensive logging"""
    
    def __init__(self, config_path=None, **kwargs):
        """Initialize trainer with config file or direct parameters"""
        self.config = self.load_config(config_path, **kwargs)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Setup experiment directory first
        self.setup_experiment_dir()
        
        # Initialize CRAZY LOGGER! 🚀
        self.logger = CrazyLogger(
            log_dir=self.exp_dir,
            experiment_name=self.config['experiment']['name']
        )
        
        # Initialize components
        self.env = None
        self.algorithm = None
        
        # Training state
        self.current_episode = 0
        self.total_frames_collected = 0
        self.best_reward = float('-inf')
        
        # Performance tracking
        self.episode_rewards = []
        self.episode_lengths = []
        
    def load_config(self, config_path=None, **kwargs):
        """Load configuration from YAML file and merge with kwargs"""
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
        else:
            config = self.get_default_config()
        
        # Override with command line arguments
        for key, value in kwargs.items():
            if value is not None:
                self.set_nested_config(config, key, value)
        
        return config
    
    def get_default_config(self):
        """Default configuration"""
        return {
            'experiment': {
                'name': 'crazy_logged_experiment',
                'save_dir': '/workspace/experiments',
                'seed': 42,
            },
            'environment': {
                'name': 'CartPole-v1',
                'normalize_observations': True,
                'max_episode_steps': 500,
            },
            'algorithm': {
                'name': 'PPO',
                'learning_rate': 3e-4,
                'gamma': 0.99,
                'lambda': 0.95,
                'clip_epsilon': 0.2,
                'entropy_coef': 1e-4,
                'num_cells': 256
            },
            'training': {
                'total_episodes': 1000,
                'max_steps_per_episode': 500,
                'eval_frequency': 50,
                'eval_episodes': 5,
                'save_frequency': 100,
                'log_frequency': 1,  # Log every episode
            },
            'logging': {
                'save_models': True,
                'save_videos': False,
                'log_activations': False,
                'activation_frequency': 100,
            }
        }
    
    def set_nested_config(self, config, key, value):
        """Set nested configuration value using dot notation"""
        keys = key.split('.')
        current = config
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]
        current[keys[-1]] = value
    
    def setup_experiment_dir(self):
        """Create experiment directory structure"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_name = f"{self.config['experiment']['name']}_{timestamp}"
        
        self.exp_dir = Path(self.config['experiment']['save_dir']) / exp_name
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"📁 Experiment directory: {self.exp_dir}")
    
    def create_environment(self):
        """Create and configure the environment"""
        env_config = self.config['environment']
        
        wrapper = GymEnvironmentWrapper(
            env_name=env_config['name'],
            normalize_observations=env_config.get('normalize_observations', True),
            max_episode_steps=env_config.get('max_episode_steps', None)
        )
        
        self.env = wrapper.create()
        
        # Log environment info
        env_info = {
            'env_name': env_config['name'],
            'obs_space_shape': self.env.observation_space.shape,
            'action_space_type': str(type(self.env.action_space)),
        }
        
        if hasattr(self.env.action_space, 'n'):
            env_info['action_space_size'] = self.env.action_space.n
        else:
            env_info['action_space_shape'] = self.env.action_space.shape
            
        self.logger.log_step(**env_info)
        
        print(f"🎮 Environment: {env_config['name']}")
        print(f"   Observation space: {self.env.observation_space}")
        print(f"   Action space: {self.env.action_space}")
    
    def create_algorithm(self):
        """Create and configure the RL algorithm"""
        algo_config = self.config['algorithm']
        
        self.algorithm = PPOAlgorithm(
            env=self.env,
            device=self.device,
            **algo_config
        )
        
        # Log initial hyperparameters
        self.logger.log_hyperparameters(self.algorithm.get_hyperparameters())
        
        print(f"🧠 Algorithm: {algo_config['name']}")
    
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
        frames = [] if self.config['logging']['save_videos'] else None
        
        for step in range(self.config['training']['max_steps_per_episode']):
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
            
            # Save frame for video
            if frames is not None:
                try:
                    frame = self.env.render(mode='rgb_array')
                    frames.append(frame)
                except:
                    pass  # Skip if rendering not available
            
            # Log step data with EVERYTHING
            step_metrics = {
                'step_reward': reward,
                'step_action': action if np.isscalar(action) else np.mean(action),
                'step_log_prob': log_prob,
                'step_obs_mean': np.mean(obs),
                'step_obs_std': np.std(obs),
                'step_obs_max': np.max(obs),
                'step_obs_min': np.min(obs),
                'action_selection_time': action_time,
                'env_step_time': env_step_time,
                'cumulative_reward': episode_reward + reward,
            }
            
            # Add info from environment if available
            if info:
                for key, value in info.items():
                    if isinstance(value, (int, float, np.number)):
                        step_metrics[f'env_info_{key}'] = value
            
            self.logger.log_step(**step_metrics)
            
            episode_reward += reward
            episode_length += 1
            obs = next_obs
            
            if done:
                break
        
        episode_collection_time = self.logger.performance_tracker.end_timer('episode_collection')
        
        # Save video if enabled
        if frames and len(frames) > 0:
            self.logger.log_video(frames, f"episode_{self.current_episode}")
        
        return episode_data, episode_reward, episode_length, episode_collection_time
    
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
        
        # Log model state periodically
        if self.current_episode % self.config['training']['save_frequency'] == 0:
            self.logger.log_model_state(
                self.algorithm.policy_net, 
                self.algorithm.optimizer,
                train_metrics.get('loss_total', 0),
                save_weights=True
            )
        
        # Log activations periodically
        if (self.config['logging']['log_activations'] and 
            self.current_episode % self.config['logging']['activation_frequency'] == 0):
            sample_obs = torch.FloatTensor(episode_data['observations'][0]).unsqueeze(0).to(self.device)
            self.logger.log_activations(self.algorithm.policy_net, sample_obs)
        
        return train_metrics
    
    def evaluate_policy(self):
        """Evaluate current policy with detailed logging"""
        self.logger.performance_tracker.start_timer('evaluation')
        
        eval_rewards = []
        eval_lengths = []
        eval_success_rate = 0
        
        for eval_ep in range(self.config['training']['eval_episodes']):
            obs = self.env.reset()
            eval_reward = 0
            eval_length = 0
            
            for step in range(self.config['training']['max_steps_per_episode']):
                action, _ = self.algorithm.get_action(obs)
                obs, reward, done, info = self.env.step(action)
                
                eval_reward += reward
                eval_length += 1
                
                if done:
                    # Check for success (environment specific)
                    if hasattr(self.env, 'spec') and self.env.spec:
                        if (hasattr(self.env.spec, 'reward_threshold') and 
                            self.env.spec.reward_threshold and 
                            eval_reward >= self.env.spec.reward_threshold):
                            eval_success_rate += 1
                    break
                    
            eval_rewards.append(eval_reward)
            eval_lengths.append(eval_length)
        
        eval_success_rate /= self.config['training']['eval_episodes']
        evaluation_time = self.logger.performance_tracker.end_timer('evaluation')
        
        eval_metrics = {
            'eval_reward_mean': np.mean(eval_rewards),
            'eval_reward_std': np.std(eval_rewards),
            'eval_reward_min': np.min(eval_rewards),
            'eval_reward_max': np.max(eval_rewards),
            'eval_length_mean': np.mean(eval_lengths),
            'eval_length_std': np.std(eval_lengths),
            'eval_success_rate': eval_success_rate,
            'evaluation_time': evaluation_time,
        }
        
        return eval_metrics
    
    def train(self):
        """Main training loop with CRAZY comprehensive logging"""
        print(f"\n🚀 Starting training with CrazyLogger...")
        print(f"Device: {self.device}")
        print("=" * 50)
        
        # Setup all components
        torch.manual_seed(self.config['experiment']['seed'])
        self.create_environment()
        self.create_algorithm()
        
        training_config = self.config['training']
        
        # Log initial configuration
        self.logger.log_step(**{
            'config_seed': self.config['experiment']['seed'],
            'config_total_episodes': training_config['total_episodes'],
            'config_device': str(self.device),
        })
        
        # Main training loop
        for episode in range(training_config['total_episodes']):
            self.current_episode = episode
            
            # Collect episode with full logging
            episode_data, episode_reward, episode_length, collection_time = self.collect_episode()
            
            # Train on episode
            train_metrics = self.train_on_episode(episode_data)
            
            # Store episode stats
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_length)
            
            # Evaluation
            eval_metrics = {}
            if episode % training_config['eval_frequency'] == 0:
                eval_metrics = self.evaluate_policy()
                
                print(f"Episode {episode:4d} | "
                      f"Reward: {eval_metrics['eval_reward_mean']:7.2f} ± {eval_metrics['eval_reward_std']:5.2f} | "
                      f"Length: {eval_metrics['eval_length_mean']:6.1f} | "
                      f"Success: {eval_metrics['eval_success_rate']:5.2f}")
                
                # Update best reward
                current_reward = eval_metrics['eval_reward_mean']
                if current_reward > self.best_reward:
                    self.best_reward = current_reward
                    print(f"🏆 New best reward: {current_reward:.2f}")
            
            # Comprehensive episode logging
            episode_metrics = {
                'episode_reward': episode_reward,
                'episode_length': episode_length,
                'episode_collection_time': collection_time,
                'best_reward_so_far': self.best_reward,
                'avg_reward_last_100': np.mean(self.episode_rewards[-100:]) if len(self.episode_rewards) >= 100 else np.mean(self.episode_rewards),
                'reward_trend': episode_reward - np.mean(self.episode_rewards[-10:]) if len(self.episode_rewards) >= 10 else 0,
            }
            
            # Combine all metrics
            all_metrics = {**episode_metrics, **train_metrics, **eval_metrics}
            
            # Performance tracker stats
            perf_stats = self.logger.performance_tracker.get_stats()
            all_metrics.update(perf_stats)
            
            # Log complete episode
            self.logger.log_episode(**all_metrics)
            
            # Log hyperparameter updates if any
            current_hyperparams = self.algorithm.get_hyperparameters()
            self.logger.log_hyperparameters(current_hyperparams, episode_reward)
            
            # Create custom plots periodically
            if episode % 200 == 0 and episode > 0:
                self.create_custom_plots()
        
        # Final evaluation and report
        final_eval = self.evaluate_policy()
        print(f"\n✅ Training completed!")
        print(f"🏆 Best reward: {self.best_reward:.2f}")
        print(f"🎯 Final reward: {final_eval['eval_reward_mean']:.2f} ± {final_eval['eval_reward_std']:.2f}")
        
        # Generate final comprehensive report
        final_summary = self.logger.generate_final_report()
        
        # Close logger
        self.logger.close()
        
        return final_summary
    
    def create_custom_plots(self):
        """Create custom analysis plots"""
        import matplotlib.pyplot as plt
        
        if len(self.episode_rewards) < 10:
            return
        
        # Reward progression plot
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Episode rewards
        axes[0, 0].plot(self.episode_rewards, alpha=0.6, label='Episode Reward')
        if len(self.episode_rewards) > 20:
            window = min(50, len(self.episode_rewards) // 5)
            moving_avg = np.convolve(self.episode_rewards, np.ones(window)/window, mode='valid')
            axes[0, 0].plot(range(window-1, len(self.episode_rewards)), moving_avg, 'r-', linewidth=2, label=f'MA({window})')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Reward')
        axes[0, 0].set_title('Training Progress')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Episode lengths
        axes[0, 1].plot(self.episode_lengths, alpha=0.6, color='green')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Episode Length')
        axes[0, 1].set_title('Episode Lengths')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Reward distribution
        axes[1, 0].hist(self.episode_rewards, bins=30, alpha=0.7, edgecolor='black')
        axes[1, 0].axvline(np.mean(self.episode_rewards), color='red', linestyle='--', label=f'Mean: {np.mean(self.episode_rewards):.2f}')
        axes[1, 0].set_xlabel('Reward')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Reward Distribution')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Recent performance (last 100 episodes)
        recent_rewards = self.episode_rewards[-100:] if len(self.episode_rewards) > 100 else self.episode_rewards
        axes[1, 1].plot(recent_rewards, alpha=0.7, color='orange')
        axes[1, 1].set_xlabel('Recent Episodes')
        axes[1, 1].set_ylabel('Reward')
        axes[1, 1].set_title(f'Recent Performance (Last {len(recent_rewards)} episodes)')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Log the plot
        self.logger.log_custom_plot('training_progress', fig, self.current_episode)
        plt.close()


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Universal RL Training with CrazyLogger')
    
    # Configuration
    parser.add_argument('--config', type=str, help='Path to YAML config file')
    
    # Experiment settings
    parser.add_argument('--experiment.name', type=str, help='Experiment name')
    parser.add_argument('--experiment.seed', type=int, help='Random seed')
    
    # Environment settings
    parser.add_argument('--environment.name', type=str, help='Environment name')
    
    # Algorithm settings  
    parser.add_argument('--algorithm.learning_rate', type=float, help='Learning rate')
    
    # Training settings
    parser.add_argument('--training.total_episodes', type=int, help='Total training episodes')
    
    # Logging settings
    parser.add_argument('--logging.save_videos', action='store_true', help='Save episode videos')
    parser.add_argument('--logging.log_activations', action='store_true', help='Log neural network activations')
    
    args = parser.parse_args()
    
    # Convert args to kwargs, filtering None values
    kwargs = {k: v for k, v in vars(args).items() if v is not None and k != 'config'}
    
    # Create and run trainer
    trainer = UniversalRLTrainer(config_path=args.config, **kwargs)
    summary = trainer.train()
    
    print(f"\n🎉 Experiment completed!")
    print(f"📊 Full results: {summary}")


if __name__ == '__main__':
    main()
