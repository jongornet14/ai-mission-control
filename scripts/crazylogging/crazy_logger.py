#!/usr/bin/env python3
"""
FILENAME: logging/crazy_logger.py
Comprehensive logging class that captures EVERYTHING for RL experiments
"""

import os
import json
import time
import psutil
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from collections import defaultdict, deque
import torch
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import seaborn as sns


class CrazyLogger:
    """
    Insanely comprehensive logger that tracks everything you could possibly want
    """
    
    def __init__(self, log_dir, experiment_name="experiment", buffer_size=10000):
        self.log_dir = Path(log_dir)
        self.experiment_name = experiment_name
        self.buffer_size = buffer_size
        
        # Create directory structure
        self.setup_directories()
        
        # Initialize logging components
        self.tb_writer = SummaryWriter(log_dir=str(self.log_dir / 'tensorboard'))
        self.start_time = time.time()
        
        # Data storage
        self.metrics_buffer = defaultdict(deque)
        self.hyperparams_history = []
        self.episode_data = []
        self.step_data = []
        self.model_checkpoints = []
        self.system_metrics = defaultdict(list)
        self.custom_plots = {}
        
        # Performance tracking
        self.performance_tracker = PerformanceTracker()
        
        # Episode and step counters
        self.current_episode = 0
        self.current_step = 0
        self.global_step = 0
        
        print(f"🚀 CrazyLogger initialized: {self.log_dir}")
        print(f"📊 TensorBoard: tensorboard --logdir {self.log_dir / 'tensorboard'}")
        
    def setup_directories(self):
        """Create comprehensive directory structure"""
        directories = [
            'tensorboard',
            'metrics',
            'episodes', 
            'models',
            'plots',
            'system',
            'hyperparams',
            'analysis',
            'raw_data',
            'videos',
            'gradients',
            'activations'
        ]
        
        for directory in directories:
            (self.log_dir / directory).mkdir(parents=True, exist_ok=True)
    
    def log_step(self, **kwargs):
        """Log everything for a single step"""
        step_data = {
            'global_step': self.global_step,
            'episode': self.current_episode,
            'step': self.current_step,
            'timestamp': time.time(),
            'elapsed_time': time.time() - self.start_time,
        }
        
        # Add all provided data
        step_data.update(kwargs)
        
        # System metrics
        step_data.update(self.get_system_metrics())
        
        # Store in buffer
        self.step_data.append(step_data)
        
        # Log to tensorboard
        for key, value in kwargs.items():
            if isinstance(value, (int, float, np.number)):
                self.tb_writer.add_scalar(f'step/{key}', value, self.global_step)
        
        self.global_step += 1
        self.current_step += 1
        
    def log_episode(self, **kwargs):
        """Log everything for a complete episode"""
        episode_data = {
            'episode': self.current_episode,
            'global_step': self.global_step,
            'timestamp': time.time(),
            'elapsed_time': time.time() - self.start_time,
            'episode_steps': self.current_step,
        }
        
        # Add all provided data
        episode_data.update(kwargs)
        
        # Calculate episode statistics from step data
        if self.step_data:
            recent_steps = [s for s in self.step_data if s['episode'] == self.current_episode]
            if recent_steps:
                episode_data.update(self.calculate_episode_stats(recent_steps))
        
        # Store episode data
        self.episode_data.append(episode_data)
        
        # Log to tensorboard
        for key, value in kwargs.items():
            if isinstance(value, (int, float, np.number)):
                self.tb_writer.add_scalar(f'episode/{key}', value, self.current_episode)
        
        # Save periodic data
        if self.current_episode % 100 == 0:
            self.save_periodic_data()
        
        self.current_episode += 1
        self.current_step = 0
        
    def log_hyperparameters(self, hyperparams, performance_metric=None):
        """Log hyperparameter changes"""
        hyperparam_entry = {
            'episode': self.current_episode,
            'global_step': self.global_step,
            'timestamp': time.time(),
            'hyperparams': hyperparams.copy(),
            'performance_metric': performance_metric
        }
        
        self.hyperparams_history.append(hyperparam_entry)
        
        # Log to tensorboard
        for key, value in hyperparams.items():
            if isinstance(value, (int, float, np.number)):
                self.tb_writer.add_scalar(f'hyperparams/{key}', value, self.current_episode)
        
        # Save hyperparameter history
        self.save_hyperparams()
        
    def log_model_state(self, model, optimizer=None, loss=None, save_weights=True):
        """Log complete model state"""
        model_data = {
            'episode': self.current_episode,
            'global_step': self.global_step,
            'timestamp': time.time(),
            'loss': loss,
        }
        
        if save_weights:
            # Save model weights
            model_path = self.log_dir / 'models' / f'model_episode_{self.current_episode}.pt'
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict() if optimizer else None,
                'episode': self.current_episode,
                'global_step': self.global_step,
                'loss': loss,
                'timestamp': time.time()
            }, model_path)
            
            model_data['model_path'] = str(model_path)
        
        # Log gradient norms
        if hasattr(model, 'parameters'):
            grad_norms = self.calculate_gradient_norms(model)
            model_data.update(grad_norms)
            
            # Log to tensorboard
            for key, value in grad_norms.items():
                self.tb_writer.add_scalar(f'gradients/{key}', value, self.global_step)
        
        self.model_checkpoints.append(model_data)
        
    def log_activations(self, model, input_data, layer_names=None):
        """Log model activations"""
        activations = {}
        
        def hook_fn(name):
            def hook(module, input, output):
                if isinstance(output, torch.Tensor):
                    activations[name] = output.detach().cpu().numpy()
            return hook
        
        # Register hooks
        hooks = []
        for name, module in model.named_modules():
            if layer_names is None or name in layer_names:
                hook = module.register_forward_hook(hook_fn(name))
                hooks.append(hook)
        
        # Forward pass to capture activations
        with torch.no_grad():
            _ = model(input_data)
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        # Save activations
        activation_path = self.log_dir / 'activations' / f'activations_step_{self.global_step}.pkl'
        with open(activation_path, 'wb') as f:
            pickle.dump(activations, f)
        
        # Log activation statistics
        for name, activation in activations.items():
            self.tb_writer.add_histogram(f'activations/{name}', activation, self.global_step)
            self.tb_writer.add_scalar(f'activations/{name}_mean', np.mean(activation), self.global_step)
            self.tb_writer.add_scalar(f'activations/{name}_std', np.std(activation), self.global_step)
    
    def log_custom_plot(self, name, figure, step=None):
        """Log custom matplotlib plots"""
        step = step or self.global_step
        
        # Save plot
        plot_path = self.log_dir / 'plots' / f'{name}_step_{step}.png'
        figure.savefig(plot_path, dpi=150, bbox_inches='tight')
        
        # Log to tensorboard
        self.tb_writer.add_figure(name, figure, step)
        
        # Store plot info
        self.custom_plots[f'{name}_{step}'] = str(plot_path)
        
    def log_video(self, frames, name="episode_video", fps=30):
        """Log video of episode"""
        try:
            import imageio
            
            video_path = self.log_dir / 'videos' / f'{name}_episode_{self.current_episode}.mp4'
            
            # Convert frames to proper format
            if isinstance(frames[0], torch.Tensor):
                frames = [frame.cpu().numpy() for frame in frames]
            
            # Save video
            imageio.mimsave(str(video_path), frames, fps=fps)
            
            print(f"📹 Video saved: {video_path}")
            
        except ImportError:
            print("⚠️  Install imageio to save videos: pip install imageio[ffmpeg]")
    
    def get_system_metrics(self):
        """Get current system performance metrics"""
        try:
            cpu_percent = psutil.cpu_percent()
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            system_metrics = {
                'system_cpu_percent': cpu_percent,
                'system_memory_percent': memory.percent,
                'system_memory_available_gb': memory.available / (1024**3),
                'system_disk_free_gb': disk.free / (1024**3),
            }
            
            # GPU metrics if available
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.memory_allocated() / (1024**3)
                gpu_memory_max = torch.cuda.max_memory_allocated() / (1024**3)
                system_metrics.update({
                    'gpu_memory_gb': gpu_memory,
                    'gpu_memory_max_gb': gpu_memory_max,
                })
            
            return system_metrics
            
        except Exception as e:
            return {'system_metrics_error': str(e)}
    
    def calculate_gradient_norms(self, model):
        """Calculate gradient norms for all parameters"""
        total_norm = 0
        param_count = 0
        layer_norms = {}
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
                param_count += 1
                
                # Layer-wise norms
                layer_name = name.split('.')[0] if '.' in name else name
                if layer_name not in layer_norms:
                    layer_norms[layer_name] = []
                layer_norms[layer_name].append(param_norm.item())
        
        gradient_metrics = {
            'grad_norm_total': total_norm ** 0.5,
            'grad_norm_avg': (total_norm / max(param_count, 1)) ** 0.5,
        }
        
        # Add layer-wise gradient norms
        for layer_name, norms in layer_norms.items():
            gradient_metrics[f'grad_norm_{layer_name}'] = np.mean(norms)
        
        return gradient_metrics
    
    def calculate_episode_stats(self, episode_steps):
        """Calculate comprehensive episode statistics"""
        if not episode_steps:
            return {}
        
        # Extract numeric data
        numeric_data = defaultdict(list)
        for step in episode_steps:
            for key, value in step.items():
                if isinstance(value, (int, float, np.number)) and not key.startswith('system_'):
                    numeric_data[key].append(value)
        
        # Calculate statistics
        stats = {}
        for key, values in numeric_data.items():
            if len(values) > 0:
                stats.update({
                    f'{key}_mean': np.mean(values),
                    f'{key}_std': np.std(values),
                    f'{key}_min': np.min(values),
                    f'{key}_max': np.max(values),
                    f'{key}_final': values[-1],
                })
        
        return stats
    
    def save_periodic_data(self):
        """Save data periodically"""
        # Save episode data
        if self.episode_data:
            episode_df = pd.DataFrame(self.episode_data)
            episode_df.to_csv(self.log_dir / 'metrics' / 'episodes.csv', index=False)
        
        # Save step data (last N steps to avoid huge files)
        if self.step_data:
            recent_steps = self.step_data[-10000:]  # Keep last 10k steps
            step_df = pd.DataFrame(recent_steps)
            step_df.to_csv(self.log_dir / 'metrics' / 'recent_steps.csv', index=False)
        
        # Save system metrics
        self.save_system_metrics()
        
    def save_hyperparams(self):
        """Save hyperparameter history"""
        if self.hyperparams_history:
            with open(self.log_dir / 'hyperparams' / 'hyperparams_history.json', 'w') as f:
                json.dump(self.hyperparams_history, f, indent=2, default=str)
    
    def save_system_metrics(self):
        """Save system performance metrics"""
        system_data = []
        for step_data in self.step_data[-1000:]:  # Last 1000 steps
            system_entry = {k: v for k, v in step_data.items() if k.startswith('system_')}
            if system_entry:
                system_entry['timestamp'] = step_data['timestamp']
                system_entry['global_step'] = step_data['global_step']
                system_data.append(system_entry)
        
        if system_data:
            system_df = pd.DataFrame(system_data)
            system_df.to_csv(self.log_dir / 'system' / 'system_metrics.csv', index=False)
    
    def create_analysis_plots(self):
        """Create comprehensive analysis plots"""
        if not self.episode_data:
            return
        
        df = pd.DataFrame(self.episode_data)
        
        # Reward progression
        if 'reward' in df.columns:
            plt.figure(figsize=(12, 4))
            
            plt.subplot(1, 2, 1)
            plt.plot(df['episode'], df['reward'], alpha=0.7)
            if len(df) > 10:
                # Moving average
                window = min(50, len(df) // 10)
                df['reward_ma'] = df['reward'].rolling(window=window).mean()
                plt.plot(df['episode'], df['reward_ma'], 'r-', linewidth=2, label=f'MA({window})')
                plt.legend()
            plt.xlabel('Episode')
            plt.ylabel('Reward')
            plt.title('Reward Progression')
            plt.grid(True, alpha=0.3)
            
            plt.subplot(1, 2, 2)
            plt.hist(df['reward'], bins=50, alpha=0.7, edgecolor='black')
            plt.xlabel('Reward')
            plt.ylabel('Frequency')
            plt.title('Reward Distribution')
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plot_path = self.log_dir / 'analysis' / 'reward_analysis.png'
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
        
        # Hyperparameter evolution
        if self.hyperparams_history:
            self.plot_hyperparameter_evolution()
        
        print(f"📊 Analysis plots saved to: {self.log_dir / 'analysis'}")
    
    def plot_hyperparameter_evolution(self):
        """Plot how hyperparameters evolved over time"""
        if not self.hyperparams_history:
            return
        
        # Extract hyperparameter data
        hyperparam_data = defaultdict(list)
        episodes = []
        
        for entry in self.hyperparams_history:
            episodes.append(entry['episode'])
            for key, value in entry['hyperparams'].items():
                if isinstance(value, (int, float, np.number)):
                    hyperparam_data[key].append(value)
                else:
                    hyperparam_data[key].append(np.nan)
        
        # Create plots
        n_params = len(hyperparam_data)
        if n_params == 0:
            return
        
        cols = min(3, n_params)
        rows = (n_params + cols - 1) // cols
        
        plt.figure(figsize=(5 * cols, 4 * rows))
        
        for i, (param_name, values) in enumerate(hyperparam_data.items()):
            plt.subplot(rows, cols, i + 1)
            valid_episodes = [e for e, v in zip(episodes, values) if not np.isnan(v)]
            valid_values = [v for v in values if not np.isnan(v)]
            
            if valid_values:
                plt.plot(valid_episodes, valid_values, 'o-', alpha=0.7)
                plt.xlabel('Episode')
                plt.ylabel(param_name)
                plt.title(f'{param_name} Evolution')
                plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = self.log_dir / 'analysis' / 'hyperparameter_evolution.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def generate_final_report(self):
        """Generate comprehensive final report"""
        self.save_periodic_data()
        self.create_analysis_plots()
        
        # Generate summary statistics
        summary = {
            'experiment_name': self.experiment_name,
            'total_episodes': self.current_episode,
            'total_steps': self.global_step,
            'total_time': time.time() - self.start_time,
            'log_directory': str(self.log_dir),
        }
        
        if self.episode_data:
            df = pd.DataFrame(self.episode_data)
            if 'reward' in df.columns:
                summary.update({
                    'final_reward': df['reward'].iloc[-1],
                    'max_reward': df['reward'].max(),
                    'mean_reward': df['reward'].mean(),
                    'reward_std': df['reward'].std(),
                })
        
        # Save summary
        with open(self.log_dir / 'experiment_summary.json', 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        print(f"\n🎯 Final Report Generated!")
        print(f"📊 Summary: {self.log_dir / 'experiment_summary.json'}")
        print(f"📈 Analysis: {self.log_dir / 'analysis'}")
        print(f"📋 Data: {self.log_dir / 'metrics'}")
        
        return summary
    
    def close(self):
        """Close logger and clean up"""
        self.tb_writer.close()
        self.generate_final_report()
        print(f"🔒 CrazyLogger closed. Data saved to: {self.log_dir}")


class PerformanceTracker:
    """Track code performance and timing"""
    
    def __init__(self):
        self.timers = defaultdict(list)
        self.active_timers = {}
    
    def start_timer(self, name):
        """Start timing an operation"""
        self.active_timers[name] = time.time()
    
    def end_timer(self, name):
        """End timing an operation"""
        if name in self.active_timers:
            duration = time.time() - self.active_timers[name]
            self.timers[name].append(duration)
            del self.active_timers[name]
            return duration
        return 0
    
    def get_stats(self):
        """Get timing statistics"""
        stats = {}
        for name, times in self.timers.items():
            if times:
                stats[f'{name}_mean'] = np.mean(times)
                stats[f'{name}_std'] = np.std(times)
                stats[f'{name}_total'] = np.sum(times)
        return stats