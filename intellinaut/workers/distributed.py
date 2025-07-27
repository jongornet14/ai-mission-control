#!/usr/bin/env python3
"""
Simple Distributed Worker - BaseWorker + Minimal Distributed Features
Just adds shared directory coordination to existing BaseWorker
"""

import torch
import json
import shutil
from pathlib import Path
from datetime import datetime

# Import your existing BaseWorker
from .base import BaseWorker


class SimpleDistributedWorker(BaseWorker):
    """
    BaseWorker + minimal distributed coordination
    
    Adds only:
    - Save models/metrics to shared directory  
    - Check for coordinator signals
    - Load new models when signaled
    """
    
    def __init__(self, worker_id, shared_dir, check_interval=10, **base_kwargs):
        """
        Args:
            worker_id: Worker ID (0, 1, 2, ...)
            shared_dir: Shared directory path for coordination
            check_interval: Episodes between checking for updates
            **base_kwargs: All BaseWorker arguments (log_dir, max_episodes, etc.)
        """
        # Initialize BaseWorker normally
        super().__init__(worker_id=worker_id, **base_kwargs)
        
        # Add minimal distributed state
        self.shared_dir = Path(shared_dir)
        self.check_interval = check_interval
        self.last_check_episode = 0
        
        # Create shared directories
        self.shared_models = self.shared_dir / "models"
        self.shared_metrics = self.shared_dir / "metrics"
        self.shared_best = self.shared_dir / "best_model"
        self.shared_signals = self.shared_dir / "signals"
        
        for d in [self.shared_models, self.shared_metrics, self.shared_best, self.shared_signals]:
            d.mkdir(parents=True, exist_ok=True)
        
        print(f"🌐 SimpleDistributedWorker {worker_id} - shared_dir: {shared_dir}")
    
    def save_to_shared_directory(self):
        """Save current model and metrics to shared directory for coordinator"""
        try:
            # Save model (copy from local to shared)
            local_model = self.models_dir / f"worker_{self.worker_id}_episode_{self.current_episode}.pt"
            shared_model = self.shared_models / f"worker_{self.worker_id}_episode_{self.current_episode}.pt"
            
            if local_model.exists():
                shutil.copy2(local_model, shared_model)
            
            # Save metrics for coordinator
            metrics = {
                'worker_id': self.worker_id,
                'total_episodes': self.total_episodes,
                'current_episode': self.current_episode,
                'avg_reward': self._get_avg_reward(min(50, len(self.episode_rewards))),
                'reward_change': self._get_reward_trend(),
                'best_reward': self.best_reward,
                'timestamp': datetime.now().isoformat()
            }
            
            shared_metrics_file = self.shared_metrics / f"worker_{self.worker_id}_performance.json"
            with open(shared_metrics_file, 'w') as f:
                json.dump(metrics, f, indent=2)
            
            print(f"📤 Worker {self.worker_id}: Saved to shared directory")
            return True
            
        except Exception as e:
            print(f"❌ Worker {self.worker_id}: Error saving to shared: {e}")
            return False
    
    def check_for_coordinator_signal(self):
        """Check if coordinator wants us to update our model"""
        try:
            signal_file = self.shared_signals / f"update_worker_{self.worker_id}.signal"
            return signal_file.exists()
        except Exception as e:
            print(f"❌ Worker {self.worker_id}: Error checking signal: {e}")
            return False
    
    def load_best_model_from_coordinator(self):
        """Load the best model provided by coordinator"""
        try:
            best_model_file = self.shared_best / "current_best.pt"
            
            if not best_model_file.exists():
                print(f"⚠️ Worker {self.worker_id}: No best model available")
                return False
            
            # Load the model
            checkpoint = torch.load(best_model_file, map_location=self.device)
            
            # Update our networks
            self.algorithm.policy_net.load_state_dict(checkpoint['policy_state_dict'])
            self.algorithm.value_net.load_state_dict(checkpoint['value_state_dict'])
            
            # Remove the signal file
            signal_file = self.shared_signals / f"update_worker_{self.worker_id}.signal"
            if signal_file.exists():
                signal_file.unlink()
            
            print(f"✅ Worker {self.worker_id}: Loaded best model from coordinator")
            
            # Log the update
            update_log = {
                "timestamp": datetime.now().isoformat(),
                "event_type": "model_update",
                "worker_id": self.worker_id,
                "episode": self.current_episode
            }
            self.json_buffer.append(update_log)
            
            return True
            
        except Exception as e:
            print(f"❌ Worker {self.worker_id}: Error loading best model: {e}")
            return False
    
    def should_check_coordinator(self):
        """Check if it's time to check for coordinator updates"""
        episodes_since_check = self.current_episode - self.last_check_episode
        return episodes_since_check >= self.check_interval
    
    def distributed_episode_hook(self):
        """Called after each episode - handles distributed coordination"""
        # Save to shared directory periodically
        if self.current_episode % 25 == 0:  # Every 25 episodes
            self.save_to_shared_directory()
        
        # Check for coordinator updates
        if self.should_check_coordinator():
            self.last_check_episode = self.current_episode
            
            if self.check_for_coordinator_signal():
                print(f"🔄 Worker {self.worker_id}: Coordinator signal received")
                self.load_best_model_from_coordinator()
    
    def run(self, env_name: str, device: str = "cuda:0", lr: float = 3e-4):
        """
        Training loop with distributed coordination
        (Identical to BaseWorker.run() but with distributed hook added)
        """
        print(f"🌐 SimpleDistributedWorker {self.worker_id}: Starting distributed training")
        
        try:
            # Setup RL components (same as BaseWorker)
            self.setup_rl_components(env_name=env_name, device=device, lr=lr)
            self._update_status("DISTRIBUTED_TRAINING")
            
            # Training loop (same as BaseWorker + distributed hook)
            for episode in range(self.max_episodes):
                self.current_episode = episode + 1
                self.total_episodes += 1
                
                # Standard BaseWorker episode logic
                episode_result = self.collect_episode()
                episode_reward = episode_result["episode_reward"] 
                episode_length = episode_result["episode_length"]
                
                train_metrics = self.train_on_episode(episode_result["episode_data"])
                
                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(episode_length)
                
                if episode_reward > self.best_reward:
                    self.best_reward = episode_reward
                
                self.log_episode(episode_reward, episode_length, train_metrics)
                
                # *** ADD DISTRIBUTED COORDINATION HERE ***
                self.distributed_episode_hook()
                
                # Standard BaseWorker progress/checkpoint logic
                if episode % self.status_update_frequency == 0:
                    avg_reward = self._get_avg_reward(100)
                    self._update_status("DISTRIBUTED_TRAINING", 
                                      avg_reward=avg_reward,
                                      episodes_remaining=self.max_episodes - episode)
                    
                    print(f"🌐 Worker {self.worker_id}: Episode {episode}/{self.max_episodes} | "
                          f"Reward: {episode_reward:.2f} | Avg: {avg_reward:.2f} | Best: {self.best_reward:.2f}")
                
                if episode % self.checkpoint_frequency == 0:
                    self.save_checkpoint()
                
                if self.should_terminate():
                    print(f"🛑 Worker {self.worker_id}: Termination requested")
                    break
            
            # Training completed (same as BaseWorker)
            self._update_status("DISTRIBUTED_COMPLETED")
            print(f"✅ SimpleDistributedWorker {self.worker_id}: Training completed!")
            print(f"🏆 Best reward: {self.best_reward:.2f}")
            
        except KeyboardInterrupt:
            self._update_status("DISTRIBUTED_INTERRUPTED")
            print(f"🛑 SimpleDistributedWorker {self.worker_id}: Training interrupted")
        except Exception as e:
            self._update_status("DISTRIBUTED_ERROR", error=str(e))
            print(f"💥 SimpleDistributedWorker {self.worker_id}: Training error: {e}")
            raise
        finally:
            # Final save to shared directory
            self.save_to_shared_directory()
            self.cleanup()


# CLI interface (same as BaseWorker + shared_dir argument)
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Simple Distributed RL Worker')
    
    # Distributed arguments
    parser.add_argument('--worker_id', type=int, required=True, help='Worker ID')
    parser.add_argument('--shared_dir', type=str, required=True, help='Shared directory')
    parser.add_argument('--check_interval', type=int, default=10, help='Episodes between coordinator checks')
    
    # BaseWorker arguments
    parser.add_argument('--max_episodes', type=int, default=1000, help='Max episodes')
    parser.add_argument('--log_dir', type=str, default=None, help='Log directory (optional)')
    
    # Training arguments
    parser.add_argument('--env', type=str, default='CartPole-v1', help='Environment')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    
    args = parser.parse_args()
    
    # Default log_dir to shared_dir/worker_logs/worker_N if not specified
    if args.log_dir is None:
        args.log_dir = f"{args.shared_dir}/worker_logs/worker_{args.worker_id}"
    
    # Create and run worker
    worker = SimpleDistributedWorker(
        worker_id=args.worker_id,
        shared_dir=args.shared_dir,
        check_interval=args.check_interval,
        log_dir=args.log_dir,
        max_episodes=args.max_episodes
    )
    
    worker.run(env_name=args.env, device=args.device, lr=args.lr)