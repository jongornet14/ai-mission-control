#!/usr/bin/env python3
"""
Record videos and log them to TensorBoard for distributed RL training
Creates videos of best agents and integrates with your existing logging setup
"""
import argparse
import torch
import gym
import os
import sys
import json
import numpy as np
from pathlib import Path
from datetime import datetime
import cv2
from torch.utils.tensorboard import SummaryWriter

# Add project root to PYTHONPATH
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from algorithms.ppo import PPOAlgorithm
from environments.gym_wrapper import GymEnvironmentWrapper


class TensorBoardVideoRecorder:
    """Records videos and logs them to TensorBoard"""
    
    def __init__(self, log_dir, experiment_name="best_agent_videos"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=str(self.log_dir / experiment_name))
        self.experiment_name = experiment_name
        
    def find_best_model(self, shared_dir):
        """Find the best performing model from distributed training"""
        shared_path = Path(shared_dir)
        
        # Check coordinator's best model first
        best_model_file = shared_path / "best_model" / "current_best.pt"
        if best_model_file.exists():
            print(f"✅ Found coordinator's best model: {best_model_file}")
            # Try to read performance info from coordinator logs
            try:
                coordinator_metrics = self._get_coordinator_performance(shared_path)
                return str(best_model_file), coordinator_metrics
            except:
                return str(best_model_file), {"source": "coordinator_best", "avg_reward": "unknown"}
        
        # Fall back to searching worker models
        return self._find_best_worker_model(shared_path)
    
    def _get_coordinator_performance(self, shared_path):
        """Extract performance info from coordinator or worker metrics"""
        metrics_dir = shared_path / "metrics"
        if not metrics_dir.exists():
            return {"source": "coordinator", "avg_reward": "unknown"}
        
        best_reward = float('-inf')
        best_info = {"source": "coordinator", "avg_reward": "unknown"}
        
        for metrics_file in metrics_dir.glob("worker_*_performance.json"):
            try:
                with open(metrics_file, 'r') as f:
                    metrics = json.load(f)
                if metrics['avg_reward'] > best_reward:
                    best_reward = metrics['avg_reward']
                    best_info = {
                        "source": f"worker_{metrics['worker_id']}",
                        "avg_reward": metrics['avg_reward'],
                        "total_episodes": metrics['total_episodes'],
                        "reward_change": metrics.get('reward_change', 0)
                    }
            except:
                continue
                
        return best_info
    
    def _find_best_worker_model(self, shared_path):
        """Find best model from worker performance files"""
        models_dir = shared_path / "models"
        metrics_dir = shared_path / "metrics"
        
        if not models_dir.exists() or not metrics_dir.exists():
            print(f"❌ Models or metrics directory not found in {shared_path}")
            return None, None
        
        best_reward = float('-inf')
        best_model_path = None
        best_worker_info = None
        
        for metrics_file in metrics_dir.glob("worker_*_performance.json"):
            try:
                with open(metrics_file, 'r') as f:
                    metrics = json.load(f)
                
                worker_id = metrics['worker_id']
                avg_reward = metrics['avg_reward']
                total_episodes = metrics['total_episodes']
                
                if avg_reward > best_reward:
                    model_pattern = f"worker_{worker_id}_episode_{total_episodes}.pt"
                    model_file = models_dir / model_pattern
                    
                    if model_file.exists():
                        best_reward = avg_reward
                        best_model_path = str(model_file)
                        best_worker_info = {
                            'source': f'worker_{worker_id}',
                            'worker_id': worker_id,
                            'avg_reward': avg_reward,
                            'total_episodes': total_episodes,
                            'reward_change': metrics.get('reward_change', 0)
                        }
                        
            except Exception as e:
                continue
        
        return best_model_path, best_worker_info
    
    def load_model(self, model_path, env, device):
        """Load model from distributed training checkpoint"""
        print(f"📂 Loading model from: {model_path}")
        
        try:
            checkpoint = torch.load(model_path, map_location=device)
            
            algorithm = PPOAlgorithm(
                env=env,
                device=device,
                learning_rate=3e-4
            )
            
            if 'policy_state_dict' in checkpoint:
                algorithm.policy_net.load_state_dict(checkpoint['policy_state_dict'])
                print("✅ Loaded policy network weights")
            else:
                print("❌ No policy_state_dict found in checkpoint")
                return None
                
            if 'value_state_dict' in checkpoint:
                algorithm.value_net.load_state_dict(checkpoint['value_state_dict'])
                print("✅ Loaded value network weights")
            
            algorithm.policy_net.eval()
            algorithm.value_net.eval()
            
            return algorithm
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return None
    
    def record_and_log_videos(self, model_path, model_info, env_name, num_episodes=3, max_episode_steps=1000, device='cuda:0'):
        """Record videos and log them to TensorBoard"""
        print(f"\n🎥 Recording videos for TensorBoard...")
        print(f"📋 Model: {model_path}")
        print(f"🎮 Environment: {env_name}")
        print(f"📹 Episodes: {num_episodes}")
        
        device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        # Create environment
        env_wrapper = GymEnvironmentWrapper(
            env_name=env_name,
            device=device,
            normalize_observations=True,
            max_episode_steps=max_episode_steps
        )
        env = env_wrapper.create()
        
        # Load model
        algorithm = self.load_model(model_path, env, device)
        if algorithm is None:
            return False
        
        # Record episodes and collect frames
        all_videos = []
        episode_rewards = []
        episode_lengths = []
        
        for episode in range(num_episodes):
            print(f"📹 Recording episode {episode + 1}/{num_episodes}...")
            
            frames = []
            obs = env.reset()
            if isinstance(obs, tuple):
                obs = obs[0]
                
            episode_reward = 0
            step_count = 0
            
            with torch.no_grad():
                while step_count < max_episode_steps:
                    # Render frame
                    try:
                        frame = env.render()
                        if frame is not None:
                            # Convert to proper format for TensorBoard (H, W, C) -> (C, H, W)
                            if len(frame.shape) == 3:
                                frames.append(frame)
                    except:
                        pass  # Skip frame if rendering fails
                    
                    # Get action and step
                    action, _ = algorithm.get_action(obs)
                    result = env.step(action)
                    
                    if len(result) == 5:
                        obs, reward, done, truncated, info = result
                        done = done or truncated
                    else:
                        obs, reward, done, info = result
                    
                    episode_reward += reward
                    step_count += 1
                    
                    if done:
                        break
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(step_count)
            
            # Add frames to video collection
            if frames:
                # Convert frames to numpy array and transpose for TensorBoard
                video_array = np.array(frames)
                if len(video_array.shape) == 4:  # (T, H, W, C)
                    video_array = video_array.transpose(0, 3, 1, 2)  # (T, C, H, W)
                    all_videos.append(video_array)
                    print(f"✅ Episode {episode + 1}: {len(frames)} frames, reward: {episode_reward:.2f}")
                else:
                    print(f"⚠️ Episode {episode + 1}: Invalid frame format")
            else:
                print(f"⚠️ Episode {episode + 1}: No frames captured")
        
        # Log videos to TensorBoard
        if all_videos:
            self._log_videos_to_tensorboard(all_videos, episode_rewards, episode_lengths, model_info, env_name)
            return True
        else:
            print("❌ No videos to log")
            return False
    
    def _log_videos_to_tensorboard(self, videos, rewards, lengths, model_info, env_name):
        """Log videos and metrics to TensorBoard"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        global_step = int(timestamp[-6:])  # Use timestamp as step
        
        # Log individual episode videos
        for i, (video, reward, length) in enumerate(zip(videos, rewards, lengths)):
            # Ensure video is in correct format (T, C, H, W) and proper data type
            if video.dtype != np.uint8:
                video = (video * 255).astype(np.uint8)
            
            # Add batch dimension if needed: (1, T, C, H, W)
            if len(video.shape) == 4:
                video = video[np.newaxis, ...]
            
            tag = f"videos/episode_{i+1}_reward_{reward:.1f}"
            self.writer.add_video(
                tag=tag,
                vid_tensor=torch.from_numpy(video),
                global_step=global_step + i,
                fps=30
            )
            
            # Log episode metrics
            self.writer.add_scalar(f"performance/episode_{i+1}_reward", reward, global_step + i)
            self.writer.add_scalar(f"performance/episode_{i+1}_length", length, global_step + i)
        
        # Log summary statistics
        avg_reward = np.mean(rewards)
        std_reward = np.std(rewards)
        avg_length = np.mean(lengths)
        
        self.writer.add_scalar("performance/avg_reward", avg_reward, global_step)
        self.writer.add_scalar("performance/std_reward", std_reward, global_step)
        self.writer.add_scalar("performance/avg_episode_length", avg_length, global_step)
        self.writer.add_scalar("performance/best_episode_reward", max(rewards), global_step)
        
        # Log model information as text
        model_text = f"""
        **Best Agent Performance**
        
        **Model Info:**
        - Source: {model_info.get('source', 'unknown')}
        - Average Reward: {model_info.get('avg_reward', 'unknown')}
        - Total Episodes: {model_info.get('total_episodes', 'unknown')}
        - Reward Change: {model_info.get('reward_change', 'unknown')}
        
        **Video Recording:**
        - Environment: {env_name}
        - Episodes Recorded: {len(videos)}
        - Average Reward: {avg_reward:.2f} ± {std_reward:.2f}
        - Best Episode: {max(rewards):.2f}
        - Average Length: {avg_length:.1f} steps
        
        **Timestamp:** {timestamp}
        """
        
        self.writer.add_text("model_info/best_agent", model_text, global_step)
        
        # Force write to disk
        self.writer.flush()
        
        print(f"\n✅ Videos logged to TensorBoard!")
        print(f"📊 Summary:")
        print(f"   Episodes: {len(videos)}")
        print(f"   Avg Reward: {avg_reward:.2f} ± {std_reward:.2f}")
        print(f"   Best Episode: {max(rewards):.2f}")
        print(f"   TensorBoard: {self.log_dir}")
        print(f"💡 View in TensorBoard under 'IMAGES' tab")
    
    def close(self):
        """Close TensorBoard writer"""
        self.writer.close()


def main():
    parser = argparse.ArgumentParser(description='Record videos and log to TensorBoard')
    
    # Model and data paths
    parser.add_argument('--shared_dir', type=str, default='./distributed_shared',
                       help='Shared directory from distributed training')
    parser.add_argument('--tensorboard_dir', type=str, default='./tensorboard_videos',
                       help='Directory for TensorBoard video logs')
    parser.add_argument('--model_path', type=str, 
                       help='Specific model path (optional)')
    
    # Recording settings
    parser.add_argument('--env_name', type=str, default='CartPole-v1',
                       help='Environment name')
    parser.add_argument('--num_episodes', type=int, default=3,
                       help='Number of episodes to record')
    parser.add_argument('--max_episode_steps', type=int, default=1000,
                       help='Max steps per episode')
    parser.add_argument('--device', type=str, 
                       default='cuda:0' if torch.cuda.is_available() else 'cpu',
                       help='Device for inference')
    
    args = parser.parse_args()
    
    # Create TensorBoard video recorder
    recorder = TensorBoardVideoRecorder(
        log_dir=args.tensorboard_dir,
        experiment_name=f"best_agent_{args.env_name}"
    )
    
    try:
        # Find or use specified model
        if args.model_path:
            if not Path(args.model_path).exists():
                print(f"❌ Specified model not found: {args.model_path}")
                return
            model_path = args.model_path
            model_info = {"source": "manual", "avg_reward": "unknown"}
            print(f"📋 Using specified model: {model_path}")
        else:
            print(f"🔍 Finding best model in: {args.shared_dir}")
            model_path, model_info = recorder.find_best_model(args.shared_dir)
            if not model_path:
                print("❌ No model found")
                return
        
        # Record and log videos
        success = recorder.record_and_log_videos(
            model_path=model_path,
            model_info=model_info,
            env_name=args.env_name,
            num_episodes=args.num_episodes,
            max_episode_steps=args.max_episode_steps,
            device=args.device
        )
        
        if success:
            print(f"\n🎉 Success! Videos logged to TensorBoard")
            print(f"🔗 Start TensorBoard: tensorboard --logdir {args.tensorboard_dir}")
            print(f"🌐 Then visit: http://localhost:6006")
            print(f"📺 Look for videos in the 'IMAGES' tab")
        else:
            print(f"\n❌ Failed to record videos")
            
    finally:
        recorder.close()


if __name__ == '__main__':
    main()