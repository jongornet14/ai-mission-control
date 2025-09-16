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
import time
from torch.utils.tensorboard import SummaryWriter

import os

os.environ["MUJOCO_GL"] = "osmesa"

# Add project root to PYTHONPATH
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Fixed imports to match your project structure
from intellinaut.algorithms.ppo import PPOAlgorithm
from intellinaut.environments.gym_wrapper import (
    GymEnvironmentWrapper,
    universal_gym_step,
)


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
            print(f"\033[92mFound coordinator's best model:\033[0m {best_model_file}")
            # Try to read performance info from coordinator logs
            try:
                coordinator_metrics = self._get_coordinator_performance(shared_path)
                return str(best_model_file), coordinator_metrics
            except:
                return str(best_model_file), {
                    "source": "coordinator_best",
                    "avg_reward": "unknown",
                }

        # Fall back to searching worker models
        return self._find_best_worker_model(shared_path)

    def _get_coordinator_performance(self, shared_path):
        """Extract performance info from coordinator or worker metrics"""
        metrics_dir = shared_path / "metrics"
        if not metrics_dir.exists():
            return {"source": "coordinator", "avg_reward": "unknown"}

        best_reward = float("-inf")
        best_info = {"source": "coordinator", "avg_reward": "unknown"}

        for metrics_file in metrics_dir.glob("worker_*_performance.json"):
            try:
                with open(metrics_file, "r") as f:
                    metrics = json.load(f)
                if metrics["avg_reward"] > best_reward:
                    best_reward = metrics["avg_reward"]
                    best_info = {
                        "source": f"worker_{metrics['worker_id']}",
                        "avg_reward": metrics["avg_reward"],
                        "total_episodes": metrics["total_episodes"],
                        "reward_change": metrics.get("reward_change", 0),
                    }
            except:
                continue

        return best_info

    def _find_best_worker_model(self, shared_path):
        """Find best model from worker performance files"""
        models_dir = shared_path / "models"
        metrics_dir = shared_path / "metrics"

        if not models_dir.exists() or not metrics_dir.exists():
            print(
                f"\033[91mModels or metrics directory not found in {shared_path}\033[0m"
            )
            return None, None

        best_reward = float("-inf")
        best_model_path = None
        best_worker_info = None

        for metrics_file in metrics_dir.glob("worker_*_performance.json"):
            try:
                with open(metrics_file, "r") as f:
                    metrics = json.load(f)

                worker_id = metrics["worker_id"]
                avg_reward = metrics["avg_reward"]
                total_episodes = metrics["total_episodes"]

                if avg_reward > best_reward:
                    # Look for any model file from this worker
                    worker_models = list(
                        models_dir.glob(f"worker_{worker_id}_episode_*.pt")
                    )
                    if worker_models:
                        # Get the most recent model
                        latest_model = max(
                            worker_models, key=lambda x: x.stat().st_mtime
                        )
                        best_reward = avg_reward
                        best_model_path = str(latest_model)
                        best_worker_info = {
                            "source": f"worker_{worker_id}",
                            "worker_id": worker_id,
                            "avg_reward": avg_reward,
                            "total_episodes": total_episodes,
                            "reward_change": metrics.get("reward_change", 0),
                        }

            except Exception as e:
                print(f"\033[93mError reading metrics file {metrics_file}:\033[0m {e}")
                continue

        return best_model_path, best_worker_info

    def load_model(self, model_path, env, device):
        """Load model from distributed training checkpoint"""
        print(f"\033[94mLoading model from:\033[0m {model_path}")

        try:
            checkpoint = torch.load(model_path, map_location=device)

            # Create algorithm with default config
            algorithm = PPOAlgorithm(
                env=env,
                device=device,
                learning_rate=3e-4,  # Default, will be overridden by checkpoint
            )

            if "policy_state_dict" in checkpoint:
                try:
                    algorithm.policy_net.load_state_dict(
                        checkpoint["policy_state_dict"]
                    )
                    print("\033[92mLoaded policy network weights\033[0m")
                except RuntimeError as e:
                    print(f"Policy network state_dict mismatch: {e}")
                    print("Skipping policy network weights loading")
            else:
                print("No policy_state_dict found in checkpoint")

            if "value_state_dict" in checkpoint:
                try:
                    algorithm.value_net.load_state_dict(checkpoint["value_state_dict"])
                    print("\033[92mLoaded value network weights\033[0m")
                except RuntimeError as e:
                    print(f"Value network state_dict mismatch: {e}")
                    print("Skipping value network weights loading")

            algorithm.policy_net.eval()
            algorithm.value_net.eval()

            return algorithm

        except Exception as e:
            print(f"\033[91mError loading model:\033[0m {e}")
            import traceback

            traceback.print_exc()
            return None

    def create_rendering_env(self, env_name, max_episode_steps=1000):
        """Create environment with proper rendering setup"""
        try:
            # Set MuJoCo to use EGL for offscreen rendering
            os.environ["MUJOCO_GL"] = "egl"

            # Try modern gym approach first
            env = gym.make(env_name, render_mode="rgb_array")
            print("\033[92m✓ Using modern gym render_mode with EGL\033[0m")
            if hasattr(env, "spec") and hasattr(env.spec, "max_episode_steps"):
                env.spec.max_episode_steps = max_episode_steps
            return env
        except TypeError:
            # Fallback for older gym versions
            try:
                env = gym.make(env_name)
                print("\033[93m✓ Using legacy gym rendering with EGL\033[0m")
                if hasattr(env, "spec") and hasattr(env.spec, "max_episode_steps"):
                    env.spec.max_episode_steps = max_episode_steps
                return env
            except Exception as e:
                print(f"\033[91mFailed to create environment:\033[0m {e}")
                return None

    def render_frame(self, env):
        """Handle rendering with fallbacks for different gym versions"""
        try:
            # Try modern approach first
            return env.render()
        except TypeError as e:
            if "unexpected keyword argument 'mode'" in str(e):
                # Try legacy approach
                try:
                    return env.render(mode="rgb_array")
                except Exception as e:
                    print(
                        f"\033[91mBoth modern and legacy rendering failed:\033[0m {e}"
                    )
                    return None
            else:
                print(f"\033[91mRender error:\033[0m {e}")
                return None

    def record_and_log_videos(
        self,
        model_path,
        model_info,
        env_name,
        num_episodes=3,
        max_episode_steps=1000,
        device="cuda:0",
    ):
        """Record videos and log them to TensorBoard"""
        print(f"\033[94mRecording videos for TensorBoard...\033[0m")

        device = torch.device(device if torch.cuda.is_available() else "cpu")

        # Create environment
        env = self.create_rendering_env(
            env_name, max_episode_steps
        )  # Use self.create_rendering_env
        if env is None:
            return False

        # Load model
        algorithm = self.load_model(model_path, env, device)
        if algorithm is None:
            return False

        # Record episodes
        all_videos = []
        episode_rewards = []
        episode_lengths = []

        for episode in range(num_episodes):
            print(f"Recording episode {episode + 1}/{num_episodes}...")
            frames = []
            obs = env.reset()
            if isinstance(obs, tuple):
                obs = obs[0]

            episode_reward = 0
            step_count = 0

            with torch.no_grad():
                while step_count < max_episode_steps:
                    # Render frame
                    frame = self.render_frame(env)  # Use self.render_frame
                    if frame is not None and isinstance(frame, np.ndarray):
                        if len(frame.shape) == 3 and frame.shape[2] == 3:
                            frames.append(frame)
                        else:
                            print(
                                f"\033[93mUnexpected frame shape:\033[0m {frame.shape}"
                            )

                    # Get action and step
                    try:
                        action, _ = algorithm.get_action(obs)
                        step_result = env.step(action)

                        # Handle different step return formats
                        if len(step_result) == 5:
                            next_obs, reward, done, truncated, info = step_result
                        else:
                            next_obs, reward, done, info = step_result
                            truncated = False

                        obs = next_obs
                        episode_reward += self._to_scalar(reward)
                        step_count += 1

                        if done or truncated:
                            break

                    except Exception as e:
                        print(f"\033[93mStep error at step {step_count}:\033[0m {e}")
                        break

            episode_rewards.append(episode_reward)
            episode_lengths.append(step_count)

            if frames:
                video_array = np.array(frames)
                if len(video_array.shape) == 4:
                    all_videos.append(video_array)
                    print(
                        f"\033[92mEpisode {episode + 1}:\033[0m {len(frames)} frames, reward: {episode_reward:.2f}"
                    )
                else:
                    print(f"\033[93mEpisode {episode + 1}:\033[0m Invalid frame format")
            else:
                print(f"\033[93mEpisode {episode + 1}:\033[0m No frames captured")

        env.close()

        # Log videos
        if all_videos:
            self._log_videos_to_tensorboard(
                all_videos, episode_rewards, episode_lengths, model_info, env_name
            )
            return True
        else:
            print("\033[91mNo videos to log\033[0m")
            return False

    def _to_scalar(self, x):
        """Convert tensor to scalar safely"""
        if torch.is_tensor(x):
            return x.item() if x.dim() == 0 else x.mean().item()
        else:
            return float(x)

    def _log_videos_to_tensorboard(
        self, videos, rewards, lengths, model_info, env_name
    ):
        """Log videos and metrics to TensorBoard"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        global_step = int(time.time())  # Use current timestamp as step

        # Log individual episode videos
        for i, (video, reward, length) in enumerate(zip(videos, rewards, lengths)):
            try:
                # Ensure video is in correct format (T, H, W, C) and proper data type
                if video.dtype != np.uint8:
                    if video.max() <= 1.0:
                        video = (video * 255).astype(np.uint8)
                    else:
                        video = video.astype(np.uint8)

                # Convert to tensor and add batch dimension: (1, T, C, H, W)
                video_tensor = torch.from_numpy(video).permute(0, 3, 1, 2).unsqueeze(0)

                tag = f"videos/episode_{i+1}_reward_{reward:.1f}"
                self.writer.add_video(
                    tag=tag,
                    vid_tensor=video_tensor,
                    global_step=global_step + i,
                    fps=30,
                )
                print(f"\033[92mLogged video {i+1} to TensorBoard\033[0m")

            except Exception as e:
                print(f"\033[93mError logging video {i+1}:\033[0m {e}")
                continue

            # Log episode metrics
            self.writer.add_scalar(
                f"performance/episode_{i+1}_reward", reward, global_step + i
            )
            self.writer.add_scalar(
                f"performance/episode_{i+1}_length", length, global_step + i
            )

        # Log summary statistics
        avg_reward = np.mean(rewards)
        std_reward = np.std(rewards)
        avg_length = np.mean(lengths)

        self.writer.add_scalar("performance/avg_reward", avg_reward, global_step)
        self.writer.add_scalar("performance/std_reward", std_reward, global_step)
        self.writer.add_scalar(
            "performance/avg_episode_length", avg_length, global_step
        )
        self.writer.add_scalar(
            "performance/best_episode_reward", max(rewards), global_step
        )

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

        print(f"\n\033[92mVideos logged to TensorBoard!\033[0m")
        print(f"\033[94mSummary:\033[0m")
        print(f"   \033[94mEpisodes:\033[0m {len(videos)}")
        print(f"   \033[94mAvg Reward:\033[0m {avg_reward:.2f} ± {std_reward:.2f}")
        print(f"   \033[94mBest Episode:\033[0m {max(rewards):.2f}")
        print(f"   \033[94mTensorBoard:\033[0m {self.log_dir}")
        print(f"\033[94mView in TensorBoard under 'IMAGES' tab\033[0m")

    def close(self):
        """Close TensorBoard writer"""
        self.writer.close()


def main():
    parser = argparse.ArgumentParser(description="Record videos and log to TensorBoard")

    # Model and data paths
    parser.add_argument(
        "--shared_dir",
        type=str,
        default="./distributed_shared",
        help="Shared directory from distributed training",
    )
    parser.add_argument(
        "--tensorboard_dir",
        type=str,
        default="./tensorboard_videos",
        help="Directory for TensorBoard video logs",
    )
    parser.add_argument("--model_path", type=str, help="Specific model path (optional)")

    # Recording settings
    parser.add_argument(
        "--env_name", type=str, default="CartPole-v1", help="Environment name"
    )
    parser.add_argument(
        "--num_episodes", type=int, default=3, help="Number of episodes to record"
    )
    parser.add_argument(
        "--max_episode_steps", type=int, default=1000, help="Max steps per episode"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="Device for inference",
    )

    args = parser.parse_args()

    print(f"TensorBoard Video Recorder")
    print(f"Shared directory: {args.shared_dir}")
    print(f"TensorBoard directory: {args.tensorboard_dir}")
    print(f"Environment: {args.env_name}")
    print(f"Device: {args.device}")

    # Create TensorBoard video recorder
    recorder = TensorBoardVideoRecorder(
        log_dir=args.tensorboard_dir, experiment_name=f"best_agent_{args.env_name}"
    )

    try:
        # Find or use specified model
        if args.model_path:
            if not Path(args.model_path).exists():
                print(f"Specified model not found: {args.model_path}")
                return
            model_path = args.model_path
            model_info = {"source": "manual", "avg_reward": "unknown"}
            print(f"Using specified model: {model_path}")
        else:
            print(f"\033[94mFinding best model in:\033[0m {args.shared_dir}")
            model_path, model_info = recorder.find_best_model(args.shared_dir)
            if not model_path:
                print("\033[91mNo model found\033[0m")
                return

        # Record and log videos
        success = recorder.record_and_log_videos(
            model_path=model_path,
            model_info=model_info,
            env_name=args.env_name,
            num_episodes=args.num_episodes,
            max_episode_steps=args.max_episode_steps,
            device=args.device,
        )

        if success:
            print(f"\n\033[92mSuccess! Videos logged to TensorBoard\033[0m")
            print(
                f"\033[94mStart TensorBoard:\033[0m tensorboard --logdir {args.tensorboard_dir}"
            )
            print(f"\033[94mThen visit:\033[0m http://localhost:6006")
            print(f"\033[94mLook for videos in the 'IMAGES' tab\033[0m")
        else:
            print(f"\n\033[91mFailed to record videos\033[0m")

    except Exception as e:
        print(f"Error during video recording: {e}")
        import traceback

        traceback.print_exc()
    finally:
        recorder.close()


if __name__ == "__main__":
    main()
