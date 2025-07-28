#!/usr/bin/env python3
import argparse
import torch
import gym
import os
import sys
from pathlib import Path

# Add project root to PYTHONPATH for imports like algorithms.ppo
# Assuming this script is at /workspace/project/scripts/
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from algorithms.ppo import PPOAlgorithm  # Adjust if your algorithm class is elsewhere
from environments.gym_wrapper import (
    GymEnvironmentWrapper,
)  # Adjust if your wrapper is elsewhere


def record_video(
    model_path,
    env_name,
    video_dir,
    num_episodes=1,
    max_episode_steps=None,
    device="cpu",
):
    """
    Loads a trained model and records video(s) of its performance.
    """
    print(f"\n🎥 Starting video recording for model: {model_path}")
    print(f"Environment: {env_name}, Episodes: {num_episodes}, Device: {device}")

    # Ensure video directory exists
    Path(video_dir).mkdir(parents=True, exist_ok=True)

    # Load the model
    if not Path(model_path).exists():
        print(f"Error: Model file not found at {model_path}")
        return

    # Initialize environment for rendering
    # Important: render_mode='rgb_array' for video recording
    # Use 'human' if you want to see a window, but it requires a full X server setup
    # For Docker, 'rgb_array' with Xvfb is the way to go.

    # Initialize a base environment first to get observation/action space for policy
    dummy_env = GymEnvironmentWrapper(
        env_name, normalize_observations=False
    )  # normalization should match training
    obs_space = dummy_env.observation_space
    action_space = dummy_env.action_space
    dummy_env.close()

    policy = PPOAlgorithm.Policy(obs_space, action_space).to(device)
    policy.load_state_dict(torch.load(model_path, map_location=device))
    policy.eval()  # Set policy to evaluation mode

    for i in range(num_episodes):
        video_filepath = Path(video_dir) / f"episode_{i+1}_{Path(model_path).stem}.mp4"
        print(f"Recording episode {i+1} to: {video_filepath}")

        # Create environment with video wrapper
        env = GymEnvironmentWrapper(
            env_name, render_mode="rgb_array", max_episode_steps=max_episode_steps
        )
        # Wrap with RecordVideo. Note: This requires ffmpeg installed in the container
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=str(video_filepath.parent),
            name_prefix=f"episode_{i+1}_{Path(model_path).stem}",
            disable_logger=True,
        )
        # gym.wrappers.RecordVideo creates a subfolder for videos. We want to control the filename.
        # A workaround is to provide a unique name_prefix and then potentially move/rename.
        # For simplicity, let's just use a base folder and let gym create its own names for now,
        # or manually handle imageio if granular control over filenames is critical.

        obs, info = env.reset()
        done = False
        truncated = False
        episode_reward = 0

        with torch.no_grad():
            while not done and not truncated:
                action, _, _ = policy(
                    torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(
                        0
                    )
                )
                obs, reward, done, truncated, info = env.step(
                    action.squeeze(0).cpu().numpy()
                )
                episode_reward += reward

        env.close()
        print(f"Episode {i+1} finished with reward: {episode_reward:.2f}")

    print("\n✅ Video recording complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Record video of a trained RL model.")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the trained model checkpoint (e.g., /workspace/experiments/my_exp/models/best_model.pth)",
    )
    parser.add_argument(
        "--env_name",
        type=str,
        default="HalfCheetah-v4",
        help="Name of the environment to record (e.g., HalfCheetah-v4)",
    )
    parser.add_argument(
        "--video_dir",
        type=str,
        default="/workspace/videos",
        help="Directory to save the recorded videos",
    )
    parser.add_argument(
        "--num_episodes", type=int, default=1, help="Number of episodes to record"
    )
    parser.add_argument(
        "--max_episode_steps",
        type=int,
        default=1000,
        help="Maximum steps per episode during recording (match training)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="Device to use for model inference (e.g., cuda:0 or cpu)",
    )

    args = parser.parse_args()

    record_video(
        model_path=args.model_path,
        env_name=args.env_name,
        video_dir=args.video_dir,
        num_episodes=args.num_episodes,
        max_episode_steps=args.max_episode_steps,
        device=args.device,
    )
