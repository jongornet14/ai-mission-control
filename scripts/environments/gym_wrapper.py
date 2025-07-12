#!/usr/bin/env python3
"""
FILENAME: environments/gym_wrapper.py
Simple Gym Environment Wrapper (no TorchRL dependencies)
"""

import gym
import numpy as np


class GymEnvironmentWrapper:
    """Simple wrapper for OpenAI Gym environments"""
    
    def __init__(self, env_name, device=None, frame_skip=1, normalize_observations=True, max_episode_steps=None):
        """Initialize Gym environment wrapper
        
        Args:
            env_name (str): Name of the Gym environment
            device: Ignored for compatibility
            frame_skip (int): Frame skipping factor  
            normalize_observations (bool): Whether to normalize observations
            max_episode_steps (int): Maximum episode steps
        """
        self.env_name = env_name
        self.frame_skip = frame_skip
        self.normalize_observations = normalize_observations
        self.max_episode_steps = max_episode_steps
        self.obs_mean = None
        self.obs_std = None
        
    def create(self):
        """Create and configure the environment"""
        # Create base gym environment
        env = gym.make(self.env_name)
        
        # Set max episode steps if specified
        if self.max_episode_steps is not None:
            env._max_episode_steps = self.max_episode_steps
            
        # Initialize observation normalization if needed
        if self.normalize_observations:
            self._init_obs_normalization(env)
            
        # Wrap environment with our simple wrapper
        wrapped_env = SimpleGymWrapper(env, self)
        
        return wrapped_env
    
    def _init_obs_normalization(self, env):
        """Initialize observation normalization statistics"""
        print("Initializing observation normalization...")
        obs_samples = []
        
        for _ in range(100):  # Sample 100 episodes for statistics
            obs, _ = env.reset()
            obs_samples.append(obs)
            
            for _ in range(100):  # Max 100 steps per episode
                action = env.action_space.sample()
                obs, _, done, _, _ = env.step(action)
                obs_samples.append(obs)
                if done:
                    break

        obs_samples = np.array(obs_samples)
        self.obs_mean = np.mean(obs_samples, axis=0)
        self.obs_std = np.std(obs_samples, axis=0) + 1e-8  # Small epsilon to avoid division by zero
        
        print(f"Observation normalization initialized: mean={self.obs_mean}, std={self.obs_std}")


class SimpleGymWrapper:
    """Simple gym environment wrapper"""
    
    def __init__(self, env, wrapper_config):
        self.env = env
        self.config = wrapper_config
        
    def reset(self):
        obs,_ = self.env.reset()
        if self.config.normalize_observations and self.config.obs_mean is not None:
            obs = (obs - self.config.obs_mean) / self.config.obs_std
        return obs
        
    def step(self, action):
        obs, reward, done, info, _ = self.env.step(action)
        if self.config.normalize_observations and self.config.obs_mean is not None:
            obs = (obs - self.config.obs_mean) / self.config.obs_std
        return obs, reward, done, info
        
    @property
    def action_space(self):
        return self.env.action_space
        
    @property 
    def observation_space(self):
        return self.env.observation_space
        
    def close(self):
        return self.env.close()


def get_available_gym_environments():
    """Get list of available Gym environments"""
    try:
        import gym
        return list(gym.envs.registry.env_specs.keys())
    except ImportError:
        return []


def get_environment_info(env_name):
    """Get information about a specific environment"""
    try:
        import gym
        env = gym.make(env_name)
        info = {
            'observation_space': env.observation_space,
            'action_space': env.action_space,
            'max_episode_steps': getattr(env, '_max_episode_steps', None),
            'reward_threshold': getattr(env.spec, 'reward_threshold', None)
        }
        env.close()
        return info
    except Exception as e:
        return {'error': str(e)}
