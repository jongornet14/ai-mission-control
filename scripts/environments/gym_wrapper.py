#!/usr/bin/env python3
"""
FILENAME: environments/gym_wrapper.py
Simple Gym Environment Wrapper (no TorchRL dependencies)
"""

import gym
import numpy as np

import torch

class TorchGymWrapper:
    """Convert gym outputs to torch tensors to avoid constant conversions"""
    
    def __init__(self, env, device='cpu'):
        self.env = env
        self.device = device
        
    def reset(self):
        obs = self.env.reset()
        return torch.FloatTensor(obs).to(self.device)
    
    def step(self, action):
        # Convert torch tensor action back to numpy for gym
        if torch.is_tensor(action):
            if action.dim() == 0:  # scalar
                action = action.item()
            else:  # array
                action = action.cpu().numpy()
                
        obs, reward, done, info = self.env.step(action)
        
        return (
            torch.FloatTensor(obs).to(self.device),
            torch.tensor(reward, dtype=torch.float32, device=self.device),
            torch.tensor(done, dtype=torch.float32, device=self.device),  # float for GAE
            info
        )
        
    def render(self, mode='rgb_array'):
        """Pass through render call to underlying environment"""
        return self.env.render()
    
    @property
    def action_space(self):
        return self.env.action_space
        
    @property 
    def observation_space(self):
        return self.env.observation_space
        
    def close(self):
        return self.env.close()

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

        self.device = device if device is not None else 'cpu'  # Store device
        
    def create(self):
        """Create and configure the environment"""

        if 'mujoco' in self.env_name.lower() or any(x in self.env_name for x in ['Ant', 'HalfCheetah', 'Hopper', 'Humanoid', 'InvertedPendulum', 'InvertedDoublePendulum', 'Reacher', 'Swimmer', 'Walker']):
            env = gym.make(self.env_name, render_mode='rgb_array')
        else:
            env = gym.make(self.env_name, render_mode='rgb_array')
        
        # MuJoCo specific: Force render initialization
        if hasattr(env, 'mujoco_renderer'):
            try:
                # Try to initialize renderer
                env.reset()
                _ = env.render()
            except:
                pass
        
        # Set max episode steps if specified
        if self.max_episode_steps is not None:
            env._max_episode_steps = self.max_episode_steps
            
        # Initialize observation normalization if needed
        if self.normalize_observations:
            self._init_obs_normalization(env)
            
        # Wrap environment with our simple wrapper
        wrapped_env = SimpleGymWrapper(env, self)
        
        # ADD THIS: Wrap with torch conversion
        torch_env = TorchGymWrapper(wrapped_env, self.device)
        
        return torch_env
    
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
        
    def render(self, mode='rgb_array'):
        """Pass through render call to underlying environment"""
        return self.env.render()
        
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
        env = gym.make(self.env_name, render_mode='rgb_array')  # ADD render_mode here
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
