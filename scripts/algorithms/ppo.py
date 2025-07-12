#!/usr/bin/env python3
"""
FILENAME: algorithms/ppo.py
Simple PPO Algorithm (no TorchRL dependencies)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque


class PPOAlgorithm:
    """Simple PPO Algorithm using plain PyTorch"""
    
    def __init__(self, env, device, **config):
        """Initialize PPO algorithm"""
        self.env = env
        self.device = device
        self.config = config
        
        # Algorithm hyperparameters
        self.learning_rate = config.get('learning_rate', 3e-4)
        self.gamma = config.get('gamma', 0.99)
        self.lambda_gae = config.get('lambda', 0.95)
        self.clip_epsilon = config.get('clip_epsilon', 0.2)
        self.entropy_coef = config.get('entropy_coef', 1e-4)
        self.critic_coef = config.get('critic_coef', 1.0)
        self.max_grad_norm = config.get('max_grad_norm', 1.0)
        self.num_epochs = config.get('num_epochs', 10)
        self.batch_size = config.get('batch_size', 64)
        self.num_cells = config.get('num_cells', 256)
        
        # Get environment dimensions
        self.obs_dim = env.observation_space.shape[0]
        if hasattr(env.action_space, 'n'):
            self.action_dim = env.action_space.n
            self.discrete = True
        else:
            self.action_dim = env.action_space.shape[0]
            self.discrete = False
            
        # Build networks
        self.policy_net = self._build_policy_network().to(device)
        self.value_net = self._build_value_network().to(device)
        
        # Optimizer
        self.optimizer = torch.optim.Adam(
            list(self.policy_net.parameters()) + list(self.value_net.parameters()),
            lr=self.learning_rate
        )
        
        print(f"🧠 Simple PPO Algorithm initialized:")
        print(f"   Environment: {self.obs_dim} obs -> {self.action_dim} actions ({'discrete' if self.discrete else 'continuous'})")
        print(f"   Learning rate: {self.learning_rate}")
        print(f"   Network size: {self.num_cells} cells")
        
    def _build_policy_network(self):
        """Build policy network"""
        if self.discrete:
            return nn.Sequential(
                nn.Linear(self.obs_dim, self.num_cells),
                nn.Tanh(),
                nn.Linear(self.num_cells, self.num_cells),
                nn.Tanh(),
                nn.Linear(self.num_cells, self.action_dim)
            )
        else:
            # Continuous actions - output mean and log_std
            return nn.Sequential(
                nn.Linear(self.obs_dim, self.num_cells),
                nn.Tanh(),
                nn.Linear(self.num_cells, self.num_cells),
                nn.Tanh(),
                nn.Linear(self.num_cells, self.action_dim * 2)  # mean + log_std
            )
            
    def _build_value_network(self):
        """Build value network"""
        return nn.Sequential(
            nn.Linear(self.obs_dim, self.num_cells),
            nn.Tanh(),
            nn.Linear(self.num_cells, self.num_cells),
            nn.Tanh(),
            nn.Linear(self.num_cells, 1)
        )
        
    def get_action(self, obs):
        """Get action from policy"""
        obs = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            if self.discrete:
                logits = self.policy_net(obs)
                action_dist = torch.distributions.Categorical(logits=logits)
                action = action_dist.sample()
                log_prob = action_dist.log_prob(action)
                return action.item(), log_prob.item()
            else:
                output = self.policy_net(obs)
                mean, log_std = output.chunk(2, dim=-1)
                std = log_std.exp()
                action_dist = torch.distributions.Normal(mean, std)
                action = action_dist.sample()
                log_prob = action_dist.log_prob(action).sum(dim=-1)
                return action.squeeze().cpu().numpy(), log_prob.item()
                
    def train_step(self, rollout_data, replay_buffer=None):
        """Perform one training step"""
        # This is a simplified version - you can expand this with your existing PPO logic
        # For now, just return dummy metrics
        return {
            'loss_total': 0.0,
            'loss_policy': 0.0,
            'loss_value': 0.0,
            'loss_entropy': 0.0,
            'reward_mean': np.mean([r for _, _, r, _, _ in rollout_data]) if rollout_data else 0.0,
            'reward_std': np.std([r for _, _, r, _, _ in rollout_data]) if rollout_data else 0.0,
            'episode_length': len(rollout_data) if rollout_data else 0,
        }
        
    def get_hyperparameters(self):
        """Get current hyperparameters for logging"""
        return {
            'learning_rate': self.learning_rate,
            'gamma': self.gamma,
            'lambda_gae': self.lambda_gae,
            'clip_epsilon': self.clip_epsilon,
            'entropy_coef': self.entropy_coef,
            'critic_coef': self.critic_coef,
            'batch_size': self.batch_size,
        }
        
    def update_hyperparameters(self, new_hyperparams):
        """Update hyperparameters during training"""
        if 'learning_rate' in new_hyperparams:
            self.learning_rate = new_hyperparams['learning_rate']
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.learning_rate
                
        # Update other hyperparameters as needed
        for param, value in new_hyperparams.items():
            if hasattr(self, param):
                setattr(self, param, value)
                
        print(f"🎛️  Updated hyperparameters: {new_hyperparams}")
        
    def state_dict(self):
        """Get algorithm state for saving"""
        return {
            'policy_net': self.policy_net.state_dict(),
            'value_net': self.value_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'hyperparameters': self.get_hyperparameters()
        }
        
    def load_state_dict(self, state_dict):
        """Load algorithm state"""
        self.policy_net.load_state_dict(state_dict['policy_net'])
        self.value_net.load_state_dict(state_dict['value_net'])
        self.optimizer.load_state_dict(state_dict['optimizer'])
