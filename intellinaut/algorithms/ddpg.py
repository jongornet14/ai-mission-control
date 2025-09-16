#!/usr/bin/env python3
"""
FILENAME: algorithms/ddpg.py
Simple DDPG Algorithm (no TorchRL dependencies)
Deep Deterministic Policy Gradient for continuous control
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import logging
from collections import deque
import random


class ReplayBuffer:
    """Simple replay buffer for DDPG"""

    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return (
            torch.FloatTensor(state),
            torch.FloatTensor(action),
            torch.FloatTensor(reward).unsqueeze(1),
            torch.FloatTensor(next_state),
            torch.FloatTensor(done).unsqueeze(1),
        )

    def __len__(self):
        return len(self.buffer)


class OrnsteinUhlenbeckNoise:
    """
    Ornstein-Uhlenbeck process for exploration noise

    Implements the correct OU process:
    dx_t = theta * (mu - x_t) * dt + sigma * dW_t

    Where dW_t is a Wiener process (Brownian motion increment)
    """

    def __init__(self, size, mu=0.0, theta=0.15, sigma=0.2, dt=1e-2):
        """
        Initialize OU noise process

        Args:
            size: Dimension of the noise
            mu: Long-term mean (drift coefficient)
            theta: Mean reversion rate (how quickly it returns to mu)
            sigma: Volatility of the process
            dt: Time step size for discretization
        """
        self.size = size
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.dt = dt
        self.reset()

    def reset(self):
        """Reset the internal state to the mean"""
        self.state = self.mu.copy()

    def sample(self):
        """
        Sample the next noise value using proper OU integration

        Uses the analytical solution for the OU process:
        x_{t+dt} = x_t + theta * (mu - x_t) * dt + sigma * sqrt(dt) * N(0,1)
        """
        # Mean-reverting drift term
        drift = self.theta * (self.mu - self.state) * self.dt

        # Diffusion term with proper scaling
        diffusion = self.sigma * np.sqrt(self.dt) * np.random.randn(self.size)

        # Update state with proper integration
        self.state = self.state + drift + diffusion

        return self.state.copy()


class DDPGAlgorithm:
    """Deep Deterministic Policy Gradient Algorithm using plain PyTorch"""

    def __init__(self, env, device, **config):
        """Initialize DDPG algorithm"""
        self.env = env
        self.device = device
        self.config = config

        # Algorithm hyperparameters
        self.learning_rate_actor = config.get("learning_rate_actor", 1e-4)
        self.learning_rate_critic = config.get("learning_rate_critic", 1e-3)
        self.learning_rate = config.get(
            "learning_rate", 1e-4
        )  # Fallback for compatibility
        self.gamma = config.get("gamma", 0.99)
        self.tau = config.get("tau", 0.005)  # Soft update parameter
        self.batch_size = config.get("batch_size", 64)
        self.buffer_size = config.get("buffer_size", 100000)
        self.num_cells = config.get("num_cells", 256)
        self.max_grad_norm = config.get("max_grad_norm", 1.0)
        self.noise_sigma = config.get("noise_sigma", 0.2)
        self.noise_theta = config.get("noise_theta", 0.15)
        self.noise_dt = config.get("noise_dt", 1e-2)  # Time step for OU process
        self.warmup_steps = config.get("warmup_steps", 1000)

        # Get environment dimensions
        self.obs_dim = env.observation_space.shape[0]

        # DDPG is only for continuous actions
        if hasattr(env.action_space, "n"):
            raise ValueError("DDPG is designed for continuous action spaces only")

        self.action_dim = env.action_space.shape[0]
        self.action_low = torch.FloatTensor(env.action_space.low).to(device)
        self.action_high = torch.FloatTensor(env.action_space.high).to(device)
        self.discrete = False

        # Build networks
        self.actor = self._build_actor_network().to(device)
        self.critic = self._build_critic_network().to(device)
        self.target_actor = self._build_actor_network().to(device)
        self.target_critic = self._build_critic_network().to(device)

        # Initialize target networks with same weights
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())

        # Optimizers
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(),
            lr=float(
                self.learning_rate_actor
                if hasattr(self, "learning_rate_actor")
                else self.learning_rate
            ),
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(),
            lr=float(
                self.learning_rate_critic
                if hasattr(self, "learning_rate_critic")
                else self.learning_rate
            ),
        )

        # Replay buffer
        self.replay_buffer = ReplayBuffer(self.buffer_size)

        # Exploration noise
        self.noise = OrnsteinUhlenbeckNoise(
            self.action_dim,
            sigma=self.noise_sigma,
            theta=self.noise_theta,
            dt=self.noise_dt,
        )

        # Training step counter
        self.step_count = 0

        print(f" Simple DDPG Algorithm initialized:")
        print(
            f"   Environment: {self.obs_dim} obs -> {self.action_dim} actions (continuous)"
        )
        print(
            f"   Actor LR: {self.learning_rate_actor if hasattr(self, 'learning_rate_actor') else self.learning_rate}"
        )
        print(
            f"   Critic LR: {self.learning_rate_critic if hasattr(self, 'learning_rate_critic') else self.learning_rate}"
        )
        print(f"   Network size: {self.num_cells} cells")
        print(f"   Buffer size: {self.buffer_size}")

    def _build_actor_network(self):
        """Build actor network (policy)"""
        return nn.Sequential(
            nn.Linear(self.obs_dim, self.num_cells),
            nn.ReLU(),
            nn.Linear(self.num_cells, self.num_cells),
            nn.ReLU(),
            nn.Linear(self.num_cells, self.action_dim),
            nn.Tanh(),  # Actions are scaled to [-1, 1]
        )

    def _build_critic_network(self):
        """Build critic network (Q-function)"""

        class Critic(nn.Module):
            def __init__(self, obs_dim, action_dim, num_cells):
                super().__init__()
                self.fc1 = nn.Linear(obs_dim, num_cells)
                self.fc2 = nn.Linear(num_cells + action_dim, num_cells)
                self.fc3 = nn.Linear(num_cells, 1)

            def forward(self, state, action):
                x = F.relu(self.fc1(state))
                x = torch.cat([x, action], dim=1)
                x = F.relu(self.fc2(x))
                x = self.fc3(x)
                return x

        return Critic(self.obs_dim, self.action_dim, self.num_cells)

    def _scale_action(self, action):
        """Scale action from [-1, 1] to environment's action space"""
        return self.action_low + (action + 1.0) * 0.5 * (
            self.action_high - self.action_low
        )

    def get_action(self, obs, add_noise=True):
        """Get action from policy"""
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        obs = obs.to(self.device)

        with torch.no_grad():
            action = self.actor(obs)
            action = self._scale_action(action)

            if add_noise:
                # Add OU noise for exploration
                # During warmup, use more exploration; after warmup, gradually reduce
                noise_scale = (
                    1.0
                    if self.step_count < self.warmup_steps
                    else max(
                        0.1, 1.0 - (self.step_count - self.warmup_steps) / 100000.0
                    )
                )
                noise = (
                    torch.FloatTensor(self.noise.sample()).to(self.device) * noise_scale
                )
                action = action + noise
                action = torch.clamp(action, self.action_low, self.action_high)

            return action.squeeze(), 0.0  # Return 0.0 for log_prob compatibility

    def train_step(self, rollout_data, replay_buffer=None):
        """Perform one training step with DDPG"""
        if not rollout_data:
            return {
                "loss_total": 0.0,
                "loss_actor": 0.0,
                "loss_critic": 0.0,
                "q_value_mean": 0.0,
            }

        # Add rollout data to replay buffer
        for step in rollout_data:
            obs, action, reward, next_obs, done = step
            # Convert tensors to numpy for storage
            if isinstance(obs, torch.Tensor):
                obs = obs.cpu().numpy()
            if isinstance(action, torch.Tensor):
                action = action.cpu().numpy()
            if isinstance(reward, torch.Tensor):
                reward = reward.cpu().numpy()
            if isinstance(next_obs, torch.Tensor):
                next_obs = next_obs.cpu().numpy()
            if isinstance(done, torch.Tensor):
                done = done.cpu().numpy()

            self.replay_buffer.push(obs, action, reward, next_obs, done)

        # Don't train until we have enough samples
        if len(self.replay_buffer) < self.batch_size:
            return {
                "loss_total": 0.0,
                "loss_actor": 0.0,
                "loss_critic": 0.0,
                "q_value_mean": 0.0,
            }

        # Sample batch from replay buffer
        state, action, reward, next_state, done = self.replay_buffer.sample(
            self.batch_size
        )
        state = state.to(self.device)
        action = action.to(self.device)
        reward = reward.to(self.device)
        next_state = next_state.to(self.device)
        done = done.to(self.device)

        # Unscale actions for critic (assuming they were stored scaled)
        action_unscaled = (
            2.0 * (action - self.action_low) / (self.action_high - self.action_low)
            - 1.0
        )

        # Compute target Q value
        with torch.no_grad():
            target_action = self.target_actor(next_state)
            target_q = self.target_critic(next_state, target_action)
            target_q = reward + (1 - done) * self.gamma * target_q

        # Update critic
        current_q = self.critic(state, action_unscaled)
        critic_loss = F.mse_loss(current_q, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
        self.critic_optimizer.step()

        # Update actor
        predicted_action = self.actor(state)
        actor_loss = -self.critic(state, predicted_action).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        self.actor_optimizer.step()

        # Soft update target networks
        self._soft_update(self.target_actor, self.actor)
        self._soft_update(self.target_critic, self.critic)

        self.step_count += 1

        return {
            "loss_total": (critic_loss.item() + actor_loss.item()),
            "loss_actor": actor_loss.item(),
            "loss_critic": critic_loss.item(),
            "q_value_mean": current_q.mean().item(),
            "reward_mean": reward.mean().item(),
            "reward_std": reward.std().item(),
            "episode_length": len(rollout_data),
        }

    def _soft_update(self, target, source):
        """Soft update target network parameters"""
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.tau) + source_param.data * self.tau
            )

    def get_hyperparameters(self):
        """Get current hyperparameters for logging"""
        params = {
            "learning_rate_actor": getattr(
                self, "learning_rate_actor", self.learning_rate
            ),
            "learning_rate_critic": getattr(
                self, "learning_rate_critic", self.learning_rate
            ),
            "gamma": self.gamma,
            "tau": self.tau,
            "batch_size": self.batch_size,
            "buffer_size": self.buffer_size,
            "noise_sigma": self.noise_sigma,
            "noise_theta": self.noise_theta,
            "noise_dt": self.noise_dt,
        }
        for k, v in params.items():
            if not isinstance(v, (int, float)) or math.isnan(v) or math.isinf(v):
                logging.warning(
                    f"[DDPG] Hyperparameter '{k}' has suspicious value: {v}"
                )
        return params

    def update_hyperparameters(self, new_hyperparams):
        """Update hyperparameters during training"""
        key_map = {
            "learning_rate": "learning_rate",
            "lr": "learning_rate",
            "learning_rate_actor": "learning_rate_actor",
            "learning_rate_critic": "learning_rate_critic",
            "actor_lr": "learning_rate_actor",
            "critic_lr": "learning_rate_critic",
            "gamma": "gamma",
            "tau": "tau",
            "batch_size": "batch_size",
            "buffer_size": "buffer_size",
            "max_grad_norm": "max_grad_norm",
            "noise_sigma": "noise_sigma",
            "noise_theta": "noise_theta",
            "noise_dt": "noise_dt",
            "num_cells": "num_cells",
        }

        # Handle learning rate updates for both optimizers
        if any(
            key in new_hyperparams
            for key in ["learning_rate", "lr", "learning_rate_actor", "actor_lr"]
        ):
            lr_key = next(
                (
                    key
                    for key in [
                        "learning_rate_actor",
                        "actor_lr",
                        "learning_rate",
                        "lr",
                    ]
                    if key in new_hyperparams
                ),
                None,
            )
            if lr_key:
                old_lr = getattr(self, "learning_rate_actor", self.learning_rate)
                new_lr = float(new_hyperparams[lr_key])
                self.learning_rate_actor = new_lr
                print(f"[DDPG] Setting actor learning rate to {new_lr}")
                for param_group in self.actor_optimizer.param_groups:
                    param_group["lr"] = new_lr
                if old_lr == new_lr:
                    logging.warning(
                        f"[DDPG] Actor learning rate not changed (still {new_lr})"
                    )

        if any(key in new_hyperparams for key in ["learning_rate_critic", "critic_lr"]):
            lr_key = next(
                (
                    key
                    for key in ["learning_rate_critic", "critic_lr"]
                    if key in new_hyperparams
                ),
                None,
            )
            if lr_key:
                old_lr = getattr(self, "learning_rate_critic", self.learning_rate)
                new_lr = float(new_hyperparams[lr_key])
                self.learning_rate_critic = new_lr
                print(f"[DDPG] Setting critic learning rate to {new_lr}")
                for param_group in self.critic_optimizer.param_groups:
                    param_group["lr"] = new_lr
                if old_lr == new_lr:
                    logging.warning(
                        f"[DDPG] Critic learning rate not changed (still {new_lr})"
                    )

        # Update other hyperparameters
        for param, value in new_hyperparams.items():
            attr = key_map.get(param, param)
            if hasattr(self, attr) and param not in [
                "learning_rate_actor",
                "learning_rate_critic",
                "actor_lr",
                "critic_lr",
            ]:
                old_value = getattr(self, attr)
                setattr(self, attr, value)
                if old_value == value:
                    logging.warning(
                        f"[DDPG] Hyperparameter '{attr}' not changed (still {value})"
                    )
            elif param not in [
                "learning_rate_actor",
                "learning_rate_critic",
                "actor_lr",
                "critic_lr",
            ]:
                logging.warning(
                    f"[DDPG] Tried to update unknown hyperparameter '{attr}'"
                )

        # Update noise parameters
        if any(
            param in new_hyperparams
            for param in ["noise_sigma", "noise_theta", "noise_dt"]
        ):
            self.noise = OrnsteinUhlenbeckNoise(
                self.action_dim,
                sigma=self.noise_sigma,
                theta=self.noise_theta,
                dt=self.noise_dt,
            )

        print(f" Updated DDPG hyperparameters: {new_hyperparams}")

    def state_dict(self):
        """Get algorithm state for saving"""
        return {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "target_actor": self.target_actor.state_dict(),
            "target_critic": self.target_critic.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "hyperparameters": self.get_hyperparameters(),
            "step_count": self.step_count,
        }

    def load_state_dict(self, state_dict):
        """Load algorithm state"""
        self.actor.load_state_dict(state_dict["actor"])
        self.critic.load_state_dict(state_dict["critic"])
        self.target_actor.load_state_dict(state_dict["target_actor"])
        self.target_critic.load_state_dict(state_dict["target_critic"])
        self.actor_optimizer.load_state_dict(state_dict["actor_optimizer"])
        self.critic_optimizer.load_state_dict(state_dict["critic_optimizer"])
        if "step_count" in state_dict:
            self.step_count = state_dict["step_count"]
