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
        self.learning_rate = config.get("learning_rate", 3e-4)
        self.gamma = config.get("gamma", 0.99)
        self.lambda_gae = config.get("lambda", 0.95)
        self.clip_epsilon = config.get("clip_epsilon", 0.2)
        self.entropy_coef = config.get("entropy_coef", 1e-4)
        self.critic_coef = config.get("critic_coef", 1.0)
        self.max_grad_norm = config.get("max_grad_norm", 1.0)
        self.num_epochs = config.get("num_epochs", 10)
        self.batch_size = config.get("batch_size", 64)
        self.num_cells = config.get("num_cells", 256)

        # Get environment dimensions
        self.obs_dim = env.observation_space.shape[0]
        if hasattr(env.action_space, "n"):
            self.action_dim = env.action_space.n
            self.discrete = True
        else:
            self.action_dim = env.action_space.shape[0]
            self.discrete = False

        # Build networks
        self.policy_net = self._build_policy_network().to(device)
        self.value_net = self._build_value_network().to(device)

        # Optimizer
        self.policy_optimizer = torch.optim.Adam(
            self.policy_net.parameters(), lr=float(self.learning_rate)
        )
        self.value_optimizer = torch.optim.Adam(
            self.value_net.parameters(), lr=float(self.learning_rate)
        )

        print(f"🧠 Simple PPO Algorithm initialized:")
        print(
            f"   Environment: {self.obs_dim} obs -> {self.action_dim} actions ({'discrete' if self.discrete else 'continuous'})"
        )
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
                nn.Linear(self.num_cells, self.action_dim),
            )
        else:
            # Continuous actions - output mean and log_std
            return nn.Sequential(
                nn.Linear(self.obs_dim, self.num_cells),
                nn.Tanh(),
                nn.Linear(self.num_cells, self.num_cells),
                nn.Tanh(),
                nn.Linear(self.num_cells, self.action_dim * 2),  # mean + log_std
            )

    def _build_value_network(self):
        """Build value network"""
        return nn.Sequential(
            nn.Linear(self.obs_dim, self.num_cells),
            nn.Tanh(),
            nn.Linear(self.num_cells, self.num_cells),
            nn.Tanh(),
            nn.Linear(self.num_cells, 1),
        )

    def get_action(self, obs):
        """Get action from policy"""
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        obs = obs.to(self.device)

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
                # Return tensor instead of numpy array
                return action.squeeze(), log_prob.item()  # Remove .cpu().numpy()

    def train_step(self, rollout_data, replay_buffer=None):
        """Perform one training step with actual PPO"""
        if not rollout_data:
            return {
                "loss_total": 0.0,
                "loss_policy": 0.0,
                "loss_value": 0.0,
                "loss_entropy": 0.0,
            }

        # Convert rollout data to tensors
        observations = torch.stack([step[0] for step in rollout_data]).to(self.device)

        # Handle actions - could be int (discrete) or tensor (continuous)
        if self.discrete:
            actions = torch.tensor(
                [step[1] for step in rollout_data], dtype=torch.long
            ).to(self.device)
        else:
            actions = torch.stack([step[1] for step in rollout_data]).to(self.device)

        rewards = torch.stack([step[2] for step in rollout_data]).to(self.device)
        next_observations = torch.stack([step[3] for step in rollout_data]).to(
            self.device
        )
        dones = torch.stack([step[4] for step in rollout_data]).to(self.device)

        # Compute values and GAE advantages
        with torch.no_grad():
            values = self.value_net(observations).squeeze()
            next_values = self.value_net(next_observations).squeeze()

            # GAE computation
            advantages = torch.zeros_like(rewards)
            gae = 0
            for t in reversed(range(len(rewards))):
                if t == len(rewards) - 1:
                    next_value = 0 if dones[t] else next_values[t]
                else:
                    next_value = values[t + 1]

                delta = rewards[t] + self.gamma * next_value - values[t]
                gae = delta + self.gamma * self.lambda_gae * gae * (1 - dones[t])
                advantages[t] = gae

            returns = advantages + values
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # Get old log probs
            if self.discrete:
                old_logits = self.policy_net(observations)
                old_log_probs = (
                    F.log_softmax(old_logits, dim=-1)
                    .gather(1, actions.unsqueeze(1))
                    .squeeze()
                )
            else:
                old_output = self.policy_net(observations)
                old_mean, old_log_std = old_output.chunk(2, dim=-1)
                old_std = old_log_std.exp()
                old_dist = torch.distributions.Normal(old_mean, old_std)
                old_log_probs = old_dist.log_prob(actions).sum(dim=-1)

        # PPO training loop
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy_loss = 0

        for epoch in range(self.num_epochs):
            # Get current policy outputs
            if self.discrete:
                logits = self.policy_net(observations)
                dist = torch.distributions.Categorical(logits=logits)
                log_probs = dist.log_prob(actions)
                entropy = dist.entropy().mean()
            else:
                output = self.policy_net(observations)
                mean, log_std = output.chunk(2, dim=-1)
                std = log_std.exp()
                dist = torch.distributions.Normal(mean, std)
                log_probs = dist.log_prob(actions).sum(dim=-1)
                entropy = dist.entropy().sum(dim=-1).mean()

            # PPO clipped objective
            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = (
                torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon)
                * advantages
            )
            policy_loss = -torch.min(surr1, surr2).mean()

            # Value loss
            current_values = self.value_net(observations).squeeze()
            value_loss = F.mse_loss(current_values, returns)

            # Entropy loss
            entropy_loss = -entropy

            # Separate optimizer updates
            # Policy update
            self.policy_optimizer.zero_grad()
            policy_total_loss = policy_loss + self.entropy_coef * entropy_loss
            policy_total_loss.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(
                self.policy_net.parameters(), self.max_grad_norm
            )
            self.policy_optimizer.step()

            # Value update
            self.value_optimizer.zero_grad()
            value_total_loss = self.critic_coef * value_loss
            value_total_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.value_net.parameters(), self.max_grad_norm
            )
            self.value_optimizer.step()

            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy_loss += entropy_loss.item()

        return {
            "loss_total": (total_policy_loss + total_value_loss + total_entropy_loss)
            / self.num_epochs,
            "loss_policy": total_policy_loss / self.num_epochs,
            "loss_value": total_value_loss / self.num_epochs,
            "loss_entropy": total_entropy_loss / self.num_epochs,
            "reward_mean": rewards.mean().item(),
            "reward_std": rewards.std().item(),
            "episode_length": len(rollout_data),
        }

    def get_hyperparameters(self):
        """Get current hyperparameters for logging"""
        return {
            "learning_rate": self.learning_rate,
            "gamma": self.gamma,
            "lambda_gae": self.lambda_gae,
            "clip_epsilon": self.clip_epsilon,
            "entropy_coef": self.entropy_coef,
            "critic_coef": self.critic_coef,
            "batch_size": self.batch_size,
        }

    def update_hyperparameters(self, new_hyperparams):
        """Update hyperparameters during training"""
        if "learning_rate" in new_hyperparams:
            self.learning_rate = new_hyperparams["learning_rate"]
            # Update both optimizers
            for param_group in self.policy_optimizer.param_groups:
                param_group["lr"] = self.learning_rate
            for param_group in self.value_optimizer.param_groups:
                param_group["lr"] = self.learning_rate

        # Update other hyperparameters as needed
        for param, value in new_hyperparams.items():
            if hasattr(self, param):
                setattr(self, param, value)

        print(f"🎛️  Updated hyperparameters: {new_hyperparams}")

    def state_dict(self):
        """Get algorithm state for saving"""
        return {
            "policy_net": self.policy_net.state_dict(),
            "value_net": self.value_net.state_dict(),
            "policy_optimizer": self.policy_optimizer.state_dict(),
            "value_optimizer": self.value_optimizer.state_dict(),
            "hyperparameters": self.get_hyperparameters(),
        }

    def load_state_dict(self, state_dict):
        """Load algorithm state"""
        self.policy_net.load_state_dict(state_dict["policy_net"])
        self.value_net.load_state_dict(state_dict["value_net"])
        self.policy_optimizer.load_state_dict(state_dict["policy_optimizer"])
        self.value_optimizer.load_state_dict(state_dict["value_optimizer"])
