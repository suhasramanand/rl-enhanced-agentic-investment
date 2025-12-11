"""
Proximal Policy Optimization (PPO) Implementation
Policy gradient method for continuous action spaces and stopping decisions.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os
from typing import Tuple, Optional, List
from .rollout_buffer import RolloutBuffer


class ActorNetwork(nn.Module):
    """
    Actor network for PPO.
    Outputs action probabilities (policy).
    """
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: List[int] = [128, 128]):
        """
        Initialize actor network.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            hidden_dims: List of hidden layer dimensions
        """
        super(ActorNetwork, self).__init__()
        
        layers = []
        input_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        
        layers.append(nn.Linear(input_dim, action_dim))
        layers.append(nn.Softmax(dim=-1))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            state: State tensor
        
        Returns:
            Action probability distribution
        """
        return self.network(state)
    
    def get_action_and_log_prob(self, state: torch.Tensor) -> Tuple[int, float]:
        """
        Sample action from policy and compute log probability.
        
        Args:
            state: State tensor
        
        Returns:
            Tuple of (action, log_probability)
        """
        probs = self.forward(state)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        return action.item(), log_prob.item()


class CriticNetwork(nn.Module):
    """
    Critic network for PPO.
    Estimates state values.
    """
    
    def __init__(self, state_dim: int, hidden_dims: List[int] = [128, 128]):
        """
        Initialize critic network.
        
        Args:
            state_dim: Dimension of state space
            hidden_dims: List of hidden layer dimensions
        """
        super(CriticNetwork, self).__init__()
        
        layers = []
        input_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        
        layers.append(nn.Linear(input_dim, 1))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            state: State tensor
        
        Returns:
            State value estimate
        """
        return self.network(state).squeeze()


class PPO:
    """
    Proximal Policy Optimization algorithm.
    Used for learning stopping policy and planning decisions.
    """
    
    # Action space: [CONTINUE, ANALYZE_MORE, STOP]
    ACTIONS = ['CONTINUE', 'ANALYZE_MORE', 'STOP']
    ACTION_TO_IDX = {action: idx for idx, action in enumerate(ACTIONS)}
    IDX_TO_ACTION = {idx: action for idx, action in enumerate(ACTIONS)}
    
    def __init__(
        self,
        state_dim: int,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        lambda_gae: float = 0.95,
        clip_epsilon: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        device: str = 'cpu'
    ):
        """
        Initialize PPO agent.
        
        Args:
            state_dim: Dimension of state space
            learning_rate: Learning rate
            gamma: Discount factor
            lambda_gae: GAE lambda parameter
            clip_epsilon: PPO clipping parameter
            value_coef: Value loss coefficient
            entropy_coef: Entropy bonus coefficient
            max_grad_norm: Maximum gradient norm for clipping
            device: Device for computation ('cpu' or 'cuda')
        """
        self.state_dim = state_dim
        self.action_dim = len(self.ACTIONS)
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.lambda_gae = lambda_gae
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.device = device
        
        # Networks
        self.actor = ActorNetwork(state_dim, self.action_dim).to(device)
        self.critic = CriticNetwork(state_dim).to(device)
        
        # Optimizer
        self.optimizer = optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=learning_rate
        )
        
        # Statistics
        self.episode_rewards = []
        self.episode_lengths = []
        self.policy_losses = []
        self.value_losses = []
        self.entropy_losses = []
    
    def select_action(self, state: np.ndarray) -> Tuple[int, str, float, float]:
        """
        Select action using current policy.
        
        Args:
            state: State observation
        
        Returns:
            Tuple of (action_index, action_name, log_prob, value)
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action, log_prob = self.actor.get_action_and_log_prob(state_tensor)
            value = self.critic(state_tensor).item()
        
        action_name = self.IDX_TO_ACTION[action]
        
        return action, action_name, log_prob, value
    
    def update(self, buffer: RolloutBuffer, epochs: int = 10):
        """
        Update policy using PPO clipped objective.
        
        Args:
            buffer: Rollout buffer with collected trajectories
            epochs: Number of update epochs
        """
        if buffer.size == 0:
            return
        
        # Compute returns and advantages
        next_value = 0.0  # Assuming terminal state value is 0
        returns, advantages = buffer.compute_returns_and_advantages(
            next_value, self.gamma, self.lambda_gae
        )
        
        # Get batch data
        states, actions, old_log_probs, old_values = buffer.get_batch()
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy_loss = 0
        
        # Multiple epochs of updates
        for epoch in range(epochs):
            # Get current policy predictions
            probs = self.actor(states)
            dist = torch.distributions.Categorical(probs)
            new_log_probs = dist.log_prob(actions)
            entropy = dist.entropy().mean()
            
            # Get current value estimates
            values = self.critic(states)
            
            # Compute policy loss (PPO clipped objective)
            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Compute value loss
            value_loss = F.mse_loss(values, returns)
            
            # Total loss
            loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
            
            # Update
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.actor.parameters()) + list(self.critic.parameters()),
                self.max_grad_norm
            )
            self.optimizer.step()
            
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy_loss += entropy.item()
        
        # Store statistics
        self.policy_losses.append(total_policy_loss / epochs)
        self.value_losses.append(total_value_loss / epochs)
        self.entropy_losses.append(total_entropy_loss / epochs)
    
    def save(self, filepath: str):
        """
        Save model weights.
        
        Args:
            filepath: Path to save file (without extension)
        """
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'policy_losses': self.policy_losses,
            'value_losses': self.value_losses,
            'entropy_losses': self.entropy_losses
        }, filepath)
    
    def load(self, filepath: str):
        """
        Load model weights.
        
        Args:
            filepath: Path to load file
        """
        if not os.path.exists(filepath):
            print(f"Warning: File {filepath} not found. Starting with random weights.")
            return
        
        checkpoint = torch.load(filepath, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.episode_rewards = checkpoint.get('episode_rewards', [])
        self.episode_lengths = checkpoint.get('episode_lengths', [])
        self.policy_losses = checkpoint.get('policy_losses', [])
        self.value_losses = checkpoint.get('value_losses', [])
        self.entropy_losses = checkpoint.get('entropy_losses', [])

