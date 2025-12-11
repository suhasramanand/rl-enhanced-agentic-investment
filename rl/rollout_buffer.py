"""
Rollout Buffer for PPO
Stores trajectories for policy gradient updates.
"""

import numpy as np
from typing import List, Tuple
import torch


class RolloutBuffer:
    """
    Buffer for storing rollout data for PPO training.
    """
    
    def __init__(self, capacity: int, state_dim: int, action_dim: int, device: str = 'cpu'):
        """
        Initialize rollout buffer.
        
        Args:
            capacity: Maximum number of steps to store
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            device: Device for tensor operations ('cpu' or 'cuda')
        """
        self.capacity = capacity
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        
        # Storage arrays
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        self.dones = []
        
        self.ptr = 0
        self.size = 0
    
    def store(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        log_prob: float,
        value: float,
        done: bool
    ):
        """
        Store a single transition.
        
        Args:
            state: State observation
            action: Action taken
            reward: Reward received
            log_prob: Log probability of action
            value: Value estimate
            done: Whether episode terminated
        """
        if self.size < self.capacity:
            self.states.append(state)
            self.actions.append(action)
            self.rewards.append(reward)
            self.log_probs.append(log_prob)
            self.values.append(value)
            self.dones.append(done)
            self.size += 1
        else:
            # Overwrite oldest entry
            idx = self.ptr % self.capacity
            self.states[idx] = state
            self.actions[idx] = action
            self.rewards[idx] = reward
            self.log_probs[idx] = log_prob
            self.values[idx] = value
            self.dones[idx] = done
            self.ptr += 1
    
    def compute_returns_and_advantages(
        self,
        next_value: float,
        gamma: float = 0.99,
        lambda_gae: float = 0.95
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute returns and advantages using Generalized Advantage Estimation (GAE).
        
        Args:
            next_value: Value estimate for next state
            gamma: Discount factor
            lambda_gae: GAE lambda parameter
        
        Returns:
            Tuple of (returns, advantages) as tensors
        """
        rewards = np.array(self.rewards)
        values = np.array(self.values + [next_value])
        dones = np.array(self.dones)
        
        # Compute returns
        returns = np.zeros_like(rewards)
        advantages = np.zeros_like(rewards)
        
        gae = 0
        for t in reversed(range(len(rewards))):
            if dones[t]:
                gae = 0
            
            delta = rewards[t] + gamma * values[t + 1] * (1 - dones[t]) - values[t]
            gae = delta + gamma * lambda_gae * (1 - dones[t]) * gae
            advantages[t] = gae
            returns[t] = advantages[t] + values[t]
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Convert to tensors
        returns_tensor = torch.FloatTensor(returns).to(self.device)
        advantages_tensor = torch.FloatTensor(advantages).to(self.device)
        
        return returns_tensor, advantages_tensor
    
    def get_batch(self) -> Tuple[torch.Tensor, ...]:
        """
        Get all stored data as tensors.
        
        Returns:
            Tuple of (states, actions, log_probs, returns, advantages, values)
        """
        states_tensor = torch.FloatTensor(np.array(self.states)).to(self.device)
        actions_tensor = torch.LongTensor(np.array(self.actions)).to(self.device)
        log_probs_tensor = torch.FloatTensor(np.array(self.log_probs)).to(self.device)
        values_tensor = torch.FloatTensor(np.array(self.values)).to(self.device)
        
        return states_tensor, actions_tensor, log_probs_tensor, values_tensor
    
    def clear(self):
        """Clear the buffer."""
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        self.dones = []
        self.ptr = 0
        self.size = 0
    
    def __len__(self) -> int:
        """Return current buffer size."""
        return self.size

