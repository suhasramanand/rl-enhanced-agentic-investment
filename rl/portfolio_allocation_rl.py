"""
RL Component for Portfolio Allocation and Rebalancing
Learns optimal weights for multiple stocks and when to rebalance.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Tuple, Optional
from collections import deque
import random


class PortfolioAllocationNetwork(nn.Module):
    """Neural network for portfolio allocation RL."""
    
    def __init__(self, state_size: int, num_stocks: int, hidden_sizes: List[int] = [128, 64]):
        super(PortfolioAllocationNetwork, self).__init__()
        self.state_size = state_size
        self.num_stocks = num_stocks
        
        layers = []
        input_size = state_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.ReLU())
            input_size = hidden_size
        # Output: weights for each stock (will be normalized to sum to 1)
        layers.append(nn.Linear(input_size, num_stocks))
        layers.append(nn.Softmax(dim=-1))  # Ensure weights sum to 1
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, state):
        """Forward pass - returns portfolio weights."""
        return self.network(state)


class PortfolioAllocationRL:
    """
    RL agent for learning optimal portfolio allocation and rebalancing.
    """
    
    def __init__(
        self,
        state_size: int,
        num_stocks: int = 5,
        learning_rate: float = 0.001,
        discount_factor: float = 0.95,
        memory_size: int = 10000,
        batch_size: int = 32,
        device: str = 'cpu'
    ):
        self.state_size = state_size
        self.num_stocks = num_stocks
        self.learning_rate = learning_rate
        self.gamma = discount_factor
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.device = torch.device(device)
        
        # Networks
        self.policy_net = PortfolioAllocationNetwork(state_size, num_stocks).to(self.device)
        self.target_net = PortfolioAllocationNetwork(state_size, num_stocks).to(self.device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        
        # Experience replay
        self.memory = deque(maxlen=memory_size)
        
        # Rebalancing threshold
        self.rebalance_threshold = 0.05  # Rebalance if weights drift by 5%
    
    def select_allocation(self, state: np.ndarray, current_weights: Optional[np.ndarray] = None) -> Tuple[np.ndarray, bool]:
        """
        Select portfolio allocation.
        
        Args:
            state: Current market state
            current_weights: Current portfolio weights (for rebalancing decision)
        
        Returns:
            Tuple of (new_weights, should_rebalance)
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            new_weights = self.policy_net(state_tensor).cpu().numpy()[0]
        
        # Decide if rebalancing is needed
        should_rebalance = False
        if current_weights is not None:
            weight_drift = np.abs(new_weights - current_weights).max()
            if weight_drift > self.rebalance_threshold:
                should_rebalance = True
        
        return new_weights, should_rebalance
    
    def store_experience(self, state: np.ndarray, weights: np.ndarray, reward: float, 
                        next_state: np.ndarray, done: bool):
        """Store experience in replay buffer."""
        self.memory.append((state, weights, reward, next_state, done))
    
    def train_step(self) -> Optional[float]:
        """Train on a batch of experiences."""
        if len(self.memory) < self.batch_size:
            return None
        
        batch = random.sample(self.memory, self.batch_size)
        states, weights, rewards, next_states, dones = zip(*batch)
        
        states = torch.FloatTensor(np.array(states)).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        dones = torch.BoolTensor(dones).to(self.device)
        
        # Compute Q-values
        current_q = self.policy_net(states)
        next_q = self.target_net(next_states)
        
        # Target Q-values
        target_q = rewards.unsqueeze(1) + (self.gamma * next_q.max(dim=1)[0] * ~dones).unsqueeze(1)
        
        # Loss (mean squared error)
        loss = nn.MSELoss()(current_q, target_q)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def update_target_network(self):
        """Update target network with policy network weights."""
        self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def save(self, filepath: str):
        """Save model."""
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, filepath)
    
    def load(self, filepath: str):
        """Load model."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])

