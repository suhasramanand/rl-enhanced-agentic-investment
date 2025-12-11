"""
RL Component for Position Sizing
Learns optimal position sizes based on confidence, volatility, and risk.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Tuple, Optional
from collections import deque
import random


class PositionSizingNetwork(nn.Module):
    """Neural network for position sizing decisions."""
    
    def __init__(self, state_size: int, hidden_sizes: List[int] = [128, 64]):
        super(PositionSizingNetwork, self).__init__()
        self.state_size = state_size
        
        layers = []
        input_size = state_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.ReLU())
            input_size = hidden_size
        # Output: position size as percentage (0-1)
        layers.append(nn.Linear(input_size, 1))
        layers.append(nn.Sigmoid())  # Ensure output is between 0 and 1
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, state):
        """Forward pass - returns position size (0-1)."""
        return self.network(state)


class PositionSizingRL:
    """
    RL agent for learning optimal position sizing.
    """
    
    def __init__(
        self,
        state_size: int,
        learning_rate: float = 0.001,
        discount_factor: float = 0.95,
        memory_size: int = 10000,
        batch_size: int = 32,
        device: str = 'cpu'
    ):
        self.state_size = state_size
        self.learning_rate = learning_rate
        self.gamma = discount_factor
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.device = torch.device(device)
        
        # Networks
        self.policy_net = PositionSizingNetwork(state_size).to(self.device)
        self.target_net = PositionSizingNetwork(state_size).to(self.device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        
        # Experience replay
        self.memory = deque(maxlen=memory_size)
    
    def select_position_size(self, state: np.ndarray) -> float:
        """
        Select position size (0-1, representing percentage of portfolio).
        
        Args:
            state: Current state (confidence, volatility, risk metrics, etc.)
        
        Returns:
            Position size as float between 0 and 1
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            position_size = self.policy_net(state_tensor).cpu().item()
        return float(position_size)
    
    def store_experience(self, state: np.ndarray, position_size: float, reward: float,
                        next_state: np.ndarray, done: bool):
        """Store experience in replay buffer."""
        self.memory.append((state, position_size, reward, next_state, done))
    
    def train_step(self) -> Optional[float]:
        """Train on a batch of experiences."""
        if len(self.memory) < self.batch_size:
            return None
        
        batch = random.sample(self.memory, self.batch_size)
        states, position_sizes, rewards, next_states, dones = zip(*batch)
        
        states = torch.FloatTensor(np.array(states)).to(self.device)
        position_sizes = torch.FloatTensor(position_sizes).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.BoolTensor(dones).to(self.device)
        
        # Current predictions
        current_pred = self.policy_net(states)
        
        # Target (reward + discounted future value)
        next_pred = self.target_net(next_states)
        target = rewards.unsqueeze(1) + (self.gamma * next_pred * ~dones.unsqueeze(1))
        
        # Loss (mean squared error)
        loss = nn.MSELoss()(current_pred, target)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def update_target_network(self):
        """Update target network."""
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

