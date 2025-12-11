"""
RL Component for Multi-Timeframe Analysis
Learns which timeframes are most predictive and combines signals.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Tuple, Optional
from collections import deque
import random


class MultiTimeframeNetwork(nn.Module):
    """Neural network for multi-timeframe analysis."""
    
    def __init__(self, state_size: int, num_timeframes: int = 4, hidden_sizes: List[int] = [128, 64]):
        super(MultiTimeframeNetwork, self).__init__()
        self.state_size = state_size
        self.num_timeframes = num_timeframes
        
        layers = []
        input_size = state_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.ReLU())
            input_size = hidden_size
        # Output: weights for each timeframe (will be normalized)
        layers.append(nn.Linear(input_size, num_timeframes))
        layers.append(nn.Softmax(dim=-1))  # Ensure weights sum to 1
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, state):
        """Forward pass - returns timeframe weights."""
        return self.network(state)


class MultiTimeframeRL:
    """
    RL agent for learning optimal multi-timeframe analysis.
    """
    
    TIMEFRAMES = ['1D', '1W', '1M', '3M']  # Daily, Weekly, Monthly, Quarterly
    
    def __init__(
        self,
        state_size: int,
        num_timeframes: int = 4,
        learning_rate: float = 0.001,
        discount_factor: float = 0.95,
        memory_size: int = 10000,
        batch_size: int = 32,
        device: str = 'cpu'
    ):
        self.state_size = state_size
        self.num_timeframes = num_timeframes
        self.learning_rate = learning_rate
        self.gamma = discount_factor
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.device = torch.device(device)
        
        # Networks
        self.policy_net = MultiTimeframeNetwork(state_size, num_timeframes).to(self.device)
        self.target_net = MultiTimeframeNetwork(state_size, num_timeframes).to(self.device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        
        # Experience replay
        self.memory = deque(maxlen=memory_size)
    
    def get_timeframe_weights(self, state: np.ndarray) -> Dict[str, float]:
        """
        Get weights for each timeframe.
        
        Returns:
            Dictionary mapping timeframe names to weights
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            weights = self.policy_net(state_tensor).cpu().numpy()[0]
        
        return {tf: float(w) for tf, w in zip(self.TIMEFRAMES, weights)}
    
    def combine_timeframe_signals(self, signals: Dict[str, float], weights: Dict[str, float]) -> float:
        """
        Combine signals from different timeframes using learned weights.
        
        Returns:
            Combined signal score
        """
        combined = sum(signals.get(tf, 0.0) * weights.get(tf, 0.0) for tf in self.TIMEFRAMES)
        return combined
    
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
        weights = torch.FloatTensor(np.array(weights)).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.BoolTensor(dones).to(self.device)
        
        # Current predictions
        current_weights = self.policy_net(states)
        
        # Target (reward + discounted future value)
        next_weights = self.target_net(next_states)
        target = weights.unsqueeze(0) + rewards.unsqueeze(1).unsqueeze(2) * (1 - dones.float()).unsqueeze(1).unsqueeze(2)
        
        # Loss (mean squared error)
        loss = nn.MSELoss()(current_weights, target.squeeze(0))
        
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

