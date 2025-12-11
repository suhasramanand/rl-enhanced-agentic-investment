"""
RL Component for Stop Loss and Take Profit Levels
Learns optimal stop-loss percentages and take-profit targets, adapting to volatility.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Tuple, Optional
from collections import deque
import random


class StopLossTakeProfitNetwork(nn.Module):
    """Neural network for stop-loss and take-profit decisions."""
    
    def __init__(self, state_size: int, hidden_sizes: List[int] = [128, 64]):
        super(StopLossTakeProfitNetwork, self).__init__()
        self.state_size = state_size
        
        layers = []
        input_size = state_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.ReLU())
            input_size = hidden_size
        # Output: [stop_loss_percentage, take_profit_percentage]
        layers.append(nn.Linear(input_size, 2))
        layers.append(nn.Sigmoid())  # Ensure outputs are between 0 and 1
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, state):
        """Forward pass - returns [stop_loss_pct, take_profit_pct]."""
        return self.network(state)


class StopLossTakeProfitRL:
    """
    RL agent for learning optimal stop-loss and take-profit levels.
    """
    
    def __init__(
        self,
        state_size: int,
        learning_rate: float = 0.001,
        discount_factor: float = 0.95,
        memory_size: int = 10000,
        batch_size: int = 32,
        device: str = 'cpu',
        max_stop_loss_pct: float = 0.10,  # Max 10% stop loss
        max_take_profit_pct: float = 0.30  # Max 30% take profit
    ):
        self.state_size = state_size
        self.learning_rate = learning_rate
        self.gamma = discount_factor
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.device = torch.device(device)
        self.max_stop_loss_pct = max_stop_loss_pct
        self.max_take_profit_pct = max_take_profit_pct
        
        # Networks
        self.policy_net = StopLossTakeProfitNetwork(state_size).to(self.device)
        self.target_net = StopLossTakeProfitNetwork(state_size).to(self.device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        
        # Experience replay
        self.memory = deque(maxlen=memory_size)
    
    def get_levels(self, state: np.ndarray, current_price: float) -> Dict[str, float]:
        """
        Get stop-loss and take-profit levels.
        
        Args:
            state: Current state (volatility, price action, etc.)
            current_price: Current stock price
        
        Returns:
            Dictionary with 'stop_loss' and 'take_profit' prices
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            percentages = self.policy_net(state_tensor).cpu().numpy()[0]
        
        stop_loss_pct = float(percentages[0] * self.max_stop_loss_pct)
        take_profit_pct = float(percentages[1] * self.max_take_profit_pct)
        
        return {
            'stop_loss': current_price * (1 - stop_loss_pct),
            'take_profit': current_price * (1 + take_profit_pct),
            'stop_loss_pct': stop_loss_pct,
            'take_profit_pct': take_profit_pct
        }
    
    def adapt_to_volatility(self, state: np.ndarray, base_stop_loss: float, 
                           base_take_profit: float) -> Dict[str, float]:
        """
        Adapt stop-loss and take-profit based on volatility.
        
        Returns:
            Adjusted levels
        """
        # Extract volatility from state (assuming it's the first feature or a specific index)
        volatility = state[0] if len(state) > 0 else 0.02
        
        # Higher volatility = wider stops
        volatility_multiplier = 1.0 + (volatility * 2)
        
        return {
            'stop_loss': base_stop_loss * volatility_multiplier,
            'take_profit': base_take_profit * volatility_multiplier
        }
    
    def store_experience(self, state: np.ndarray, levels: np.ndarray, reward: float,
                        next_state: np.ndarray, done: bool):
        """Store experience in replay buffer."""
        self.memory.append((state, levels, reward, next_state, done))
    
    def train_step(self) -> Optional[float]:
        """Train on a batch of experiences."""
        if len(self.memory) < self.batch_size:
            return None
        
        batch = random.sample(self.memory, self.batch_size)
        states, levels, rewards, next_states, dones = zip(*batch)
        
        states = torch.FloatTensor(np.array(states)).to(self.device)
        levels = torch.FloatTensor(np.array(levels)).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.BoolTensor(dones).to(self.device)
        
        # Current predictions
        current_pred = self.policy_net(states)
        
        # Target (reward + discounted future value)
        next_pred = self.target_net(next_states)
        target = levels.unsqueeze(0) + rewards.unsqueeze(1).unsqueeze(2) * (1 - dones.float()).unsqueeze(1).unsqueeze(2)
        
        # Loss (mean squared error)
        loss = nn.MSELoss()(current_pred, target.squeeze(0))
        
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

