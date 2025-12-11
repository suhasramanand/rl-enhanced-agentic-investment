"""
RL Component for Risk Management
Learns when to reduce exposure, adjust stop-loss levels, and manage drawdowns.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Tuple, Optional
from collections import deque
import random


class RiskManagementNetwork(nn.Module):
    """Neural network for risk management decisions."""
    
    def __init__(self, state_size: int, hidden_sizes: List[int] = [128, 64]):
        super(RiskManagementNetwork, self).__init__()
        self.state_size = state_size
        
        # Actions: REDUCE_EXPOSURE, ADJUST_STOP_LOSS, HOLD, INCREASE_EXPOSURE
        self.action_size = 4
        
        layers = []
        input_size = state_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.ReLU())
            input_size = hidden_size
        layers.append(nn.Linear(input_size, self.action_size))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, state):
        """Forward pass - returns Q-values for each action."""
        return self.network(state)


class RiskManagementRL:
    """
    RL agent for learning optimal risk management strategies.
    """
    
    ACTIONS = ['REDUCE_EXPOSURE', 'ADJUST_STOP_LOSS', 'HOLD', 'INCREASE_EXPOSURE']
    ACTION_TO_IDX = {action: idx for idx, action in enumerate(ACTIONS)}
    IDX_TO_ACTION = {idx: action for idx, action in enumerate(ACTIONS)}
    
    def __init__(
        self,
        state_size: int,
        learning_rate: float = 0.001,
        discount_factor: float = 0.95,
        epsilon: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.995,
        memory_size: int = 10000,
        batch_size: int = 32,
        device: str = 'cpu'
    ):
        self.state_size = state_size
        self.learning_rate = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.device = torch.device(device)
        
        # Networks
        self.q_network = RiskManagementNetwork(state_size).to(self.device)
        self.target_network = RiskManagementNetwork(state_size).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Experience replay
        self.memory = deque(maxlen=memory_size)
    
    def select_action(self, state: np.ndarray, training: bool = True) -> Tuple[int, str]:
        """
        Select risk management action.
        
        Returns:
            Tuple of (action_idx, action_name)
        """
        if training and random.random() < self.epsilon:
            action_idx = random.randrange(len(self.ACTIONS))
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.q_network(state_tensor)
                action_idx = q_values.argmax().item()
        
        return action_idx, self.ACTIONS[action_idx]
    
    def get_stop_loss_adjustment(self, state: np.ndarray, current_stop_loss: float, 
                                 current_price: float) -> float:
        """
        Get recommended stop-loss adjustment.
        
        Returns:
            New stop-loss price
        """
        action_idx, action_name = self.select_action(state, training=False)
        
        if action_name == 'ADJUST_STOP_LOSS':
            # Adjust based on volatility in state
            volatility = state[0] if len(state) > 0 else 0.02
            # Move stop loss based on volatility (trailing stop)
            adjustment = current_price * volatility * 2  # 2x volatility buffer
            new_stop_loss = current_price - adjustment
            return max(new_stop_loss, current_stop_loss * 0.95)  # Don't move stop loss too close
        else:
            return current_stop_loss
    
    def store_experience(self, state: np.ndarray, action: int, reward: float,
                        next_state: np.ndarray, done: bool):
        """Store experience in replay buffer."""
        self.memory.append((state, action, reward, next_state, done))
    
    def train_step(self) -> Optional[float]:
        """Train on a batch of experiences."""
        if len(self.memory) < self.batch_size:
            return None
        
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.BoolTensor(dones).to(self.device)
        
        # Current Q-values
        current_q = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Next Q-values
        next_q = self.target_network(next_states).max(1)[0].detach()
        target_q = rewards + (self.gamma * next_q * ~dones)
        
        # Loss
        loss = nn.MSELoss()(current_q.squeeze(), target_q)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        return loss.item()
    
    def update_target_network(self):
        """Update target network."""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def save(self, filepath: str):
        """Save model."""
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, filepath)
    
    def load(self, filepath: str):
        """Load model."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint.get('epsilon', self.epsilon_min)

