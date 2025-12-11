"""
RL Component for Confidence Calibration
Learns to calibrate confidence scores based on outcomes and improve accuracy.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Tuple, Optional
from collections import deque
import random


class ConfidenceCalibrationNetwork(nn.Module):
    """Neural network for confidence calibration."""
    
    def __init__(self, state_size: int, hidden_sizes: List[int] = [128, 64]):
        super(ConfidenceCalibrationNetwork, self).__init__()
        self.state_size = state_size
        
        layers = []
        input_size = state_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.ReLU())
            input_size = hidden_size
        # Output: calibrated confidence (0-1)
        layers.append(nn.Linear(input_size, 1))
        layers.append(nn.Sigmoid())  # Ensure output is between 0 and 1
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, state):
        """Forward pass - returns calibrated confidence."""
        return self.network(state)


class ConfidenceCalibrationRL:
    """
    RL agent for learning optimal confidence calibration.
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
        self.policy_net = ConfidenceCalibrationNetwork(state_size).to(self.device)
        self.target_net = ConfidenceCalibrationNetwork(state_size).to(self.device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        
        # Experience replay
        self.memory = deque(maxlen=memory_size)
        
        # Buy/Hold/Sell thresholds (learned)
        self.buy_threshold = 0.6
        self.sell_threshold = 0.4
    
    def calibrate_confidence(self, raw_confidence: float, state: np.ndarray) -> float:
        """
        Calibrate confidence score based on state.
        
        Args:
            raw_confidence: Raw confidence from recommendation agent
            state: Current state (features, indicators, etc.)
        
        Returns:
            Calibrated confidence (0-1)
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            calibration_factor = self.policy_net(state_tensor).cpu().item()
        
        # Apply calibration
        calibrated = raw_confidence * calibration_factor
        return float(np.clip(calibrated, 0.0, 1.0))
    
    def get_recommendation_threshold(self, state: np.ndarray) -> Dict[str, float]:
        """
        Get learned thresholds for Buy/Hold/Sell recommendations.
        
        Returns:
            Dictionary with 'buy_threshold' and 'sell_threshold'
        """
        # Could learn these dynamically, for now return current values
        return {
            'buy_threshold': self.buy_threshold,
            'sell_threshold': self.sell_threshold
        }
    
    def store_experience(self, state: np.ndarray, confidence: float, actual_outcome: float,
                        next_state: np.ndarray, done: bool):
        """
        Store experience in replay buffer.
        
        Args:
            actual_outcome: Actual return/performance (for reward calculation)
        """
        # Reward: how well confidence predicted outcome
        reward = 1.0 - abs(confidence - actual_outcome)
        self.memory.append((state, confidence, reward, next_state, done))
    
    def train_step(self) -> Optional[float]:
        """Train on a batch of experiences."""
        if len(self.memory) < self.batch_size:
            return None
        
        batch = random.sample(self.memory, self.batch_size)
        states, confidences, rewards, next_states, dones = zip(*batch)
        
        states = torch.FloatTensor(np.array(states)).to(self.device)
        confidences = torch.FloatTensor(confidences).unsqueeze(1).to(self.device)
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
            'buy_threshold': self.buy_threshold,
            'sell_threshold': self.sell_threshold
        }, filepath)
    
    def load(self, filepath: str):
        """Load model."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.buy_threshold = checkpoint.get('buy_threshold', 0.6)
        self.sell_threshold = checkpoint.get('sell_threshold', 0.4)

