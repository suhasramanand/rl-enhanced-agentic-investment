"""
RL Component for News Sentiment Weighting
Learns how much weight to give different news sources and adjusts based on stock characteristics.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Tuple, Optional
from collections import deque
import random


class SentimentWeightingNetwork(nn.Module):
    """Neural network for sentiment weighting decisions."""
    
    def __init__(self, state_size: int, num_sources: int = 5, hidden_sizes: List[int] = [128, 64]):
        super(SentimentWeightingNetwork, self).__init__()
        self.state_size = state_size
        self.num_sources = num_sources
        
        layers = []
        input_size = state_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.ReLU())
            input_size = hidden_size
        # Output: weights for each news source (will be normalized)
        layers.append(nn.Linear(input_size, num_sources))
        layers.append(nn.Softmax(dim=-1))  # Ensure weights sum to 1
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, state):
        """Forward pass - returns source weights."""
        return self.network(state)


class SentimentWeightingRL:
    """
    RL agent for learning optimal news sentiment weighting.
    """
    
    NEWS_SOURCES = ['yahoo_finance', 'marketwatch', 'bloomberg', 'reuters', 'general']
    
    def __init__(
        self,
        state_size: int,
        num_sources: int = 5,
        learning_rate: float = 0.001,
        discount_factor: float = 0.95,
        memory_size: int = 10000,
        batch_size: int = 32,
        device: str = 'cpu'
    ):
        self.state_size = state_size
        self.num_sources = num_sources
        self.learning_rate = learning_rate
        self.gamma = discount_factor
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.device = torch.device(device)
        
        # Networks
        self.policy_net = SentimentWeightingNetwork(state_size, num_sources).to(self.device)
        self.target_net = SentimentWeightingNetwork(state_size, num_sources).to(self.device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        
        # Experience replay
        self.memory = deque(maxlen=memory_size)
    
    def get_source_weights(self, state: np.ndarray) -> Dict[str, float]:
        """
        Get weights for each news source.
        
        Returns:
            Dictionary mapping source names to weights
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            weights = self.policy_net(state_tensor).cpu().numpy()[0]
        
        return {source: float(w) for source, w in zip(self.NEWS_SOURCES, weights)}
    
    def apply_weights(self, sentiments: Dict[str, float], weights: Dict[str, float]) -> float:
        """
        Apply learned weights to sentiment scores.
        
        Returns:
            Weighted sentiment score
        """
        weighted_sum = sum(sentiments.get(source, 0.0) * weights.get(source, 0.0) 
                          for source in self.NEWS_SOURCES)
        return weighted_sum
    
    def filter_noise(self, sentiment_scores: List[float], weights: Dict[str, float]) -> float:
        """
        Filter noise from signal using learned weights.
        
        Returns:
            Filtered sentiment score
        """
        # Weighted average with outlier removal
        weighted_scores = [s * w for s, w in zip(sentiment_scores, weights.values())]
        # Remove outliers (beyond 2 standard deviations)
        if len(weighted_scores) > 2:
            mean = np.mean(weighted_scores)
            std = np.std(weighted_scores)
            filtered = [s for s in weighted_scores if abs(s - mean) <= 2 * std]
            return float(np.mean(filtered) if filtered else mean)
        return float(np.mean(weighted_scores))
    
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

