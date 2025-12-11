"""
RL Component for Portfolio-Level Optimization
Learns correlations between stocks, optimizes portfolio-level risk/return, and learns hedging strategies.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Tuple, Optional
from collections import deque
import random


class PortfolioOptimizationNetwork(nn.Module):
    """Neural network for portfolio-level optimization."""
    
    def __init__(self, state_size: int, num_stocks: int, hidden_sizes: List[int] = [128, 64]):
        super(PortfolioOptimizationNetwork, self).__init__()
        self.state_size = state_size
        self.num_stocks = num_stocks
        
        layers = []
        input_size = state_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.ReLU())
            input_size = hidden_size
        # Output: [portfolio_return_prediction, risk_score, hedging_signal]
        layers.append(nn.Linear(input_size, 3))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, state):
        """Forward pass - returns [return_pred, risk_score, hedge_signal]."""
        return self.network(state)


class PortfolioOptimizationRL:
    """
    RL agent for learning portfolio-level optimization strategies.
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
        self.policy_net = PortfolioOptimizationNetwork(state_size, num_stocks).to(self.device)
        self.target_net = PortfolioOptimizationNetwork(state_size, num_stocks).to(self.device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        
        # Experience replay
        self.memory = deque(maxlen=memory_size)
        
        # Correlation matrix (learned)
        self.correlation_matrix = np.eye(num_stocks)
    
    def compute_correlations(self, returns: np.ndarray) -> np.ndarray:
        """
        Compute correlation matrix from stock returns.
        
        Args:
            returns: Array of shape (num_stocks, time_periods)
        
        Returns:
            Correlation matrix
        """
        if returns.shape[0] < 2:
            return self.correlation_matrix
        
        correlation = np.corrcoef(returns)
        # Update learned correlation matrix (exponential moving average)
        self.correlation_matrix = 0.9 * self.correlation_matrix + 0.1 * correlation
        return self.correlation_matrix
    
    def optimize_portfolio(self, state: np.ndarray, current_weights: np.ndarray,
                          returns: np.ndarray) -> Dict[str, any]:
        """
        Optimize portfolio-level risk/return.
        
        Returns:
            Dictionary with optimized weights, risk score, and hedging signal
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            output = self.policy_net(state_tensor).cpu().numpy()[0]
        
        return_pred = float(output[0])
        risk_score = float(output[1])
        hedge_signal = float(output[2])
        
        # Compute correlations
        correlation_matrix = self.compute_correlations(returns)
        
        # Simple optimization: adjust weights based on risk/return tradeoff
        # Higher risk score = reduce exposure to high-risk stocks
        risk_adjusted_weights = current_weights * (1 - risk_score * 0.5)
        risk_adjusted_weights = risk_adjusted_weights / risk_adjusted_weights.sum()
        
        return {
            'optimized_weights': risk_adjusted_weights,
            'expected_return': return_pred,
            'risk_score': risk_score,
            'hedge_signal': hedge_signal,
            'correlation_matrix': correlation_matrix
        }
    
    def get_hedging_strategy(self, state: np.ndarray, portfolio_weights: np.ndarray,
                            correlation_matrix: np.ndarray) -> Dict[str, any]:
        """
        Get hedging strategy based on correlations.
        
        Returns:
            Dictionary with hedging recommendations
        """
        # Find negatively correlated pairs for hedging
        hedge_pairs = []
        for i in range(len(portfolio_weights)):
            for j in range(i + 1, len(portfolio_weights)):
                if correlation_matrix[i, j] < -0.5:  # Strong negative correlation
                    hedge_pairs.append({
                        'stock1': i,
                        'stock2': j,
                        'correlation': float(correlation_matrix[i, j]),
                        'hedge_ratio': abs(correlation_matrix[i, j])
                    })
        
        return {
            'hedge_pairs': hedge_pairs,
            'recommend_hedge': len(hedge_pairs) > 0
        }
    
    def store_experience(self, state: np.ndarray, action: np.ndarray, reward: float,
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
        actions = torch.FloatTensor(np.array(actions)).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.BoolTensor(dones).to(self.device)
        
        # Current predictions
        current_pred = self.policy_net(states)
        
        # Target (reward + discounted future value)
        next_pred = self.target_net(next_states)
        target = actions.unsqueeze(0) + rewards.unsqueeze(1).unsqueeze(2) * (1 - dones.float()).unsqueeze(1).unsqueeze(2)
        
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
            'correlation_matrix': self.correlation_matrix
        }, filepath)
    
    def load(self, filepath: str):
        """Load model."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.correlation_matrix = checkpoint.get('correlation_matrix', np.eye(self.num_stocks))

