"""
Portfolio State Encoder
Encodes portfolio-level state for RL algorithms.
"""

from typing import Dict, List, Any
import numpy as np


class PortfolioStateEncoder:
    """
    Encodes portfolio state into continuous vector for Portfolio DQN.
    """
    
    def __init__(self, state_dim: int = 50):
        """
        Initialize portfolio state encoder.
        
        Args:
            state_dim: Dimension of state vector
        """
        self.state_dim = state_dim
    
    def encode_continuous(self, state: Dict[str, Any]) -> np.ndarray:
        """
        Encode portfolio state into continuous vector.
        
        Args:
            state: Portfolio state dictionary
        
        Returns:
            Feature vector
        """
        features = []
        
        # Portfolio metrics (10 features)
        features.append(state.get('total_allocated', 0.0))  # Total allocation percentage
        features.append(state.get('num_stocks_selected', 0) / 10.0)  # Normalized
        features.append(state.get('num_stocks_allocated', 0) / 10.0)  # Normalized
        features.append(state.get('diversity', 0.0))  # Portfolio diversity
        features.append(state.get('max_allocation', 0.0))  # Concentration
        features.append(state.get('avg_confidence', 0.0))  # Average confidence
        features.append(min(1.0, state.get('steps_taken', 0) / 50.0))  # Progress
        features.append(state.get('stocks_remaining', 10) / 10.0)  # Normalized
        features.append(1.0 if state.get('portfolio_finalized', False) else 0.0)
        features.append(state.get('watchlist_size', 10) / 10.0)  # Normalized
        
        # Current stock state (5 features)
        features.append(1.0 if state.get('current_stock') else 0.0)
        features.append(state.get('current_stock_allocated', 0.0))
        features.append(1.0 if state.get('current_stock_analyzed', False) else 0.0)
        
        current_rec = state.get('current_recommendation')
        if current_rec == 'Buy':
            features.append(1.0)
            features.append(0.0)
        elif current_rec == 'Sell':
            features.append(0.0)
            features.append(1.0)
        else:
            features.append(0.0)
            features.append(0.0)
        
        features.append(state.get('current_confidence', 0.0))
        
        # Allocation distribution features (10 features)
        # These would be filled from actual allocations if available
        # For now, use derived metrics
        allocations = state.get('allocations', {})
        if allocations:
            alloc_values = list(allocations.values())
            features.append(np.mean(alloc_values))  # Mean allocation
            features.append(np.std(alloc_values))  # Std (concentration measure)
            features.append(max(alloc_values))  # Max allocation
            features.append(min([a for a in alloc_values if a > 0], default=0.0))  # Min non-zero
            features.append(len([a for a in alloc_values if a > 0]))  # Number of positions
        else:
            features.extend([0.0] * 5)
        
        # Add more features to reach state_dim
        # Market conditions (could add more)
        features.append(0.5)  # Placeholder for market regime
        features.append(0.5)  # Placeholder for volatility
        
        # Pad or truncate to state_dim
        feature_vector = np.array(features, dtype=np.float32)
        
        if len(feature_vector) < self.state_dim:
            padding = np.zeros(self.state_dim - len(feature_vector), dtype=np.float32)
            feature_vector = np.concatenate([feature_vector, padding])
        elif len(feature_vector) > self.state_dim:
            feature_vector = feature_vector[:self.state_dim]
        
        return feature_vector

