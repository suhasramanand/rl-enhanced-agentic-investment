"""
State Encoder for Reinforcement Learning
Encodes environment state into discrete or continuous representations for RL algorithms.
"""

from typing import Dict, List, Any, Tuple
import numpy as np


class StateEncoder:
    """
    Encodes environment state into formats suitable for RL algorithms.
    """
    
    def __init__(self, state_dim: int = 32):
        """
        Initialize state encoder.
        
        Args:
            state_dim: Dimension of continuous state representation
        """
        self.state_dim = state_dim
        self.feature_names = [
            'has_news', 'news_sentiment', 'has_fundamentals', 'has_sentiment', 'has_macro',
            'has_ta_basic', 'has_ta_advanced', 'has_insights', 'has_recommendation',
            'rsi', 'macd_signal', 'trend', 'atr_normalized',
            'price_change', 'volume_change', 'volatility',
            'num_insights', 'confidence', 'steps_taken',
            'num_tools_used', 'diversity_score'
        ]
    
    def encode_discrete(self, state: Dict[str, Any]) -> int:
        """
        Encode state into discrete integer representation for Q-Learning.
        
        Args:
            state: Dictionary containing state information
        
        Returns:
            Discrete state index
        """
        # Create binary feature vector
        features = []
        
        # Data availability flags (8 bits)
        features.append(1 if state.get('has_news', False) else 0)
        features.append(1 if state.get('has_fundamentals', False) else 0)
        features.append(1 if state.get('has_sentiment', False) else 0)
        features.append(1 if state.get('has_macro', False) else 0)
        features.append(1 if state.get('has_ta_basic', False) else 0)
        features.append(1 if state.get('has_ta_advanced', False) else 0)
        features.append(1 if state.get('has_insights', False) else 0)
        features.append(1 if state.get('has_recommendation', False) else 0)
        
        # Technical indicators (discretized)
        rsi = state.get('rsi', 50.0)
        if rsi < 30:
            features.append(0)  # Oversold
        elif rsi > 70:
            features.append(2)  # Overbought
        else:
            features.append(1)  # Neutral
        
        macd_signal = state.get('macd_signal', 0.0)
        if macd_signal > 0:
            features.append(1)  # Bullish
        else:
            features.append(0)  # Bearish
        
        trend = state.get('trend', 'sideways')
        if trend == 'uptrend':
            features.append(2)
        elif trend == 'downtrend':
            features.append(0)
        else:
            features.append(1)
        
        # Steps taken (discretized)
        steps = state.get('steps_taken', 0)
        if steps < 5:
            features.append(0)
        elif steps < 10:
            features.append(1)
        elif steps < 15:
            features.append(2)
        else:
            features.append(3)
        
        # Convert binary features to integer state
        state_index = 0
        for i, feature in enumerate(features):
            state_index += feature * (4 ** i)  # Base-4 encoding
        
        return state_index
    
    def encode_continuous(self, state: Dict[str, Any]) -> np.ndarray:
        """
        Encode state into continuous vector representation for PPO.
        
        Args:
            state: Dictionary containing state information
        
        Returns:
            Continuous state vector
        """
        features = []
        
        # Binary flags (8 features)
        features.append(1.0 if state.get('has_news', False) else 0.0)
        
        # News sentiment (0-1, where 0.5 is neutral)
        news_sentiment = state.get('news_sentiment', 0.5)
        features.append(news_sentiment)  # Actual sentiment value from news
        
        features.append(1.0 if state.get('has_fundamentals', False) else 0.0)
        features.append(1.0 if state.get('has_sentiment', False) else 0.0)
        features.append(1.0 if state.get('has_macro', False) else 0.0)
        features.append(1.0 if state.get('has_ta_basic', False) else 0.0)
        features.append(1.0 if state.get('has_ta_advanced', False) else 0.0)
        features.append(1.0 if state.get('has_insights', False) else 0.0)
        features.append(1.0 if state.get('has_recommendation', False) else 0.0)
        
        # Normalized technical indicators
        rsi = state.get('rsi', 50.0) / 100.0  # Normalize to [0, 1]
        features.append(rsi)
        
        macd_signal = state.get('macd_signal', 0.0)
        features.append(np.tanh(macd_signal))  # Normalize with tanh
        
        # Trend encoding (one-hot like)
        trend = state.get('trend', 'sideways')
        features.append(1.0 if trend == 'uptrend' else 0.0)
        features.append(1.0 if trend == 'downtrend' else 0.0)
        features.append(1.0 if trend == 'sideways' else 0.0)
        
        # Normalized ATR
        atr = state.get('atr_normalized', 0.0)
        features.append(np.clip(atr, 0.0, 1.0))
        
        # Price and volume changes
        price_change = state.get('price_change', 0.0)
        features.append(np.tanh(price_change))
        
        volume_change = state.get('volume_change', 0.0)
        features.append(np.tanh(volume_change))
        
        # Volatility
        volatility = state.get('volatility', 0.0)
        features.append(np.clip(volatility, 0.0, 1.0))
        
        # Counts and scores (normalized)
        num_insights = state.get('num_insights', 0)
        features.append(min(1.0, num_insights / 10.0))
        
        confidence = state.get('confidence', 0.0)
        features.append(confidence)
        
        steps_taken = state.get('steps_taken', 0)
        features.append(min(1.0, steps_taken / 20.0))
        
        num_tools_used = state.get('num_tools_used', 0)
        features.append(min(1.0, num_tools_used / 10.0))
        
        diversity_score = state.get('diversity_score', 0.0)
        features.append(diversity_score)
        
        # Pad or truncate to desired dimension
        feature_vector = np.array(features, dtype=np.float32)
        
        if len(feature_vector) < self.state_dim:
            # Pad with zeros
            padding = np.zeros(self.state_dim - len(feature_vector), dtype=np.float32)
            feature_vector = np.concatenate([feature_vector, padding])
        elif len(feature_vector) > self.state_dim:
            # Truncate
            feature_vector = feature_vector[:self.state_dim]
        
        return feature_vector
    
    def get_state_dict(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get normalized state dictionary with all features.
        
        Args:
            state: Raw state dictionary
        
        Returns:
            Normalized state dictionary
        """
        normalized_state = {
            'has_news': state.get('has_news', False),
            'news_sentiment': state.get('news_sentiment', 0.5),
            'has_fundamentals': state.get('has_fundamentals', False),
            'has_sentiment': state.get('has_sentiment', False),
            'has_macro': state.get('has_macro', False),
            'has_ta_basic': state.get('has_ta_basic', False),
            'has_ta_advanced': state.get('has_ta_advanced', False),
            'has_insights': state.get('has_insights', False),
            'has_recommendation': state.get('has_recommendation', False),
            'rsi': state.get('rsi', 50.0),
            'macd_signal': state.get('macd_signal', 0.0),
            'trend': state.get('trend', 'sideways'),
            'atr_normalized': state.get('atr_normalized', 0.0),
            'price_change': state.get('price_change', 0.0),
            'volume_change': state.get('volume_change', 0.0),
            'volatility': state.get('volatility', 0.0),
            'num_insights': state.get('num_insights', 0),
            'confidence': state.get('confidence', 0.0),
            'steps_taken': state.get('steps_taken', 0),
            'num_tools_used': state.get('num_tools_used', 0),
            'diversity_score': state.get('diversity_score', 0.0)
        }
        
        return normalized_state

