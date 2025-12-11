"""
Q-Learning Implementation
Discrete action-value learning for tool selection in agentic system.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import json
import os


class QLearning:
    """
    Q-Learning algorithm for discrete state-action spaces.
    Used for selecting which tool/agent to call next.
    """
    
    # Action space definitions
    ACTIONS = [
        'FETCH_NEWS',
        'FETCH_FUNDAMENTALS',
        'FETCH_SENTIMENT',
        'FETCH_MACRO',
        'RUN_TA_BASIC',
        'RUN_TA_ADVANCED',
        'GENERATE_INSIGHT',
        'GENERATE_RECOMMENDATION',
        'STOP'
    ]
    
    ACTION_TO_IDX = {action: idx for idx, action in enumerate(ACTIONS)}
    IDX_TO_ACTION = {idx: action for idx, action in enumerate(ACTIONS)}
    
    def __init__(
        self,
        learning_rate: float = 0.1,
        discount_factor: float = 0.95,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        state_space_size: int = 1000000  # Large discrete state space
    ):
        """
        Initialize Q-Learning agent.
        
        Args:
            learning_rate: Learning rate (alpha)
            discount_factor: Discount factor (gamma)
            epsilon_start: Initial exploration rate
            epsilon_end: Final exploration rate
            epsilon_decay: Epsilon decay rate per episode
            state_space_size: Size of discrete state space
        """
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        # Q-table: state -> action -> Q-value
        # Using dictionary for sparse representation
        self.q_table: Dict[int, np.ndarray] = {}
        self.state_space_size = state_space_size
        
        # Statistics
        self.episode_rewards = []
        self.episode_lengths = []
        self.q_value_history = []
    
    def _get_q_values(self, state: int) -> np.ndarray:
        """
        Get Q-values for a given state.
        
        Args:
            state: Discrete state index
        
        Returns:
            Array of Q-values for each action
        """
        if state not in self.q_table:
            # Initialize with zeros
            self.q_table[state] = np.zeros(len(self.ACTIONS))
        
        return self.q_table[state]
    
    def select_action(self, state: int, training: bool = True) -> Tuple[int, str]:
        """
        Select action using epsilon-greedy policy.
        
        Args:
            state: Current state index
            training: Whether in training mode (affects exploration)
        
        Returns:
            Tuple of (action_index, action_name)
        """
        q_values = self._get_q_values(state)
        
        if training and np.random.random() < self.epsilon:
            # Explore: random action
            action_idx = np.random.randint(0, len(self.ACTIONS))
            exploration_type = "exploration"
        else:
            # Exploit: best action
            action_idx = np.argmax(q_values)
            exploration_type = "exploitation"
        
        action_name = self.IDX_TO_ACTION[action_idx]
        
        # Log exploration vs exploitation
        if training:
            print(f"      - Action selection: {exploration_type} (epsilon: {self.epsilon:.4f})")
            if exploration_type == "exploitation":
                print(f"      - Best Q-value: {q_values[action_idx]:.4f}")
        
        return action_idx, action_name
    
    def update(
        self,
        state: int,
        action: int,
        reward: float,
        next_state: int,
        done: bool
    ):
        """
        Update Q-value using Q-Learning update rule.
        
        Q(s,a) = Q(s,a) + alpha * [r + gamma * max Q(s',a') - Q(s,a)]
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode terminated
        """
        q_values_current = self._get_q_values(state)
        q_values_next = self._get_q_values(next_state)
        
        # Current Q-value
        current_q = q_values_current[action]
        
        # Target Q-value
        if done:
            target_q = reward
        else:
            target_q = reward + self.discount_factor * np.max(q_values_next)
        
        # Q-Learning update
        q_values_current[action] = current_q + self.learning_rate * (target_q - current_q)
        
        # Store updated Q-values
        self.q_table[state] = q_values_current
    
    def decay_epsilon(self):
        """Decay exploration rate."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
    
    def get_q_value_heatmap_data(self, max_states: int = 100) -> np.ndarray:
        """
        Get Q-value data for visualization.
        
        Args:
            max_states: Maximum number of states to include
        
        Returns:
            Array of shape (num_states, num_actions) with Q-values
        """
        if not self.q_table:
            return np.zeros((1, len(self.ACTIONS)))
        
        states = list(self.q_table.keys())[:max_states]
        q_matrix = np.array([self.q_table[state] for state in states])
        
        return q_matrix
    
    def save(self, filepath: str):
        """
        Save Q-table to file.
        
        Args:
            filepath: Path to save file
        """
        # Convert numpy arrays to lists for JSON serialization
        q_table_serializable = {
            str(state): q_values.tolist()
            for state, q_values in self.q_table.items()
        }
        
        data = {
            'q_table': q_table_serializable,
            'epsilon': self.epsilon,
            'learning_rate': self.learning_rate,
            'discount_factor': self.discount_factor,
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths
        }
        
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load(self, filepath: str):
        """
        Load Q-table from file.
        
        Args:
            filepath: Path to load file
        """
        if not os.path.exists(filepath):
            print(f"Warning: File {filepath} not found. Starting with empty Q-table.")
            return
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Convert back to numpy arrays
        self.q_table = {
            int(state): np.array(q_values)
            for state, q_values in data['q_table'].items()
        }
        
        self.epsilon = data.get('epsilon', self.epsilon_start)
        self.learning_rate = data.get('learning_rate', self.learning_rate)
        self.discount_factor = data.get('discount_factor', self.discount_factor)
        self.episode_rewards = data.get('episode_rewards', [])
        self.episode_lengths = data.get('episode_lengths', [])

