"""
Deep Q-Network (DQN) implementation for ORION-AI — Optimized Research & Investment Orchestration Network.
Uses neural networks to approximate Q-values instead of Q-table.
"""

import numpy as np
import json
import os
from typing import Dict, List, Tuple, Optional
from collections import deque
import random

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("⚠️  PyTorch not available. Install with: pip install torch")


class DQNNetwork(nn.Module):
    """Neural network for DQN."""
    
    def __init__(self, state_size: int, action_size: int, hidden_sizes: List[int] = [128, 128, 64]):
        super(DQNNetwork, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        
        # Build network layers
        layers = []
        input_size = state_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.ReLU())
            input_size = hidden_size
        layers.append(nn.Linear(input_size, action_size))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, state):
        """Forward pass through the network."""
        return self.network(state)


class DQN:
    """
    Deep Q-Network (DQN) implementation with experience replay and target network.
    """
    
    ACTIONS = [
        'FETCH_NEWS',
        'FETCH_FUNDAMENTALS',
        'FETCH_SENTIMENT',
        'FETCH_MACRO',
        'RUN_TA_BASIC',
        'RUN_TA_ADVANCED',
        'GENERATE_INSIGHT',
        'GENERATE_RECOMMENDATION',
        'EVALUATE_PERFORMANCE',  # Evaluator Agent - validates and evaluates
        'STOP'
    ]
    
    ACTION_TO_IDX = {action: idx for idx, action in enumerate(ACTIONS)}
    IDX_TO_ACTION = {idx: action for idx, action in enumerate(ACTIONS)}
    
    def __init__(
        self,
        state_size: int = 20,  # Size of state vector
        learning_rate: float = 0.001,
        discount_factor: float = 0.95,
        epsilon: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.995,
        memory_size: int = 10000,
        batch_size: int = 32,
        target_update_freq: int = 100,
        hidden_sizes: List[int] = [128, 128, 64]
    ):
        """
        Initialize DQN.
        
        Args:
            state_size: Size of state vector
            learning_rate: Learning rate for optimizer
            discount_factor: Discount factor (gamma)
            epsilon: Initial exploration rate
            epsilon_min: Minimum exploration rate
            epsilon_decay: Epsilon decay rate
            memory_size: Size of replay buffer
            batch_size: Batch size for training
            target_update_freq: Frequency of target network updates
            hidden_sizes: Hidden layer sizes
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for DQN. Install with: pip install torch")
        
        self.state_size = state_size
        self.action_size = len(self.ACTIONS)
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        
        # Device (GPU if available, else CPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🔧 DQN using device: {self.device}")
        
        # Q-Network (main network)
        self.q_network = DQNNetwork(state_size, self.action_size, hidden_sizes).to(self.device)
        
        # Target network (for stable learning)
        self.target_network = DQNNetwork(state_size, self.action_size, hidden_sizes).to(self.device)
        self.update_target_network()
        
        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Experience replay buffer
        self.memory = deque(maxlen=memory_size)
        
        # Training step counter
        self.train_step = 0
    
    def update_target_network(self):
        """Copy weights from main network to target network."""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def remember(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool):
        """
        Store experience in replay buffer.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode terminated
        """
        self.memory.append((state, action, reward, next_state, done))
    
    def select_action(self, state: np.ndarray, training: bool = True) -> Tuple[int, str]:
        """
        Select action using epsilon-greedy policy.
        
        Args:
            state: Current state vector
            training: Whether in training mode
        
        Returns:
            Tuple of (action_index, action_name)
        """
        if training and np.random.random() < self.epsilon:
            # Exploration: random action
            action_idx = np.random.randint(0, self.action_size)
        else:
            # Exploitation: best action according to Q-network
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor)
            action_idx = q_values.argmax().item()
        
        action_name = self.IDX_TO_ACTION[action_idx]
        return action_idx, action_name
    
    def replay(self) -> Optional[float]:
        """
        Train the network on a batch of experiences from replay buffer.
        
        Returns:
            Loss value if training occurred, None otherwise
        """
        if len(self.memory) < self.batch_size:
            return None
        
        # Sample batch from replay buffer
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to tensors
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Current Q-values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Next Q-values from target network
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.discount_factor * next_q_values
        
        # Compute loss
        loss = F.mse_loss(current_q_values, target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        
        # Update target network periodically
        self.train_step += 1
        if self.train_step % self.target_update_freq == 0:
            self.update_target_network()
        
        return loss.item()
    
    def decay_epsilon(self):
        """Decay exploration rate."""
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def get_q_values(self, state: np.ndarray) -> np.ndarray:
        """
        Get Q-values for a state.
        
        Args:
            state: State vector
        
        Returns:
            Array of Q-values for each action
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_network(state_tensor).cpu().numpy()[0]
        return q_values
    
    def save(self, filepath: str):
        """Save DQN model."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'train_step': self.train_step,
            'state_size': self.state_size,
            'action_size': self.action_size,
            'learning_rate': self.learning_rate,
            'discount_factor': self.discount_factor,
        }, filepath)
        print(f"💾 DQN model saved to: {filepath}")
    
    def load(self, filepath: str):
        """Load DQN model."""
        if not os.path.exists(filepath):
            print(f"⚠️  DQN model not found at: {filepath}")
            return
        
        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint.get('epsilon', self.epsilon)
        self.train_step = checkpoint.get('train_step', 0)
        print(f"✅ DQN model loaded from: {filepath}")
        print(f"   - Epsilon: {self.epsilon:.4f}")
        print(f"   - Train steps: {self.train_step}")

