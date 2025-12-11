"""
Hyperparameter Optimization Module
Implements Bayesian optimization and grid search for hyperparameter tuning.
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Callable
from dataclasses import dataclass
import json
from pathlib import Path
import time

try:
    from skopt import gp_minimize
    from skopt.space import Real, Integer
    from skopt.utils import use_named_args
    SKOPT_AVAILABLE = True
except ImportError:
    SKOPT_AVAILABLE = False
    print("⚠️  scikit-optimize not available. Install with: pip install scikit-optimize")
    print("   Falling back to grid search.")


@dataclass
class HyperparameterSpace:
    """Define hyperparameter search space."""
    learning_rate: Tuple[float, float] = (1e-5, 1e-2)  # (min, max)
    discount_factor: Tuple[float, float] = (0.9, 0.99)
    epsilon_decay: Tuple[float, float] = (0.99, 0.9999)
    batch_size: Tuple[int, int] = (16, 128)
    memory_size: Tuple[int, int] = (5000, 20000)
    hidden_size_1: Tuple[int, int] = (64, 256)
    hidden_size_2: Tuple[int, int] = (64, 256)
    hidden_size_3: Tuple[int, int] = (32, 128)


class BayesianOptimizer:
    """
    Bayesian optimization for hyperparameter tuning using Gaussian Processes.
    """
    
    def __init__(
        self,
        hyperparameter_space: HyperparameterSpace,
        n_calls: int = 50,
        n_initial_points: int = 10,
        random_state: int = 42
    ):
        """
        Initialize Bayesian optimizer.
        
        Args:
            hyperparameter_space: Hyperparameter search space
            n_calls: Number of optimization iterations
            n_initial_points: Number of random initial points
            random_state: Random seed
        """
        if not SKOPT_AVAILABLE:
            raise ImportError("scikit-optimize is required for Bayesian optimization")
        
        self.space = hyperparameter_space
        self.n_calls = n_calls
        self.n_initial_points = n_initial_points
        self.random_state = random_state
        
        # Define search space for skopt
        self.dimensions = [
            Real(*hyperparameter_space.learning_rate, name='learning_rate', prior='log-uniform'),
            Real(*hyperparameter_space.discount_factor, name='discount_factor'),
            Real(*hyperparameter_space.epsilon_decay, name='epsilon_decay'),
            Integer(*hyperparameter_space.batch_size, name='batch_size'),
            Integer(*hyperparameter_space.memory_size, name='memory_size'),
            Integer(*hyperparameter_space.hidden_size_1, name='hidden_size_1'),
            Integer(*hyperparameter_space.hidden_size_2, name='hidden_size_2'),
            Integer(*hyperparameter_space.hidden_size_3, name='hidden_size_3')
        ]
        
        self.results = []
    
    def optimize(
        self,
        objective_function: Callable[[Dict[str, Any]], float],
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Run Bayesian optimization.
        
        Args:
            objective_function: Function that takes hyperparameters dict and returns score (higher is better)
            verbose: Whether to print progress
        
        Returns:
            Dictionary with best hyperparameters and results
        """
        @use_named_args(dimensions=self.dimensions)
        def objective(**params):
            # Convert to dict format expected by objective function
            hyperparams = {
                'learning_rate': params['learning_rate'],
                'discount_factor': params['discount_factor'],
                'epsilon_decay': params['epsilon_decay'],
                'batch_size': params['batch_size'],
                'memory_size': params['memory_size'],
                'hidden_sizes': [
                    params['hidden_size_1'],
                    params['hidden_size_2'],
                    params['hidden_size_3']
                ]
            }
            
            score = objective_function(hyperparams)
            self.results.append({
                'hyperparameters': hyperparams,
                'score': score,
                'timestamp': time.time()
            })
            
            # Return negative score (gp_minimize minimizes)
            return -score
        
        # Run optimization
        result = gp_minimize(
            func=objective,
            dimensions=self.dimensions,
            n_calls=self.n_calls,
            n_initial_points=self.n_initial_points,
            random_state=self.random_state,
            verbose=verbose
        )
        
        # Extract best hyperparameters
        best_params = {
            'learning_rate': result.x[0],
            'discount_factor': result.x[1],
            'epsilon_decay': result.x[2],
            'batch_size': result.x[3],
            'memory_size': result.x[4],
            'hidden_sizes': [result.x[5], result.x[6], result.x[7]]
        }
        
        return {
            'best_hyperparameters': best_params,
            'best_score': -result.fun,  # Negate because we minimized negative
            'n_iterations': len(result.func_vals),
            'all_results': self.results,
            'optimization_result': result
        }


class GridSearchOptimizer:
    """
    Grid search for hyperparameter tuning.
    """
    
    def __init__(self, hyperparameter_grid: Dict[str, List[Any]]):
        """
        Initialize grid search optimizer.
        
        Args:
            hyperparameter_grid: Dictionary with parameter names and lists of values to try
        """
        self.grid = hyperparameter_grid
        self.results = []
    
    def optimize(
        self,
        objective_function: Callable[[Dict[str, Any]], float],
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Run grid search.
        
        Args:
            objective_function: Function that takes hyperparameters dict and returns score
            verbose: Whether to print progress
        
        Returns:
            Dictionary with best hyperparameters and results
        """
        from itertools import product
        
        # Generate all combinations
        param_names = list(self.grid.keys())
        param_values = list(self.grid.values())
        combinations = list(product(*param_values))
        
        total_combinations = len(combinations)
        print(f"Grid search: {total_combinations} combinations to evaluate")
        
        best_score = float('-inf')
        best_params = None
        
        for idx, combination in enumerate(combinations):
            hyperparams = dict(zip(param_names, combination))
            
            if verbose:
                print(f"\n[{idx+1}/{total_combinations}] Testing: {hyperparams}")
            
            score = objective_function(hyperparams)
            
            self.results.append({
                'hyperparameters': hyperparams,
                'score': score,
                'timestamp': time.time()
            })
            
            if score > best_score:
                best_score = score
                best_params = hyperparams
            
            if verbose:
                print(f"  Score: {score:.4f} | Best: {best_score:.4f}")
        
        return {
            'best_hyperparameters': best_params,
            'best_score': best_score,
            'n_iterations': total_combinations,
            'all_results': self.results
        }


def create_validation_objective(
    train_function: Callable,
    validation_stocks: List[str],
    validation_episodes: int = 100
) -> Callable[[Dict[str, Any]], float]:
    """
    Create an objective function for hyperparameter optimization.
    
    Args:
        train_function: Function that trains model with given hyperparameters
        validation_stocks: Stocks to use for validation
        validation_episodes: Number of episodes for validation
    
    Returns:
        Objective function that returns validation score
    """
    def objective(hyperparams: Dict[str, Any]) -> float:
        """
        Objective function for optimization.
        
        Args:
            hyperparams: Dictionary of hyperparameters
        
        Returns:
            Validation score (higher is better)
        """
        # Train model with given hyperparameters
        model, training_metrics = train_function(hyperparams, validation_episodes)
        
        # Evaluate on validation set
        validation_score = evaluate_hyperparameters(
            model,
            validation_stocks,
            validation_episodes
        )
        
        return validation_score
    
    return objective


def evaluate_hyperparameters(
    model: Any,
    stocks: List[str],
    episodes: int = 100
) -> float:
    """
    Evaluate model with given hyperparameters on validation set.
    
    Args:
        model: Trained model
        stocks: Stocks to evaluate on
        episodes: Number of episodes
    
    Returns:
        Average validation score (combination of reward and accuracy)
    """
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    from env.stock_research_env import StockResearchEnv
    from utils.data_cache import DataCache
    
    cache = DataCache()
    all_rewards = []
    all_accuracies = []
    
    episodes_per_stock = episodes // len(stocks)
    
    for stock_symbol in stocks:
        env = StockResearchEnv(
            stock_symbol=stock_symbol,
            max_steps=100,
            use_latest_date=False
        )
        
        for _ in range(episodes_per_stock):
            state = env.reset()
            state_vector = env._get_state()
            state_array = np.array(list(state_vector.values()))
            
            total_reward = 0.0
            
            while not env.done:
                action_idx = model.select_action(state_array)
                action = model.IDX_TO_ACTION[action_idx]
                
                next_state, reward, done, info = env.step(action)
                next_state_vector = env._get_state()
                next_state_array = np.array(list(next_state_vector.values()))
                
                state_array = next_state_array
                total_reward += reward
            
            all_rewards.append(total_reward)
            
            # Calculate accuracy
            if env.recommendation is not None and env.future_return is not None:
                future_return = env.future_return
                is_correct = False
                
                if env.recommendation == 'Buy' and future_return > 0.02:
                    is_correct = True
                elif env.recommendation == 'Sell' and future_return < -0.02:
                    is_correct = True
                elif env.recommendation == 'Hold' and -0.02 <= future_return <= 0.02:
                    is_correct = True
                
                all_accuracies.append(1 if is_correct else 0)
    
    # Combined score: weighted average of reward and accuracy
    avg_reward = np.mean(all_rewards) if all_rewards else 0.0
    avg_accuracy = np.mean(all_accuracies) if all_accuracies else 0.0
    
    # Normalize and combine (both in [0, 1] range)
    normalized_reward = (avg_reward + 1) / 2  # Assuming rewards in [-1, 1]
    score = 0.6 * normalized_reward + 0.4 * avg_accuracy
    
    return float(score)

