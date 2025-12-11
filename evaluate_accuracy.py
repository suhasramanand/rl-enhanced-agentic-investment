#!/usr/bin/env python3
"""
Evaluate the accuracy of ORION-AI — Optimized Research & Investment Orchestration Network.

This script evaluates:
1. Recommendation accuracy (Buy/Hold/Sell vs actual returns)
2. DQN model performance
3. Action selection quality
4. Overall system metrics
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import yfinance as yf

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from env.stock_research_env import StockResearchEnv
from rl.dqn import DQN
from utils.state_encoder import StateEncoder
from agents.recommendation_agent import RecommendationAgent

class AccuracyEvaluator:
    """Evaluate system accuracy on historical data."""
    
    def __init__(self, model_path: str = None):
        """Initialize evaluator with trained model."""
        self.model_path = model_path or str(Path.home() / "Downloads" / "dqn_model.pth")
        self.state_encoder = StateEncoder(state_dim=21)
        
        # Load DQN model
        self.dqn = DQN(state_size=21)
        if os.path.exists(self.model_path):
            try:
                self.dqn.load(self.model_path)
                self.dqn.epsilon = 0.0  # No exploration
                print(f"✅ Loaded DQN model from: {self.model_path}")
            except Exception as e:
                print(f"⚠️  Could not load model: {e}")
                print("   Using untrained model for evaluation")
        else:
            print(f"⚠️  Model not found at: {self.model_path}")
            print("   Using untrained model for evaluation")
    
    def evaluate_recommendations(self, stock_symbol: str, num_tests: int = 50, 
                                 future_days: int = 30) -> Dict:
        """
        Evaluate recommendation accuracy on historical data.
        
        Args:
            stock_symbol: Stock to evaluate
            num_tests: Number of test cases
            future_days: Days ahead to check actual return
        
        Returns:
            Dictionary with accuracy metrics
        """
        print(f"\n{'='*70}")
        print(f"📊 EVALUATING RECOMMENDATION ACCURACY")
        print(f"{'='*70}")
        print(f"Stock: {stock_symbol}")
        print(f"Test cases: {num_tests}")
        print(f"Future days: {future_days}")
        print(f"{'='*70}\n")
        
        results = {
            'recommendations': [],
            'actual_returns': [],
            'correct_predictions': [],
            'buy_returns': [],
            'hold_returns': [],
            'sell_returns': [],
            'confidences': []
        }
        
        correct_count = 0
        total_tests = 0
        
        for i in range(num_tests):
            try:
                # Create environment
                env = StockResearchEnv(stock_symbol=stock_symbol, max_steps=20)
                state = env.reset()
                
                # Run analysis with DQN
                step_count = 0
                max_steps = 20
                
                while not env.done and step_count < max_steps:
                    # Force FETCH_NEWS first
                    if step_count == 0 and env.news_data is None:
                        action_name = 'FETCH_NEWS'
                    else:
                        state_vector = self.state_encoder.encode_continuous(state)
                        if len(state_vector) != self.dqn.state_size:
                            if len(state_vector) > self.dqn.state_size:
                                state_vector = state_vector[:self.dqn.state_size]
                            else:
                                padding = np.zeros(self.dqn.state_size - len(state_vector), dtype=np.float32)
                                state_vector = np.concatenate([state_vector, padding])
                        
                        action_idx, action_name = self.dqn.select_action(state_vector, training=False)
                    
                    next_state, reward, done, info = env.step(action_name)
                    state = next_state
                    step_count += 1
                    
                    # Stop if recommendation generated
                    if env.recommendation is not None:
                        break
                
                # Get recommendation
                recommendation = env.recommendation
                confidence = env.confidence
                
                if recommendation is None:
                    continue
                
                # Calculate actual return
                try:
                    current_date = env.current_date
                    future_date = current_date + timedelta(days=future_days)
                    
                    # Get actual prices
                    ticker = yf.Ticker(stock_symbol)
                    hist = ticker.history(start=current_date, end=future_date + timedelta(days=5))
                    
                    if len(hist) < 2:
                        continue
                    
                    current_price = hist.iloc[0]['Close']
                    future_price = hist.iloc[-1]['Close']
                    actual_return = (future_price - current_price) / current_price
                    
                    # Evaluate prediction accuracy
                    is_correct = False
                    if recommendation == 'Buy' and actual_return > 0.02:  # >2% gain
                        is_correct = True
                    elif recommendation == 'Sell' and actual_return < -0.02:  # >2% loss
                        is_correct = True
                    elif recommendation == 'Hold' and -0.02 <= actual_return <= 0.02:  # Within ±2%
                        is_correct = True
                    
                    results['recommendations'].append(recommendation)
                    results['actual_returns'].append(actual_return)
                    results['correct_predictions'].append(is_correct)
                    results['confidences'].append(confidence)
                    
                    if recommendation == 'Buy':
                        results['buy_returns'].append(actual_return)
                    elif recommendation == 'Hold':
                        results['hold_returns'].append(actual_return)
                    elif recommendation == 'Sell':
                        results['sell_returns'].append(actual_return)
                    
                    if is_correct:
                        correct_count += 1
                    total_tests += 1
                    
                    if (i + 1) % 10 == 0:
                        print(f"  Progress: {i+1}/{num_tests} | Accuracy: {correct_count/total_tests*100:.1f}%")
                
                except Exception as e:
                    continue
            
            except Exception as e:
                continue
        
        # Calculate metrics
        if total_tests == 0:
            return {'error': 'No valid test cases'}
        
        accuracy = correct_count / total_tests
        
        metrics = {
            'total_tests': total_tests,
            'accuracy': accuracy,
            'correct_predictions': correct_count,
            'avg_confidence': np.mean(results['confidences']) if results['confidences'] else 0,
            'avg_return': np.mean(results['actual_returns']) if results['actual_returns'] else 0,
            'buy_accuracy': self._calculate_buy_accuracy(results),
            'hold_accuracy': self._calculate_hold_accuracy(results),
            'sell_accuracy': self._calculate_sell_accuracy(results),
            'buy_avg_return': np.mean(results['buy_returns']) if results['buy_returns'] else 0,
            'hold_avg_return': np.mean(results['hold_returns']) if results['hold_returns'] else 0,
            'sell_avg_return': np.mean(results['sell_returns']) if results['sell_returns'] else 0,
            'recommendation_distribution': self._get_distribution(results['recommendations'])
        }
        
        return metrics
    
    def _calculate_buy_accuracy(self, results: Dict) -> float:
        """Calculate accuracy for Buy recommendations."""
        buy_indices = [i for i, rec in enumerate(results['recommendations']) if rec == 'Buy']
        if not buy_indices:
            return 0.0
        
        buy_correct = sum([results['correct_predictions'][i] for i in buy_indices])
        return buy_correct / len(buy_indices)
    
    def _calculate_hold_accuracy(self, results: Dict) -> float:
        """Calculate accuracy for Hold recommendations."""
        hold_indices = [i for i, rec in enumerate(results['recommendations']) if rec == 'Hold']
        if not hold_indices:
            return 0.0
        
        hold_correct = sum([results['correct_predictions'][i] for i in hold_indices])
        return hold_correct / len(hold_indices)
    
    def _calculate_sell_accuracy(self, results: Dict) -> float:
        """Calculate accuracy for Sell recommendations."""
        sell_indices = [i for i, rec in enumerate(results['recommendations']) if rec == 'Sell']
        if not sell_indices:
            return 0.0
        
        sell_correct = sum([results['correct_predictions'][i] for i in sell_indices])
        return sell_correct / len(sell_indices)
    
    def _get_distribution(self, recommendations: List[str]) -> Dict[str, int]:
        """Get distribution of recommendations."""
        dist = {'Buy': 0, 'Hold': 0, 'Sell': 0}
        for rec in recommendations:
            if rec in dist:
                dist[rec] += 1
        return dist
    
    def print_results(self, metrics: Dict):
        """Print evaluation results in a formatted way."""
        if 'error' in metrics:
            print(f"❌ Error: {metrics['error']}")
            return
        
        print(f"\n{'='*70}")
        print(f"📊 EVALUATION RESULTS")
        print(f"{'='*70}\n")
        
        print(f"🎯 Overall Accuracy: {metrics['accuracy']*100:.2f}%")
        print(f"   ({metrics['correct_predictions']}/{metrics['total_tests']} correct predictions)\n")
        
        print(f"📈 Average Return: {metrics['avg_return']*100:.2f}%")
        print(f"💪 Average Confidence: {metrics['avg_confidence']*100:.1f}%\n")
        
        print(f"📊 Recommendation-Specific Accuracy:")
        print(f"   Buy:  {metrics['buy_accuracy']*100:.2f}% (avg return: {metrics['buy_avg_return']*100:.2f}%)")
        print(f"   Hold: {metrics['hold_accuracy']*100:.2f}% (avg return: {metrics['hold_avg_return']*100:.2f}%)")
        print(f"   Sell: {metrics['sell_accuracy']*100:.2f}% (avg return: {metrics['sell_avg_return']*100:.2f}%)\n")
        
        print(f"📋 Recommendation Distribution:")
        dist = metrics['recommendation_distribution']
        total = sum(dist.values())
        for rec, count in dist.items():
            pct = (count / total * 100) if total > 0 else 0
            print(f"   {rec}: {count} ({pct:.1f}%)")
        
        print(f"\n{'='*70}\n")
        
        # Performance assessment
        if metrics['accuracy'] >= 0.60:
            print("✅ EXCELLENT - System is performing well!")
        elif metrics['accuracy'] >= 0.50:
            print("✅ GOOD - System is performing adequately")
        elif metrics['accuracy'] >= 0.40:
            print("⚠️  FAIR - System needs improvement")
        else:
            print("❌ POOR - System needs significant improvement")


def main():
    """Main evaluation function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate system accuracy')
    parser.add_argument('--symbol', type=str, default='NVDA', help='Stock symbol to evaluate')
    parser.add_argument('--tests', type=int, default=50, help='Number of test cases')
    parser.add_argument('--days', type=int, default=30, help='Future days to check return')
    parser.add_argument('--model', type=str, default=None, help='Path to DQN model')
    
    args = parser.parse_args()
    
    evaluator = AccuracyEvaluator(model_path=args.model)
    metrics = evaluator.evaluate_recommendations(
        stock_symbol=args.symbol,
        num_tests=args.tests,
        future_days=args.days
    )
    
    evaluator.print_results(metrics)
    
    return metrics


if __name__ == '__main__':
    main()

