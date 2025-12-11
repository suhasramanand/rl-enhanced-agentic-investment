"""
Outcome Tracker for Learning from Actual Stock Performance
Tracks recommendations and their actual outcomes for outcome-based learning.
"""

import json
import os
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import yfinance as yf
import pandas as pd
import numpy as np


class OutcomeTracker:
    """
    Tracks stock recommendations and their actual outcomes.
    Enables outcome-based learning by comparing predictions to reality.
    """
    
    def __init__(self, storage_path: str = 'outcomes_history.json'):
        """
        Initialize outcome tracker.
        
        Args:
            storage_path: Path to store outcome history
        """
        self.storage_path = storage_path
        self.recommendations = []  # Pending recommendations
        self.outcomes = []  # Completed outcomes
        self._load_history()
    
    def _load_history(self):
        """Load outcome history from file."""
        if os.path.exists(self.storage_path):
            try:
                with open(self.storage_path, 'r') as f:
                    data = json.load(f)
                    self.recommendations = data.get('recommendations', [])
                    self.outcomes = data.get('outcomes', [])
                print(f"✅ Loaded {len(self.outcomes)} historical outcomes")
            except Exception as e:
                print(f"⚠️  Error loading history: {e}")
                self.recommendations = []
                self.outcomes = []
    
    def _save_history(self):
        """Save outcome history to file."""
        try:
            with open(self.storage_path, 'w') as f:
                json.dump({
                    'recommendations': self.recommendations,
                    'outcomes': self.outcomes
                }, f, indent=2, default=str)
        except Exception as e:
            print(f"⚠️  Error saving history: {e}")
    
    def record_recommendation(
        self,
        stock_symbol: str,
        recommendation: str,
        confidence: float,
        date: Optional[str] = None,
        allocation: float = 0.0,
        portfolio_id: Optional[str] = None
    ):
        """
        Record a recommendation for future outcome tracking.
        
        Args:
            stock_symbol: Stock ticker
            recommendation: Buy/Hold/Sell
            confidence: Confidence level (0-1)
            date: Date of recommendation (default: today)
            allocation: Portfolio allocation (0-1)
            portfolio_id: ID of portfolio this belongs to
        """
        if date is None:
            date = datetime.now().strftime('%Y-%m-%d')
        
        rec = {
            'id': f"{stock_symbol}_{date}_{len(self.recommendations)}",
            'stock_symbol': stock_symbol,
            'recommendation': recommendation,
            'confidence': float(confidence),
            'date': date,
            'allocation': float(allocation),
            'portfolio_id': portfolio_id,
            'status': 'pending',
            'created_at': datetime.now().isoformat()
        }
        
        self.recommendations.append(rec)
        self._save_history()
        print(f"📝 Recorded recommendation: {stock_symbol} {recommendation} @ {date}")
    
    def calculate_outcome(
        self,
        recommendation_id: str,
        future_days: int = 30
    ) -> Optional[Dict[str, Any]]:
        """
        Calculate actual outcome for a recommendation.
        
        Args:
            recommendation_id: ID of recommendation
            future_days: Days ahead to check outcome
        
        Returns:
            Outcome dictionary with actual return and reward
        """
        # Find recommendation
        rec = None
        for r in self.recommendations:
            if r['id'] == recommendation_id:
                rec = r
                break
        
        if not rec:
            print(f"⚠️  Recommendation {recommendation_id} not found")
            return None
        
        if rec['status'] != 'pending':
            # Already calculated
            for outcome in self.outcomes:
                if outcome['recommendation_id'] == recommendation_id:
                    return outcome
        
        # Get actual stock prices
        stock_symbol = rec['stock_symbol']
        rec_date = datetime.strptime(rec['date'], '%Y-%m-%d').date()
        future_date = rec_date + timedelta(days=future_days)
        
        try:
            ticker = yf.Ticker(stock_symbol)
            hist = ticker.history(start=str(rec_date), end=str(future_date))
            
            if len(hist) == 0:
                print(f"⚠️  No price data for {stock_symbol} from {rec_date} to {future_date}")
                return None
            
            # Get prices
            rec_price = float(hist.iloc[0]['Close'])
            future_price = float(hist.iloc[-1]['Close'])
            
            # Calculate return
            actual_return = (future_price - rec_price) / rec_price
            
            # Calculate reward based on recommendation correctness
            recommendation = rec['recommendation']
            confidence = rec['confidence']
            
            if recommendation == 'Buy':
                # Reward positive returns, penalize negative
                reward = actual_return * confidence
            elif recommendation == 'Sell':
                # Reward negative returns (we sold, price went down = good)
                reward = -actual_return * confidence
            else:  # Hold
                # Reward small moves (neutral), penalize large moves
                reward = -abs(actual_return) * confidence
            
            outcome = {
                'recommendation_id': recommendation_id,
                'stock_symbol': stock_symbol,
                'recommendation': recommendation,
                'confidence': confidence,
                'date': rec['date'],
                'allocation': rec['allocation'],
                'rec_price': rec_price,
                'future_price': future_price,
                'actual_return': actual_return,
                'reward': reward,
                'future_days': future_days,
                'calculated_at': datetime.now().isoformat()
            }
            
            # Update recommendation status
            rec['status'] = 'completed'
            rec['outcome_id'] = outcome['recommendation_id']
            
            # Store outcome
            self.outcomes.append(outcome)
            self._save_history()
            
            print(f"✅ Outcome calculated: {stock_symbol} return={actual_return:.2%}, reward={reward:.4f}")
            
            return outcome
            
        except Exception as e:
            print(f"❌ Error calculating outcome: {e}")
            return None
    
    def calculate_portfolio_outcome(
        self,
        portfolio_id: str,
        future_days: int = 30
    ) -> Optional[Dict[str, Any]]:
        """
        Calculate outcome for entire portfolio.
        
        Args:
            portfolio_id: ID of portfolio
            future_days: Days ahead to check
        
        Returns:
            Portfolio outcome with total return and reward
        """
        # Get all recommendations for this portfolio
        portfolio_recs = [r for r in self.recommendations if r.get('portfolio_id') == portfolio_id]
        
        if not portfolio_recs:
            print(f"⚠️  No recommendations found for portfolio {portfolio_id}")
            return None
        
        # Calculate outcomes for each stock
        total_weighted_return = 0.0
        total_weighted_reward = 0.0
        individual_outcomes = []
        
        for rec in portfolio_recs:
            outcome = self.calculate_outcome(rec['id'], future_days)
            if outcome:
                allocation = rec.get('allocation', 0.0)
                weighted_return = outcome['actual_return'] * allocation
                weighted_reward = outcome['reward'] * allocation
                
                total_weighted_return += weighted_return
                total_weighted_reward += weighted_reward
                
                individual_outcomes.append({
                    'stock': rec['stock_symbol'],
                    'allocation': allocation,
                    'return': outcome['actual_return'],
                    'reward': outcome['reward'],
                    'weighted_return': weighted_return
                })
        
        portfolio_outcome = {
            'portfolio_id': portfolio_id,
            'num_stocks': len(individual_outcomes),
            'total_weighted_return': total_weighted_return,
            'total_weighted_reward': total_weighted_reward,
            'individual_outcomes': individual_outcomes,
            'future_days': future_days,
            'calculated_at': datetime.now().isoformat()
        }
        
        print(f"📊 Portfolio {portfolio_id}: Total return={total_weighted_return:.2%}, Reward={total_weighted_reward:.4f}")
        
        return portfolio_outcome
    
    def get_learning_statistics(self) -> Dict[str, Any]:
        """Get statistics for learning evaluation."""
        if not self.outcomes:
            return {'error': 'No outcomes available'}
        
        returns = [o['actual_return'] for o in self.outcomes]
        rewards = [o['reward'] for o in self.outcomes]
        
        buy_recs = [o for o in self.outcomes if o['recommendation'] == 'Buy']
        sell_recs = [o for o in self.outcomes if o['recommendation'] == 'Sell']
        
        stats = {
            'total_outcomes': len(self.outcomes),
            'avg_return': np.mean(returns),
            'avg_reward': np.mean(rewards),
            'total_return': np.sum(returns),
            'total_reward': np.sum(rewards),
            'buy_accuracy': len([r for r in buy_recs if r['actual_return'] > 0]) / len(buy_recs) if buy_recs else 0.0,
            'sell_accuracy': len([r for r in sell_recs if r['actual_return'] < 0]) / len(sell_recs) if sell_recs else 0.0,
        }
        
        return stats

