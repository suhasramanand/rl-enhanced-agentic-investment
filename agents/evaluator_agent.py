"""
Evaluator Agent
Computes reward scores and evaluates system performance.
"""

from typing import Dict, List, Any, Optional
from utils.reward_functions import calculate_comprehensive_reward


class EvaluatorAgent:
    """
    Agent responsible for evaluating performance and computing rewards.
    """
    
    def __init__(self):
        """Initialize Evaluator Agent."""
        self.name = "EvaluatorAgent"
        self.capabilities = ['evaluate_performance', 'compute_reward']
    
    def evaluate_performance(
        self,
        insights: List[str],
        recommendation: str,
        confidence: float,
        actual_outcome: str,
        steps_taken: int,
        tools_used: List[str],
        sources_used: List[str],
        stock_symbol: str
    ) -> Dict[str, Any]:
        """
        Evaluate overall system performance.
        
        Args:
            insights: List of generated insights
            recommendation: Buy/Hold/Sell recommendation
            confidence: Confidence score
            actual_outcome: Actual market outcome
            steps_taken: Number of steps taken
            tools_used: List of tools called
            sources_used: List of data sources used
            stock_symbol: Stock symbol analyzed
        
        Returns:
            Performance evaluation dictionary
        """
        # Compute comprehensive reward
        reward = calculate_comprehensive_reward(
            insights=insights,
            recommendation=recommendation,
            confidence=confidence,
            actual_outcome=actual_outcome,
            steps_taken=steps_taken,
            tools_used=tools_used,
            sources_used=sources_used,
            stock_symbol=stock_symbol
        )
        
        # Additional metrics
        efficiency_score = max(0.0, 1.0 - (steps_taken / 20.0))
        diversity_score = len(set(sources_used)) / 4.0
        
        # Recommendation correctness
        recommendation_map = {
            'Buy': 'positive',
            'Hold': 'neutral',
            'Sell': 'negative'
        }
        expected_outcome = recommendation_map.get(recommendation, 'neutral')
        correctness = 1.0 if expected_outcome == actual_outcome else 0.0
        
        evaluation = {
            'reward': reward,
            'efficiency_score': efficiency_score,
            'diversity_score': diversity_score,
            'correctness': correctness,
            'confidence': confidence,
            'num_insights': len(insights),
            'steps_taken': steps_taken,
            'tools_used_count': len(set(tools_used)),
            'sources_used_count': len(set(sources_used))
        }
        
        return evaluation
    
    def compute_reward(
        self,
        insights: List[str],
        recommendation: str,
        confidence: float,
        actual_outcome: str,
        steps_taken: int,
        tools_used: List[str],
        sources_used: List[str],
        stock_symbol: str
    ) -> float:
        """
        Compute reward score.
        
        Args:
            insights: List of generated insights
            recommendation: Buy/Hold/Sell recommendation
            confidence: Confidence score
            actual_outcome: Actual market outcome
            steps_taken: Number of steps taken
            tools_used: List of tools called
            sources_used: List of data sources used
            stock_symbol: Stock symbol analyzed
        
        Returns:
            Reward score
        """
        return calculate_comprehensive_reward(
            insights=insights,
            recommendation=recommendation,
            confidence=confidence,
            actual_outcome=actual_outcome,
            steps_taken=steps_taken,
            tools_used=tools_used,
            sources_used=sources_used,
            stock_symbol=stock_symbol
        )

