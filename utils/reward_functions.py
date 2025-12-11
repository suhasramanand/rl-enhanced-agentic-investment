"""
Reward Functions for Reinforcement Learning
Defines reward shaping functions for the RL environment.
"""

from typing import Dict, List, Any
import numpy as np


def calculate_insight_quality_reward(insights: List[str], sources_used: List[str]) -> float:
    """
    Calculate reward based on insight quality and diversity.
    
    Args:
        insights: List of generated insights
        sources_used: List of data sources used
    
    Returns:
        Reward score between 0 and 1.0
    """
    if not insights:
        return -0.1
    
    # Base reward for having insights
    base_reward = 0.3
    
    # Diversity bonus: more unique sources = better
    unique_sources = len(set(sources_used))
    diversity_bonus = min(0.3, unique_sources * 0.05)
    
    # Insight depth: longer, more detailed insights get bonus
    avg_insight_length = np.mean([len(insight.split()) for insight in insights])
    depth_bonus = min(0.2, avg_insight_length / 50.0)
    
    # Number of insights (diminishing returns)
    count_bonus = min(0.2, len(insights) * 0.05)
    
    total_reward = base_reward + diversity_bonus + depth_bonus + count_bonus
    
    return min(1.0, total_reward)


def calculate_recommendation_reward(
    recommendation: str,
    confidence: float,
    actual_outcome: str,
    stock_symbol: str
) -> float:
    """
    Calculate reward based on recommendation correctness.
    
    Args:
        recommendation: Buy/Hold/Sell recommendation
        confidence: Confidence score (0-1)
        actual_outcome: Actual market outcome ('positive', 'negative', 'neutral')
        stock_symbol: Stock symbol for context
    
    Returns:
        Reward score between -1.0 and 1.0
    """
    # Map recommendations to expected outcomes
    recommendation_map = {
        'Buy': 'positive',
        'Hold': 'neutral',
        'Sell': 'negative'
    }
    
    expected_outcome = recommendation_map.get(recommendation, 'neutral')
    
    # Calculate correctness
    if expected_outcome == actual_outcome:
        correctness_reward = 0.5
    elif (expected_outcome == 'positive' and actual_outcome == 'neutral') or \
         (expected_outcome == 'negative' and actual_outcome == 'neutral'):
        correctness_reward = 0.1
    else:
        correctness_reward = -0.5
    
    # Confidence multiplier
    confidence_multiplier = confidence
    
    # Total reward
    total_reward = correctness_reward * confidence_multiplier
    
    return np.clip(total_reward, -1.0, 1.0)


def calculate_efficiency_penalty(steps_taken: int, tools_used: List[str], max_steps: int = 20) -> float:
    """
    Calculate penalty for inefficient use of steps and redundant tool calls.
    
    Args:
        steps_taken: Number of steps taken
        tools_used: List of tools called
        max_steps: Maximum allowed steps
    
    Returns:
        Penalty score (negative value)
    """
    # Step penalty: more steps = more penalty
    step_penalty = -0.01 * steps_taken
    
    # Redundancy penalty: calling same tool multiple times
    tool_counts = {}
    for tool in tools_used:
        tool_counts[tool] = tool_counts.get(tool, 0) + 1
    
    redundancy_penalty = 0.0
    for count in tool_counts.values():
        if count > 1:
            redundancy_penalty -= 0.05 * (count - 1)
    
    # Overstep penalty: exceeding max steps
    overstep_penalty = 0.0
    if steps_taken > max_steps:
        overstep_penalty = -0.1 * (steps_taken - max_steps)
    
    total_penalty = step_penalty + redundancy_penalty + overstep_penalty
    
    return total_penalty


def calculate_diversity_bonus(sources_used: List[str]) -> float:
    """
    Calculate bonus for using diverse information sources.
    
    Args:
        sources_used: List of data sources used
    
    Returns:
        Bonus score (positive value)
    """
    unique_sources = set(sources_used)
    
    # Bonus increases with diversity, but with diminishing returns
    diversity_score = len(unique_sources)
    
    if diversity_score == 0:
        return 0.0
    elif diversity_score == 1:
        return 0.05
    elif diversity_score == 2:
        return 0.15
    elif diversity_score == 3:
        return 0.25
    elif diversity_score >= 4:
        return 0.3
    
    return 0.0


def calculate_comprehensive_reward(
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
    Calculate comprehensive reward combining all factors.
    
    Args:
        insights: List of generated insights
        recommendation: Buy/Hold/Sell recommendation
        confidence: Confidence score
        actual_outcome: Actual market outcome
        steps_taken: Number of steps taken
        tools_used: List of tools called
        sources_used: List of data sources used
        stock_symbol: Stock symbol
    
    Returns:
        Total reward score
    """
    insight_reward = calculate_insight_quality_reward(insights, sources_used)
    recommendation_reward = calculate_recommendation_reward(
        recommendation, confidence, actual_outcome, stock_symbol
    )
    efficiency_penalty = calculate_efficiency_penalty(steps_taken, tools_used)
    diversity_bonus = calculate_diversity_bonus(sources_used)
    
    total_reward = insight_reward + recommendation_reward + efficiency_penalty + diversity_bonus
    
    return total_reward

