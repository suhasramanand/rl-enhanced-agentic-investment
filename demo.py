"""
Demo Script
Demonstrates ORION-AI — Optimized Research & Investment Orchestration Network in action.
"""

import os
import sys

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from env.stock_research_env import StockResearchEnv
from agents.controller_agent import ControllerAgent
from utils.state_encoder import StateEncoder


def run_demo(stock_symbol: str = 'NVDA', max_steps: int = 15):
    """
    Run a demonstration of the system.
    
    Args:
        stock_symbol: Stock symbol to analyze
        max_steps: Maximum number of steps
    """
    print("=" * 70)
    print("ORION-AI — Optimized Research & Investment Orchestration Network")
    print("=" * 70)
    print(f"\nAnalyzing: {stock_symbol}\n")
    
    # Initialize components
    env = StockResearchEnv(stock_symbol=stock_symbol, max_steps=max_steps)
    controller = ControllerAgent(use_q_learning=True, use_ppo=True)
    state_encoder = StateEncoder()
    
    # Reset environment
    state = env.reset()
    step_count = 0
    
    print("Starting Analysis...")
    print("-" * 70)
    
    while not env.done and step_count < max_steps:
        step_count += 1
        
        # Select action
        action_name, action_info = controller.select_action(state, training=False)
        
        print(f"\nStep {step_count}: {action_name}")
        
        # Execute action
        next_state, reward, done, info = env.step(action_name)
        
        # Display results based on action
        if action_name == 'FETCH_NEWS' and env.news_data:
            print(f"  ✓ Fetched {len(env.news_data['headlines'])} news articles")
            print(f"    Sentiment Score: {env.news_data['sentiment_score']:.2f}")
        
        elif action_name == 'FETCH_FUNDAMENTALS' and env.fundamentals_data:
            print(f"  ✓ Fetched fundamental data")
            print(f"    P/E Ratio: {env.fundamentals_data['pe_ratio']:.2f}")
            print(f"    Revenue Growth: {env.fundamentals_data['revenue_growth']*100:.1f}%")
        
        elif action_name == 'FETCH_SENTIMENT' and env.sentiment_data:
            print(f"  ✓ Fetched sentiment data")
            print(f"    Social Sentiment: {env.sentiment_data['social_sentiment']:.2f}")
            print(f"    Analyst Rating: {env.sentiment_data['analyst_rating']}")
        
        elif action_name == 'RUN_TA_BASIC' and env.ta_basic_data:
            print(f"  ✓ Completed basic technical analysis")
            print(f"    RSI: {env.ta_basic_data['rsi']:.2f}")
            print(f"    MA20: ${env.ta_basic_data['ma20']:.2f}")
        
        elif action_name == 'RUN_TA_ADVANCED' and env.ta_advanced_data:
            print(f"  ✓ Completed advanced technical analysis")
            print(f"    Trend: {env.ta_advanced_data['trend']}")
            print(f"    MACD Signal: {env.ta_advanced_data['macd_signal']:.2f}")
        
        elif action_name == 'GENERATE_INSIGHT' and env.insights:
            print(f"  ✓ Generated {len(env.insights)} insights:")
            for i, insight in enumerate(env.insights, 1):
                print(f"    {i}. {insight}")
        
        elif action_name == 'GENERATE_RECOMMENDATION' and env.recommendation:
            print(f"  ✓ Generated recommendation:")
            print(f"    Recommendation: {env.recommendation}")
            print(f"    Confidence: {env.confidence*100:.1f}%")
        
        elif action_name == 'STOP':
            print(f"  ✓ Stopping analysis")
        
        print(f"  Reward: {reward:.3f}")
        
        state = next_state
    
    # Final summary
    print("\n" + "=" * 70)
    print("Analysis Complete")
    print("=" * 70)
    
    print(f"\nTotal Steps: {step_count}")
    print(f"Tools Used: {len(set(env.tools_used))} unique tools")
    print(f"Sources Used: {len(set(env.sources_used))} unique sources")
    
    if env.insights:
        print(f"\nGenerated Insights ({len(env.insights)}):")
        for i, insight in enumerate(env.insights, 1):
            print(f"  {i}. {insight}")
    
    if env.recommendation:
        print(f"\nFinal Recommendation:")
        print(f"  Action: {env.recommendation}")
        print(f"  Confidence: {env.confidence*100:.1f}%")
        print(f"  Expected Outcome: {env.scenario['outcome']}")
        
        # Check correctness
        recommendation_map = {'Buy': 'positive', 'Hold': 'neutral', 'Sell': 'negative'}
        expected_outcome = recommendation_map.get(env.recommendation, 'neutral')
        actual_outcome = env.scenario['outcome']
        correct = "✓ CORRECT" if expected_outcome == actual_outcome else "✗ INCORRECT"
        print(f"  {correct}")
    
    # Final reward
    final_reward = env._calculate_final_reward()
    print(f"\nFinal Reward: {final_reward:.3f}")
    
    print("\n" + "=" * 70)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Demo: RL-Enhanced Investment System')
    parser.add_argument('--stock', type=str, default='NVDA',
                       choices=['NVDA', 'AAPL', 'TSLA', 'JPM', 'XOM'],
                       help='Stock symbol to analyze')
    parser.add_argument('--max-steps', type=int, default=15,
                       help='Maximum number of steps')
    
    args = parser.parse_args()
    
    run_demo(stock_symbol=args.stock, max_steps=args.max_steps)

