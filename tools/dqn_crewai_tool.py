"""
CrewAI Tool Wrapper for DQN Stock Research Model

This module provides a CrewAI-compatible tool that wraps the trained DQN model
from the Colab notebook. The tool can be used by CrewAI agents to analyze stocks
using the Reinforcement Learning model.

Usage:
    from tools.dqn_crewai_tool import dqn_stock_research
    
    agent = Agent(
        role="Stock Analyst",
        tools=[dqn_stock_research],
        ...
    )
"""

import os
import sys
from typing import Dict, Any, Optional
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from crewai.tools import tool
except ImportError:
    raise ImportError(
        "CrewAI is not installed. Install it with: pip install crewai"
    )

from rl.dqn import DQN
from utils.state_encoder import StateEncoder
from env.stock_research_env import StockResearchEnv
from utils.data_cache import DataCache


def find_model_file(default_path: str = 'dqn_model.pth') -> str:
    """
    Find the DQN model file in common locations.
    
    Checks in order:
    1. Current directory
    2. Project root
    3. Downloads folder
    4. Environment variable DQN_MODEL_PATH
    
    Args:
        default_path: Default relative path to look for
        
    Returns:
        Path to model file
        
    Raises:
        FileNotFoundError: If model file not found anywhere
    """
    # Get project root (parent of tools directory)
    project_root = Path(__file__).parent.parent
    
    # Possible locations
    possible_paths = [
        default_path,  # Current directory
        project_root / default_path,  # Project root
        Path.home() / 'Downloads' / default_path,  # Downloads folder
        os.getenv('DQN_MODEL_PATH', ''),  # Environment variable
    ]
    
    # Filter out empty paths
    possible_paths = [p for p in possible_paths if p]
    
    for path in possible_paths:
        path_obj = Path(path)
        if path_obj.exists() and path_obj.is_file():
            print(f"✅ Found model at: {path_obj.absolute()}")
            return str(path_obj.absolute())
    
    # If not found, raise error with helpful message
    raise FileNotFoundError(
        f"DQN model not found. Searched in:\n"
        f"  - {default_path}\n"
        f"  - {project_root / default_path}\n"
        f"  - {Path.home() / 'Downloads' / default_path}\n"
        f"  - {os.getenv('DQN_MODEL_PATH', 'Not set')}\n\n"
        f"Please download the model from Colab and place it in one of these locations.\n"
        f"See tools/COLAB_MODEL_DOWNLOAD_GUIDE.md for instructions."
    )


class DQNStockResearchTool:
    """
    CrewAI-compatible tool wrapper for DQN stock research system.
    This allows the trained DQN model to be used as a tool in CrewAI agents.
    """
    
    def __init__(self, dqn_model_path: Optional[str] = None):
        """
        Initialize the tool with a trained DQN model.
        
        Args:
            dqn_model_path: Path to saved DQN model (.pth file). 
                           If None, will search common locations.
        """
        # Find model file if path not provided
        if dqn_model_path is None:
            dqn_model_path = find_model_file()
        elif not os.path.exists(dqn_model_path):
            # Try to find it if provided path doesn't exist
            dqn_model_path = find_model_file(dqn_model_path)
        
        if not os.path.exists(dqn_model_path):
            raise FileNotFoundError(
                f"DQN model not found at: {dqn_model_path}\n"
                f"Please download the model from Colab and place it in the project directory.\n"
                f"See tools/COLAB_MODEL_DOWNLOAD_GUIDE.md for instructions."
            )
        
        # Initialize DQN with default parameters (must match training)
        # Try state_size=21 first (with news sentiment), fallback to 20 for old models
        try:
            self.dqn = DQN(state_size=21)
            self.dqn.load(dqn_model_path)
            state_dim = 21
        except Exception:
            # Fallback to old model format (state_size=20)
            self.dqn = DQN(state_size=20)
            self.dqn.load(dqn_model_path)
            state_dim = 20
        
        # Set epsilon to 0 for inference (no exploration)
        self.dqn.epsilon = 0.0
        
        # Initialize state encoder (matches DQN state_size)
        self.state_encoder = StateEncoder(state_dim=state_dim)
        
        # Initialize data cache
        self.cache = DataCache()
        
        print(f"✅ DQN Stock Research Tool initialized with model: {dqn_model_path}")
    
    def analyze_stock(self, stock_symbol: str) -> Dict[str, Any]:
        """
        Analyze a stock using the trained DQN model.
        
        Args:
            stock_symbol: Stock ticker symbol (e.g., 'NVDA', 'AAPL')
        
        Returns:
            Dictionary containing analysis results, recommendation, and insights
        """
        # Initialize environment
        env = StockResearchEnv(
            stock_symbol=stock_symbol,
            max_steps=20,
            cache=self.cache
        )
        
        state = env.reset()
        state_vector = self.state_encoder.encode_continuous(state)
        
        actions_taken = []
        total_reward = 0.0
        
        # Run DQN inference (no exploration)
        while not env.done:
            action_idx, action_name = self.dqn.select_action(
                state_vector, 
                training=False  # No exploration during inference
            )
            
            next_state, reward, done, info = env.step(action_name)
            next_state_vector = self.state_encoder.encode_continuous(next_state)
            
            actions_taken.append(action_name)
            total_reward += reward
            state_vector = next_state_vector
            
            if done:
                break
        
        # Compile results
        result = {
            'stock_symbol': stock_symbol,
            'recommendation': env.recommendation or 'Hold',
            'confidence': float(env.confidence) if env.confidence else 0.5,
            'insights': env.insights or [],
            'actions_taken': actions_taken,
            'total_reward': float(total_reward),
            'news_data': env.news_data,
            'fundamentals_data': env.fundamentals_data,
            'ta_basic': env.ta_basic_data,
            'ta_advanced': env.ta_advanced_data,
            'num_steps': len(actions_taken)
        }
        
        return result
    
    def get_recommendation_summary(self, stock_symbol: str) -> str:
        """
        Get a formatted summary string for CrewAI.
        
        Args:
            stock_symbol: Stock ticker symbol
        
        Returns:
            Formatted string summary
        """
        result = self.analyze_stock(stock_symbol)
        
        summary = f"""
Stock Analysis Report for {stock_symbol}
========================================

Recommendation: {result['recommendation']}
Confidence: {result['confidence']:.1%}

Key Insights:
"""
        if result['insights']:
            for i, insight in enumerate(result['insights'], 1):
                summary += f"  {i}. {insight}\n"
        else:
            summary += "  No insights generated.\n"
        
        summary += f"\nAnalysis Steps: {result['num_steps']}\n"
        summary += f"Actions Taken: {', '.join(result['actions_taken'])}\n"
        
        if result['ta_basic']:
            summary += f"\nTechnical Analysis:\n"
            summary += f"  RSI: {result['ta_basic'].get('rsi', 'N/A'):.2f}\n"
            summary += f"  Current Price: ${result['ta_basic'].get('current_price', 0):.2f}\n"
        
        if result['news_data']:
            summary += f"\nNews Sentiment: {result['news_data'].get('sentiment_score', 0.5):.1%}\n"
            summary += f"Articles Analyzed: {result['news_data'].get('num_articles', 0)}\n"
        
        return summary


# Global tool instance (lazy initialization)
_tool_instance: Optional[DQNStockResearchTool] = None


@tool("DQN Stock Research Tool")
def dqn_stock_research(stock_symbol: str) -> str:
    """
    Analyze a stock using a trained Deep Q-Network (DQN) RL model.
    
    The DQN model intelligently orchestrates multiple agents:
    - Research Agent: Fetches news, fundamentals, sentiment
    - Technical Analysis Agent: Calculates indicators (RSI, MACD, etc.)
    - Insight Agent: Generates actionable insights
    - Recommendation Agent: Provides Buy/Hold/Sell recommendations
    
    Args:
        stock_symbol: Stock ticker symbol (e.g., 'NVDA', 'AAPL', 'TSLA', 'MSFT')
    
    Returns:
        Comprehensive stock analysis report including:
        - Investment recommendation (Buy/Hold/Sell)
        - Confidence level (0-100%)
        - Key insights from multiple data sources
        - Technical analysis indicators
        - News sentiment analysis
        - Actions taken by RL agent
    
    Example:
        result = dqn_stock_research("NVDA")
        # Returns formatted analysis report
    """
    global _tool_instance
    
    if _tool_instance is None:
        model_path = os.getenv('DQN_MODEL_PATH', None)
        try:
            _tool_instance = DQNStockResearchTool(dqn_model_path=model_path)
        except FileNotFoundError as e:
            return f"Error: {str(e)}"
        except Exception as e:
            return f"Error initializing DQN tool: {str(e)}"
    
    try:
        return _tool_instance.get_recommendation_summary(stock_symbol)
    except Exception as e:
        return f"Error analyzing {stock_symbol}: {str(e)}"

