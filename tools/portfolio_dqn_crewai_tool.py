"""
CrewAI Tool for Portfolio-Level DQN with Outcome-Based Learning

This tool uses Portfolio DQN to:
1. Select which stocks to analyze from a watchlist
2. Allocate capital optimally across selected stocks
3. Learn from actual returns (outcome-based learning)

Fits Assignment Requirements:
- Value-Based Learning (DQN) ✅
- Agent Orchestration Systems ✅ (decides which stocks/agents to use)
- Research/Analysis Agents ✅ (learns effective information gathering)
"""

import os
import sys
from typing import Dict, Any, Optional, List

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from crewai.tools import tool
except ImportError:
    raise ImportError("CrewAI is not installed. Install with: pip install crewai")

from env.portfolio_env import PortfolioEnv
from rl.portfolio_dqn import PortfolioDQN
from utils.portfolio_state_encoder import PortfolioStateEncoder
from utils.outcome_tracker import OutcomeTracker
from utils.data_cache import DataCache


class PortfolioDQNTool:
    """
    CrewAI-compatible tool for portfolio-level stock analysis.
    Uses DQN to optimize stock selection and capital allocation.
    """
    
    def __init__(self, model_path: str = 'experiments/results/portfolio_dqn/portfolio_dqn_model.pth'):
        """
        Initialize portfolio DQN tool.
        
        Args:
            model_path: Path to trained Portfolio DQN model
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Portfolio DQN model not found at: {model_path}\n"
                f"Please train the model first using train_portfolio_dqn.py"
            )
        
        # Initialize DQN
        self.dqn = PortfolioDQN(state_size=50)
        self.dqn.load(model_path)
        self.dqn.epsilon = 0.0  # No exploration during inference
        
        # Initialize components
        self.state_encoder = PortfolioStateEncoder(state_dim=50)
        self.cache = DataCache()
        self.outcome_tracker = OutcomeTracker()
        
        print(f"✅ Portfolio DQN Tool initialized with model: {model_path}")
    
    def optimize_portfolio(
        self,
        watchlist: Optional[List[str]] = None,
        initial_capital: float = 100000.0
    ) -> Dict[str, Any]:
        """
        Optimize portfolio using trained DQN.
        
        Args:
            watchlist: List of stocks to choose from (default: env watchlist)
            initial_capital: Starting capital
        
        Returns:
            Portfolio optimization results
        """
        # Initialize environment
        env = PortfolioEnv(
            watchlist=watchlist,
            initial_capital=initial_capital,
            use_latest_date=True,  # Use latest date for inference
            cache=self.cache
        )
        
        state_dict = env.reset()
        state_vector = self.state_encoder.encode_continuous(state_dict)
        
        actions_taken = []
        portfolio_id = f"portfolio_{os.getpid()}_{len(self.outcome_tracker.recommendations)}"
        
        # Run DQN inference
        while not env.done:
            action_idx, action_name = self.dqn.select_action(state_vector, training=False)
            
            next_state_dict, reward, done, info = env.step(action_name)
            next_state_vector = self.state_encoder.encode_continuous(next_state_dict)
            
            actions_taken.append(action_name)
            state_vector = next_state_vector
            
            # Track recommendations
            if action_name == 'ANALYZE_STOCK' and 'analysis' in info:
                analysis = info['analysis']
                stock = env.current_stock
                if stock:
                    self.outcome_tracker.record_recommendation(
                        stock_symbol=stock,
                        recommendation=analysis.get('recommendation', 'Hold'),
                        confidence=analysis.get('confidence', 0.5),
                        allocation=env.allocations.get(stock, 0.0),
                        portfolio_id=portfolio_id
                    )
            
            if done:
                break
        
        # Get portfolio summary
        portfolio_summary = env.get_portfolio_summary()
        
        return {
            'portfolio_id': portfolio_id,
            'allocations': portfolio_summary.get('allocations', {}),
            'selected_stocks': list(portfolio_summary.get('allocations', {}).keys()),
            'num_stocks': len([a for a in portfolio_summary.get('allocations', {}).values() if a > 0]),
            'actions_taken': actions_taken,
            'analyzed_stocks': env.analyzed_stocks,
            'individual_returns': portfolio_summary.get('individual_returns', {}),
            'recommendations': portfolio_summary.get('recommendations', [])
        }
    
    def get_portfolio_summary(self, watchlist: Optional[List[str]] = None) -> str:
        """
        Get formatted portfolio summary for CrewAI.
        
        Args:
            watchlist: List of stocks to analyze
        
        Returns:
            Formatted summary string
        """
        result = self.optimize_portfolio(watchlist=watchlist)
        
        summary = f"""
Portfolio Optimization Report
============================

Selected Stocks: {', '.join(result['selected_stocks']) if result['selected_stocks'] else 'None'}
Number of Positions: {result['num_stocks']}

Capital Allocation:
"""
        for stock, allocation in result['allocations'].items():
            if allocation > 0:
                summary += f"  {stock}: {allocation:.1%}\n"
        
        summary += "\nStock Analysis:\n"
        for stock, analysis in result['analyzed_stocks'].items():
            summary += f"\n{stock}:\n"
            summary += f"  Recommendation: {analysis.get('recommendation', 'N/A')}\n"
            summary += f"  Confidence: {analysis.get('confidence', 0):.1%}\n"
            if analysis.get('insights'):
                summary += f"  Insights:\n"
                for insight in analysis['insights'][:3]:  # Top 3 insights
                    summary += f"    - {insight}\n"
        
        if result.get('individual_returns'):
            summary += "\nExpected Returns (based on analysis):\n"
            for stock, ret in result['individual_returns'].items():
                summary += f"  {stock}: {ret:.2%}\n"
        
        return summary


# Global tool instance
_tool_instance: Optional[PortfolioDQNTool] = None


@tool("Portfolio DQN Optimization Tool")
def portfolio_dqn_optimize(watchlist: str = None) -> str:
    """
    Optimize a stock portfolio using Reinforcement Learning.
    
    The Portfolio DQN model:
    1. Selects which stocks to analyze from a watchlist
    2. Allocates capital optimally across selected stocks
    3. Learns from actual returns (outcome-based learning)
    
    This uses Value-Based Learning (DQN) for Agent Orchestration:
    - Decides which stocks/agents to use
    - Optimizes capital allocation
    - Learns effective information gathering strategies
    
    Args:
        watchlist: Comma-separated list of stock symbols (e.g., "NVDA,AAPL,TSLA,MSFT")
                  If None, uses default watchlist
    
    Returns:
        Portfolio optimization report with:
        - Selected stocks and allocations
        - Investment recommendations
        - Expected returns
        - Analysis insights
    
    Example:
        result = portfolio_dqn_optimize("NVDA,AAPL,TSLA")
        # Returns optimized portfolio allocation
    """
    global _tool_instance
    
    if _tool_instance is None:
        model_path = os.getenv('PORTFOLIO_DQN_MODEL_PATH', 
                              'experiments/results/portfolio_dqn/portfolio_dqn_model.pth')
        try:
            _tool_instance = PortfolioDQNTool(model_path=model_path)
        except FileNotFoundError as e:
            return f"Error: {str(e)}"
        except Exception as e:
            return f"Error initializing Portfolio DQN tool: {str(e)}"
    
    try:
        # Parse watchlist if provided
        stock_list = None
        if watchlist:
            stock_list = [s.strip().upper() for s in watchlist.split(',')]
        
        return _tool_instance.get_portfolio_summary(watchlist=stock_list)
    except Exception as e:
        return f"Error optimizing portfolio: {str(e)}"

