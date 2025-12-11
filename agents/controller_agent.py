"""
Controller Agent
RL-powered agent that orchestrates other agents using Q-Learning and PPO.
"""

from typing import Dict, List, Any, Optional, Tuple
import time
import numpy as np
from rl.q_learning import QLearning
from rl.ppo import PPO
from utils.state_encoder import StateEncoder
from agents.research_agent import ResearchAgent
from agents.ta_agent import TechnicalAnalysisAgent
from agents.insight_agent import InsightAgent
from agents.recommendation_agent import RecommendationAgent
from agents.evaluator_agent import EvaluatorAgent
from utils.data_cache import DataCache


class ControllerAgent:
    """
    Central controller agent that uses RL to optimize agent orchestration.
    """
    
    def __init__(
        self,
        use_q_learning: bool = True,
        use_ppo: bool = True,
        state_dim: int = 32,
        cache: Optional[DataCache] = None
    ):
        """
        Initialize Controller Agent.
        
        Args:
            use_q_learning: Whether to use Q-Learning for tool selection
            use_ppo: Whether to use PPO for stopping policy
            state_dim: Dimension of state space
            cache: Data cache instance for real data
        """
        self.name = "ControllerAgent"
        self.use_q_learning = use_q_learning
        self.use_ppo = use_ppo
        
        # Initialize RL algorithms
        if use_q_learning:
            self.q_learning = QLearning()
        else:
            self.q_learning = None
        
        if use_ppo:
            self.ppo = PPO(state_dim=state_dim)
        else:
            self.ppo = None
        
        self.state_encoder = StateEncoder(state_dim=state_dim)
        
        # Initialize cache if not provided
        self.cache = cache or DataCache()
        
        # Initialize sub-agents with cache
        self.research_agent = ResearchAgent(cache=self.cache)
        self.ta_agent = TechnicalAnalysisAgent()
        self.insight_agent = InsightAgent()
        self.recommendation_agent = RecommendationAgent()
        self.evaluator_agent = EvaluatorAgent()
        
        # State tracking
        self.current_state = None
        self.episode_history = []
    
    def select_action(self, state: Dict[str, Any], training: bool = True) -> Tuple[str, Dict[str, Any]]:
        """
        Select next action using RL algorithms.
        
        Args:
            state: Current state dictionary
            training: Whether in training mode
        
        Returns:
            Tuple of (action_name, action_info)
        """
        print(f"  [ControllerAgent.select_action] Called with training={training}")
        self.current_state = state
        
        # Encode state
        if self.use_q_learning:
            print(f"  [ControllerAgent] Using Q-Learning for action selection")
            discrete_state = self.state_encoder.encode_discrete(state)
            print(f"  [ControllerAgent] Encoded discrete state: {discrete_state}")
            
            if hasattr(self.q_learning, 'q_table'):
                if discrete_state in self.q_learning.q_table:
                    q_values = self.q_learning.q_table[discrete_state]
                    print(f"  [ControllerAgent] Q-values for state {discrete_state}: {q_values}")
                else:
                    print(f"  [ControllerAgent] State {discrete_state} not in Q-table (will use default)")
            
            action_idx, action_name = self.q_learning.select_action(discrete_state, training)
            print(f"  [ControllerAgent] Q-Learning selected: action_idx={action_idx}, action_name={action_name}")
        else:
            # Fallback: simple heuristic
            print(f"  [ControllerAgent] Q-Learning disabled, using heuristic")
            action_name = self._heuristic_action_selection(state)
            print(f"  [ControllerAgent] Heuristic selected: {action_name}")
        
        # Use PPO for stopping decision if available
        if self.use_ppo and action_name == 'STOP':
            print(f"  [ControllerAgent] PPO enabled, checking STOP decision")
            continuous_state = self.state_encoder.encode_continuous(state)
            ppo_action, ppo_action_name, _, _ = self.ppo.select_action(continuous_state)
            print(f"  [ControllerAgent] PPO decision: {ppo_action_name}")
            
            if ppo_action_name == 'CONTINUE':
                # Override STOP decision
                action_name = self._select_continue_action(state)
                print(f"  [ControllerAgent] PPO overrode STOP → {action_name}")
            elif ppo_action_name == 'ANALYZE_MORE':
                # Force more analysis
                action_name = self._select_analysis_action(state)
                print(f"  [ControllerAgent] PPO forced more analysis → {action_name}")
        else:
            if self.use_ppo:
                print(f"  [ControllerAgent] PPO enabled but action is not STOP, skipping PPO")
            else:
                print(f"  [ControllerAgent] PPO disabled")
        
        action_info = {
            'action': action_name,
            'state': state,
            'training': training
        }
        
        return action_name, action_info
    
    def _heuristic_action_selection(self, state: Dict[str, Any]) -> str:
        """
        Heuristic action selection (fallback when Q-Learning not used).
        
        Args:
            state: Current state dictionary
        
        Returns:
            Action name
        """
        # Priority order: gather data -> analyze -> generate insights -> recommend -> stop
        if not state.get('has_news', False):
            return 'FETCH_NEWS'
        if not state.get('has_fundamentals', False):
            return 'FETCH_FUNDAMENTALS'
        if not state.get('has_ta_basic', False):
            return 'RUN_TA_BASIC'
        if not state.get('has_insights', False):
            return 'GENERATE_INSIGHT'
        if not state.get('has_recommendation', False):
            return 'GENERATE_RECOMMENDATION'
        
        return 'STOP'
    
    def _select_continue_action(self, state: Dict[str, Any]) -> str:
        """
        Select action to continue analysis.
        
        Args:
            state: Current state dictionary
        
        Returns:
            Action name
        """
        # Select action that adds most value
        if not state.get('has_ta_advanced', False) and state.get('has_ta_basic', False):
            return 'RUN_TA_ADVANCED'
        if not state.get('has_sentiment', False):
            return 'FETCH_SENTIMENT'
        if not state.get('has_macro', False):
            return 'FETCH_MACRO'
        
        return 'GENERATE_INSIGHT'
    
    def _select_analysis_action(self, state: Dict[str, Any]) -> str:
        """
        Select action for additional analysis.
        
        Args:
            state: Current state dictionary
        
        Returns:
            Action name
        """
        if not state.get('has_ta_advanced', False):
            return 'RUN_TA_ADVANCED'
        if not state.get('has_sentiment', False):
            return 'FETCH_SENTIMENT'
        if not state.get('has_macro', False):
            return 'FETCH_MACRO'
        
        return 'GENERATE_INSIGHT'
    
    def update_q_learning(
        self,
        state: int,
        action: int,
        reward: float,
        next_state: int,
        done: bool
    ):
        """
        Update Q-Learning algorithm.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode terminated
        """
        if self.q_learning:
            self.q_learning.update(state, action, reward, next_state, done)
    
    def update_ppo(self, buffer, epochs: int = 10):
        """
        Update PPO algorithm.
        
        Args:
            buffer: Rollout buffer with trajectories
            epochs: Number of update epochs
        """
        if self.ppo:
            self.ppo.update(buffer, epochs)
    
    def execute_action(
        self,
        action: str,
        env_state: Dict[str, Any],
        price_history: List[float],
        high_history: List[float],
        low_history: List[float]
    ) -> Dict[str, Any]:
        """
        Execute selected action by calling appropriate agent.
        
        Args:
            action: Action to execute
            env_state: Current environment state
            price_history: Price history for TA
            high_history: High price history
            low_history: Low price history
        
        Returns:
            Result dictionary
        """
        result = {}
        
        if action == 'FETCH_NEWS':
            result = self.research_agent.fetch_news(
                env_state.get('stock_symbol', 'UNKNOWN'),
                env_state.get('news_data')
            )
        
        elif action == 'FETCH_FUNDAMENTALS':
            result = self.research_agent.fetch_fundamentals(
                env_state.get('stock_symbol', 'UNKNOWN'),
                env_state.get('fundamentals_data')
            )
        
        elif action == 'FETCH_SENTIMENT':
            result = self.research_agent.fetch_sentiment(
                env_state.get('stock_symbol', 'UNKNOWN'),
                env_state.get('sentiment_data')
            )
        
        elif action == 'FETCH_MACRO':
            result = self.research_agent.fetch_macro(env_state.get('macro_data'))
        
        elif action == 'RUN_TA_BASIC':
            result = self.ta_agent.run_ta_basic(
                price_history,
                env_state.get('ta_basic_data')
            )
        
        elif action == 'RUN_TA_ADVANCED':
            result = self.ta_agent.run_ta_advanced(
                price_history,
                high_history,
                low_history,
                env_state.get('ta_advanced_data')
            )
        
        elif action == 'GENERATE_INSIGHT':
            result = self.insight_agent.generate_insight(
                news_data=env_state.get('news_data'),
                fundamentals_data=env_state.get('fundamentals_data'),
                sentiment_data=env_state.get('sentiment_data'),
                macro_data=env_state.get('macro_data'),
                ta_basic_data=env_state.get('ta_basic_data'),
                ta_advanced_data=env_state.get('ta_advanced_data'),
                existing_insights=env_state.get('insights', [])
            )
        
        elif action == 'GENERATE_RECOMMENDATION':
            recommendation, confidence = self.recommendation_agent.generate_recommendation(
                insights=env_state.get('insights', []),
                news_data=env_state.get('news_data'),
                fundamentals_data=env_state.get('fundamentals_data'),
                sentiment_data=env_state.get('sentiment_data'),
                ta_basic_data=env_state.get('ta_basic_data'),
                ta_advanced_data=env_state.get('ta_advanced_data')
            )
            result = {
                'recommendation': recommendation,
                'confidence': confidence
            }
        
        elif action == 'STOP':
            result = {'status': 'stopped'}
        
        return result
    
    def orchestrate(
        self,
        stock_symbol: str,
        env: Any,
        callback_queue: Any = None
    ) -> Dict[str, Any]:
        """
        Orchestrate the entire agent workflow.
        Controller Agent calls each agent sequentially, waits for completion,
        and stops on any failure.
        
        Args:
            stock_symbol: Stock symbol to analyze
            env: StockResearchEnv instance
            callback_queue: Optional queue for sending updates to frontend
        
        Returns:
            Dictionary with orchestration results and status
        """
        # Prevent multiple simultaneous orchestrations
        if hasattr(self, '_orchestrating') and self._orchestrating:
            print(f"⚠️  Orchestration already in progress! Ignoring duplicate call.")
            return {
                'orchestration_status': {'success': False, 'error': 'Orchestration already in progress'},
                'agent_calls': [],
                'total_steps': 0
            }
        
        self._orchestrating = True
        
        print(f"\n{'='*70}")
        print("🎯 CONTROLLER AGENT ORCHESTRATION")
        print(f"{'='*70}")
        print(f"Stock: {stock_symbol}")
        print(f"Controller Agent will orchestrate all agents sequentially")
        print(f"{'='*70}\n")
        
        # Define the sequence of agents to call
        agent_sequence = [
            ('FETCH_NEWS', 'ResearchAgent'),
            ('FETCH_FUNDAMENTALS', 'ResearchAgent'),
            ('FETCH_SENTIMENT', 'ResearchAgent'),
            ('RUN_TA_BASIC', 'TechnicalAnalysisAgent'),
            ('RUN_TA_ADVANCED', 'TechnicalAnalysisAgent'),
            ('GENERATE_INSIGHT', 'InsightAgent'),
            ('GENERATE_RECOMMENDATION', 'RecommendationAgent'),
            ('EVALUATE_PERFORMANCE', 'EvaluatorAgent')
        ]
        
        step_count = 0
        agent_calls = []
        orchestration_status = {
            'success': True,
            'error': None,
            'failed_agent': None,
            'failed_action': None,
            'steps_completed': 0
        }
        
        # Orchestrate each agent in sequence
        # Track which actions have been executed to prevent loops
        executed_actions = set()
        
        print(f"\n📋 Starting orchestration of {len(agent_sequence)} agents...")
        for idx, (action_name, agent_name) in enumerate(agent_sequence, 1):
            step_count += 1
            
            print(f"\n[ORCHESTRATION {idx}/{len(agent_sequence)}] Processing: {action_name}")
            
            # Prevent executing the same action multiple times
            if action_name in executed_actions:
                print(f"   ⚠️  SKIPPING {action_name} - already executed in this orchestration")
                print(f"   Executed actions so far: {executed_actions}")
                continue
            
            # Check prerequisites before calling agent
            if action_name == 'GENERATE_INSIGHT':
                has_ta = env.ta_basic_data is not None or env.ta_advanced_data is not None
                if not has_ta:
                    print(f"   ⚠️  Skipping {action_name} - prerequisites not met (TA data required)")
                    executed_actions.add(action_name)  # Mark as executed even if skipped
                    continue
                # Also check if insights already exist
                if env.insights and len(env.insights) > 0:
                    print(f"   ⚠️  Skipping {action_name} - insights already exist (count: {len(env.insights)})")
                    executed_actions.add(action_name)  # Mark as executed
                    continue
            
            if action_name == 'GENERATE_RECOMMENDATION':
                has_ta = env.ta_basic_data is not None or env.ta_advanced_data is not None
                has_insights = env.insights and len(env.insights) > 0
                if not (has_ta and has_insights):
                    print(f"   ⚠️  Skipping {action_name} - prerequisites not met (TA={has_ta}, Insights={has_insights})")
                    executed_actions.add(action_name)  # Mark as executed even if skipped
                    continue
            
            if action_name == 'EVALUATE_PERFORMANCE':
                if not env.recommendation:
                    print(f"   ⚠️  Skipping {action_name} - no recommendation to evaluate")
                    executed_actions.add(action_name)  # Mark as executed even if skipped
                    break  # Last action, so break instead of continue
            
            print(f"\n[STEP {step_count}] 🎯 Controller Agent calling: {agent_name}")
            print(f"   Action: {action_name}")
            print(f"   Progress: {idx}/{len(agent_sequence)} in sequence")
            print(f"   Already executed: {executed_actions}")
            
            # Record agent call
            agent_call = {
                'step': step_count,
                'agent': agent_name,
                'action': action_name,
                'timestamp': time.time(),
                'status': 'calling'
            }
            agent_calls.append(agent_call)
            
            # Send update to frontend if callback queue provided
            if callback_queue:
                callback_queue.put({
                    'type': 'agent_call',
                    'agent': agent_name,
                    'action': action_name,
                    'step': step_count,
                    'source': 'controller_orchestration',
                    'message': f'Step {step_count}: Controller Agent calling {agent_name} for {action_name}'
                })
            
            # Execute action through environment (which calls the agent)
            try:
                print(f"   🔄 Executing {action_name}...")
                next_state, reward, done, info = env.step(action_name)
                
                # Update agent call status
                agent_call['status'] = 'completed'
                agent_call['reward'] = reward
                agent_call['done'] = done
                
                print(f"   ✅ {agent_name} completed successfully")
                print(f"   - Reward: {reward:.4f}")
                print(f"   - Done: {done}")
                
                # Send result update to frontend
                if callback_queue:
                    callback_queue.put({
                        'type': 'action_result',
                        'action': action_name,
                        'agent': agent_name,
                        'result': self._get_action_result(action_name, env),
                        'reward': reward,
                        'step': step_count
                    })
                
                # Controller Agent continues orchestration regardless of env.done
                # Only stop if we've completed all agents or if there's an error
                # The environment's done flag is for RL training, not for orchestration control
                if env.done and action_name != 'EVALUATE_PERFORMANCE':
                    # Reset done flag to allow Controller Agent to continue orchestration
                    print(f"   ℹ️  Environment marked as done, but Controller Agent will continue orchestration")
                    env.done = False  # Reset to allow continuation
                
                orchestration_status['steps_completed'] = step_count
                
                # Mark action as executed to prevent loops
                executed_actions.add(action_name)
                
            except Exception as e:
                # Agent failed - Controller Agent stops execution and reports error
                error_msg = f"Agent '{agent_name}' failed during action '{action_name}': {str(e)}"
                print(f"\n❌ ERROR: {error_msg}")
                import traceback
                traceback.print_exc()
                
                # Update orchestration status
                orchestration_status['success'] = False
                orchestration_status['error'] = str(e)
                orchestration_status['failed_agent'] = agent_name
                orchestration_status['failed_action'] = action_name
                orchestration_status['steps_completed'] = step_count - 1
                
                # Update agent call status
                agent_call['status'] = 'failed'
                agent_call['error'] = str(e)
                
                # Send error to frontend
                if callback_queue:
                    callback_queue.put({
                        'type': 'error',
                        'message': error_msg,
                        'error': str(e),
                        'agent': agent_name,
                        'action': action_name,
                        'step': step_count
                    })
                
                # Stop the workflow
                env.done = True
                break
            
            # Small delay for visualization
            time.sleep(0.5)
        
        print(f"\n{'='*70}")
        print("📊 ORCHESTRATION COMPLETE")
        print(f"{'='*70}")
        print(f"Status: {'✅ SUCCESS' if orchestration_status['success'] else '❌ FAILED'}")
        print(f"Steps Completed: {orchestration_status['steps_completed']}")
        print(f"Total Actions Executed: {len(executed_actions)}")
        print(f"Executed Actions: {executed_actions}")
        if not orchestration_status['success']:
            print(f"Failed Agent: {orchestration_status['failed_agent']}")
            print(f"Failed Action: {orchestration_status['failed_action']}")
            print(f"Error: {orchestration_status['error']}")
        print(f"{'='*70}\n")
        
        # Always reset orchestration flag (use finally-like pattern)
        self._orchestrating = False
        
        return {
            'orchestration_status': orchestration_status,
            'agent_calls': agent_calls,
            'total_steps': step_count
        }
    
    def _get_action_result(self, action: str, env: Any) -> dict:
        """Get result data for an action (helper for orchestration)."""
        result = {}
        
        if action == 'FETCH_NEWS' and env.news_data:
            result = {
                'type': 'news',
                'headlines': env.news_data.get('headlines', []),
                'sentiment': env.news_data.get('sentiment_score', 0.0),
                'num_articles': env.news_data.get('num_articles', 0),
                'articles': env.news_data.get('articles', [])
            }
        elif action == 'FETCH_FUNDAMENTALS' and env.fundamentals_data:
            result = {
                'type': 'fundamentals',
                'pe_ratio': env.fundamentals_data.get('pe_ratio', 0.0),
                'revenue_growth': env.fundamentals_data.get('revenue_growth', 0.0),
                'profit_margin': env.fundamentals_data.get('profit_margin', 0.0)
            }
        elif action == 'FETCH_SENTIMENT' and env.sentiment_data:
            result = {
                'type': 'sentiment',
                'social_sentiment': env.sentiment_data.get('social_sentiment', 0.0),
                'analyst_rating': env.sentiment_data.get('analyst_rating', 'Hold')
            }
        elif action == 'RUN_TA_BASIC' and env.ta_basic_data:
            result = {
                'type': 'ta_basic',
                'rsi': env.ta_basic_data.get('rsi', 50.0),
                'ma20': env.ta_basic_data.get('ma20', 0.0),
                'current_price': env.ta_basic_data.get('current_price', 0.0)
            }
        elif action == 'RUN_TA_ADVANCED' and env.ta_advanced_data:
            result = {
                'type': 'ta_advanced',
                'trend': env.ta_advanced_data.get('trend', 'sideways'),
                'macd_signal': env.ta_advanced_data.get('macd_signal', 0.0),
                'ma50': env.ta_advanced_data.get('ma50', 0.0),
                'ma200': env.ta_advanced_data.get('ma200', 0.0)
            }
        elif action == 'GENERATE_INSIGHT' and env.insights:
            result = {
                'type': 'insights',
                'insights': env.insights
            }
        elif action == 'GENERATE_RECOMMENDATION' and env.recommendation:
            result = {
                'type': 'recommendation',
                'recommendation': env.recommendation,
                'confidence': env.confidence
            }
        elif action == 'EVALUATE_PERFORMANCE' and env.evaluation_results:
            result = {
                'type': 'evaluation',
                'efficiency_score': env.evaluation_results.get('efficiency_score', 0.0),
                'diversity_score': env.evaluation_results.get('diversity_score', 0.0),
                'evaluation_quality': env.evaluation_results.get('evaluation_quality', 0.0)
            }
        
        return result

