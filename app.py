"""
Flask Web Application for ORION-AI — Optimized Research & Investment Orchestration Network
Provides a web UI for stock analysis with real-time agent visualization.
"""

import os
import sys
import json
import time
import numpy as np
from flask import Flask, request, jsonify, Response, send_from_directory, send_file
from flask_cors import CORS
import os
import threading
import queue

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from env.stock_research_env import StockResearchEnv
from agents.controller_agent import ControllerAgent
from agents.llm_query_interpreter import LLMQueryInterpreterAgent
from agents.llm_output_formatter import LLMOutputFormatterAgent
from agents.evaluator_agent import EvaluatorAgent
from agents.risk_agent import RiskAgent
from utils.state_encoder import StateEncoder
from utils.data_cache import DataCache
from utils.contradiction_detector import ContradictionDetector
from rl.dqn import DQN
from pathlib import Path

# Disable Flask's default static file handler to use our custom one
app = Flask(__name__, static_folder=None)
CORS(app)

# Global state
analysis_queue = queue.Queue()
current_analysis = None

# Stock symbol mapping
STOCK_SYMBOLS = {
    'nvidia': 'NVDA',
    'nvda': 'NVDA',
    'apple': 'AAPL',
    'aapl': 'AAPL',
    'tesla': 'TSLA',
    'tsla': 'TSLA',
    'jpmorgan': 'JPM',
    'jpm': 'JPM',
    'exxon': 'XOM',
    'xom': 'XOM',
    'exxonmobil': 'XOM'
}


def normalize_stock_symbol(company_name: str) -> str:
    """Normalize company name to stock symbol."""
    company_lower = company_name.strip().lower()
    
    # Direct mapping
    if company_lower in STOCK_SYMBOLS:
        return STOCK_SYMBOLS[company_lower]
    
    # Check if it's already a symbol
    if company_lower.upper() in ['NVDA', 'AAPL', 'TSLA', 'JPM', 'XOM']:
        return company_lower.upper()
    
    # Default to NVDA if not found
    return 'NVDA'


def run_analysis(company_name: str, callback_queue: queue.Queue, use_llm: bool = True):
    """
    Run stock analysis using RL for action selection.
    LLM is ONLY used for query interpretation (extracting ticker) and output formatting.
    RL (Q-Learning) is ALWAYS used for action selection - this is the core of the assignment.
    
    Args:
        company_name: Company name or stock symbol (or natural language query)
        callback_queue: Queue to send updates to
        use_llm: Whether to use Groq LLM for query interpretation and output formatting (NOT for action selection)
    """
    try:
        # Step 1: Interpret query with LLM (if enabled) - ONLY to extract ticker and preferences
        # LLM does NOT determine action sequence - RL does that!
        stock_symbol = normalize_stock_symbol(company_name)
        user_preferences = {}
        
        if use_llm:
            try:
                query_interpreter = LLMQueryInterpreterAgent()
                instructions = query_interpreter.interpret_query(company_name)
                stock_symbol = instructions.get('ticker', stock_symbol)
                # Store preferences but DON'T use agent_sequence - RL will decide actions
                user_preferences = {
                    'analysis_horizon': instructions.get('analysis_horizon', '1-3 months'),
                    'risk_tolerance': instructions.get('risk_tolerance', 'medium'),
                    'needs_fundamentals': instructions.get('needs_fundamentals', True),
                    'needs_sentiment': instructions.get('needs_sentiment', True),
                    'needs_news': instructions.get('needs_news', True),
                    'needs_technical_analysis': instructions.get('needs_technical_analysis', True),
                }
                callback_queue.put({
                    'type': 'llm_interpretation',
                    'message': f'LLM interpreted query: Analyzing {stock_symbol}',
                    'instructions': instructions,
                    'stock_symbol': stock_symbol
                })
                callback_queue.put({
                    'type': 'status',
                    'message': f'✅ Ticker extracted: {stock_symbol}. RL will now select actions...',
                })
            except Exception as e:
                print(f"LLM query interpretation failed: {e}. Using fallback.")
                stock_symbol = normalize_stock_symbol(company_name)
        
        # Initialize cache for real-world data
        cache = DataCache()
        
        # Initialize components with cache
        # Use latest date for inference/UI mode (not random date for training)
        env = StockResearchEnv(
            stock_symbol=stock_symbol, 
            max_steps=100,  # Increased to allow full orchestration (8 agents + RL components)
            cache=cache,
            use_latest_date=True  # Use latest date for UI/inference
        )
        
        # ========== DQN INITIALIZATION ==========
        print("\n" + "="*70)
        print("🤖 DQN RL SYSTEM INITIALIZATION")
        print("="*70)
        
        # Initialize DQN - try to load model first to determine state_size
        model_paths = [
            Path.home() / "Downloads" / "dqn_model.pth",
            Path.home() / "Downloads" / "dqn_model (1).pth",
            "experiments/results/dqn/dqn_model.pth",
            "dqn_model.pth",
            os.getenv('DQN_MODEL_PATH', '')
        ]
        
        model_loaded = False
        loaded_path = None
        state_size = 21  # Default to new size with news sentiment
        
        # Try to load model to determine its state_size
        for model_path in model_paths:
            if not model_path:
                continue
            model_path = Path(model_path)
            if model_path.exists():
                try:
                    # Try loading with state_size=21 first (new format)
                    try:
                        dqn = DQN(state_size=21)
                        dqn.load(str(model_path))
                        state_size = 21
                        model_loaded = True
                        loaded_path = str(model_path)
                        print(f"✅ DQN model loaded from: {model_path} (state_size=21 with news sentiment)")
                        print(f"   - Epsilon: {dqn.epsilon:.4f}")
                        print(f"   - Train steps: {dqn.train_step}")
                        break
                    except Exception:
                        # Fallback to old format (state_size=20)
                        dqn = DQN(state_size=20)
                        dqn.load(str(model_path))
                        state_size = 20
                        model_loaded = True
                        loaded_path = str(model_path)
                        print(f"✅ DQN model loaded from: {model_path} (state_size=20, old format)")
                        print(f"   ⚠️  Model doesn't include news sentiment - consider retraining")
                        print(f"   - Epsilon: {dqn.epsilon:.4f}")
                        print(f"   - Train steps: {dqn.train_step}")
                        break
                except Exception as e:
                    print(f"⚠️  Error loading {model_path}: {e}")
                    continue
        
        # If no model found, initialize with new format
        if not model_loaded:
            dqn = DQN(state_size=21)
            state_size = 21
            print(f"⚠️  DQN model not found. Using untrained model (state_size=21 with news sentiment)")
            print(f"   Searched in: {[str(p) for p in model_paths if p]}")
        
        # Initialize state encoder to match DQN state_size
        state_encoder = StateEncoder(state_dim=state_size)
        
        if not model_loaded:
            print(f"⚠️  DQN model not found. Using untrained model (random exploration)")
            print(f"   Searched in: {[str(p) for p in model_paths if p]}")
        
        # Set epsilon to 0 for inference (no exploration in UI)
        dqn.epsilon = 0.0
        print(f"✅ DQN initialized (epsilon=0.0 for inference)")
        print(f"✅ StateEncoder initialized")
        print("="*70 + "\n")
        
        # Initialize output formatter (if LLM enabled)
        output_formatter = None
        if use_llm:
            try:
                output_formatter = LLMOutputFormatterAgent()
            except Exception as e:
                print(f"LLM output formatter initialization failed: {e}")
        
        # Send initial status
        callback_queue.put({
            'type': 'status',
            'message': f'Initializing analysis for {stock_symbol}...',
            'stock_symbol': stock_symbol
        })
        
        # Reset environment
        state = env.reset()
        
        callback_queue.put({
            'type': 'status',
            'message': 'Environment initialized. Starting agent workflow...',
            'stock_symbol': stock_symbol
        })
        
        step_count = 0
        total_reward = 0.0
        agent_calls = []
        
        # Controller Agent will orchestrate all agents
        print("\n" + "="*70)
        print("🎯 AGENT ORCHESTRATION STRATEGY")
        print("="*70)
        print(f"✅ Controller Agent will orchestrate all agents")
        print(f"   - Controller Agent calls each agent sequentially")
        print(f"   - Waits for completion before calling next agent")
        print(f"   - Stops and reports error if any agent fails")
        print(f"   - RL will be used AFTER agents run for optimization")
        print("="*70 + "\n")
        
        # Initialize Controller Agent for orchestration
        controller_agent = ControllerAgent(
            use_q_learning=False,  # Not using RL for orchestration, just sequential
            use_ppo=False,
            cache=cache
        )
        
        # Controller Agent orchestrates all agents
        print("\n" + "="*70)
        print("🎯 CONTROLLER AGENT ORCHESTRATION")
        print("="*70)
        print(f"✅ Controller Agent will orchestrate all agents")
        print(f"   - Calls each agent sequentially")
        print(f"   - Waits for completion before calling next")
        print(f"   - Stops and reports error if any agent fails")
        print("="*70 + "\n")
        
        # Let Controller Agent orchestrate the workflow
        print("\n" + "="*70)
        print("🚀 CALLING CONTROLLER AGENT.ORCHESTRATE()")
        print("="*70)
        print(f"Stock: {stock_symbol}")
        print(f"Environment: {type(env).__name__}")
        print(f"Callback Queue: {callback_queue is not None}")
        print("="*70 + "\n")
        
        orchestration_result = controller_agent.orchestrate(
            stock_symbol=stock_symbol,
            env=env,
            callback_queue=callback_queue
        )
        
        print("\n" + "="*70)
        print("✅ CONTROLLER AGENT.ORCHESTRATE() COMPLETED")
        print("="*70)
        print(f"Result keys: {list(orchestration_result.keys())}")
        print(f"Agent calls: {len(orchestration_result.get('agent_calls', []))}")
        print(f"Total steps: {orchestration_result.get('total_steps', 0)}")
        print("="*70 + "\n")
        
        # Extract results from orchestration
        agent_calls = orchestration_result['agent_calls']
        step_count = orchestration_result['total_steps']
        orchestration_status = orchestration_result['orchestration_status']
        
        # Calculate total reward from agent calls
        total_reward = sum(call.get('reward', 0.0) for call in agent_calls if 'reward' in call)
        
        # Check if orchestration failed
        if not orchestration_status['success']:
            # Error already reported by Controller Agent
            print(f"\n❌ Orchestration failed at step {orchestration_status['steps_completed']}")
            print(f"   Failed Agent: {orchestration_status['failed_agent']}")
            print(f"   Failed Action: {orchestration_status['failed_action']}")
            print(f"   Error: {orchestration_status['error']}")
        
        # After all agents have run, use RL for analysis and optimization
        print(f"\n{'='*70}")
        print("📊 All Agents Executed - Now Using RL for Analysis & Optimization")
        print(f"{'='*70}")
        
        # Collect all data from agents
        state = env._get_state()
        state_vector = state_encoder.encode_continuous(state)
        
        # Initialize RL components for the 10 areas
        print("\n🤖 Initializing RL Components...")
        
        # Model paths
        rl_models_dir = Path("experiments/results/rl_components")
        
        # 1. Portfolio Allocation & Rebalancing
        from rl.portfolio_allocation_rl import PortfolioAllocationRL
        portfolio_rl = PortfolioAllocationRL(state_size=len(state_vector), num_stocks=1, device='cpu')
        # Load trained model if available
        portfolio_model_path = rl_models_dir / "portfolio_allocation_rl.pth"
        if portfolio_model_path.exists():
            try:
                portfolio_rl.load(str(portfolio_model_path))
                print(f"   ✅ Loaded trained portfolio allocation model")
            except:
                print(f"   ⚠️  Could not load portfolio allocation model, using untrained")
        
        # 2. Entry/Exit Timing
        from rl.entry_exit_timing_rl import EntryExitTimingRL
        entry_exit_rl = EntryExitTimingRL(state_size=len(state_vector), device='cpu')
        entry_exit_model_path = rl_models_dir / "entry_exit_timing_rl.pth"
        if entry_exit_model_path.exists():
            try:
                entry_exit_rl.load(str(entry_exit_model_path))
                print(f"   ✅ Loaded trained entry/exit timing model")
            except:
                print(f"   ⚠️  Could not load entry/exit timing model, using untrained")
        
        # 3. Position Sizing
        from rl.position_sizing_rl import PositionSizingRL
        position_sizing_rl = PositionSizingRL(state_size=len(state_vector), device='cpu')
        position_sizing_model_path = rl_models_dir / "position_sizing_rl.pth"
        if position_sizing_model_path.exists():
            try:
                position_sizing_rl.load(str(position_sizing_model_path))
                print(f"   ✅ Loaded trained position sizing model")
            except:
                print(f"   ⚠️  Could not load position sizing model, using untrained")
        
        # 4. Risk Management
        from rl.risk_management_rl import RiskManagementRL
        risk_mgmt_rl = RiskManagementRL(state_size=len(state_vector), device='cpu')
        risk_mgmt_model_path = rl_models_dir / "risk_management_rl.pth"
        if risk_mgmt_model_path.exists():
            try:
                risk_mgmt_rl.load(str(risk_mgmt_model_path))
                print(f"   ✅ Loaded trained risk management model")
            except:
                print(f"   ⚠️  Could not load risk management model, using untrained")
        
        # 5. Feature/Indicator Weighting
        from rl.feature_weighting_rl import FeatureWeightingRL
        feature_weighting_rl = FeatureWeightingRL(state_size=len(state_vector), num_features=10, device='cpu')
        feature_weighting_model_path = rl_models_dir / "feature_weighting_rl.pth"
        if feature_weighting_model_path.exists():
            try:
                feature_weighting_rl.load(str(feature_weighting_model_path))
                print(f"   ✅ Loaded trained feature weighting model")
            except:
                print(f"   ⚠️  Could not load feature weighting model, using untrained")
        
        # 6. Confidence Calibration
        from rl.confidence_calibration_rl import ConfidenceCalibrationRL
        confidence_calibration_rl = ConfidenceCalibrationRL(state_size=len(state_vector), device='cpu')
        confidence_calibration_model_path = rl_models_dir / "confidence_calibration_rl.pth"
        if confidence_calibration_model_path.exists():
            try:
                confidence_calibration_rl.load(str(confidence_calibration_model_path))
                print(f"   ✅ Loaded trained confidence calibration model")
            except:
                print(f"   ⚠️  Could not load confidence calibration model, using untrained")
        
        # 7. Multi-Timeframe Analysis
        from rl.multi_timeframe_rl import MultiTimeframeRL
        multi_timeframe_rl = MultiTimeframeRL(state_size=len(state_vector), device='cpu')
        multi_timeframe_model_path = rl_models_dir / "multi_timeframe_rl.pth"
        if multi_timeframe_model_path.exists():
            try:
                multi_timeframe_rl.load(str(multi_timeframe_model_path))
                print(f"   ✅ Loaded trained multi-timeframe model")
            except:
                print(f"   ⚠️  Could not load multi-timeframe model, using untrained")
        
        # 8. News Sentiment Weighting
        from rl.sentiment_weighting_rl import SentimentWeightingRL
        sentiment_weighting_rl = SentimentWeightingRL(state_size=len(state_vector), device='cpu')
        sentiment_weighting_model_path = rl_models_dir / "sentiment_weighting_rl.pth"
        if sentiment_weighting_model_path.exists():
            try:
                sentiment_weighting_rl.load(str(sentiment_weighting_model_path))
                print(f"   ✅ Loaded trained sentiment weighting model")
            except:
                print(f"   ⚠️  Could not load sentiment weighting model, using untrained")
        
        # 9. Stop Loss & Take Profit
        from rl.stop_loss_take_profit_rl import StopLossTakeProfitRL
        stop_loss_tp_rl = StopLossTakeProfitRL(state_size=len(state_vector), device='cpu')
        stop_loss_tp_model_path = rl_models_dir / "stop_loss_take_profit_rl.pth"
        if stop_loss_tp_model_path.exists():
            try:
                stop_loss_tp_rl.load(str(stop_loss_tp_model_path))
                print(f"   ✅ Loaded trained stop-loss/take-profit model")
            except:
                print(f"   ⚠️  Could not load stop-loss/take-profit model, using untrained")
        
        # 10. Portfolio-Level Optimization
        from rl.portfolio_optimization_rl import PortfolioOptimizationRL
        portfolio_opt_rl = PortfolioOptimizationRL(state_size=len(state_vector), num_stocks=1, device='cpu')
        portfolio_opt_model_path = rl_models_dir / "portfolio_optimization_rl.pth"
        if portfolio_opt_model_path.exists():
            try:
                portfolio_opt_rl.load(str(portfolio_opt_model_path))
                print(f"   ✅ Loaded trained portfolio optimization model")
            except:
                print(f"   ⚠️  Could not load portfolio optimization model, using untrained")
        
        print("✅ All RL components initialized")
        
        # Apply RL optimizations
        print("\n🔧 Applying RL Optimizations...")
        
        # Get current price for calculations
        current_price = env.ta_basic_data.get('current_price', 100.0) if env.ta_basic_data else 100.0
        
        # 1. Portfolio Allocation (if multiple stocks, otherwise skip)
        # portfolio_weights, should_rebalance = portfolio_rl.select_allocation(state_vector)
        # print(f"   📊 Portfolio weights: {portfolio_weights}")
        
        # 2. Entry/Exit Timing
        entry_exit_action_idx, entry_exit_action = entry_exit_rl.select_action(state_vector, training=False)
        print(f"   ⏰ Entry/Exit Timing: {entry_exit_action}")
        
        # 3. Position Sizing
        position_size = position_sizing_rl.select_position_size(state_vector)
        print(f"   📏 Position Size: {position_size:.2%}")
        
        # 4. Risk Management
        risk_action_idx, risk_action = risk_mgmt_rl.select_action(state_vector, training=False)
        print(f"   ⚠️  Risk Management: {risk_action}")
        
        # 5. Feature Weighting
        feature_weights = feature_weighting_rl.get_feature_weights(state_vector)
        print(f"   🎯 Top 3 Feature Weights: {sorted(feature_weights.items(), key=lambda x: x[1], reverse=True)[:3]}")
        
        # 6. Confidence Calibration
        raw_confidence = env.confidence if env.confidence else 0.5
        calibrated_confidence = confidence_calibration_rl.calibrate_confidence(raw_confidence, state_vector)
        print(f"   📊 Confidence: {raw_confidence:.2f} → {calibrated_confidence:.2f} (calibrated)")
        
        # 7. Multi-Timeframe Analysis
        timeframe_weights = multi_timeframe_rl.get_timeframe_weights(state_vector)
        print(f"   📈 Top Timeframe: {max(timeframe_weights.items(), key=lambda x: x[1])[0]}")
        
        # 8. Sentiment Weighting
        sentiment_weights = sentiment_weighting_rl.get_source_weights(state_vector)
        print(f"   📰 Top Sentiment Source: {max(sentiment_weights.items(), key=lambda x: x[1])[0]}")
        
        # 9. Stop Loss & Take Profit
        stop_loss_tp_levels = stop_loss_tp_rl.get_levels(state_vector, current_price)
        print(f"   🛑 Stop Loss: ${stop_loss_tp_levels['stop_loss']:.2f} ({stop_loss_tp_levels['stop_loss_pct']:.2%})")
        print(f"   🎯 Take Profit: ${stop_loss_tp_levels['take_profit']:.2f} ({stop_loss_tp_levels['take_profit_pct']:.2%})")
        
        # 10. Portfolio Optimization (if multiple stocks)
        # portfolio_opt = portfolio_opt_rl.optimize_portfolio(state_vector, portfolio_weights, returns)
        
        print("✅ RL optimizations applied")
        
        # Store RL results for output
        rl_results = {
            'entry_exit_timing': entry_exit_action,
            'position_size': position_size,
            'risk_management': risk_action,
            'feature_weights': feature_weights,
            'calibrated_confidence': calibrated_confidence,
            'timeframe_weights': timeframe_weights,
            'sentiment_weights': sentiment_weights,
            'stop_loss_tp': stop_loss_tp_levels
        }
        
        print(f"\n✅ Collected data from {len(agent_calls)} agent executions")
        
        # Collect raw outputs for formatting
        raw_outputs = {
            'research': {},
            'technical': {},
            'insights': env.insights,
            'recommendation': env.recommendation,
            'confidence': env.confidence,
            'price_levels': env.price_levels,  # Include price levels
            'rl_optimizations': rl_results  # Include RL optimization results
        }
        
        # Collect research data
        if env.news_data:
            raw_outputs['research']['news'] = {
                'num_articles': env.news_data.get('num_articles', 0),
                'sentiment': env.news_data.get('sentiment_score', 0.5),
                'headlines': env.news_data.get('headlines', []),
                'articles': env.news_data.get('articles', [])  # Include full articles with links
            }
        
        # Collect fundamentals data (check if it exists, even if not 'available')
        if env.fundamentals_data:
            if env.fundamentals_data.get('available'):
                raw_outputs['research']['fundamentals'] = {
                    'pe_ratio': env.fundamentals_data.get('pe_ratio'),
                    'revenue_growth': env.fundamentals_data.get('revenue_growth'),
                    'profit_margin': env.fundamentals_data.get('profit_margin')
                }
            else:
                # Still include fundamentals data even if marked as not available
                raw_outputs['research']['fundamentals'] = {
                    'available': False,
                    'message': env.fundamentals_data.get('message', 'Fundamentals data not available for this stock')
                }
        else:
            # No fundamentals data at all
            raw_outputs['research']['fundamentals'] = {
                'available': False,
                'message': 'Fundamentals data was not fetched'
            }
        
        if env.sentiment_data:
            raw_outputs['research']['sentiment'] = {
                'social_sentiment': env.sentiment_data.get('social_sentiment', 0.5),
                'analyst_rating': env.sentiment_data.get('analyst_rating', 'Hold')
            }
        
        # Collect technical data
        if env.ta_basic_data:
            raw_outputs['technical'].update({
                'rsi': env.ta_basic_data.get('rsi'),
                'ma20': env.ta_basic_data.get('ma20'),
                'current_price': env.ta_basic_data.get('current_price')
            })
        else:
            # Add message if TA basic data is missing
            raw_outputs['technical']['message'] = 'Technical analysis data was not calculated'
        
        if env.ta_advanced_data:
            raw_outputs['technical'].update({
                'trend': env.ta_advanced_data.get('trend'),
                'macd_signal': env.ta_advanced_data.get('macd_signal'),
                'ma50': env.ta_advanced_data.get('ma50'),
                'ma200': env.ta_advanced_data.get('ma200')
            })
        
        # Generate Risk Analysis
        try:
            risk_agent = RiskAgent()
            risks = risk_agent.analyze_risks(
                stock_symbol=stock_symbol,
                news_data=env.news_data,
                fundamentals_data=env.fundamentals_data,
                sentiment_data=env.sentiment_data,
                ta_basic_data=env.ta_basic_data,
                ta_advanced_data=env.ta_advanced_data,
                recommendation=env.recommendation,
                confidence=env.confidence
            )
            raw_outputs['risks'] = risks if risks else []
            print(f"\n📊 Risk Analysis: {len(risks) if risks else 0} risks identified")
            if risks:
                for risk in risks:
                    print(f"   [{risk['severity']}] {risk['category']}: {risk['description']}")
            else:
                print("   ⚠️  No risks identified")
        except Exception as e:
            print(f"\n⚠️  Risk Analysis failed: {e}")
            raw_outputs['risks'] = []
            import traceback
            traceback.print_exc()
        
        # Collect price history and indicators for charts
        if hasattr(env, 'price_history') and env.price_history:
            raw_outputs['technical']['price_history'] = env.price_history[-50:]  # Last 50 days
        if hasattr(env, 'ohlcv_data') and env.ohlcv_data is not None and not env.ohlcv_data.empty:
            # Convert OHLCV data to list format for charts
            ohlcv_list = []
            for idx, row in env.ohlcv_data.tail(50).iterrows():
                ohlcv_list.append({
                    'date': str(idx),
                    'open': float(row['Open']),
                    'high': float(row['High']),
                    'low': float(row['Low']),
                    'close': float(row['Close']),
                    'volume': float(row['Volume'])
                })
            raw_outputs['technical']['ohlcv'] = ohlcv_list
        
        # Format output with LLM (if available)
        formatted_report = None
        if output_formatter and output_formatter.groq_client.is_available():
            try:
                formatted_report = output_formatter.format_output(
                    stock_symbol=stock_symbol,
                    research_data=raw_outputs['research'] if raw_outputs['research'] else None,
                    technical_data=raw_outputs['technical'] if raw_outputs['technical'] else None,
                    insights=raw_outputs['insights'],
                    recommendation=raw_outputs['recommendation'],
                    confidence=raw_outputs['confidence'],
                    output_style='professional'
                )
            except Exception as e:
                print(f"LLM formatting failed: {e}")
        
        # ========== ORCHESTRATION SUMMARY ==========
        print("\n" + "="*70)
        print("📊 CONTROLLER AGENT ORCHESTRATION SUMMARY")
        print("="*70)
        print(f"Total Steps: {step_count}")
        print(f"🔄 Agents Orchestrated: {len(agent_calls)}")
        print(f"\n✅ AGENTS CALLED BY CONTROLLER:")
        for call in agent_calls:
            status_icon = "✅" if call.get('status') == 'completed' else "❌" if call.get('status') == 'failed' else "🔄"
            print(f"   {status_icon} Step {call['step']}: {call['action']} ({call['agent']})")
            if call.get('status') == 'failed':
                print(f"      Error: {call.get('error', 'Unknown error')}")
        print(f"\n📈 Performance:")
        print(f"   - Total Reward: {total_reward:.4f}")
        print(f"   - Average Reward per Step: {total_reward/step_count if step_count > 0 else 0:.4f}")
        print(f"   - Orchestration Status: {'✅ SUCCESS' if orchestration_status['success'] else '❌ FAILED'}")
        print("="*70 + "\n")
        
        # Debug: Check what data was collected
        print("\n" + "="*70)
        print("🔍 DATA COLLECTION STATUS")
        print("="*70)
        print(f"News Data: {'✅ Available' if env.news_data else '❌ Not available'}")
        print(f"Fundamentals Data: {'✅ Available' if env.fundamentals_data and env.fundamentals_data.get('available') else '❌ Not available'}")
        print(f"Sentiment Data: {'✅ Available' if env.sentiment_data else '❌ Not available'}")
        print(f"TA Basic Data: {'✅ Available' if env.ta_basic_data else '❌ Not available'}")
        print(f"TA Advanced Data: {'✅ Available' if env.ta_advanced_data else '❌ Not available'}")
        print(f"Insights: {'✅ Available' if env.insights and len(env.insights) > 0 else '❌ Not available'}")
        print(f"Recommendation: {'✅ Available' if env.recommendation else '❌ Not available'}")
        print(f"Risks: {'✅ Available' if raw_outputs.get('risks') and len(raw_outputs.get('risks', [])) > 0 else '❌ Not available'}")
        print("="*70 + "\n")
        
        # Final results
        final_results = {
            'stock_symbol': stock_symbol,
            'total_steps': step_count,
            'total_reward': total_reward,
            'insights': env.insights,
            'recommendation': env.recommendation,
            'confidence': env.confidence,
            'evaluation_results': env.evaluation_results,  # Include evaluation results
            'tools_used': list(set(env.tools_used)),
            'sources_used': list(set(env.sources_used)),
            'agent_calls': agent_calls,
            'formatted_report': formatted_report,
            'raw_outputs': raw_outputs,
            'agent_execution': {
                'execution_mode': 'controller_orchestration',
                'orchestrated_by': 'ControllerAgent',
                'orchestration_status': orchestration_status,
                'total_steps': step_count,
                'total_reward': total_reward,
                'avg_reward_per_step': total_reward / step_count if step_count > 0 else 0,
                'agent_actions': [call['action'] for call in agent_calls],
                'rl_optimizations_applied': True
            }
        }
        
        # Send completion
        callback_queue.put({
            'type': 'complete',
            'results': final_results
        })
        
    except Exception as e:
        callback_queue.put({
            'type': 'error',
            'message': f'Error during analysis: {str(e)}',
            'error': str(e)
        })


def get_agent_name(action: str) -> str:
    """Map action to agent name."""
    agent_map = {
        'FETCH_NEWS': 'ResearchAgent',
        'FETCH_FUNDAMENTALS': 'ResearchAgent',
        'FETCH_SENTIMENT': 'ResearchAgent',
        'FETCH_MACRO': 'ResearchAgent',
        'RUN_TA_BASIC': 'TechnicalAnalysisAgent',
        'RUN_TA_ADVANCED': 'TechnicalAnalysisAgent',
        'GENERATE_INSIGHT': 'InsightAgent',
        'GENERATE_RECOMMENDATION': 'RecommendationAgent',
        'STOP': 'ControllerAgent'
    }
    return agent_map.get(action, 'ControllerAgent')


def get_action_result(action: str, env: StockResearchEnv) -> dict:
    """Get result data for an action."""
    result = {}
    
    if action == 'FETCH_NEWS' and env.news_data:
        result = {
            'type': 'news',
            'headlines': env.news_data.get('headlines', []),
            'sentiment': env.news_data.get('sentiment_score', 0.0),
            'num_articles': env.news_data.get('num_articles', 0),
            'articles': env.news_data.get('articles', [])  # Include full article data with links
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
            # Add price history for charts
            if hasattr(env, 'price_history') and env.price_history:
                result['price_history'] = env.price_history[-50:]
            if hasattr(env, 'ohlcv_data') and env.ohlcv_data is not None and not env.ohlcv_data.empty:
                ohlcv_list = []
                for idx, row in env.ohlcv_data.tail(50).iterrows():
                    ohlcv_list.append({
                        'date': str(idx),
                        'open': float(row['Open']),
                        'high': float(row['High']),
                        'low': float(row['Low']),
                        'close': float(row['Close']),
                        'volume': float(row['Volume'])
                    })
                result['ohlcv'] = ohlcv_list
    elif action == 'RUN_TA_ADVANCED' and env.ta_advanced_data:
            result = {
                'type': 'ta_advanced',
                'trend': env.ta_advanced_data.get('trend', 'sideways'),
                'macd_signal': env.ta_advanced_data.get('macd_signal', 0.0),
                'ma50': env.ta_advanced_data.get('ma50', 0.0),
                'ma200': env.ta_advanced_data.get('ma200', 0.0)
            }
            # Add price history for charts
            if hasattr(env, 'price_history') and env.price_history:
                result['price_history'] = env.price_history[-50:]
            if hasattr(env, 'ohlcv_data') and env.ohlcv_data is not None and not env.ohlcv_data.empty:
                ohlcv_list = []
                for idx, row in env.ohlcv_data.tail(50).iterrows():
                    ohlcv_list.append({
                        'date': str(idx),
                        'open': float(row['Open']),
                        'high': float(row['High']),
                        'low': float(row['Low']),
                        'close': float(row['Close']),
                        'volume': float(row['Volume']),
                        'ma50': env.ta_advanced_data.get('ma50', 0.0),
                        'ma200': env.ta_advanced_data.get('ma200', 0.0)
                    })
                result['ohlcv'] = ohlcv_list
    elif action == 'GENERATE_INSIGHT' and env.insights:
        result = {
            'type': 'insights',
            'insights': env.insights
        }
    elif action == 'EVALUATE_PERFORMANCE' and env.evaluation_results:
        result = {
            'type': 'evaluation',
            'efficiency_score': env.evaluation_results.get('efficiency_score', 0.0),
            'diversity_score': env.evaluation_results.get('diversity_score', 0.0),
            'evaluation_quality': env.evaluation_results.get('evaluation_quality', 0.0),
            'steps_taken': env.evaluation_results.get('steps_taken', 0),
            'tools_used_count': env.evaluation_results.get('tools_used_count', 0),
            'sources_used_count': env.evaluation_results.get('sources_used_count', 0),
            'confidence': env.evaluation_results.get('confidence', 0.0)
        }
    elif action == 'GENERATE_RECOMMENDATION' and env.recommendation:
        result = {
            'type': 'recommendation',
            'recommendation': env.recommendation,
            'confidence': env.confidence
        }
    
    return result


# IMPORTANT: Route order matters! Static files must come BEFORE the catch-all route

# Serve React static files FIRST
@app.route('/static/<path:path>')
def serve_react_static(path):
    """Serve React static files (JS, CSS, images, etc.)"""
    react_build_path = os.path.join(os.path.dirname(__file__), 'frontend', 'build')
    static_dir = os.path.join(react_build_path, 'static')
    
    # Handle nested paths (js/main.xxx.js, css/main.xxx.css)
    if '/' in path:
        subdir, filename = path.split('/', 1)
        subdir_path = os.path.join(static_dir, subdir)
        file_path = os.path.join(subdir_path, filename)
        
        if os.path.exists(subdir_path) and os.path.exists(file_path):
            try:
                return send_from_directory(subdir_path, filename)
            except Exception as e:
                print(f"[STATIC] Error serving {path}: {e}")
                return '', 404
    else:
        # Direct file in static root
        file_path = os.path.join(static_dir, path)
        if os.path.exists(file_path):
            try:
                return send_from_directory(static_dir, path)
            except Exception as e:
                print(f"[STATIC] Error serving {path}: {e}")
                return '', 404
    
    print(f"[STATIC] File not found: /static/{path}")
    return '', 404

# Serve React app root - MUST come after static route but before catch-all
@app.route('/')
def index():
    """Main page - serve React app ONLY - no fallback to templates."""
    react_build_path = os.path.join(os.path.dirname(__file__), 'frontend', 'build')
    index_path = os.path.join(react_build_path, 'index.html')
    
    if not os.path.exists(index_path):
        return jsonify({
            'error': 'React app not built',
            'message': 'Please run: cd frontend && npm run build',
            'instructions': 'The React frontend must be built before running the server.'
        }), 500
    
    # Force serve React - no template fallback
    try:
        return send_from_directory(react_build_path, 'index.html')
    except Exception as e:
        print(f"[ERROR] Failed to serve React app: {e}")
        return jsonify({
            'error': 'Failed to serve React app',
            'message': str(e)
        }), 500


@app.route('/api/analyze', methods=['POST'])
def analyze():
    """Start analysis endpoint."""
    data = request.json
    company_name = data.get('company', '').strip()
    
    if not company_name:
        return jsonify({'error': 'Company name is required'}), 400
    
    # Create a new queue for this analysis
    callback_queue = queue.Queue()
    
    # Start analysis in background thread (with LLM enabled)
    thread = threading.Thread(
        target=run_analysis,
        args=(company_name, callback_queue, True),  # use_llm=True
        daemon=True
    )
    thread.start()
    
    return jsonify({
        'status': 'started',
        'message': f'Analysis started for {company_name}',
        'queue_id': id(callback_queue)
    })


@app.route('/api/stream')
def stream():
    """Server-sent events stream for real-time updates."""
    def generate():
        # Get the latest queue (in production, use proper session management)
        callback_queue = queue.Queue()
        
        # For demo, we'll create a new analysis
        # In production, associate queue with session
        
        while True:
            try:
                # Get update from queue (with timeout)
                update = callback_queue.get(timeout=1.0)
                yield f"data: {json.dumps(update)}\n\n"
                
                if update.get('type') == 'complete' or update.get('type') == 'error':
                    break
            except queue.Empty:
                # Send heartbeat
                yield f"data: {json.dumps({'type': 'heartbeat'})}\n\n"
            except Exception as e:
                yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
                break
    
    return Response(generate(), mimetype='text/event-stream')


@app.route('/api/analyze-stream', methods=['POST'])
def analyze_stream():
    """Start analysis and stream results."""
    data = request.json
    company_name = data.get('company', '').strip()
    force_rl = data.get('force_rl', False)  # Option to force RL usage
    
    if not company_name:
        return jsonify({'error': 'Company name is required'}), 400
    
    # Create queue for this analysis
    callback_queue = queue.Queue()
    
    def generate():
        # Start analysis in background
        # If force_rl=True, pass it to run_analysis (will need to modify function signature)
        use_llm = not force_rl  # Disable LLM if forcing RL
        thread = threading.Thread(
            target=run_analysis,
            args=(company_name, callback_queue, use_llm),  # use_llm based on force_rl
            daemon=True
        )
        thread.start()
        
        # Stream updates
        while True:
            try:
                update = callback_queue.get(timeout=30.0)
                yield f"data: {json.dumps(update)}\n\n"
                
                if update.get('type') == 'complete' or update.get('type') == 'error':
                    break
            except queue.Empty:
                yield f"data: {json.dumps({'type': 'timeout', 'message': 'Analysis timeout'})}\n\n"
                break
            except Exception as e:
                yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
                break
    
    return Response(generate(), mimetype='text/event-stream')


@app.route('/<path:path>')
def serve_react_app(path):
    # Don't handle static or api routes here
    if path.startswith('static') or path.startswith('api'):
        return '', 404
    
    react_build_path = os.path.join(os.path.dirname(__file__), 'frontend', 'build')
    if os.path.exists(react_build_path):
        try:
            file_path = os.path.join(react_build_path, path)
            if os.path.exists(file_path) and os.path.isfile(file_path):
                return send_from_directory(react_build_path, path)
            else:
                # SPA fallback - serve index.html for all routes
                return send_from_directory(react_build_path, 'index.html')
        except Exception as e:
            print(f"Error serving {path}: {e}")
            return send_from_directory(react_build_path, 'index.html')
    return '', 404

if __name__ == '__main__':
    print("=" * 70)
    print("ORION-AI - Optimized Research & Investment Orchestration Network - Web UI")
    print("=" * 70)
    print("\nStarting web server...")
    print("React UI: http://localhost:5001")
    print("API: http://localhost:5001/api/analyze-stream")
    print("\nPress Ctrl+C to stop the server\n")
    
    app.run(debug=False, host='0.0.0.0', port=5001, threaded=True, use_reloader=False)

