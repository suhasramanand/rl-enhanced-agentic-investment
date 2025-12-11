"""
Streamlit Dashboard for DQN Stock Analysis
Interactive UI for analyzing stocks and viewing portfolio recommendations
"""

import streamlit as st
import sys
from pathlib import Path
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import requests

# Add project to path
project_root = Path(__file__).parent
if (project_root / "rl-enhanced-agentic-investment").exists():
    sys.path.insert(0, str(project_root / "rl-enhanced-agentic-investment"))
elif (project_root.parent / "rl-enhanced-agentic-investment").exists():
    sys.path.insert(0, str(project_root.parent / "rl-enhanced-agentic-investment"))

try:
    from tools.dqn_crewai_tool import DQNStockResearchTool
    from tools.portfolio_dqn_crewai_tool import PortfolioDQNTool
    TOOLS_AVAILABLE = True
except ImportError:
    TOOLS_AVAILABLE = False
    st.error("⚠️ Tools not available. Please check project setup.")

# Page configuration
st.set_page_config(
    page_title="DQN Stock Analysis Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .recommendation-buy {
        color: #00cc00;
        font-weight: bold;
        font-size: 1.5rem;
    }
    .recommendation-sell {
        color: #ff0000;
        font-weight: bold;
        font-size: 1.5rem;
    }
    .recommendation-hold {
        color: #ffaa00;
        font-weight: bold;
        font-size: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'dqn_tool' not in st.session_state and TOOLS_AVAILABLE:
    try:
        st.session_state.dqn_tool = DQNStockResearchTool()
        st.session_state.portfolio_tool = PortfolioDQNTool()
    except Exception as e:
        st.error(f"Error initializing tools: {e}")
        st.session_state.dqn_tool = None
        st.session_state.portfolio_tool = None

# Header
st.markdown('<div class="main-header">📈 DQN Stock Analysis Dashboard</div>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    
    page = st.radio(
        "Select Page",
        ["Stock Analysis", "Portfolio Optimization", "Batch Analysis", "Performance Metrics"]
    )
    
    st.divider()
    
    # API Configuration (optional)
    use_api = st.checkbox("Use API (if available)", value=False)
    api_url = st.text_input("API URL", value="http://localhost:8000", disabled=not use_api)
    
    st.divider()
    
    st.info("""
    **About This Dashboard:**
    - Uses Reinforcement Learning (DQN) for stock analysis
    - Provides Buy/Hold/Sell recommendations
    - Shows confidence levels and insights
    """)

# Stock Analysis Page
if page == "Stock Analysis":
    st.header("🔍 Single Stock Analysis")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        symbol = st.text_input("Stock Symbol", value="NVDA", placeholder="Enter stock symbol (e.g., NVDA, AAPL)")
    
    with col2:
        analyze_button = st.button("Analyze Stock", type="primary", use_container_width=True)
    
    if analyze_button and symbol:
        if st.session_state.get('dqn_tool') is None:
            st.error("❌ DQN tool not initialized. Please check model files.")
        else:
            with st.spinner(f"Analyzing {symbol.upper()}..."):
                try:
                    if use_api:
                        # Use API
                        response = requests.post(
                            f"{api_url}/analyze",
                            json={"symbol": symbol.upper()}
                        )
                        if response.status_code == 200:
                            result = response.json()
                        else:
                            st.error(f"API Error: {response.text}")
                            result = None
                    else:
                        # Use direct tool
                        result = st.session_state.dqn_tool.analyze_stock(symbol.upper())
                    
                    if result:
                        # Display results
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            rec = result.get('recommendation', 'Hold')
                            if rec == 'Buy':
                                st.markdown(f'<div class="recommendation-buy">🟢 {rec}</div>', unsafe_allow_html=True)
                            elif rec == 'Sell':
                                st.markdown(f'<div class="recommendation-sell">🔴 {rec}</div>', unsafe_allow_html=True)
                            else:
                                st.markdown(f'<div class="recommendation-hold">🟡 {rec}</div>', unsafe_allow_html=True)
                        
                        with col2:
                            confidence = result.get('confidence', 0.5)
                            st.metric("Confidence", f"{confidence:.1%}")
                        
                        with col3:
                            st.metric("Total Reward", f"{result.get('total_reward', 0):.2f}")
                        
                        with col4:
                            st.metric("Steps Taken", len(result.get('actions_taken', [])))
                        
                        # Insights
                        st.subheader("💡 Insights")
                        insights = result.get('insights', [])
                        if insights:
                            for insight in insights:
                                st.info(f"• {insight}")
                        else:
                            st.info("No insights generated")
                        
                        # Actions taken
                        with st.expander("📋 Actions Taken"):
                            actions = result.get('actions_taken', [])
                            st.write("Agent decision sequence:")
                            for i, action in enumerate(actions, 1):
                                st.write(f"{i}. {action}")
                        
                        # Technical data
                        if result.get('ta_basic'):
                            st.subheader("📊 Technical Analysis")
                            ta_data = result['ta_basic']
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("RSI", f"{ta_data.get('rsi', 0):.2f}")
                            with col2:
                                st.metric("20-day MA", f"${ta_data.get('ma20', 0):.2f}")
                            with col3:
                                st.metric("Current Price", f"${ta_data.get('current_price', 0):.2f}")
                
                except Exception as e:
                    st.error(f"Error analyzing stock: {e}")
                    import traceback
                    st.code(traceback.format_exc())

# Portfolio Optimization Page
elif page == "Portfolio Optimization":
    st.header("💼 Portfolio Optimization")
    
    st.write("Optimize a portfolio using Portfolio DQN model")
    
    watchlist_input = st.text_input(
        "Watchlist (comma-separated)",
        value="NVDA,AAPL,TSLA,MSFT,GOOGL",
        help="Enter stock symbols separated by commas"
    )
    
    initial_capital = st.number_input("Initial Capital ($)", value=100000.0, min_value=1000.0)
    
    if st.button("Optimize Portfolio", type="primary"):
        if st.session_state.get('portfolio_tool') is None:
            st.error("❌ Portfolio tool not initialized")
        else:
            watchlist = [s.strip().upper() for s in watchlist_input.split(',')]
            
            with st.spinner("Optimizing portfolio..."):
                try:
                    result = st.session_state.portfolio_tool.optimize_portfolio(
                        watchlist=watchlist,
                        initial_capital=initial_capital
                    )
                    
                    # Display portfolio
                    st.subheader("📊 Optimized Portfolio")
                    
                    allocations = result.get('allocations', {})
                    if allocations:
                        # Create allocation chart
                        fig = go.Figure(data=[go.Pie(
                            labels=list(allocations.keys()),
                            values=list(allocations.values()),
                            hole=0.3
                        )])
                        fig.update_layout(title="Portfolio Allocation")
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Allocation table
                        df = pd.DataFrame([
                            {"Stock": stock, "Allocation": f"{alloc:.1%}", "Amount": f"${alloc * initial_capital:,.2f}"}
                            for stock, alloc in allocations.items() if alloc > 0
                        ])
                        st.dataframe(df, use_container_width=True)
                    
                    # Recommendations
                    recommendations = result.get('recommendations', [])
                    if recommendations:
                        st.subheader("💡 Stock Recommendations")
                        for rec in recommendations:
                            st.write(f"**{rec.get('stock', 'N/A')}**: {rec.get('recommendation', 'N/A')} "
                                   f"(Confidence: {rec.get('confidence', 0):.1%})")
                
                except Exception as e:
                    st.error(f"Error optimizing portfolio: {e}")

# Batch Analysis Page
elif page == "Batch Analysis":
    st.header("📊 Batch Stock Analysis")
    
    symbols_input = st.text_area(
        "Stock Symbols (one per line or comma-separated)",
        value="NVDA\nAAPL\nTSLA\nMSFT\nGOOGL",
        height=150
    )
    
    if st.button("Analyze All", type="primary"):
        # Parse symbols
        symbols = []
        for line in symbols_input.split('\n'):
            symbols.extend([s.strip().upper() for s in line.split(',') if s.strip()])
        
        if symbols and st.session_state.get('dqn_tool'):
            results = []
            progress_bar = st.progress(0)
            
            for i, symbol in enumerate(symbols):
                try:
                    result = st.session_state.dqn_tool.analyze_stock(symbol)
                    results.append({
                        'Symbol': symbol,
                        'Recommendation': result.get('recommendation', 'N/A'),
                        'Confidence': f"{result.get('confidence', 0):.1%}",
                        'Reward': f"{result.get('total_reward', 0):.2f}"
                    })
                except Exception as e:
                    results.append({
                        'Symbol': symbol,
                        'Recommendation': 'Error',
                        'Confidence': 'N/A',
                        'Reward': str(e)
                    })
                
                progress_bar.progress((i + 1) / len(symbols))
            
            # Display results table
            df = pd.DataFrame(results)
            st.dataframe(df, use_container_width=True)
            
            # Summary statistics
            col1, col2, col3 = st.columns(3)
            with col1:
                buy_count = len([r for r in results if r['Recommendation'] == 'Buy'])
                st.metric("Buy Recommendations", buy_count)
            with col2:
                sell_count = len([r for r in results if r['Recommendation'] == 'Sell'])
                st.metric("Sell Recommendations", sell_count)
            with col3:
                hold_count = len([r for r in results if r['Recommendation'] == 'Hold'])
                st.metric("Hold Recommendations", hold_count)

# Performance Metrics Page
elif page == "Performance Metrics":
    st.header("📈 Performance Metrics")
    
    st.info("Performance tracking and model evaluation metrics will be displayed here.")
    
    # Placeholder for future metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Analyses", "0")
    with col2:
        st.metric("Avg Confidence", "0%")
    with col3:
        st.metric("Success Rate", "0%")
    with col4:
        st.metric("Avg Reward", "0.00")

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>DQN Stock Analysis Dashboard | Powered by Reinforcement Learning</p>
    <p>Model trained on historical stock data | Use at your own risk</p>
</div>
""", unsafe_allow_html=True)

