# ORION-AI — **O**ptimized **R**esearch & **I**nvestment **O**rchestration **N**etwork

A comprehensive reinforcement learning system for automated stock research, technical analysis, and investment recommendations using DQN and PPO algorithms with portfolio-level decision making and outcome-based learning.

## Features

- **Two RL Approaches**: DQN (Value-Based) + PPO (Policy Gradient)
- **Portfolio-Level Decision Making**: Optimizes stock selection and capital allocation
- **Outcome-Based Learning**: Learns from actual stock returns
- **Any Stock Symbol**: Works with any ticker symbol, pulls live data automatically
- **Production API**: FastAPI REST API for integration
- **Interactive Dashboard**: Streamlit web interface
- **Model Evaluation**: Performance metrics and hyperparameter tuning tools
- **Local Training**: Train models on your machine

## Project Structure

```
rl-enhanced-agentic-investment/
├── api/                        # Production API
│   └── main.py                 # FastAPI application
├── env/                        # RL Environments
│   ├── stock_research_env.py   # Stock-level environment
│   └── portfolio_env.py        # Portfolio-level environment
├── rl/                         # RL Algorithms
│   ├── dqn.py                  # DQN implementation
│   ├── portfolio_dqn.py        # Portfolio DQN
│   ├── ppo.py                  # PPO implementation
│   └── rollout_buffer.py       # PPO rollout buffer
├── utils/                      # Utilities
│   ├── state_encoder.py        # State encoding
│   ├── portfolio_state_encoder.py
│   ├── outcome_tracker.py      # Outcome-based learning
│   └── data_cache.py           # Data caching
├── tools/                      # CrewAI Tools
│   ├── dqn_crewai_tool.py      # DQN tool wrapper
│   └── portfolio_dqn_crewai_tool.py
├── experiments/                # Training results
├── RL_Stock_Research_DQN.ipynb # Colab notebook (self-contained)
├── train_portfolio_dqn.py      # Portfolio training script
├── train_local.py              # Local training script
├── dashboard.py                # Streamlit dashboard
├── model_improvements.py       # Evaluation & tuning
├── test_local_agents.py        # Local testing
├── requirements.txt            # Dependencies
├── requirements_api.txt         # API dependencies
├── Dockerfile                  # Docker config
└── README.md                   # This file
```

## Quick Start

### 1. Installation

```bash
# Install dependencies
pip install -r requirements.txt

# For API and dashboard
pip install -r requirements_api.txt
```

### 2. Train Models Locally

```bash
# Train stock-level DQN
python train_local.py --type stock --symbol NVDA --episodes 1000

# Train portfolio DQN
python train_local.py --type portfolio --watchlist "NVDA,AAPL,TSLA,MSFT" --episodes 500
```

### 3. Test Models

```bash
# Test locally
python test_local_agents.py
```

### 4. Run API

```bash
# Start FastAPI server
uvicorn api.main:app --reload

# API available at: http://localhost:8000
# Docs at: http://localhost:8000/docs
```

### 5. Run Dashboard

```bash
# Start Streamlit dashboard
streamlit run dashboard.py

# Dashboard at: http://localhost:8501
```

## Training

### Stock-Level DQN

Trains a DQN to optimize research actions for a single stock:

```bash
python train_local.py \
    --type stock \
    --symbol NVDA \
    --episodes 1000 \
    --learning-rate 0.001 \
    --save-path models/dqn_nvda.pth
```

### Portfolio-Level DQN

Trains a DQN to optimize portfolio decisions (stock selection, allocation):

```bash
python train_local.py \
    --type portfolio \
    --watchlist "NVDA,AAPL,TSLA,MSFT,GOOGL" \
    --episodes 500 \
    --future-days 30 \
    --save-path models/portfolio_dqn.pth
```

### Using Colab Notebook

The `RL_Stock_Research_DQN.ipynb` notebook is self-contained and can be run in Google Colab:
- All classes defined inline (no external imports needed)
- Works with any stock symbol
- Pulls live data automatically
- Includes both DQN and PPO implementations

## API Usage

### Endpoints

- `GET /health` - Health check
- `POST /analyze` - Analyze single stock
- `GET /analyze/{symbol}` - Quick analysis
- `POST /portfolio` - Optimize portfolio
- `POST /analyze/batch` - Batch analysis

### Example

```python
import requests

# Analyze stock
response = requests.post(
    "http://localhost:8000/analyze",
    json={"symbol": "NVDA"}
)
result = response.json()
print(f"Recommendation: {result['recommendation']}")
print(f"Confidence: {result['confidence']:.1%}")
```

## Dashboard Features

- **Stock Analysis**: Enter any symbol, get recommendations
- **Portfolio Optimization**: Optimize multi-stock portfolios
- **Batch Analysis**: Analyze multiple stocks at once
- **Performance Metrics**: Track model performance
- **Visualizations**: Interactive charts with Plotly

## Assignment Requirements

**Two RL Approaches**:
- DQN (Value-Based Learning)
- PPO (Policy Gradient Methods)

**Integration with Agentic Systems**:
- Agent Orchestration (Portfolio DQN)
- Research/Analysis Agents (Stock DQN + PPO)

**Portfolio-Level Decision Making**:
- Stock selection
- Capital allocation
- Portfolio rebalancing

**Outcome-Based Learning**:
- Learns from actual stock returns
- Tracks recommendations vs outcomes
- Uses real future prices for rewards

## Model Evaluation

```python
from model_improvements import ModelEvaluator

# Evaluate model
evaluator = ModelEvaluator("models/dqn_model.pth")
results = evaluator.evaluate_multiple_stocks(['NVDA', 'AAPL', 'TSLA'])
metrics = evaluator.calculate_metrics(results)
```

## Docker Deployment

```bash
# Build and run
docker build -t dqn-stock-api .
docker run -p 8000:8000 dqn-stock-api

# Or use deploy script
./deploy.sh
```

## Key Features

- **Any Stock Symbol**: Change symbol and train/analyze any stock
- **Live Data**: Automatically pulls from yfinance
- **Rate Limiting**: Built-in delays to prevent API limits
- **Self-Contained**: Notebook works standalone in Colab
- **Production Ready**: API and dashboard for deployment
- **Local Training**: Train models on your machine

## Development

### Project Components

- **Environments**: `env/stock_research_env.py`, `env/portfolio_env.py`
- **RL Algorithms**: `rl/dqn.py`, `rl/portfolio_dqn.py`, `rl/ppo.py`
- **Training**: `train_local.py`, `train_portfolio_dqn.py`
- **API**: `api/main.py`
- **Dashboard**: `dashboard.py`
- **Tools**: `tools/dqn_crewai_tool.py`, `tools/portfolio_dqn_crewai_tool.py`

### Testing

```bash
# Test models
python test_local_agents.py

# Test with CrewAI
python test_with_crewai_agents.py
```

## Model Files

Models are saved to:
- `models/dqn_{symbol}_{episodes}ep.pth` - Stock DQN
- `models/portfolio_dqn_{episodes}ep.pth` - Portfolio DQN
- `experiments/results/` - Training checkpoints

Models can be loaded from:
- Project directory
- `~/Downloads/` folder
- Environment variable `DQN_MODEL_PATH`

## Deployment

### Local
```bash
uvicorn api.main:app --reload
streamlit run dashboard.py
```

### Docker
```bash
./deploy.sh
```

### Cloud
- Deploy Docker image to AWS/GCP/Heroku
- Set environment variables
- Models auto-load from Downloads or specified path

## Usage Examples

### Train on Any Stock
```bash
python train_local.py --type stock --symbol AAPL --episodes 1000
python train_local.py --type stock --symbol TSLA --episodes 1000
```

### Portfolio with Custom Watchlist
```bash
python train_local.py --type portfolio --watchlist "NVDA,AMD,INTC,MU" --episodes 500
```

### Analyze via API
```bash
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"symbol": "NVDA"}'
```

## Important Notes

- **Rate Limiting**: Built-in 0.5s delay between API calls
- **Data Caching**: Data cached locally for faster training
- **Model Paths**: Auto-detects models in Downloads folder
- **Live Data**: All data pulled from yfinance in real-time
- **GPU Support**: Automatically uses GPU if available

## Integration

### With CrewAI
```python
from tools.dqn_crewai_tool import dqn_stock_research
from crewai import Agent, Task, Crew

agent = Agent(
    role="Stock Analyst",
    tools=[dqn_stock_research]
)
```

### With Your Code
```python
from tools.dqn_crewai_tool import DQNStockResearchTool

tool = DQNStockResearchTool()
result = tool.analyze_stock("NVDA")
```

## License

Educational project for course assignment.

---

**Version**: 1.0
**Last Updated**: 2024
