"""
Production API for DQN Stock Analysis
FastAPI-based REST API with async support
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import sys
from pathlib import Path
import logging
from datetime import datetime

# Add project to path
project_root = Path(__file__).parent.parent
if (project_root / "rl-enhanced-agentic-investment").exists():
    sys.path.insert(0, str(project_root / "rl-enhanced-agentic-investment"))
elif (project_root.parent / "rl-enhanced-agentic-investment").exists():
    sys.path.insert(0, str(project_root.parent / "rl-enhanced-agentic-investment"))

try:
    from tools.dqn_crewai_tool import DQNStockResearchTool
    from tools.portfolio_dqn_crewai_tool import PortfolioDQNTool
except ImportError as e:
    print(f"⚠️  Warning: {e}")
    DQNStockResearchTool = None
    PortfolioDQNTool = None

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="DQN Stock Analysis API",
    description="Reinforcement Learning-based Stock Analysis API",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global tool instances
dqn_tool: Optional[DQNStockResearchTool] = None
portfolio_tool: Optional[PortfolioDQNTool] = None

# Request/Response models
class StockAnalysisRequest(BaseModel):
    symbol: str
    current_state: Optional[str] = None

class PortfolioRequest(BaseModel):
    watchlist: List[str]
    initial_capital: Optional[float] = 100000.0

class AnalysisResponse(BaseModel):
    success: bool
    symbol: str
    recommendation: str
    confidence: float
    insights: List[str]
    actions_taken: List[str]
    total_reward: float
    timestamp: str

class PortfolioResponse(BaseModel):
    success: bool
    portfolio_id: str
    allocations: Dict[str, float]
    selected_stocks: List[str]
    num_stocks: int
    recommendations: List[Dict[str, Any]]
    timestamp: str

# Initialize tools on startup
@app.on_event("startup")
async def startup_event():
    """Initialize tools on API startup."""
    global dqn_tool, portfolio_tool
    
    logger.info("Initializing DQN tools...")
    
    try:
        if DQNStockResearchTool:
            dqn_tool = DQNStockResearchTool()
            logger.info("✅ DQN Stock Research Tool initialized")
    except Exception as e:
        logger.error(f"❌ Error initializing DQN tool: {e}")
    
    try:
        if PortfolioDQNTool:
            portfolio_tool = PortfolioDQNTool()
            logger.info("✅ Portfolio DQN Tool initialized")
    except Exception as e:
        logger.warning(f"⚠️  Portfolio tool not available: {e}")

# Health check
@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "DQN Stock Analysis API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "/health": "Health check",
            "/analyze": "Analyze single stock",
            "/portfolio": "Optimize portfolio",
            "/docs": "API documentation"
        }
    }

@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "dqn_tool_loaded": dqn_tool is not None,
        "portfolio_tool_loaded": portfolio_tool is not None,
        "timestamp": datetime.now().isoformat()
    }

# Stock analysis endpoint
@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_stock(request: StockAnalysisRequest):
    """
    Analyze a single stock using DQN model.
    
    Args:
        request: Stock analysis request with symbol
        
    Returns:
        Analysis results with recommendation and confidence
    """
    if dqn_tool is None:
        raise HTTPException(status_code=503, detail="DQN tool not initialized")
    
    try:
        logger.info(f"Analyzing stock: {request.symbol}")
        result = dqn_tool.analyze_stock(request.symbol.upper())
        
        return AnalysisResponse(
            success=True,
            symbol=request.symbol.upper(),
            recommendation=result.get('recommendation', 'Hold'),
            confidence=result.get('confidence', 0.5),
            insights=result.get('insights', []),
            actions_taken=result.get('actions_taken', []),
            total_reward=result.get('total_reward', 0.0),
            timestamp=datetime.now().isoformat()
        )
    except Exception as e:
        logger.error(f"Error analyzing {request.symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/analyze/{symbol}")
async def analyze_stock_get(symbol: str):
    """Quick analysis endpoint (GET method)."""
    request = StockAnalysisRequest(symbol=symbol)
    return await analyze_stock(request)

# Portfolio optimization endpoint
@app.post("/portfolio", response_model=PortfolioResponse)
async def optimize_portfolio(request: PortfolioRequest):
    """
    Optimize portfolio using Portfolio DQN.
    
    Args:
        request: Portfolio request with watchlist
        
    Returns:
        Portfolio optimization results
    """
    if portfolio_tool is None:
        raise HTTPException(status_code=503, detail="Portfolio tool not initialized")
    
    try:
        logger.info(f"Optimizing portfolio: {request.watchlist}")
        result = portfolio_tool.optimize_portfolio(
            watchlist=request.watchlist,
            initial_capital=request.initial_capital
        )
        
        return PortfolioResponse(
            success=True,
            portfolio_id=result.get('portfolio_id', 'unknown'),
            allocations=result.get('allocations', {}),
            selected_stocks=result.get('selected_stocks', []),
            num_stocks=result.get('num_stocks', 0),
            recommendations=result.get('recommendations', []),
            timestamp=datetime.now().isoformat()
        )
    except Exception as e:
        logger.error(f"Error optimizing portfolio: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Batch analysis endpoint
@app.post("/analyze/batch")
async def analyze_batch(symbols: List[str]):
    """
    Analyze multiple stocks in batch.
    
    Args:
        symbols: List of stock symbols
        
    Returns:
        List of analysis results
    """
    if dqn_tool is None:
        raise HTTPException(status_code=503, detail="DQN tool not initialized")
    
    results = []
    for symbol in symbols:
        try:
            result = dqn_tool.analyze_stock(symbol.upper())
            results.append({
                "symbol": symbol.upper(),
                "recommendation": result.get('recommendation', 'Hold'),
                "confidence": result.get('confidence', 0.5),
                "success": True
            })
        except Exception as e:
            results.append({
                "symbol": symbol.upper(),
                "error": str(e),
                "success": False
            })
    
    return {"results": results, "timestamp": datetime.now().isoformat()}

# Recommendation summary endpoint
@app.get("/recommendation/{symbol}")
async def get_recommendation_summary(symbol: str):
    """Get formatted recommendation summary."""
    if dqn_tool is None:
        raise HTTPException(status_code=503, detail="DQN tool not initialized")
    
    try:
        summary = dqn_tool.get_recommendation_summary(symbol.upper())
        return {
            "symbol": symbol.upper(),
            "summary": summary,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

