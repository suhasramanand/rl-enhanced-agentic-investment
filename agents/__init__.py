"""
Agent Modules
"""

from .controller_agent import ControllerAgent
from .research_agent import ResearchAgent
from .ta_agent import TechnicalAnalysisAgent
from .insight_agent import InsightAgent
from .recommendation_agent import RecommendationAgent
from .evaluator_agent import EvaluatorAgent

__all__ = [
    'ControllerAgent',
    'ResearchAgent',
    'TechnicalAnalysisAgent',
    'InsightAgent',
    'RecommendationAgent',
    'EvaluatorAgent'
]

