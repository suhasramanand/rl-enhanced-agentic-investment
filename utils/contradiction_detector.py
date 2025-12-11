"""
Contradiction Detection and Uncertainty Assessment

Detects contradictions between agent outputs and assesses uncertainty
to determine if more agents need to be called.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np


class ContradictionDetector:
    """Detect contradictions between agent outputs."""
    
    def __init__(self):
        """Initialize contradiction detector."""
        pass
    
    def detect_contradictions(self, agent_outputs: Dict) -> Dict:
        """
        Detect contradictions between different agent outputs.
        
        Args:
            agent_outputs: Dictionary with outputs from different agents
                {
                    'news': {'sentiment': 0.7, ...},
                    'fundamentals': {'pe_ratio': 30, 'revenue_growth': 0.1, ...},
                    'technical': {'rsi': 70, 'trend': 'uptrend', ...},
                    'insights': [...],
                    'recommendation': 'Buy',
                    'confidence': 0.65
                }
        
        Returns:
            Dictionary with contradiction analysis
        """
        contradictions = []
        uncertainty_score = 0.0
        
        # Check news vs technical sentiment
        if agent_outputs.get('news') and agent_outputs.get('technical'):
            news_sentiment = agent_outputs['news'].get('sentiment', 0.5)
            technical_trend = agent_outputs['technical'].get('trend', 'sideways')
            rsi = agent_outputs['technical'].get('rsi', 50)
            
            # News positive but technical bearish
            if news_sentiment > 0.6 and technical_trend in ['downtrend', 'bearish']:
                contradictions.append({
                    'type': 'news_technical',
                    'severity': 'high',
                    'description': f'Positive news sentiment ({news_sentiment:.2f}) contradicts bearish technical trend ({technical_trend})',
                    'agents': ['ResearchAgent', 'TechnicalAnalysisAgent']
                })
                uncertainty_score += 0.3
            
            # News negative but technical bullish
            elif news_sentiment < 0.4 and technical_trend in ['uptrend', 'bullish']:
                contradictions.append({
                    'type': 'news_technical',
                    'severity': 'high',
                    'description': f'Negative news sentiment ({news_sentiment:.2f}) contradicts bullish technical trend ({technical_trend})',
                    'agents': ['ResearchAgent', 'TechnicalAnalysisAgent']
                })
                uncertainty_score += 0.3
            
            # RSI overbought but news negative
            if rsi > 70 and news_sentiment < 0.4:
                contradictions.append({
                    'type': 'rsi_news',
                    'severity': 'medium',
                    'description': f'Overbought RSI ({rsi:.1f}) with negative news sentiment ({news_sentiment:.2f})',
                    'agents': ['TechnicalAnalysisAgent', 'ResearchAgent']
                })
                uncertainty_score += 0.2
        
        # Check fundamentals vs technical
        if agent_outputs.get('fundamentals') and agent_outputs.get('technical'):
            pe_ratio = agent_outputs['fundamentals'].get('pe_ratio', 0)
            revenue_growth = agent_outputs['fundamentals'].get('revenue_growth', 0)
            technical_trend = agent_outputs['technical'].get('trend', 'sideways')
            
            # Strong fundamentals but bearish technical
            if pe_ratio > 0 and revenue_growth > 0.1 and technical_trend in ['downtrend', 'bearish']:
                contradictions.append({
                    'type': 'fundamentals_technical',
                    'severity': 'medium',
                    'description': f'Strong fundamentals (PE: {pe_ratio:.1f}, Growth: {revenue_growth*100:.1f}%) contradict bearish technical trend',
                    'agents': ['ResearchAgent', 'TechnicalAnalysisAgent']
                })
                uncertainty_score += 0.25
        
        # Check recommendation vs confidence
        recommendation = agent_outputs.get('recommendation')
        confidence = agent_outputs.get('confidence', 0.5)
        
        if recommendation and confidence < 0.5:
            contradictions.append({
                'type': 'low_confidence',
                'severity': 'high',
                'description': f'Recommendation ({recommendation}) has low confidence ({confidence*100:.1f}%)',
                'agents': ['RecommendationAgent']
            })
            uncertainty_score += 0.4
        
        # Check if insights contradict recommendation
        if agent_outputs.get('insights') and recommendation:
            insights = agent_outputs['insights']
            insight_sentiment = self._analyze_insight_sentiment(insights)
            
            if recommendation == 'Buy' and insight_sentiment < -0.2:
                contradictions.append({
                    'type': 'insight_recommendation',
                    'severity': 'medium',
                    'description': f'Buy recommendation contradicts negative insights sentiment',
                    'agents': ['InsightAgent', 'RecommendationAgent']
                })
                uncertainty_score += 0.2
            elif recommendation == 'Sell' and insight_sentiment > 0.2:
                contradictions.append({
                    'type': 'insight_recommendation',
                    'severity': 'medium',
                    'description': f'Sell recommendation contradicts positive insights sentiment',
                    'agents': ['InsightAgent', 'RecommendationAgent']
                })
                uncertainty_score += 0.2
        
        return {
            'has_contradictions': len(contradictions) > 0,
            'contradictions': contradictions,
            'uncertainty_score': min(1.0, uncertainty_score),
            'needs_more_agents': uncertainty_score > 0.3 or len(contradictions) > 0
        }
    
    def _analyze_insight_sentiment(self, insights: List[str]) -> float:
        """Analyze sentiment from insights text."""
        if not insights:
            return 0.0
        
        positive_words = ['positive', 'growth', 'bullish', 'strong', 'increase', 'gain', 'profit']
        negative_words = ['negative', 'decline', 'bearish', 'weak', 'decrease', 'loss', 'risk']
        
        text = ' '.join(insights).lower()
        positive_count = sum(1 for word in positive_words if word in text)
        negative_count = sum(1 for word in negative_words if word in text)
        
        if positive_count + negative_count == 0:
            return 0.0
        
        return (positive_count - negative_count) / max(positive_count + negative_count, 1)
    
    def assess_uncertainty(self, agent_outputs: Dict, q_values: Optional[np.ndarray] = None) -> Dict:
        """
        Assess overall uncertainty in the system.
        
        Args:
            agent_outputs: Current agent outputs
            q_values: Q-values from DQN (optional)
        
        Returns:
            Uncertainty assessment
        """
        uncertainty_factors = []
        total_uncertainty = 0.0
        
        # Low confidence
        confidence = agent_outputs.get('confidence', 0.5)
        if confidence < 0.6:
            uncertainty_factors.append({
                'factor': 'low_confidence',
                'value': 1.0 - confidence,
                'description': f'Low confidence: {confidence*100:.1f}%'
            })
            total_uncertainty += (1.0 - confidence) * 0.4
        
        # Missing data sources
        required_sources = ['news', 'technical']
        missing_sources = [src for src in required_sources if not agent_outputs.get(src)]
        if missing_sources:
            uncertainty_factors.append({
                'factor': 'missing_data',
                'value': len(missing_sources) / len(required_sources),
                'description': f'Missing data sources: {", ".join(missing_sources)}'
            })
            total_uncertainty += (len(missing_sources) / len(required_sources)) * 0.3
        
        # Q-value uncertainty (if available)
        if q_values is not None:
            q_std = np.std(q_values)
            q_max = np.max(q_values)
            q_second_max = np.partition(q_values, -2)[-2]
            q_gap = q_max - q_second_max
            
            # Small gap between best actions indicates uncertainty
            if q_gap < 0.1:
                uncertainty_factors.append({
                    'factor': 'q_value_uncertainty',
                    'value': 1.0 - (q_gap * 10),
                    'description': f'DQN uncertain: small gap ({q_gap:.3f}) between actions'
                })
                total_uncertainty += (1.0 - (q_gap * 10)) * 0.3
        
        # Contradictions
        contradiction_analysis = self.detect_contradictions(agent_outputs)
        if contradiction_analysis['has_contradictions']:
            uncertainty_factors.append({
                'factor': 'contradictions',
                'value': contradiction_analysis['uncertainty_score'],
                'description': f'{len(contradiction_analysis["contradictions"])} contradiction(s) detected'
            })
            total_uncertainty += contradiction_analysis['uncertainty_score'] * 0.3
        
        return {
            'uncertainty_score': min(1.0, total_uncertainty),
            'uncertainty_factors': uncertainty_factors,
            'is_uncertain': total_uncertainty > 0.4,
            'needs_more_agents': total_uncertainty > 0.3 or contradiction_analysis['needs_more_agents']
        }

