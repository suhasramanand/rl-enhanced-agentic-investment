"""
Risk Analysis Agent
Identifies and analyzes investment risks for stocks.
"""

from typing import Dict, List, Any, Optional


class RiskAgent:
    """
    Agent responsible for identifying and analyzing investment risks.
    """
    
    def __init__(self):
        """Initialize Risk Agent."""
        self.name = "RiskAgent"
        self.capabilities = ['analyze_risks']
    
    def analyze_risks(
        self,
        stock_symbol: str,
        news_data: Optional[Dict[str, Any]] = None,
        fundamentals_data: Optional[Dict[str, Any]] = None,
        sentiment_data: Optional[Dict[str, Any]] = None,
        ta_basic_data: Optional[Dict[str, Any]] = None,
        ta_advanced_data: Optional[Dict[str, Any]] = None,
        recommendation: Optional[str] = None,
        confidence: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        Analyze and identify investment risks.
        
        Args:
            stock_symbol: Stock symbol
            news_data: News data dictionary
            fundamentals_data: Fundamentals data dictionary
            sentiment_data: Sentiment data dictionary
            ta_basic_data: Basic technical analysis data
            ta_advanced_data: Advanced technical analysis data
            recommendation: Current recommendation (Buy/Hold/Sell)
            confidence: Confidence level
        
        Returns:
            List of risk dictionaries with 'category', 'severity', and 'description'
        """
        risks = []
        
        # 1. Valuation Risks
        if fundamentals_data and fundamentals_data.get('available'):
            pe_ratio = fundamentals_data.get('pe_ratio')
            if pe_ratio and pe_ratio > 40:
                risks.append({
                    'category': 'Valuation',
                    'severity': 'High',
                    'description': f'High P/E ratio ({pe_ratio:.1f}) suggests stock may be overvalued relative to earnings'
                })
            elif pe_ratio and pe_ratio < 5:
                risks.append({
                    'category': 'Valuation',
                    'severity': 'Medium',
                    'description': f'Very low P/E ratio ({pe_ratio:.1f}) may indicate underlying issues or market concerns'
                })
            
            debt_to_equity = fundamentals_data.get('debt_to_equity')
            if debt_to_equity and debt_to_equity > 2.0:
                risks.append({
                    'category': 'Financial Health',
                    'severity': 'High',
                    'description': f'High debt-to-equity ratio ({debt_to_equity:.2f}) indicates significant financial leverage risk'
                })
        
        # 2. Technical Risks
        if ta_basic_data:
            rsi = ta_basic_data.get('rsi', 50.0)
            if rsi > 80:
                risks.append({
                    'category': 'Technical',
                    'severity': 'High',
                    'description': f'Extremely overbought conditions (RSI: {rsi:.1f}) suggest potential reversal risk'
                })
            elif rsi < 20:
                risks.append({
                    'category': 'Technical',
                    'severity': 'Medium',
                    'description': f'Oversold conditions (RSI: {rsi:.1f}) may indicate continued downward pressure'
                })
        
        if ta_advanced_data:
            trend = ta_advanced_data.get('trend', 'sideways')
            if trend == 'downtrend':
                risks.append({
                    'category': 'Technical',
                    'severity': 'Medium',
                    'description': 'Downtrend pattern indicates bearish momentum and potential further decline'
                })
            
            ma50 = ta_advanced_data.get('ma50')
            ma200 = ta_advanced_data.get('ma200')
            if ma50 and ma200 and ma50 < ma200:
                risks.append({
                    'category': 'Technical',
                    'severity': 'Medium',
                    'description': 'Death cross pattern (50-day MA below 200-day MA) suggests long-term bearish trend'
                })
        
        # 3. Sentiment Risks
        if news_data:
            sentiment = news_data.get('sentiment_score', 0.5)
            if sentiment < 0.3:
                risks.append({
                    'category': 'Sentiment',
                    'severity': 'High',
                    'description': f'Very negative news sentiment ({sentiment:.2f}) may indicate fundamental concerns'
                })
            
            num_articles = news_data.get('num_articles', 0)
            if num_articles == 0:
                risks.append({
                    'category': 'Information',
                    'severity': 'Low',
                    'description': 'Limited news coverage may indicate low market interest or information asymmetry'
                })
        
        if sentiment_data:
            social_sentiment = sentiment_data.get('social_sentiment', 0.5)
            if social_sentiment < 0.3:
                risks.append({
                    'category': 'Sentiment',
                    'severity': 'Medium',
                    'description': f'Negative social sentiment ({social_sentiment:.2f}) may reflect broader market concerns'
                })
        
        # 4. Confidence-Based Risks
        if confidence < 0.6:
            risks.append({
                'category': 'Uncertainty',
                'severity': 'Medium',
                'description': f'Low confidence level ({confidence:.2f}) suggests high uncertainty in recommendation'
            })
        
        # 5. Recommendation-Specific Risks
        if recommendation == 'Buy':
            if ta_advanced_data and ta_advanced_data.get('trend') == 'downtrend':
                risks.append({
                    'category': 'Timing',
                    'severity': 'High',
                    'description': 'Buy recommendation during downtrend may be premature - consider waiting for trend reversal'
                })
        elif recommendation == 'Sell':
            if ta_advanced_data and ta_advanced_data.get('trend') == 'uptrend':
                risks.append({
                    'category': 'Timing',
                    'severity': 'Medium',
                    'description': 'Sell recommendation during uptrend may result in missing further gains'
                })
        elif recommendation == 'Hold':
            risks.append({
                'category': 'Opportunity',
                'severity': 'Low',
                'description': 'Hold recommendation may result in missed opportunities if market conditions change'
            })
        
        # 6. Data Availability Risks
        data_sources = []
        if news_data: data_sources.append('news')
        if fundamentals_data and fundamentals_data.get('available'): data_sources.append('fundamentals')
        if sentiment_data: data_sources.append('sentiment')
        if ta_basic_data: data_sources.append('technical')
        
        if len(data_sources) < 3:
            risks.append({
                'category': 'Information',
                'severity': 'Medium',
                'description': f'Limited data sources ({len(data_sources)}/4) may reduce analysis reliability'
            })
        
        # 7. Market Volatility Risk (if we have price data)
        if ta_basic_data:
            current_price = ta_basic_data.get('current_price')
            ma20 = ta_basic_data.get('ma20')
            if current_price and ma20:
                price_deviation = abs(current_price - ma20) / ma20 if ma20 > 0 else 0
                if price_deviation > 0.15:  # 15% deviation
                    risks.append({
                        'category': 'Volatility',
                        'severity': 'Medium',
                        'description': f'High price volatility ({price_deviation*100:.1f}% deviation from 20-day MA) indicates market uncertainty'
                    })
        
        # Ensure at least one risk if none identified
        if not risks:
            risks.append({
                'category': 'General',
                'severity': 'Low',
                'description': 'Standard market risks apply: price volatility, market conditions, and economic factors'
            })
        
        return risks

