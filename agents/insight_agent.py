"""
Insight Agent with LLM Enhancement
Synthesizes research and technical analysis data into actionable insights using LLM.
"""

import os
from typing import Dict, List, Any, Optional

# Try to import OpenAI, but make it optional
try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False
    print("Warning: OpenAI not installed. LLM insights will use fallback mode.")


class InsightAgent:
    """
    Agent responsible for generating insights from research and TA data.
    Uses LLM for enhanced insight generation when available.
    """
    
    def __init__(self, use_llm: bool = True):
        """
        Initialize Insight Agent.
        
        Args:
            use_llm: Whether to use LLM for enhanced insights
        """
        self.name = "InsightAgent"
        self.capabilities = ['generate_insight']
        self.use_llm = use_llm and HAS_OPENAI
        
        if self.use_llm:
            api_key = os.getenv('OPENAI_API_KEY')
            if api_key:
                self.client = OpenAI(api_key=api_key)
            else:
                self.use_llm = False
                print("Warning: OPENAI_API_KEY not set. Using fallback insights.")
    
    def generate_insight(
        self,
        news_data: Optional[Dict[str, Any]] = None,
        fundamentals_data: Optional[Dict[str, Any]] = None,
        sentiment_data: Optional[Dict[str, Any]] = None,
        macro_data: Optional[Dict[str, Any]] = None,
        ta_basic_data: Optional[Dict[str, Any]] = None,
        ta_advanced_data: Optional[Dict[str, Any]] = None,
        existing_insights: Optional[List[str]] = None,
        stock_symbol: Optional[str] = None
    ) -> List[str]:
        """
        Generate insights from available data using LLM when available.
        
        Args:
            news_data: News data dictionary
            fundamentals_data: Fundamentals data dictionary
            sentiment_data: Sentiment data dictionary
            macro_data: Macroeconomic data dictionary
            ta_basic_data: Basic technical analysis data
            ta_advanced_data: Advanced technical analysis data
            existing_insights: Previously generated insights
            stock_symbol: Stock symbol for context
        
        Returns:
            List of insight strings
        """
        insights = existing_insights.copy() if existing_insights else []
        
        # Collect data summary for LLM
        data_summary = self._prepare_data_summary(
            news_data, fundamentals_data, sentiment_data,
            macro_data, ta_basic_data, ta_advanced_data
        )
        
        # Use LLM if available, otherwise use rule-based
        if self.use_llm and data_summary:
            try:
                llm_insights = self._generate_llm_insights(data_summary, stock_symbol)
                if llm_insights:
                    insights.extend(llm_insights)
            except Exception as e:
                print(f"LLM insight generation failed: {e}. Using fallback.")
                insights.extend(self._generate_rule_based_insights(
                    news_data, fundamentals_data, sentiment_data,
                    macro_data, ta_basic_data, ta_advanced_data
                ))
        else:
            insights.extend(self._generate_rule_based_insights(
                news_data, fundamentals_data, sentiment_data,
                macro_data, ta_basic_data, ta_advanced_data
            ))
        
        # Ensure at least one insight
        if not insights:
            insights.append("Insufficient data for comprehensive insights")
        
        return insights
    
    def _prepare_data_summary(self, news_data, fundamentals_data, sentiment_data,
                             macro_data, ta_basic_data, ta_advanced_data) -> str:
        """Prepare data summary for LLM."""
        summary_parts = []
        
        if news_data:
            sentiment = news_data.get('sentiment_score', 0.5)
            num_articles = news_data.get('num_articles', 0)
            summary_parts.append(f"News: {num_articles} articles with sentiment {sentiment:.2f}")
        
        if fundamentals_data and fundamentals_data.get('available'):
            pe = fundamentals_data.get('pe_ratio')
            growth = fundamentals_data.get('revenue_growth')
            margin = fundamentals_data.get('profit_margin')
            if pe:
                summary_parts.append(f"P/E Ratio: {pe:.2f}")
            if growth:
                summary_parts.append(f"Revenue Growth: {growth*100:.1f}%")
            if margin:
                summary_parts.append(f"Profit Margin: {margin*100:.1f}%")
        
        if ta_basic_data:
            rsi = ta_basic_data.get('rsi', 50.0)
            summary_parts.append(f"RSI: {rsi:.2f}")
        
        if ta_advanced_data:
            trend = ta_advanced_data.get('trend', 'sideways')
            macd = ta_advanced_data.get('macd_signal', 0.0)
            summary_parts.append(f"Trend: {trend}, MACD Signal: {macd:.2f}")
        
        return "; ".join(summary_parts)
    
    def _generate_llm_insights(self, data_summary: str, stock_symbol: Optional[str] = None) -> List[str]:
        """Generate insights using LLM."""
        if not self.use_llm or not data_summary:
            return []
        
        try:
            prompt = f"""You are a financial analyst. Analyze the following stock data and provide 2-3 concise, actionable insights.

Stock: {stock_symbol or 'Unknown'}
Data: {data_summary}

Provide insights in bullet point format, focusing on:
1. Key strengths or concerns
2. Market sentiment implications
3. Technical indicators significance

Keep each insight to one sentence. Be specific and data-driven."""

            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a financial analyst providing concise, actionable investment insights."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=200,
                temperature=0.7
            )
            
            content = response.choices[0].message.content.strip()
            # Parse bullet points
            insights = [line.strip('- •').strip() for line in content.split('\n') if line.strip() and line.strip()[0] in ['-', '•', '*']]
            
            return insights if insights else [content]
            
        except Exception as e:
            print(f"Error generating LLM insights: {e}")
            return []
    
    def _generate_rule_based_insights(
        self,
        news_data: Optional[Dict[str, Any]],
        fundamentals_data: Optional[Dict[str, Any]],
        sentiment_data: Optional[Dict[str, Any]],
        macro_data: Optional[Dict[str, Any]],
        ta_basic_data: Optional[Dict[str, Any]],
        ta_advanced_data: Optional[Dict[str, Any]]
    ) -> List[str]:
        """Generate insights using rule-based approach (fallback)."""
        insights = []
        
        # News insights
        if news_data:
            sentiment = news_data.get('sentiment_score', 0.5)
            num_articles = news_data.get('num_articles', 0)
            
            if sentiment > 0.6:
                insights.append(f"Positive news sentiment ({sentiment:.2f}) with {num_articles} recent articles indicating bullish momentum")
            elif sentiment < 0.4:
                insights.append(f"Negative news sentiment ({sentiment:.2f}) with {num_articles} recent articles indicating bearish pressure")
            else:
                insights.append(f"Neutral news sentiment ({sentiment:.2f}) with {num_articles} articles providing mixed signals")
        
        # Fundamental insights
        if fundamentals_data and fundamentals_data.get('available', False):
            pe_ratio = fundamentals_data.get('pe_ratio')
            revenue_growth = fundamentals_data.get('revenue_growth')
            profit_margin = fundamentals_data.get('profit_margin')
            
            if pe_ratio and revenue_growth:
                if pe_ratio < 20 and revenue_growth > 0.1:
                    insights.append(f"Attractive valuation: P/E ratio of {pe_ratio:.1f} combined with strong revenue growth of {revenue_growth*100:.1f}%")
                elif pe_ratio > 30:
                    insights.append(f"High valuation concern: P/E ratio of {pe_ratio:.1f} suggests stock may be overvalued relative to earnings")
            
            if profit_margin and profit_margin > 0.2:
                insights.append(f"Strong profitability: Profit margin of {profit_margin*100:.1f}% indicates efficient operations")
        
        # Sentiment insights
        if sentiment_data:
            social_sentiment = sentiment_data.get('social_sentiment', 0.5)
            analyst_rating = sentiment_data.get('analyst_rating', 'Hold')
            
            if social_sentiment > 0.6:
                insights.append(f"Positive social sentiment ({social_sentiment:.2f}) aligns with analyst rating of {analyst_rating}")
            elif social_sentiment < 0.4:
                insights.append(f"Negative social sentiment ({social_sentiment:.2f}) contrasts with analyst rating of {analyst_rating}")
        
        # Technical analysis insights
        if ta_basic_data:
            rsi = ta_basic_data.get('rsi', 50.0)
            current_price = ta_basic_data.get('current_price', 0.0)
            ma20 = ta_basic_data.get('ma20', 0.0)
            
            if rsi < 30:
                insights.append(f"RSI of {rsi:.1f} indicates oversold conditions, potential buying opportunity")
            elif rsi > 70:
                insights.append(f"RSI of {rsi:.1f} indicates overbought conditions, potential selling pressure")
            
            if current_price and ma20:
                if current_price > ma20:
                    insights.append(f"Price ({current_price:.2f}) above 20-day MA ({ma20:.2f}), suggesting short-term bullish momentum")
                elif current_price < ma20:
                    insights.append(f"Price ({current_price:.2f}) below 20-day MA ({ma20:.2f}), suggesting short-term bearish pressure")
        
        if ta_advanced_data:
            trend = ta_advanced_data.get('trend', 'sideways')
            macd_signal = ta_advanced_data.get('macd_signal', 0.0)
            ma50 = ta_advanced_data.get('ma50', 0.0)
            ma200 = ta_advanced_data.get('ma200', 0.0)
            
            insights.append(f"Technical trend analysis shows {trend} pattern with MACD signal at {macd_signal:.2f}")
            
            if ma50 and ma200:
                if ma50 > ma200:
                    insights.append(f"Golden cross pattern: 50-day MA ({ma50:.2f}) above 200-day MA ({ma200:.2f}), indicating long-term bullish trend")
                elif ma50 < ma200:
                    insights.append(f"Death cross pattern: 50-day MA ({ma50:.2f}) below 200-day MA ({ma200:.2f}), indicating long-term bearish trend")
        
        # Macro insights
        if macro_data:
            gdp_growth = macro_data.get('gdp_growth', 0.0)
            interest_rate = macro_data.get('interest_rate', 0.0)
            
            if gdp_growth > 0.03:
                insights.append(f"Supportive macro environment: GDP growth of {gdp_growth*100:.1f}% and interest rate of {interest_rate*100:.1f}%")
        
        return insights
