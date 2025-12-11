"""
Recommendation Agent
Generates Buy/Hold/Sell recommendations with confidence scores.
"""

from typing import Dict, List, Any, Optional, Tuple


class RecommendationAgent:
    """
    Agent responsible for generating investment recommendations.
    """
    
    def __init__(self):
        """Initialize Recommendation Agent."""
        self.name = "RecommendationAgent"
        self.capabilities = ['generate_recommendation']
    
    def generate_recommendation(
        self,
        insights: List[str],
        news_data: Optional[Dict[str, Any]] = None,
        fundamentals_data: Optional[Dict[str, Any]] = None,
        sentiment_data: Optional[Dict[str, Any]] = None,
        ta_basic_data: Optional[Dict[str, Any]] = None,
        ta_advanced_data: Optional[Dict[str, Any]] = None
    ) -> Tuple[str, float, Dict[str, Any]]:
        """
        Generate Buy/Hold/Sell recommendation with confidence.
        
        Args:
            insights: List of generated insights
            news_data: News data dictionary
            fundamentals_data: Fundamentals data dictionary
            sentiment_data: Sentiment data dictionary
            ta_basic_data: Basic technical analysis data
            ta_advanced_data: Advanced technical analysis data
        
        Returns:
            Tuple of (recommendation, confidence_score)
        """
        if not insights:
            return 'Hold', 0.5
        
        buy_signals = 0
        sell_signals = 0
        neutral_signals = 0
        
        # Analyze insights for signals
        insight_text = ' '.join(insights).lower()
        
        if 'positive' in insight_text or 'bullish' in insight_text or 'buying opportunity' in insight_text:
            buy_signals += 1
        if 'negative' in insight_text or 'bearish' in insight_text or 'selling pressure' in insight_text:
            sell_signals += 1
        if 'oversold' in insight_text:
            buy_signals += 1
        if 'overbought' in insight_text:
            sell_signals += 1
        if 'uptrend' in insight_text or 'golden cross' in insight_text:
            buy_signals += 1
        if 'downtrend' in insight_text or 'death cross' in insight_text:
            sell_signals += 1
        
        # News sentiment signals
        if news_data:
            sentiment = news_data.get('sentiment_score', 0.5)
            if sentiment > 0.6:
                buy_signals += 1
            elif sentiment < 0.4:
                sell_signals += 1
            else:
                neutral_signals += 1
        
        # Fundamental signals
        if fundamentals_data:
            pe_ratio = fundamentals_data.get('pe_ratio', 20.0)
            revenue_growth = fundamentals_data.get('revenue_growth', 0.0)
            
            if pe_ratio < 20 and revenue_growth > 0.1:
                buy_signals += 1
            elif pe_ratio > 30:
                sell_signals += 1
            else:
                neutral_signals += 1
        
        # Sentiment signals
        if sentiment_data:
            social_sentiment = sentiment_data.get('social_sentiment', 0.5)
            analyst_rating = sentiment_data.get('analyst_rating', 'Hold')
            
            if social_sentiment > 0.6 or analyst_rating == 'Buy':
                buy_signals += 1
            elif social_sentiment < 0.4 or analyst_rating == 'Sell':
                sell_signals += 1
            else:
                neutral_signals += 1
        
        # Technical analysis signals
        if ta_basic_data:
            rsi = ta_basic_data.get('rsi', 50.0)
            if rsi < 30:
                buy_signals += 1
            elif rsi > 70:
                sell_signals += 1
            else:
                neutral_signals += 1
        
        if ta_advanced_data:
            trend = ta_advanced_data.get('trend', 'sideways')
            macd_signal = ta_advanced_data.get('macd_signal', 0.0)
            
            if trend == 'uptrend' or macd_signal > 0:
                buy_signals += 1
            elif trend == 'downtrend' or macd_signal < 0:
                sell_signals += 1
            else:
                neutral_signals += 1
        
        # Determine recommendation - IMPROVED: More selective Hold, better Sell detection
        total_signals = buy_signals + sell_signals + neutral_signals
        
        if total_signals == 0:
            return 'Hold', 0.5
        
        buy_ratio = buy_signals / total_signals
        sell_ratio = sell_signals / total_signals
        signal_diff = buy_signals - sell_signals
        
        # IMPROVED: More selective thresholds
        # Buy: Need clear bullish signal (buy_signals > sell_signals + 1)
        # Sell: Need clear bearish signal (sell_signals > buy_signals + 1)  
        # Hold: Only when signals are truly balanced (within 1 signal)
        if signal_diff > 1:  # Clear buy signal
            recommendation = 'Buy'
            confidence = min(0.95, 0.6 + (signal_diff - 1) * 0.1)
        elif signal_diff < -1:  # Clear sell signal
            recommendation = 'Sell'
            confidence = min(0.95, 0.6 + (abs(signal_diff) - 1) * 0.1)
        else:  # Balanced signals - Hold is appropriate
            recommendation = 'Hold'
            # Higher confidence when signals are very balanced (diff = 0 or 1)
            if abs(signal_diff) <= 1:
                confidence = 0.7  # High confidence for balanced signals
            else:
                confidence = 0.55  # Lower confidence when slightly unbalanced
        
        # Ensure minimum confidence
        confidence = max(0.5, confidence)
        
        # Calculate price levels (entry, stop loss, resistance, exit)
        price_levels = self._calculate_price_levels(
            recommendation, confidence, ta_basic_data, ta_advanced_data
        )
        
        return recommendation, confidence, price_levels
    
    def _calculate_price_levels(
        self,
        recommendation: str,
        confidence: float,
        ta_basic_data: Optional[Dict[str, Any]],
        ta_advanced_data: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Calculate entry price, stop loss, resistance levels, and exit price.
        
        Returns:
            Dictionary with entry_price, stop_loss, resistance_levels, exit_price
        """
        current_price = None
        if ta_basic_data:
            current_price = ta_basic_data.get('current_price')
        elif ta_advanced_data:
            # Try to get from advanced data
            pass
        
        if current_price is None:
            return {
                'entry_price': None,
                'stop_loss': None,
                'resistance_levels': [],
                'support_levels': [],
                'exit_price': None
            }
        
        price_levels = {}
        
        if recommendation == 'Buy':
            # Entry: Current price or slightly below for better entry
            price_levels['entry_price'] = round(current_price * 0.998, 2)  # 0.2% below current
            
            # Stop Loss: 2-5% below entry based on volatility
            stop_loss_pct = 0.03 if confidence > 0.7 else 0.05  # 3% or 5%
            price_levels['stop_loss'] = round(price_levels['entry_price'] * (1 - stop_loss_pct), 2)
            
            # Resistance levels: Based on MA50, MA200, or 5-10% above current
            resistance_levels = []
            if ta_advanced_data:
                ma50 = ta_advanced_data.get('ma50')
                ma200 = ta_advanced_data.get('ma200')
                if ma50 and ma50 > current_price:
                    resistance_levels.append(round(ma50, 2))
                if ma200 and ma200 > current_price:
                    resistance_levels.append(round(ma200, 2))
            
            # Add percentage-based resistance levels
            for pct in [0.05, 0.10, 0.15]:  # 5%, 10%, 15% above
                resistance_levels.append(round(current_price * (1 + pct), 2))
            
            price_levels['resistance_levels'] = sorted(list(set(resistance_levels)))[:3]  # Top 3
            price_levels['support_levels'] = [price_levels['stop_loss']]
            
            # Exit: First major resistance or 10% gain
            price_levels['exit_price'] = price_levels['resistance_levels'][0] if price_levels['resistance_levels'] else round(current_price * 1.10, 2)
        
        elif recommendation == 'Sell':
            # Entry: Current price or slightly above for better exit
            price_levels['entry_price'] = round(current_price * 1.002, 2)  # 0.2% above current
            
            # Stop Loss: 2-5% above entry (for short positions, this is the stop)
            stop_loss_pct = 0.03 if confidence > 0.7 else 0.05
            price_levels['stop_loss'] = round(price_levels['entry_price'] * (1 + stop_loss_pct), 2)
            
            # Support levels: Based on MA50, MA200, or 5-10% below current
            support_levels = []
            if ta_advanced_data:
                ma50 = ta_advanced_data.get('ma50')
                ma200 = ta_advanced_data.get('ma200')
                if ma50 and ma50 < current_price:
                    support_levels.append(round(ma50, 2))
                if ma200 and ma200 < current_price:
                    support_levels.append(round(ma200, 2))
            
            # Add percentage-based support levels
            for pct in [0.05, 0.10, 0.15]:  # 5%, 10%, 15% below
                support_levels.append(round(current_price * (1 - pct), 2))
            
            price_levels['support_levels'] = sorted(list(set(support_levels)), reverse=True)[:3]  # Top 3
            price_levels['resistance_levels'] = [price_levels['stop_loss']]
            
            # Exit: First major support or 10% decline
            price_levels['exit_price'] = price_levels['support_levels'][0] if price_levels['support_levels'] else round(current_price * 0.90, 2)
        
        else:  # Hold
            # For Hold, show current price and nearby levels
            price_levels['entry_price'] = round(current_price, 2)
            price_levels['stop_loss'] = round(current_price * 0.97, 2)  # 3% below
            
            # Resistance and support based on MAs
            resistance_levels = []
            support_levels = []
            if ta_advanced_data:
                ma50 = ta_advanced_data.get('ma50')
                ma200 = ta_advanced_data.get('ma200')
                if ma50:
                    if ma50 > current_price:
                        resistance_levels.append(round(ma50, 2))
                    else:
                        support_levels.append(round(ma50, 2))
                if ma200:
                    if ma200 > current_price:
                        resistance_levels.append(round(ma200, 2))
                    else:
                        support_levels.append(round(ma200, 2))
            
            # Add percentage-based levels
            resistance_levels.extend([round(current_price * 1.05, 2), round(current_price * 1.10, 2)])
            support_levels.extend([round(current_price * 0.95, 2), round(current_price * 0.90, 2)])
            
            price_levels['resistance_levels'] = sorted(list(set(resistance_levels)))[:3]
            price_levels['support_levels'] = sorted(list(set(support_levels)), reverse=True)[:3]
            price_levels['exit_price'] = None  # No exit for Hold
        
        return price_levels

