"""
Technical Analysis Agent
Performs technical analysis using various indicators.
"""

from typing import Dict, List, Any, Optional
from utils.ta_indicators import (
    calculate_rsi, calculate_macd, calculate_moving_average,
    calculate_atr, identify_trend, calculate_bollinger_bands,
    calculate_fibonacci_retracements, calculate_support_resistance,
    calculate_stochastic, calculate_adx, calculate_ichimoku,
    calculate_volume_indicators
)


class TechnicalAnalysisAgent:
    """
    Agent responsible for technical analysis of stocks.
    """
    
    def __init__(self):
        """Initialize Technical Analysis Agent."""
        self.name = "TechnicalAnalysisAgent"
        self.capabilities = [
            'run_ta_basic',
            'run_ta_advanced'
        ]
    
    def run_ta_basic(
        self,
        price_history: List[float],
        env_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Run basic technical analysis.
        
        Args:
            price_history: List of historical closing prices
            env_data: Optional environment data (if already calculated)
        
        Returns:
            Basic TA results dictionary
        """
        if env_data:
            return env_data
        
        if len(price_history) < 14:
            return {
                'rsi': 50.0,
                'ma20': price_history[-1] if price_history else 0.0,
                'current_price': price_history[-1] if price_history else 0.0
            }
        
        rsi = calculate_rsi(price_history)
        ma20 = calculate_moving_average(price_history, 20)
        
        return {
            'rsi': rsi,
            'ma20': ma20,
            'current_price': price_history[-1]
        }
    
    def run_ta_advanced(
        self,
        price_history: List[float],
        high_history: List[float],
        low_history: List[float],
        volume_history: Optional[List[float]] = None,
        env_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Run advanced technical analysis including Fibonacci, support/resistance, etc.
        
        Args:
            price_history: List of historical closing prices
            high_history: List of historical high prices
            low_history: List of historical low prices
            volume_history: Optional list of volume values
            env_data: Optional environment data (if already calculated)
        
        Returns:
            Advanced TA results dictionary
        """
        if env_data:
            return env_data
        
        if len(price_history) < 50:
            current_price = price_history[-1] if price_history else 0.0
            return {
                'macd': 0.0,
                'macd_signal': 0.0,
                'macd_histogram': 0.0,
                'ma50': current_price,
                'ma200': current_price,
                'atr': 0.0,
                'trend': 'sideways',
                'bollinger_upper': current_price,
                'bollinger_middle': current_price,
                'bollinger_lower': current_price
            }
        
        # Basic advanced indicators
        macd, macd_signal, macd_hist = calculate_macd(price_history)
        ma50 = calculate_moving_average(price_history, 50)
        ma200 = calculate_moving_average(price_history, 200)
        atr = calculate_atr(high_history, low_history, price_history)
        trend = identify_trend(price_history)
        
        # Bollinger Bands
        upper_band, middle_band, lower_band = calculate_bollinger_bands(price_history)
        
        # Fibonacci Retracements
        recent_high = max(high_history[-50:])
        recent_low = min(low_history[-50:])
        fib_levels = calculate_fibonacci_retracements(recent_high, recent_low)
        
        # Support and Resistance Levels
        support_levels, resistance_levels = calculate_support_resistance(price_history)
        
        # Stochastic Oscillator
        stoch_k, stoch_d = calculate_stochastic(high_history, low_history, price_history)
        
        # ADX (Average Directional Index)
        adx = calculate_adx(high_history, low_history, price_history)
        
        # Ichimoku Cloud
        ichimoku = calculate_ichimoku(high_history, low_history, price_history)
        
        # Volume Indicators
        volume_data = {}
        if volume_history and len(volume_history) >= 2:
            volume_data = calculate_volume_indicators(price_history, volume_history)
        
        result = {
            'macd': macd,
            'macd_signal': macd_signal,
            'macd_histogram': macd_hist,
            'ma50': ma50,
            'ma200': ma200,
            'atr': atr,
            'trend': trend,
            'bollinger_upper': upper_band,
            'bollinger_middle': middle_band,
            'bollinger_lower': lower_band,
            'fibonacci': fib_levels,
            'support_levels': support_levels[:3],  # Top 3 support levels
            'resistance_levels': resistance_levels[:3],  # Top 3 resistance levels
            'stochastic_k': stoch_k,
            'stochastic_d': stoch_d,
            'adx': adx,
            'ichimoku': ichimoku,
            **volume_data
        }
        
        return result

