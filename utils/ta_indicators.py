"""
Technical Analysis Indicators
Provides functions for calculating various technical indicators used in stock analysis.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict


def calculate_rsi(prices: List[float], period: int = 14) -> float:
    """
    Calculate Relative Strength Index (RSI).
    
    Args:
        prices: List of closing prices
        period: Period for RSI calculation (default 14)
    
    Returns:
        RSI value between 0 and 100
    """
    if len(prices) < period + 1:
        return 50.0  # Neutral RSI if insufficient data
    
    deltas = np.diff(prices[-period-1:])
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    
    avg_gain = np.mean(gains)
    avg_loss = np.mean(losses)
    
    if avg_loss == 0:
        return 100.0
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    return float(rsi)


def calculate_macd(prices: List[float], fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[float, float, float]:
    """
    Calculate MACD (Moving Average Convergence Divergence).
    
    Args:
        prices: List of closing prices
        fast: Fast EMA period (default 12)
        slow: Slow EMA period (default 26)
        signal: Signal line EMA period (default 9)
    
    Returns:
        Tuple of (MACD line, Signal line, Histogram)
    """
    if len(prices) < slow + signal:
        return (0.0, 0.0, 0.0)
    
    prices_array = np.array(prices)
    
    # Calculate EMAs
    ema_fast = calculate_ema(prices_array, fast)
    ema_slow = calculate_ema(prices_array, slow)
    
    # Align EMA arrays to same length (use the longer one as reference)
    min_len = min(len(ema_fast), len(ema_slow))
    if min_len == 0:
        return (0.0, 0.0, 0.0)
    
    # Take the last min_len elements from each
    ema_fast_aligned = ema_fast[-min_len:]
    ema_slow_aligned = ema_slow[-min_len:]
    
    macd_line = ema_fast_aligned - ema_slow_aligned
    
    # Calculate signal line (EMA of MACD)
    macd_values = macd_line
    signal_line = calculate_ema(macd_values, signal)
    
    # Align signal line with macd_line
    if len(signal_line) > 0 and len(macd_line) > 0:
        histogram = macd_line[-1] - signal_line[-1]
    else:
        histogram = 0.0
    
    return (float(macd_line[-1]), float(signal_line[-1]) if len(signal_line) > 0 else 0.0, float(histogram))


def calculate_ema(prices: np.ndarray, period: int) -> np.ndarray:
    """
    Calculate Exponential Moving Average (EMA).
    
    Args:
        prices: Array of prices
        period: EMA period
    
    Returns:
        Array of EMA values
    """
    if len(prices) < period:
        return np.array([np.mean(prices)])
    
    ema = np.zeros(len(prices))
    multiplier = 2 / (period + 1)
    
    # Start with SMA
    ema[period - 1] = np.mean(prices[:period])
    
    # Calculate EMA for remaining values
    for i in range(period, len(prices)):
        ema[i] = (prices[i] - ema[i - 1]) * multiplier + ema[i - 1]
    
    return ema[period - 1:]


def calculate_moving_average(prices: List[float], period: int) -> float:
    """
    Calculate Simple Moving Average (SMA).
    
    Args:
        prices: List of prices
        period: Period for moving average
    
    Returns:
        Moving average value
    """
    if len(prices) < period:
        return float(np.mean(prices)) if prices else 0.0
    
    return float(np.mean(prices[-period:]))


def calculate_atr(highs: List[float], lows: List[float], closes: List[float], period: int = 14) -> float:
    """
    Calculate Average True Range (ATR).
    
    Args:
        highs: List of high prices
        lows: List of low prices
        closes: List of closing prices
        period: Period for ATR calculation (default 14)
    
    Returns:
        ATR value
    """
    if len(highs) < period + 1 or len(lows) < period + 1 or len(closes) < period + 1:
        return 0.0
    
    true_ranges = []
    for i in range(1, len(highs)):
        tr1 = highs[i] - lows[i]
        tr2 = abs(highs[i] - closes[i - 1])
        tr3 = abs(lows[i] - closes[i - 1])
        true_ranges.append(max(tr1, tr2, tr3))
    
    if len(true_ranges) < period:
        return float(np.mean(true_ranges)) if true_ranges else 0.0
    
    return float(np.mean(true_ranges[-period:]))


def identify_trend(prices: List[float], short_period: int = 20, long_period: int = 50) -> str:
    """
    Identify trend direction based on moving averages.
    
    Args:
        prices: List of closing prices
        short_period: Short-term MA period (default 20)
        long_period: Long-term MA period (default 50)
    
    Returns:
        Trend direction: 'uptrend', 'downtrend', or 'sideways'
    """
    if len(prices) < long_period:
        return 'sideways'
    
    ma_short = calculate_moving_average(prices, short_period)
    ma_long = calculate_moving_average(prices, long_period)
    current_price = prices[-1]
    
    if ma_short > ma_long and current_price > ma_short:
        return 'uptrend'
    elif ma_short < ma_long and current_price < ma_short:
        return 'downtrend'
    else:
        return 'sideways'


def calculate_bollinger_bands(prices: List[float], period: int = 20, num_std: float = 2.0) -> Tuple[float, float, float]:
    """
    Calculate Bollinger Bands.
    
    Args:
        prices: List of closing prices
        period: Period for calculation (default 20)
        num_std: Number of standard deviations (default 2.0)
    
    Returns:
        Tuple of (upper_band, middle_band, lower_band)
    """
    if len(prices) < period:
        ma = float(np.mean(prices)) if prices else 0.0
        return (ma, ma, ma)
    
    ma = calculate_moving_average(prices, period)
    std = float(np.std(prices[-period:]))
    
    upper_band = ma + (num_std * std)
    lower_band = ma - (num_std * std)
    
    return (float(upper_band), float(ma), float(lower_band))


def calculate_fibonacci_retracements(high: float, low: float) -> Dict[str, float]:
    """
    Calculate Fibonacci retracement levels.
    
    Args:
        high: Highest price in the period
        low: Lowest price in the period
    
    Returns:
        Dictionary with Fibonacci levels (0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0)
    """
    diff = high - low
    fib_levels = {
        'fib_0': high,
        'fib_236': high - (diff * 0.236),
        'fib_382': high - (diff * 0.382),
        'fib_500': high - (diff * 0.5),
        'fib_618': high - (diff * 0.618),
        'fib_786': high - (diff * 0.786),
        'fib_100': low
    }
    return fib_levels


def calculate_support_resistance(prices: List[float], window: int = 20) -> Tuple[List[float], List[float]]:
    """
    Identify support and resistance levels using local minima and maxima.
    
    Args:
        prices: List of closing prices
        window: Window size for finding local extrema
    
    Returns:
        Tuple of (support_levels, resistance_levels)
    """
    if len(prices) < window * 2:
        return ([], [])
    
    support_levels = []
    resistance_levels = []
    
    for i in range(window, len(prices) - window):
        local_min = min(prices[i-window:i+window])
        local_max = max(prices[i-window:i+window])
        
        if prices[i] == local_min:
            support_levels.append(prices[i])
        if prices[i] == local_max:
            resistance_levels.append(prices[i])
    
    # Remove duplicates and sort
    support_levels = sorted(list(set(support_levels)))
    resistance_levels = sorted(list(set(resistance_levels)), reverse=True)
    
    return (support_levels[:5], resistance_levels[:5])  # Return top 5 levels


def calculate_stochastic(highs: List[float], lows: List[float], closes: List[float], 
                        k_period: int = 14, d_period: int = 3) -> Tuple[float, float]:
    """
    Calculate Stochastic Oscillator (%K and %D).
    
    Args:
        highs: List of high prices
        lows: List of low prices
        closes: List of closing prices
        k_period: Period for %K calculation (default 14)
        d_period: Period for %D smoothing (default 3)
    
    Returns:
        Tuple of (%K, %D)
    """
    if len(closes) < k_period + d_period:
        return (50.0, 50.0)
    
    # Calculate %K
    k_values = []
    for i in range(k_period - 1, len(closes)):
        period_high = max(highs[i - k_period + 1:i + 1])
        period_low = min(lows[i - k_period + 1:i + 1])
        
        if period_high == period_low:
            k = 50.0
        else:
            k = ((closes[i] - period_low) / (period_high - period_low)) * 100
        k_values.append(k)
    
    # Calculate %D (SMA of %K)
    if len(k_values) < d_period:
        return (k_values[-1] if k_values else 50.0, 50.0)
    
    d = float(np.mean(k_values[-d_period:]))
    k = k_values[-1]
    
    return (float(k), float(d))


def calculate_adx(highs: List[float], lows: List[float], closes: List[float], 
                 period: int = 14) -> float:
    """
    Calculate Average Directional Index (ADX).
    
    Args:
        highs: List of high prices
        lows: List of low prices
        closes: List of closing prices
        period: Period for ADX calculation (default 14)
    
    Returns:
        ADX value (0-100)
    """
    if len(highs) < period + 1:
        return 0.0
    
    # Calculate True Range
    tr_values = []
    for i in range(1, len(highs)):
        tr1 = highs[i] - lows[i]
        tr2 = abs(highs[i] - closes[i-1])
        tr3 = abs(lows[i] - closes[i-1])
        tr_values.append(max(tr1, tr2, tr3))
    
    # Calculate Directional Movement
    plus_dm = []
    minus_dm = []
    
    for i in range(1, len(highs)):
        up_move = highs[i] - highs[i-1]
        down_move = lows[i-1] - lows[i]
        
        if up_move > down_move and up_move > 0:
            plus_dm.append(up_move)
        else:
            plus_dm.append(0.0)
        
        if down_move > up_move and down_move > 0:
            minus_dm.append(down_move)
        else:
            minus_dm.append(0.0)
    
    # Calculate smoothed values
    atr = np.mean(tr_values[-period:]) if len(tr_values) >= period else np.mean(tr_values) if tr_values else 1.0
    plus_di = (np.mean(plus_dm[-period:]) / atr * 100) if len(plus_dm) >= period else 0.0
    minus_di = (np.mean(minus_dm[-period:]) / atr * 100) if len(minus_dm) >= period else 0.0
    
    # Calculate DX
    if plus_di + minus_di == 0:
        return 0.0
    
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    
    return float(dx)


def calculate_ichimoku(highs: List[float], lows: List[float], closes: List[float]) -> Dict[str, float]:
    """
    Calculate Ichimoku Cloud components.
    
    Args:
        highs: List of high prices
        lows: List of low prices
        closes: List of closing prices
    
    Returns:
        Dictionary with Tenkan-sen, Kijun-sen, Senkou Span A, Senkou Span B, Chikou Span
    """
    if len(closes) < 52:
        current = closes[-1] if closes else 0.0
        return {
            'tenkan_sen': current,
            'kijun_sen': current,
            'senkou_span_a': current,
            'senkou_span_b': current,
            'chikou_span': current
        }
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    tenkan_high = max(highs[-9:])
    tenkan_low = min(lows[-9:])
    tenkan_sen = (tenkan_high + tenkan_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    kijun_high = max(highs[-26:])
    kijun_low = min(lows[-26:])
    kijun_sen = (kijun_high + kijun_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen) / 2, plotted 26 periods ahead
    senkou_span_a = (tenkan_sen + kijun_sen) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2, plotted 26 periods ahead
    senkou_high = max(highs[-52:])
    senkou_low = min(lows[-52:])
    senkou_span_b = (senkou_high + senkou_low) / 2
    
    # Chikou Span (Lagging Span): Current closing price plotted 26 periods back
    chikou_span = closes[-1]
    
    return {
        'tenkan_sen': float(tenkan_sen),
        'kijun_sen': float(kijun_sen),
        'senkou_span_a': float(senkou_span_a),
        'senkou_span_b': float(senkou_span_b),
        'chikou_span': float(chikou_span)
    }


def calculate_volume_indicators(prices: List[float], volumes: List[float], period: int = 20) -> Dict[str, float]:
    """
    Calculate volume-based indicators.
    
    Args:
        prices: List of closing prices
        volumes: List of volume values
        period: Period for calculation
    
    Returns:
        Dictionary with OBV, Volume MA, Price-Volume Trend
    """
    if len(prices) < 2 or len(volumes) < 2:
        return {
            'obv': 0.0,
            'volume_ma': float(np.mean(volumes)) if volumes else 0.0,
            'pvt': 0.0
        }
    
    # On-Balance Volume (OBV)
    obv = 0.0
    for i in range(1, len(prices)):
        if prices[i] > prices[i-1]:
            obv += volumes[i]
        elif prices[i] < prices[i-1]:
            obv -= volumes[i]
        # If price unchanged, OBV unchanged
    
    # Volume Moving Average
    volume_ma = float(np.mean(volumes[-period:])) if len(volumes) >= period else float(np.mean(volumes))
    
    # Price-Volume Trend (PVT)
    pvt = 0.0
    for i in range(1, len(prices)):
        price_change = (prices[i] - prices[i-1]) / prices[i-1] if prices[i-1] != 0 else 0
        pvt += volumes[i] * price_change
    
    return {
        'obv': float(obv),
        'volume_ma': volume_ma,
        'pvt': float(pvt)
    }

