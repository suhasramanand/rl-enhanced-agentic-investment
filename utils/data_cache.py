"""
Data Cache Utility
Caches real-world stock data to avoid repeated API calls during training.
"""

import os
import json
import pickle
import hashlib
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
import pandas as pd


class DataCache:
    """
    Cache for storing real-world stock data.
    Reduces API calls during RL training.
    """
    
    def __init__(self, cache_dir: str = 'data_cache'):
        """
        Initialize data cache.
        
        Args:
            cache_dir: Directory to store cached data
        """
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
        # Cache expiration (days)
        self.cache_expiry = {
            'ohlcv': 1,  # Daily data - refresh daily
            'fundamentals': 7,  # Fundamentals - refresh weekly
            'news': 1,  # News - refresh daily
        }
    
    def _get_cache_key(self, symbol: str, data_type: str) -> str:
        """Generate cache key for symbol and data type."""
        key = f"{symbol}_{data_type}"
        return hashlib.md5(key.encode()).hexdigest()
    
    def _get_cache_path(self, symbol: str, data_type: str) -> str:
        """Get cache file path."""
        cache_key = self._get_cache_key(symbol, data_type)
        return os.path.join(self.cache_dir, f"{cache_key}.pkl")
    
    def _is_cache_valid(self, cache_path: str, expiry_days: int) -> bool:
        """Check if cache is still valid."""
        if not os.path.exists(cache_path):
            return False
        
        file_time = datetime.fromtimestamp(os.path.getmtime(cache_path))
        age = datetime.now() - file_time
        
        return age.days < expiry_days
    
    def get_ohlcv(self, symbol: str) -> Optional[pd.DataFrame]:
        """
        Get cached OHLCV data.
        
        Args:
            symbol: Stock symbol
        
        Returns:
            Cached DataFrame or None if not found/invalid
        """
        cache_path = self._get_cache_path(symbol, 'ohlcv')
        
        if self._is_cache_valid(cache_path, self.cache_expiry['ohlcv']):
            try:
                with open(cache_path, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                print(f"Error loading cache for {symbol}: {e}")
                return None
        
        return None
    
    def save_ohlcv(self, symbol: str, data: pd.DataFrame):
        """
        Save OHLCV data to cache.
        
        Args:
            symbol: Stock symbol
            data: DataFrame with OHLCV data
        """
        cache_path = self._get_cache_path(symbol, 'ohlcv')
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(data, f)
        except Exception as e:
            print(f"Error saving cache for {symbol}: {e}")
    
    def get_fundamentals(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get cached fundamentals data.
        
        Args:
            symbol: Stock symbol
        
        Returns:
            Cached fundamentals dict or None
        """
        cache_path = self._get_cache_path(symbol, 'fundamentals')
        
        if self._is_cache_valid(cache_path, self.cache_expiry['fundamentals']):
            try:
                with open(cache_path, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                print(f"Error loading fundamentals cache for {symbol}: {e}")
                return None
        
        return None
    
    def save_fundamentals(self, symbol: str, data: Dict[str, Any]):
        """
        Save fundamentals data to cache.
        
        Args:
            symbol: Stock symbol
            data: Fundamentals dictionary
        """
        cache_path = self._get_cache_path(symbol, 'fundamentals')
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(data, f)
        except Exception as e:
            print(f"Error saving fundamentals cache for {symbol}: {e}")
    
    def get_news(self, symbol: str) -> Optional[list]:
        """
        Get cached news data.
        
        Args:
            symbol: Stock symbol
        
        Returns:
            Cached news list or None
        """
        cache_path = self._get_cache_path(symbol, 'news')
        
        if self._is_cache_valid(cache_path, self.cache_expiry['news']):
            try:
                with open(cache_path, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                print(f"Error loading news cache for {symbol}: {e}")
                return None
        
        return None
    
    def save_news(self, symbol: str, data: list):
        """
        Save news data to cache.
        
        Args:
            symbol: Stock symbol
            data: News list
        """
        cache_path = self._get_cache_path(symbol, 'news')
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(data, f)
        except Exception as e:
            print(f"Error saving news cache for {symbol}: {e}")
    
    def clear_cache(self, symbol: Optional[str] = None):
        """
        Clear cache for a symbol or all symbols.
        
        Args:
            symbol: Stock symbol (None for all)
        """
        if symbol:
            # Clear specific symbol
            for data_type in ['ohlcv', 'fundamentals', 'news']:
                cache_path = self._get_cache_path(symbol, data_type)
                if os.path.exists(cache_path):
                    os.remove(cache_path)
        else:
            # Clear all
            for filename in os.listdir(self.cache_dir):
                if filename.endswith('.pkl'):
                    os.remove(os.path.join(self.cache_dir, filename))

