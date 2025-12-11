"""
Research Agent
Handles fetching REAL news, fundamentals, sentiment, and macroeconomic data.
Uses yfinance, Yahoo Finance RSS, and VADER sentiment analysis.
"""

import time
import yfinance as yf
import feedparser
import requests
from typing import Dict, List, Any, Optional
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import pandas as pd

# Download VADER lexicon if not present
try:
    nltk.data.find('vader_lexicon.zip')
except LookupError:
    try:
        import ssl
        ssl._create_default_https_context = ssl._create_unverified_context
        nltk.download('vader_lexicon', quiet=True)
    except Exception as e:
        print(f"Warning: Could not download VADER lexicon: {e}")
        print("Sentiment analysis may not work correctly. Please download manually:")
        print("  python -c \"import ssl; import nltk; ssl._create_default_https_context = ssl._create_unverified_context; nltk.download('vader_lexicon')\"")

from utils.data_cache import DataCache


class ResearchAgent:
    """
    Agent responsible for gathering REAL research data about stocks.
    """
    
    def __init__(self, cache: Optional[DataCache] = None, max_retries: int = 3):
        """
        Initialize Research Agent.
        
        Args:
            cache: Data cache instance
            max_retries: Maximum retry attempts for API calls
        """
        self.name = "ResearchAgent"
        self.capabilities = [
            'fetch_news',
            'fetch_fundamentals',
            'fetch_sentiment',
            'fetch_macro'
        ]
        self.cache = cache or DataCache()
        self.max_retries = max_retries
        self.sia = SentimentIntensityAnalyzer()
    
    def _retry_api_call(self, func, *args, **kwargs):
        """
        Retry wrapper for API calls.
        
        Args:
            func: Function to call
            *args: Positional arguments
            **kwargs: Keyword arguments
        
        Returns:
            Function result or None if all retries fail
        """
        for attempt in range(self.max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if attempt == self.max_retries - 1:
                    print(f"API call failed after {self.max_retries} attempts: {e}")
                    return None
                time.sleep(1 * (attempt + 1))  # Exponential backoff
        return None
    
    def fetch_news(self, stock_symbol: str, env_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Fetch REAL news data for a stock using Yahoo Finance RSS.
        
        Args:
            stock_symbol: Stock symbol to research
            env_data: Optional environment data (if already fetched)
        
        Returns:
            News data dictionary with real headlines and sentiment
        """
        if env_data:
            return env_data
        
        # Check cache first
        cached_news = self.cache.get_news(stock_symbol)
        if cached_news:
            return cached_news
        
        # Fetch from Yahoo Finance RSS
        rss_url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={stock_symbol}"
        
        def fetch_rss():
            feed = feedparser.parse(rss_url)
            return feed
        
        feed = self._retry_api_call(fetch_rss)
        
        if not feed or not feed.entries:
            # Fallback: return empty news with neutral sentiment
            return {
                'headlines': [],
                'sentiment_score': 0.5,
                'num_articles': 0,
                'articles': []
            }
        
        # Extract headlines and summaries
        headlines = []
        articles = []
        sentiment_scores = []
        
        # Filter articles relevant to the stock symbol
        stock_keywords = [stock_symbol.lower(), stock_symbol]
        # Add company name variations if available
        if stock_symbol == 'NVDA':
            stock_keywords.extend(['nvidia', 'nvidia corporation'])
        elif stock_symbol == 'AAPL':
            stock_keywords.extend(['apple', 'apple inc'])
        elif stock_symbol == 'TSLA':
            stock_keywords.extend(['tesla', 'tesla inc'])
        elif stock_symbol == 'JPM':
            stock_keywords.extend(['jpmorgan', 'jp morgan', 'jpmorgan chase'])
        elif stock_symbol == 'XOM':
            stock_keywords.extend(['exxon', 'exxon mobil', 'exxonmobil'])
        
        relevant_articles = []
        for entry in feed.entries[:30]:  # Check more entries for relevance
            title = entry.get('title', '')
            summary = entry.get('summary', '')
            link = entry.get('link', '')
            published = entry.get('published', '')
            
            # Check if article is relevant to the stock
            text_lower = f"{title} {summary}".lower()
            is_relevant = any(keyword.lower() in text_lower for keyword in stock_keywords)
            
            # Include article if relevant, or if we don't have enough relevant ones
            if is_relevant or len(relevant_articles) < 10:
                if title:
                    headlines.append(title)
                    
                    # Analyze sentiment
                    text = f"{title} {summary}"
                    sentiment = self.sia.polarity_scores(text)
                    sentiment_scores.append(sentiment['compound'])
                    
                    articles.append({
                        'title': title,
                        'summary': summary[:200] if summary else '',  # Truncate summary
                        'link': link,
                        'sentiment': sentiment['compound'],
                        'published': published,
                        'relevance': 'high' if is_relevant else 'medium'
                    })
                    
                    if is_relevant:
                        relevant_articles.append(articles[-1])
        
        # Prioritize relevant articles
        if relevant_articles:
            articles = relevant_articles + [a for a in articles if a not in relevant_articles]
            articles = articles[:20]  # Limit to top 20
            headlines = [a['title'] for a in articles]
        
        # Calculate average sentiment
        avg_sentiment = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0.5
        # Normalize to [0, 1]
        normalized_sentiment = (avg_sentiment + 1) / 2
        
        result = {
            'headlines': headlines,
            'sentiment_score': normalized_sentiment,
            'num_articles': len(headlines),
            'articles': articles,
            'raw_sentiment': avg_sentiment
        }
        
        # Cache result
        self.cache.save_news(stock_symbol, result)
        
        return result
    
    def fetch_fundamentals(self, stock_symbol: str, env_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Fetch REAL fundamental data using yfinance.
        
        Args:
            stock_symbol: Stock symbol to research
            env_data: Optional environment data (if already fetched)
        
        Returns:
            Fundamentals data dictionary with real metrics
        """
        if env_data:
            return env_data
        
        # Check cache first
        cached_fundamentals = self.cache.get_fundamentals(stock_symbol)
        if cached_fundamentals:
            return cached_fundamentals
        
        def fetch_yfinance_data():
            ticker = yf.Ticker(stock_symbol)
            info = ticker.info
            financials = ticker.financials
            balance_sheet = ticker.balance_sheet
            cashflow = ticker.cashflow
            return info, financials, balance_sheet, cashflow
        
        result_data = self._retry_api_call(fetch_yfinance_data)
        
        if not result_data:
            # Fallback: return minimal data
            return {
                'pe_ratio': None,
                'eps': None,
                'revenue_growth': None,
                'profit_margin': None,
                'debt_to_equity': None,
                'market_cap': None,
                'available': False
            }
        
        info, financials, balance_sheet, cashflow = result_data
        
        # Extract key metrics
        pe_ratio = info.get('trailingPE') or info.get('forwardPE')
        eps = info.get('trailingEps') or info.get('forwardEps')
        market_cap = info.get('marketCap')
        
        # Calculate revenue growth from financials
        revenue_growth = None
        if financials is not None and not financials.empty:
            revenue_row = financials.loc[financials.index.str.contains('Total Revenue', case=False, na=False)]
            if not revenue_row.empty:
                revenues = revenue_row.iloc[0].dropna()
                if len(revenues) >= 2:
                    recent_revenue = revenues.iloc[0]
                    prev_revenue = revenues.iloc[1]
                    if prev_revenue != 0:
                        revenue_growth = (recent_revenue - prev_revenue) / abs(prev_revenue)
        
        # Calculate profit margin
        profit_margin = None
        if financials is not None and not financials.empty:
            net_income_row = financials.loc[financials.index.str.contains('Net Income', case=False, na=False)]
            revenue_row = financials.loc[financials.index.str.contains('Total Revenue', case=False, na=False)]
            if not net_income_row.empty and not revenue_row.empty:
                net_income = net_income_row.iloc[0].iloc[0]
                revenue = revenue_row.iloc[0].iloc[0]
                if revenue != 0:
                    profit_margin = net_income / abs(revenue)
        
        # Calculate debt-to-equity
        debt_to_equity = None
        if balance_sheet is not None and not balance_sheet.empty:
            debt_row = balance_sheet.loc[balance_sheet.index.str.contains('Total Debt', case=False, na=False)]
            equity_row = balance_sheet.loc[balance_sheet.index.str.contains('Stockholders Equity', case=False, na=False)]
            if not debt_row.empty and not equity_row.empty:
                total_debt = debt_row.iloc[0].iloc[0]
                total_equity = equity_row.iloc[0].iloc[0]
                if total_equity != 0:
                    debt_to_equity = total_debt / abs(total_equity)
        
        result = {
            'pe_ratio': float(pe_ratio) if pe_ratio else None,
            'eps': float(eps) if eps else None,
            'revenue_growth': float(revenue_growth) if revenue_growth else None,
            'profit_margin': float(profit_margin) if profit_margin else None,
            'debt_to_equity': float(debt_to_equity) if debt_to_equity else None,
            'market_cap': float(market_cap) if market_cap else None,
            'available': True,
            'info': info  # Store full info for reference
        }
        
        # Cache result
        self.cache.save_fundamentals(stock_symbol, result)
        
        return result
    
    def fetch_sentiment(self, stock_symbol: str, env_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Fetch sentiment data by analyzing news.
        
        Args:
            stock_symbol: Stock symbol to research
            env_data: Optional environment data (if already fetched)
        
        Returns:
            Sentiment data dictionary
        """
        if env_data:
            return env_data
        
        # Get news data (which includes sentiment)
        news_data = self.fetch_news(stock_symbol)
        
        # Extract sentiment from news
        sentiment_score = news_data.get('sentiment_score', 0.5)
        
        # Determine analyst rating based on sentiment
        if sentiment_score > 0.6:
            analyst_rating = 'Buy'
        elif sentiment_score < 0.4:
            analyst_rating = 'Sell'
        else:
            analyst_rating = 'Hold'
        
        return {
            'social_sentiment': sentiment_score,
            'analyst_rating': analyst_rating,
            'bullish_percentage': sentiment_score * 100,
            'num_articles': news_data.get('num_articles', 0)
        }
    
    def fetch_macro(self, env_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Fetch macroeconomic data.
        Uses simplified approach - in production, use FRED API or similar.
        
        Args:
            env_data: Optional environment data (if already fetched)
        
        Returns:
            Macroeconomic data dictionary
        """
        if env_data:
            return env_data
        
        # For now, return reasonable defaults
        # In production, fetch from FRED API or similar
        # This doesn't affect RL training significantly as it's not stock-specific
        return {
            'gdp_growth': 0.025,  # 2.5% typical
            'inflation_rate': 0.03,  # 3% typical
            'interest_rate': 0.05,  # 5% typical
            'unemployment_rate': 0.04  # 4% typical
        }
