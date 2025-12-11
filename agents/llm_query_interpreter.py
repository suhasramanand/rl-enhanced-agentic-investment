"""
LLM Query Interpreter Agent
Uses Groq LLM to interpret user queries and extract structured instructions.
NOT used in RL training - only for user interaction.
"""

import json
import re
from typing import Dict, Any, Optional
from llm.groq_client import get_groq_client


class LLMQueryInterpreterAgent:
    """
    Agent that interprets natural language queries using Groq LLM.
    Converts user questions into structured instructions for the RL system.
    """
    
    def __init__(self):
        """Initialize LLM Query Interpreter Agent."""
        self.name = "LLMQueryInterpreterAgent"
        self.capabilities = ['interpret_query']
        self.groq_client = get_groq_client()
    
    def interpret_query(self, user_query: str) -> Dict[str, Any]:
        """
        Interpret user query and extract structured instructions.
        
        Args:
            user_query: Natural language query from user
        
        Returns:
            Structured dictionary with ticker, analysis parameters, etc.
        """
        # Try LLM interpretation first
        if self.groq_client.is_available():
            try:
                llm_result = self._interpret_with_llm(user_query)
                if llm_result:
                    return llm_result
            except Exception as e:
                print(f"LLM interpretation failed: {e}. Using fallback.")
        
        # Fallback to rule-based parsing
        return self._interpret_fallback(user_query)
    
    def _interpret_with_llm(self, user_query: str) -> Optional[Dict[str, Any]]:
        """Interpret query using Groq LLM."""
        # Enhanced prompt for agent orchestration
        prompt_template = """You are an AI assistant that transforms user investment questions into structured JSON instructions.

Extract the following information:
- ticker: Stock symbol (e.g., "AAPL", "NVDA")
- analysis_horizon: Time frame ("short-term", "1-3 months", "long-term")
- risk_tolerance: "low", "medium", or "high"
- needs_fundamentals: true/false (P/E, revenue, financials)
- needs_sentiment: true/false (news sentiment, social sentiment)
- needs_news: true/false (recent news articles)
- needs_technical_analysis: true/false (RSI, MACD, charts, trends)
- output_style: "professional", "simple", or "detailed"

Additionally, determine which agents should be called and in what order:
- agent_sequence: Array of agent actions to execute (e.g., ["FETCH_NEWS", "FETCH_FUNDAMENTALS", "RUN_TA_BASIC", "GENERATE_INSIGHT", "GENERATE_RECOMMENDATION"])

Available agent actions:
- FETCH_NEWS: Get recent news articles
- FETCH_FUNDAMENTALS: Get financial metrics (P/E, revenue, etc.)
- FETCH_SENTIMENT: Get sentiment analysis
- FETCH_MACRO: Get macroeconomic data
- RUN_TA_BASIC: Basic technical analysis (RSI, MA20)
- RUN_TA_ADVANCED: Advanced technical analysis (MACD, trends, MA50/200)
- GENERATE_INSIGHT: Synthesize insights from collected data
- GENERATE_RECOMMENDATION: Generate Buy/Hold/Sell recommendation

Return ONLY valid JSON. Do not add commentary or markdown formatting.

Example:
{
  "ticker": "AAPL",
  "analysis_horizon": "1-3 months",
  "risk_tolerance": "medium",
  "needs_fundamentals": true,
  "needs_sentiment": true,
  "needs_news": true,
  "needs_technical_analysis": true,
  "output_style": "professional",
  "agent_sequence": ["FETCH_NEWS", "FETCH_FUNDAMENTALS", "RUN_TA_BASIC", "GENERATE_INSIGHT", "GENERATE_RECOMMENDATION"]
}"""
        
        # Construct prompt
        prompt = f"{prompt_template}\n\nUser Query: {user_query}\n\nJSON Response:"
        
        # Call LLM
        response = self.groq_client.call_llm(
            prompt,
            model="llama-3.3-70b-versatile",  # Updated to latest model
            temperature=0.3,  # Lower temperature for more structured output
            max_tokens=500
        )
        
        if not response:
            return None
        
        # Parse JSON from response
        try:
            # Try to extract JSON from response (might have markdown code blocks)
            json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
            else:
                json_str = response
            
            result = json.loads(json_str)
            
            # Validate and normalize
            return self._validate_and_normalize(result)
            
        except json.JSONDecodeError as e:
            print(f"Failed to parse LLM JSON response: {e}")
            print(f"Response was: {response}")
            return None
    
    def _validate_and_normalize(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and normalize LLM output."""
        # Default values
        defaults = {
            "ticker": "NVDA",
            "analysis_horizon": "1-3 months",
            "risk_tolerance": "medium",
            "needs_fundamentals": True,
            "needs_sentiment": True,
            "needs_news": True,
            "needs_technical_analysis": True,
            "output_style": "professional",
            "agent_sequence": ["FETCH_NEWS", "FETCH_FUNDAMENTALS", "RUN_TA_BASIC", "GENERATE_INSIGHT", "GENERATE_RECOMMENDATION"]
        }
        
        # Normalize ticker (uppercase, remove spaces)
        if "ticker" in result:
            result["ticker"] = str(result["ticker"]).upper().strip()
        
        # Ensure boolean fields are boolean
        bool_fields = [
            "needs_fundamentals",
            "needs_sentiment",
            "needs_news",
            "needs_technical_analysis"
        ]
        for field in bool_fields:
            if field in result:
                result[field] = bool(result[field])
        
        # Validate agent_sequence
        valid_actions = [
            "FETCH_NEWS", "FETCH_FUNDAMENTALS", "FETCH_SENTIMENT", "FETCH_MACRO",
            "RUN_TA_BASIC", "RUN_TA_ADVANCED", "GENERATE_INSIGHT", "GENERATE_RECOMMENDATION", "STOP"
        ]
        
        if "agent_sequence" in result:
            # Filter to only valid actions
            result["agent_sequence"] = [
                action for action in result["agent_sequence"] 
                if action in valid_actions
            ]
            # Ensure we have at least some actions
            if not result["agent_sequence"]:
                result["agent_sequence"] = defaults["agent_sequence"]
        else:
            # Build agent_sequence based on needs
            agent_sequence = []
            if result.get("needs_news", True):
                agent_sequence.append("FETCH_NEWS")
            if result.get("needs_fundamentals", True):
                agent_sequence.append("FETCH_FUNDAMENTALS")
            if result.get("needs_sentiment", True):
                agent_sequence.append("FETCH_SENTIMENT")
            if result.get("needs_technical_analysis", True):
                agent_sequence.append("RUN_TA_BASIC")
                agent_sequence.append("RUN_TA_ADVANCED")
            agent_sequence.append("GENERATE_INSIGHT")
            agent_sequence.append("GENERATE_RECOMMENDATION")
            result["agent_sequence"] = agent_sequence
        
        # Merge with defaults
        normalized = {**defaults, **result}
        
        return normalized
    
    def _interpret_fallback(self, user_query: str) -> Dict[str, Any]:
        """
        Fallback rule-based query interpretation.
        Used when LLM is unavailable.
        """
        query_lower = user_query.lower()
        
        # Extract ticker (common patterns)
        ticker = "NVDA"  # default
        ticker_patterns = [
            r'\b([A-Z]{2,5})\b',  # Stock symbols
            r'(?:ticker|symbol|stock)\s+([A-Z]{2,5})',
            r'\b(nvda|aapl|tsla|jpm|xom)\b'
        ]
        
        for pattern in ticker_patterns:
            match = re.search(pattern, user_query, re.IGNORECASE)
            if match:
                ticker = match.group(1).upper()
                break
        
        # Determine analysis needs (default to all)
        needs_fundamentals = any(word in query_lower for word in ['fundamental', 'financial', 'pe', 'eps', 'revenue'])
        needs_sentiment = any(word in query_lower for word in ['sentiment', 'feeling', 'mood'])
        needs_news = any(word in query_lower for word in ['news', 'article', 'headline'])
        needs_technical_analysis = any(word in query_lower for word in ['technical', 'rsi', 'macd', 'chart', 'trend'])
        
        # If no specific needs mentioned, enable all
        if not any([needs_fundamentals, needs_sentiment, needs_news, needs_technical_analysis]):
            needs_fundamentals = needs_sentiment = needs_news = needs_technical_analysis = True
        
        # Build agent sequence based on needs
        agent_sequence = []
        if needs_news:
            agent_sequence.append("FETCH_NEWS")
        if needs_fundamentals:
            agent_sequence.append("FETCH_FUNDAMENTALS")
        if needs_sentiment:
            agent_sequence.append("FETCH_SENTIMENT")
        if needs_technical_analysis:
            agent_sequence.append("RUN_TA_BASIC")
            agent_sequence.append("RUN_TA_ADVANCED")
        agent_sequence.append("GENERATE_INSIGHT")
        agent_sequence.append("GENERATE_RECOMMENDATION")
        
        # Determine risk tolerance
        risk_tolerance = "medium"
        if any(word in query_lower for word in ['conservative', 'safe', 'low risk']):
            risk_tolerance = "low"
        elif any(word in query_lower for word in ['aggressive', 'high risk', 'risky']):
            risk_tolerance = "high"
        
        # Determine horizon
        analysis_horizon = "1-3 months"
        if any(word in query_lower for word in ['short', 'week', 'days']):
            analysis_horizon = "short-term"
        elif any(word in query_lower for word in ['long', 'year', 'years']):
            analysis_horizon = "long-term"
        
        # Output style
        output_style = "professional"
        if any(word in query_lower for word in ['simple', 'brief', 'quick']):
            output_style = "simple"
        elif any(word in query_lower for word in ['detailed', 'comprehensive', 'full']):
            output_style = "detailed"
        
        return {
            "ticker": ticker,
            "analysis_horizon": analysis_horizon,
            "risk_tolerance": risk_tolerance,
            "needs_fundamentals": needs_fundamentals,
            "needs_sentiment": needs_sentiment,
            "needs_news": needs_news,
            "needs_technical_analysis": needs_technical_analysis,
            "output_style": output_style,
            "agent_sequence": agent_sequence
        }

