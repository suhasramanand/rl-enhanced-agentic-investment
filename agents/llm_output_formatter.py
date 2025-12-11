"""
LLM Output Formatter Agent
Uses Groq LLM to format raw agent outputs into polished financial reports.
NOT used in RL training - only for final output formatting.
"""

import json
from typing import Dict, Any, Optional, List
from llm.groq_client import get_groq_client


class LLMOutputFormatterAgent:
    """
    Agent that formats raw analysis outputs using Groq LLM.
    Creates professional financial reports from agent outputs.
    """
    
    def __init__(self):
        """Initialize LLM Output Formatter Agent."""
        self.name = "LLMOutputFormatterAgent"
        self.capabilities = ['format_output']
        self.groq_client = get_groq_client()
    
    def format_output(
        self,
        stock_symbol: str,
        research_data: Optional[Dict[str, Any]] = None,
        technical_data: Optional[Dict[str, Any]] = None,
        insights: Optional[List[str]] = None,
        recommendation: Optional[str] = None,
        confidence: Optional[float] = None,
        output_style: str = "professional"
    ) -> str:
        """
        Format raw outputs into polished financial report.
        
        Args:
            stock_symbol: Stock symbol analyzed
            research_data: Output from ResearchAgent
            technical_data: Output from TechnicalAnalysisAgent
            insights: List of insights from InsightAgent
            recommendation: Buy/Hold/Sell recommendation
            confidence: Confidence score
            output_style: Output style (professional, simple, detailed)
        
        Returns:
            Formatted markdown report
        """
        # Try LLM formatting first
        if self.groq_client.is_available():
            try:
                llm_result = self._format_with_llm(
                    stock_symbol, research_data, technical_data,
                    insights, recommendation, confidence, output_style
                )
                if llm_result:
                    return llm_result
            except Exception as e:
                print(f"LLM formatting failed: {e}. Using fallback.")
        
        # Fallback to rule-based formatting
        return self._format_fallback(
            stock_symbol, research_data, technical_data,
            insights, recommendation, confidence, output_style
        )
    
    def _format_with_llm(
        self,
        stock_symbol: str,
        research_data: Optional[Dict[str, Any]],
        technical_data: Optional[Dict[str, Any]],
        insights: Optional[List[str]],
        recommendation: Optional[str],
        confidence: Optional[float],
        output_style: str
    ) -> Optional[str]:
        """Format output using Groq LLM."""
        # Load prompt template
        try:
            prompt_path = "prompts/output_formatter_prompt.txt"
            with open(prompt_path, 'r') as f:
                prompt_template = f.read()
        except FileNotFoundError:
            prompt_template = """You are a financial analysis formatter.
Create a clear, concise, professional stock analysis report.
Include: Summary, Fundamentals, Technicals, Sentiment, Risks, and Final Recommendation.

IMPORTANT FORMATTING RULES:
- For news articles: Summarize the key points from the articles provided. Do NOT show raw Python dictionaries or lists.
- Format news as: "Key News: [summary of top 3-5 articles with their main points]"
- Use bullet points for clarity
- Do NOT hallucinate new data. Use only what is provided.
- Return clean, readable markdown."""
        
        # Prepare data summary
        data_summary = self._prepare_data_summary(
            stock_symbol, research_data, technical_data,
            insights, recommendation, confidence
        )
        
        # Construct prompt
        prompt = f"""{prompt_template}

Stock Symbol: {stock_symbol}
Output Style: {output_style}

Raw Analysis Data:
{data_summary}

Format this into a professional financial analysis report:"""
        
        # Call LLM
        response = self.groq_client.call_llm(
            prompt,
            model="llama-3.3-70b-versatile",  # Updated to latest model
            temperature=0.5,
            max_tokens=2000
        )
        
        return response
    
    def _prepare_data_summary(
        self,
        stock_symbol: str,
        research_data: Optional[Dict[str, Any]],
        technical_data: Optional[Dict[str, Any]],
        insights: Optional[List[str]],
        recommendation: Optional[str],
        confidence: Optional[float]
    ) -> str:
        """Prepare data summary for LLM."""
        summary_parts = [f"Stock: {stock_symbol}\n"]
        
        # Research data
        if research_data:
            summary_parts.append("\n=== Research Data ===")
            if research_data.get('news'):
                news = research_data['news']
                num_articles = news.get('num_articles', len(news.get('articles', [])))
                summary_parts.append(f"News: {num_articles} articles")
                summary_parts.append(f"News Sentiment: {news.get('sentiment_score', news.get('sentiment', 0)):.2f}")
                
                # Format news articles as a clean summary
                articles = news.get('articles', [])
                if articles:
                    summary_parts.append("\nTop News Articles:")
                    for i, article in enumerate(articles[:5], 1):  # Top 5 articles
                        title = article.get('title', 'No title')
                        summary = article.get('summary', '')
                        sentiment = article.get('sentiment', 0)
                        sentiment_label = "Positive" if sentiment > 0.3 else "Negative" if sentiment < -0.3 else "Neutral"
                        summary_parts.append(f"  {i}. {title}")
                        if summary:
                            summary_parts.append(f"     Summary: {summary[:150]}...")
                        summary_parts.append(f"     Sentiment: {sentiment_label} ({sentiment:.2f})")
                
                # Also include headlines if available
                if news.get('headlines') and not articles:
                    summary_parts.append("\nKey Headlines:")
                    for i, headline in enumerate(news['headlines'][:5], 1):
                        summary_parts.append(f"  {i}. {headline}")
            
            if research_data.get('fundamentals'):
                fund = research_data['fundamentals']
                if fund.get('pe_ratio'):
                    summary_parts.append(f"P/E Ratio: {fund['pe_ratio']:.2f}")
                if fund.get('revenue_growth') is not None:
                    summary_parts.append(f"Revenue Growth: {fund['revenue_growth']*100:.1f}%")
                if fund.get('profit_margin') is not None:
                    summary_parts.append(f"Profit Margin: {fund['profit_margin']*100:.1f}%")
            
            if research_data.get('sentiment'):
                sent = research_data['sentiment']
                summary_parts.append(f"Social Sentiment: {sent.get('social_sentiment', 0):.2f}")
                summary_parts.append(f"Analyst Rating: {sent.get('analyst_rating', 'N/A')}")
        
        # Technical data
        if technical_data:
            summary_parts.append("\n=== Technical Analysis ===")
            if technical_data.get('rsi') is not None:
                summary_parts.append(f"RSI: {technical_data['rsi']:.2f}")
            if technical_data.get('trend'):
                summary_parts.append(f"Trend: {technical_data['trend']}")
            if technical_data.get('macd_signal') is not None:
                summary_parts.append(f"MACD Signal: {technical_data['macd_signal']:.2f}")
            if technical_data.get('ma50') and technical_data.get('ma200'):
                summary_parts.append(f"MA50: ${technical_data['ma50']:.2f}, MA200: ${technical_data['ma200']:.2f}")
        
        # Insights
        if insights:
            summary_parts.append("\n=== Insights ===")
            for insight in insights:
                summary_parts.append(f"- {insight}")
        
        # Recommendation
        if recommendation:
            summary_parts.append(f"\n=== Recommendation ===")
            summary_parts.append(f"Action: {recommendation}")
            if confidence is not None:
                summary_parts.append(f"Confidence: {confidence*100:.1f}%")
        
        return "\n".join(summary_parts)
    
    def _format_fallback(
        self,
        stock_symbol: str,
        research_data: Optional[Dict[str, Any]],
        technical_data: Optional[Dict[str, Any]],
        insights: Optional[List[str]],
        recommendation: Optional[str],
        confidence: Optional[float],
        output_style: str
    ) -> str:
        """
        Fallback rule-based formatting.
        Used when LLM is unavailable.
        """
        report = [f"# Stock Analysis Report: {stock_symbol}\n"]
        
        # Summary
        report.append("## Summary")
        report.append(f"Analysis of {stock_symbol} based on multiple data sources.")
        if recommendation:
            report.append(f"Recommendation: **{recommendation}** (Confidence: {confidence*100:.1f}%)\n")
        else:
            report.append("Recommendation: Pending\n")
        
        # Fundamentals
        if research_data and research_data.get('fundamentals'):
            report.append("## Fundamentals")
            fund = research_data['fundamentals']
            if fund.get('pe_ratio'):
                report.append(f"- **P/E Ratio:** {fund['pe_ratio']:.2f}")
            if fund.get('revenue_growth') is not None:
                report.append(f"- **Revenue Growth:** {fund['revenue_growth']*100:.1f}%")
            if fund.get('profit_margin') is not None:
                report.append(f"- **Profit Margin:** {fund['profit_margin']*100:.1f}%")
            report.append("")
        
        # Technical Analysis
        if technical_data:
            report.append("## Technical Analysis")
            if technical_data.get('rsi') is not None:
                rsi = technical_data['rsi']
                rsi_status = "Oversold" if rsi < 30 else "Overbought" if rsi > 70 else "Neutral"
                report.append(f"- **RSI:** {rsi:.2f} ({rsi_status})")
            if technical_data.get('trend'):
                report.append(f"- **Trend:** {technical_data['trend']}")
            if technical_data.get('macd_signal') is not None:
                report.append(f"- **MACD Signal:** {technical_data['macd_signal']:.2f}")
            report.append("")
        
        # Sentiment & News
        if research_data:
            report.append("## Sentiment & News")
            if research_data.get('news'):
                news = research_data['news']
                num_articles = news.get('num_articles', len(news.get('articles', [])))
                sentiment_score = news.get('sentiment_score', news.get('sentiment', 0))
                report.append(f"- **News Articles:** {num_articles} articles analyzed")
                report.append(f"- **News Sentiment:** {sentiment_score*100:.1f}%")
                
                # Format news articles summary
                articles = news.get('articles', [])
                if articles:
                    report.append("\n**Key News Highlights:**")
                    for i, article in enumerate(articles[:5], 1):  # Top 5 articles
                        title = article.get('title', 'No title')
                        summary = article.get('summary', '')
                        sentiment = article.get('sentiment', 0)
                        sentiment_icon = "📈" if sentiment > 0.3 else "📉" if sentiment < -0.3 else "➡️"
                        
                        report.append(f"\n{i}. **{title}** {sentiment_icon}")
                        if summary:
                            # Truncate summary to 200 chars
                            clean_summary = summary[:200] + "..." if len(summary) > 200 else summary
                            report.append(f"   {clean_summary}")
                        if article.get('link'):
                            report.append(f"   [Read more]({article['link']})")
                
                # Fallback to headlines if articles not available
                elif news.get('headlines'):
                    report.append("\n**Key Headlines:**")
                    for i, headline in enumerate(news['headlines'][:5], 1):
                        report.append(f"{i}. {headline}")
            
            if research_data.get('sentiment'):
                sent = research_data['sentiment']
                report.append(f"\n- **Social Sentiment:** {sent.get('social_sentiment', 0)*100:.1f}%")
                report.append(f"- **Analyst Rating:** {sent.get('analyst_rating', 'N/A')}")
            report.append("")
        
        # Insights
        if insights:
            report.append("## Key Insights")
            for insight in insights:
                report.append(f"- {insight}")
            report.append("")
        
        # Recommendation
        if recommendation:
            report.append("## Final Recommendation")
            report.append(f"**{recommendation}** (Confidence: {confidence*100:.1f}%)")
            report.append("")
            report.append("### Justification")
            if recommendation == 'Buy':
                report.append("Positive signals from fundamentals, technical analysis, and sentiment suggest buying opportunity.")
            elif recommendation == 'Sell':
                report.append("Negative signals from fundamentals, technical analysis, and sentiment suggest selling.")
            else:
                report.append("Mixed signals suggest maintaining current position.")
        
        return "\n".join(report)

