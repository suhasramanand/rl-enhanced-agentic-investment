import React from 'react';
import { Newspaper, TrendingUp, BarChart3 } from 'lucide-react';
import './Panel.css';

function ResearchPanel({ data }) {
  if (!data) {
    return (
      <div className="panel">
        <div className="panel-header">
          <Newspaper size={20} />
          <h3>Research Data</h3>
        </div>
        <div className="panel-content empty">
          <p>No research data yet. Start analysis to see results.</p>
        </div>
      </div>
    );
  }

  return (
    <div className="panel">
      <div className="panel-header">
        <Newspaper size={20} />
        <h3>Research Data</h3>
      </div>
      <div className="panel-content">
        {data.news && (
          <div className="data-section">
            <h4><Newspaper size={16} /> News</h4>
            <div className="data-item">
              <span className="label">Articles:</span>
              <span className="value">{data.news.articles}</span>
            </div>
            <div className="data-item">
              <span className="label">Sentiment:</span>
              <span className="value sentiment" data-sentiment={data.news.sentiment}>
                {(data.news.sentiment * 100).toFixed(1)}%
              </span>
            </div>
            {data.news.headlines && data.news.headlines.length > 0 && (
              <div className="headlines">
                <strong>Recent Headlines:</strong>
                <ul>
                  {data.news.headlines.slice(0, 3).map((headline, idx) => (
                    <li key={idx}>{headline.substring(0, 80)}...</li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        )}

        {data.fundamentals && (
          <div className="data-section">
            <h4><TrendingUp size={16} /> Fundamentals</h4>
            {data.fundamentals.pe_ratio && (
              <div className="data-item">
                <span className="label">P/E Ratio:</span>
                <span className="value">{data.fundamentals.pe_ratio.toFixed(2)}</span>
              </div>
            )}
            {data.fundamentals.revenue_growth !== null && (
              <div className="data-item">
                <span className="label">Revenue Growth:</span>
                <span className="value positive">
                  {data.fundamentals.revenue_growth > 0 ? '+' : ''}
                  {(data.fundamentals.revenue_growth * 100).toFixed(1)}%
                </span>
              </div>
            )}
            {data.fundamentals.profit_margin !== null && (
              <div className="data-item">
                <span className="label">Profit Margin:</span>
                <span className="value">{(data.fundamentals.profit_margin * 100).toFixed(1)}%</span>
              </div>
            )}
          </div>
        )}

        {data.sentiment && (
          <div className="data-section">
            <h4><BarChart3 size={16} /> Sentiment</h4>
            <div className="data-item">
              <span className="label">Social Sentiment:</span>
              <span className="value">{(data.sentiment.social_sentiment * 100).toFixed(1)}%</span>
            </div>
            <div className="data-item">
              <span className="label">Analyst Rating:</span>
              <span className="value rating">{data.sentiment.analyst_rating}</span>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default ResearchPanel;

