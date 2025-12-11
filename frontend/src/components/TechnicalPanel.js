import React from 'react';
import { Activity, TrendingUp, TrendingDown, Minus } from 'lucide-react';
import './Panel.css';

function TechnicalPanel({ data }) {
  if (!data) {
    return (
      <div className="panel">
        <div className="panel-header">
          <Activity size={20} />
          <h3>Technical Analysis</h3>
        </div>
        <div className="panel-content empty">
          <p>No technical analysis yet. Start analysis to see results.</p>
        </div>
      </div>
    );
  }

  const getTrendIcon = (trend) => {
    if (trend === 'uptrend') return <TrendingUp size={16} className="trend-icon positive" />;
    if (trend === 'downtrend') return <TrendingDown size={16} className="trend-icon negative" />;
    return <Minus size={16} className="trend-icon neutral" />;
  };

  return (
    <div className="panel">
      <div className="panel-header">
        <Activity size={20} />
        <h3>Technical Analysis</h3>
      </div>
      <div className="panel-content">
        {data.rsi !== undefined && (
          <div className="data-item">
            <span className="label">RSI:</span>
            <span className={`value ${data.rsi < 30 ? 'oversold' : data.rsi > 70 ? 'overbought' : ''}`}>
              {data.rsi.toFixed(2)}
            </span>
          </div>
        )}

        {data.current_price && (
          <div className="data-item">
            <span className="label">Current Price:</span>
            <span className="value">${data.current_price.toFixed(2)}</span>
          </div>
        )}

        {data.ma20 !== undefined && (
          <div className="data-item">
            <span className="label">MA20:</span>
            <span className="value">${data.ma20.toFixed(2)}</span>
          </div>
        )}

        {data.ma50 !== undefined && (
          <div className="data-item">
            <span className="label">MA50:</span>
            <span className="value">${data.ma50.toFixed(2)}</span>
          </div>
        )}

        {data.ma200 !== undefined && (
          <div className="data-item">
            <span className="label">MA200:</span>
            <span className="value">${data.ma200.toFixed(2)}</span>
          </div>
        )}

        {data.trend && (
          <div className="data-item trend-item">
            <span className="label">Trend:</span>
            <span className="value trend-value">
              {getTrendIcon(data.trend)}
              {data.trend}
            </span>
          </div>
        )}

        {data.macd_signal !== undefined && (
          <div className="data-item">
            <span className="label">MACD Signal:</span>
            <span className={`value ${data.macd_signal > 0 ? 'positive' : 'negative'}`}>
              {data.macd_signal.toFixed(2)}
            </span>
          </div>
        )}
      </div>
    </div>
  );
}

export default TechnicalPanel;

