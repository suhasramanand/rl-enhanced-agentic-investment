import React from 'react';
import { Target, TrendingUp, TrendingDown, Minus } from 'lucide-react';
import './Panel.css';

function RecommendationPanel({ recommendation }) {
  if (!recommendation) {
    return (
      <div className="panel">
        <div className="panel-header">
          <Target size={20} />
          <h3>Investment Recommendation</h3>
        </div>
        <div className="panel-content empty">
          <p>No recommendation yet. Complete analysis to see recommendation.</p>
        </div>
      </div>
    );
  }

  const rec = recommendation.recommendation;
  const confidence = recommendation.confidence || 0.5;

  const getRecIcon = () => {
    if (rec === 'Buy') return <TrendingUp size={24} />;
    if (rec === 'Sell') return <TrendingDown size={24} />;
    return <Minus size={24} />;
  };

  const getRecClass = () => {
    if (rec === 'Buy') return 'buy';
    if (rec === 'Sell') return 'sell';
    return 'hold';
  };

  return (
    <div className="panel recommendation-panel">
      <div className="panel-header">
        <Target size={20} />
        <h3>Investment Recommendation</h3>
      </div>
      <div className="panel-content">
        <div className={`recommendation-badge ${getRecClass()}`}>
          <div className="rec-icon">{getRecIcon()}</div>
          <div className="rec-text">{rec}</div>
        </div>

        <div className="confidence-section">
          <div className="confidence-label">Confidence</div>
          <div className="confidence-bar-container">
            <div
              className={`confidence-bar ${getRecClass()}`}
              style={{ width: `${confidence * 100}%` }}
            >
              <span className="confidence-text">{(confidence * 100).toFixed(1)}%</span>
            </div>
          </div>
        </div>

        <div className="recommendation-note">
          {rec === 'Buy' && 'Based on positive signals from fundamentals, technical analysis, and sentiment.'}
          {rec === 'Sell' && 'Based on negative signals from fundamentals, technical analysis, and sentiment.'}
          {rec === 'Hold' && 'Mixed signals suggest maintaining current position.'}
        </div>
      </div>
    </div>
  );
}

export default RecommendationPanel;

