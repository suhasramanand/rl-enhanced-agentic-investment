import React from 'react';
import { Lightbulb } from 'lucide-react';
import './Panel.css';

function InsightsPanel({ insights }) {
  return (
    <div className="panel">
      <div className="panel-header">
        <Lightbulb size={20} />
        <h3>AI-Generated Insights</h3>
      </div>
      <div className="panel-content">
        {insights && insights.length > 0 ? (
          <div className="insights-list">
            {insights.map((insight, idx) => (
              <div key={idx} className="insight-item">
                <div className="insight-bullet">💡</div>
                <div className="insight-text">{insight}</div>
              </div>
            ))}
          </div>
        ) : (
          <div className="empty">
            <p>No insights generated yet. Complete analysis to see AI-generated insights.</p>
          </div>
        )}
      </div>
    </div>
  );
}

export default InsightsPanel;

