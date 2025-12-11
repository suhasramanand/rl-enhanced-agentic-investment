import React from 'react';
import { Loader2, CheckCircle, AlertCircle } from 'lucide-react';
import './StatusBar.css';

function StatusBar({ status, isAnalyzing }) {
  const getIcon = () => {
    if (isAnalyzing) return <Loader2 className="spinning" size={18} />;
    if (status.includes('Error')) return <AlertCircle size={18} />;
    if (status.includes('complete')) return <CheckCircle size={18} />;
    return null;
  };

  return (
    <div className={`status-bar ${isAnalyzing ? 'analyzing' : ''}`}>
      <div className="status-content">
        {getIcon()}
        <span className="status-text">{status || 'Ready to analyze'}</span>
      </div>
      {isAnalyzing && (
        <div className="progress-indicator">
          <div className="progress-bar"></div>
        </div>
      )}
    </div>
  );
}

export default StatusBar;

