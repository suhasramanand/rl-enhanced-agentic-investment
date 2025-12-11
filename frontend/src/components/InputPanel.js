import React, { useState } from 'react';
import { Search, Play } from 'lucide-react';
import './InputPanel.css';

function InputPanel({ onAnalyze, isAnalyzing }) {
  const [company, setCompany] = useState('NVDA');

  const handleSubmit = (e) => {
    e.preventDefault();
    if (company.trim() && !isAnalyzing) {
      onAnalyze(company.trim());
    }
  };

  return (
    <div className="input-panel">
      <form onSubmit={handleSubmit} className="input-form">
        <div className="input-wrapper">
          <Search className="search-icon" size={20} />
          <input
            type="text"
            value={company}
            onChange={(e) => setCompany(e.target.value)}
            placeholder="Enter company name or stock symbol (e.g., NVDA, Apple, Tesla)"
            className="company-input"
            disabled={isAnalyzing}
          />
          <button
            type="submit"
            className="analyze-button"
            disabled={isAnalyzing || !company.trim()}
          >
            <Play size={18} />
            {isAnalyzing ? 'Analyzing...' : 'Analyze'}
          </button>
        </div>
        <div className="hint">
          Supported: NVDA (Nvidia), AAPL (Apple), TSLA (Tesla), JPM (JPMorgan), XOM (ExxonMobil)
        </div>
      </form>
    </div>
  );
}

export default InputPanel;

