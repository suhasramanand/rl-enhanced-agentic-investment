// Main application JavaScript with Visual Agent Network

let eventSource = null;
let network = null;
let nodes = null;
let edges = null;
let agentNodes = {};
let currentStep = 0;

// Agent definitions
const AGENTS = {
    'ControllerAgent': { id: 'controller', label: '🎮 Controller\nAgent', color: '#9b59b6', shape: 'box' },
    'ResearchAgent': { id: 'research', label: '📰 Research\nAgent', color: '#3498db', shape: 'ellipse' },
    'TechnicalAnalysisAgent': { id: 'technical', label: '📊 Technical\nAnalysis', color: '#e74c3c', shape: 'ellipse' },
    'InsightAgent': { id: 'insight', label: '💡 Insight\nAgent', color: '#f39c12', shape: 'ellipse' },
    'RecommendationAgent': { id: 'recommendation', label: '🎯 Recommendation\nAgent', color: '#27ae60', shape: 'ellipse' },
    'EvaluatorAgent': { id: 'evaluator', label: '📈 Evaluator\nAgent', color: '#16a085', shape: 'ellipse' }
};

function initializeNetwork() {
    // Create nodes
    nodes = new vis.DataSet([
        { id: 'controller', label: '🎮 Controller\nAgent', color: { background: '#9b59b6', border: '#7d3c98' }, shape: 'box', font: { size: 16, color: 'white' }, x: 0, y: 0, fixed: { x: true, y: true } },
        { id: 'research', label: '📰 Research\nAgent', color: { background: '#3498db', border: '#2980b9' }, shape: 'ellipse', font: { size: 14, color: 'white' }, x: -200, y: -150, fixed: { x: true, y: true } },
        { id: 'technical', label: '📊 Technical\nAnalysis', color: { background: '#e74c3c', border: '#c0392b' }, shape: 'ellipse', font: { size: 14, color: 'white' }, x: 200, y: -150, fixed: { x: true, y: true } },
        { id: 'insight', label: '💡 Insight\nAgent', color: { background: '#f39c12', border: '#d68910' }, shape: 'ellipse', font: { size: 14, color: 'white' }, x: -200, y: 150, fixed: { x: true, y: true } },
        { id: 'recommendation', label: '🎯 Recommendation\nAgent', color: { background: '#27ae60', border: '#229954' }, shape: 'ellipse', font: { size: 14, color: 'white' }, x: 200, y: 150, fixed: { x: true, y: true } },
        { id: 'evaluator', label: '📈 Evaluator\nAgent', color: { background: '#16a085', border: '#138d75' }, shape: 'ellipse', font: { size: 14, color: 'white' }, x: 0, y: 300, fixed: { x: true, y: true } }
    ]);

    // Create initial edges (static connections)
    edges = new vis.DataSet([
        { from: 'controller', to: 'research', arrows: 'to', color: { color: '#95a5a6' }, width: 2 },
        { from: 'controller', to: 'technical', arrows: 'to', color: { color: '#95a5a6' }, width: 2 },
        { from: 'controller', to: 'insight', arrows: 'to', color: { color: '#95a5a6' }, width: 2 },
        { from: 'controller', to: 'recommendation', arrows: 'to', color: { color: '#95a5a6' }, width: 2 },
        { from: 'research', to: 'insight', arrows: 'to', color: { color: '#95a5a6' }, width: 1, dashes: true },
        { from: 'technical', to: 'insight', arrows: 'to', color: { color: '#95a5a6' }, width: 1, dashes: true },
        { from: 'insight', to: 'recommendation', arrows: 'to', color: { color: '#95a5a6' }, width: 1, dashes: true },
        { from: 'recommendation', to: 'evaluator', arrows: 'to', color: { color: '#95a5a6' }, width: 2 }
    ]);

    // Network options
    const options = {
        nodes: {
            borderWidth: 3,
            shadow: true,
            font: {
                size: 14,
                face: 'Arial'
            }
        },
        edges: {
            width: 2,
            shadow: true,
            smooth: {
                type: 'continuous',
                roundness: 0.5
            }
        },
        physics: {
            enabled: false
        },
        interaction: {
            dragNodes: false,
            dragView: true,
            zoomView: true
        }
    };

    const container = document.getElementById('agent-network');
    const data = { nodes: nodes, edges: edges };
    network = new vis.Network(container, data, options);
}

function highlightAgent(agentName, action) {
    const agentInfo = AGENTS[agentName];
    if (!agentInfo) return;

    const agentId = agentInfo.id;
    
    // Highlight the node
    nodes.update({
        id: agentId,
        color: {
            background: agentInfo.color,
            border: '#2c3e50'
        },
        borderWidth: 4
    });

    // Highlight the edge from controller
    const edgeId = `controller-${agentId}`;
    edges.update({
        from: 'controller',
        to: agentId,
        color: { color: agentInfo.color },
        width: 4,
        label: action || ''
    });

    // Reset after 1 second
    setTimeout(() => {
        nodes.update({
            id: agentId,
            borderWidth: 3
        });
        edges.update({
            from: 'controller',
            to: agentId,
            color: { color: '#95a5a6' },
            width: 2,
            label: ''
        });
    }, 1000);
}

function startAnalysis() {
    const companyInput = document.getElementById('company-input');
    const analyzeBtn = document.getElementById('analyze-btn');
    const company = companyInput.value.trim();
    
    if (!company) {
        alert('Please enter a company name or stock symbol');
        return;
    }
    
    // Disable button
    analyzeBtn.disabled = true;
    analyzeBtn.textContent = 'Analyzing...';
    
    // Reset UI
    resetUI();
    
    // Show status
    document.getElementById('status-section').classList.remove('hidden');
    updateStatus(`Starting analysis for ${company}...`);
    
    // Start SSE connection
    startSSE(company);
}

function startSSE(company) {
    // Close existing connection
    if (eventSource) {
        eventSource.close();
    }
    
    const url = `/api/analyze-stream`;
    
    fetch(url, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ company: company })
    })
    .then(response => {
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        
        function readStream() {
            reader.read().then(({ done, value }) => {
                if (done) {
                    return;
                }
                
                const chunk = decoder.decode(value);
                const lines = chunk.split('\n');
                
                for (const line of lines) {
                    if (line.startsWith('data: ')) {
                        try {
                            const data = JSON.parse(line.substring(6));
                            handleUpdate(data);
                        } catch (e) {
                            console.error('Error parsing SSE data:', e);
                        }
                    }
                }
                
                readStream();
            });
        }
        
        readStream();
    })
    .catch(error => {
        console.error('Error:', error);
        showError('Failed to start analysis. Please try again.');
        resetButton();
    });
}

function handleUpdate(data) {
    console.log('Update:', data);
    
    switch (data.type) {
        case 'llm_interpretation':
            // Show LLM interpretation results
            if (data.agent_sequence && data.agent_sequence.length > 0) {
                const sequenceText = data.agent_sequence.join(' → ');
                updateStatus(`🤖 LLM Agent Sequence: ${sequenceText}`);
                // Add visual indicator for LLM-managed sequence
                showLLMSequence(data.agent_sequence);
            }
            break;
            
        case 'status':
            updateStatus(data.message);
            if (data.agent_sequence && data.agent_sequence.length > 0) {
                showLLMSequence(data.agent_sequence);
            }
            break;
            
        case 'agent_call':
            highlightAgent(data.agent, data.action);
            const sourceLabel = data.source === 'llm' ? '🤖 LLM-Managed' : '🎯 RL-Managed';
            updateStatus(`${sourceLabel} → ${data.agent} executing: ${data.action}`);
            break;
            
        case 'action_result':
            updateOutputs(data.action, data.result);
            break;
            
        case 'complete':
            handleComplete(data.results);
            break;
            
        case 'error':
            showError(data.message);
            resetButton();
            break;
    }
}

function showLLMSequence(sequence) {
    // Create or update a visual indicator for the LLM sequence
    const statusSection = document.getElementById('status-section');
    let sequenceDiv = document.getElementById('llm-sequence-indicator');
    
    if (!sequenceDiv) {
        sequenceDiv = document.createElement('div');
        sequenceDiv.id = 'llm-sequence-indicator';
        sequenceDiv.className = 'llm-sequence-indicator';
        sequenceDiv.style.cssText = 'margin-top: 10px; padding: 10px; background: #e8f5e9; border-radius: 5px; border-left: 4px solid #4caf50;';
        statusSection.appendChild(sequenceDiv);
    }
    
    const sequenceHTML = sequence.map((action, idx) => {
        const agentName = getAgentNameFromAction(action);
        return `<span style="padding: 5px 10px; margin: 0 5px; background: white; border-radius: 3px; display: inline-block;">${idx + 1}. ${agentName}</span>`;
    }).join(' → ');
    
    sequenceDiv.innerHTML = `<strong>🤖 LLM-Managed Agent Sequence:</strong><br>${sequenceHTML}`;
}

function getAgentNameFromAction(action) {
    const actionMap = {
        'FETCH_NEWS': '📰 ResearchAgent',
        'FETCH_FUNDAMENTALS': '📰 ResearchAgent',
        'FETCH_SENTIMENT': '📰 ResearchAgent',
        'FETCH_MACRO': '📰 ResearchAgent',
        'RUN_TA_BASIC': '📊 TechnicalAnalysisAgent',
        'RUN_TA_ADVANCED': '📊 TechnicalAnalysisAgent',
        'GENERATE_INSIGHT': '💡 InsightAgent',
        'GENERATE_RECOMMENDATION': '🎯 RecommendationAgent',
        'STOP': '🎮 ControllerAgent'
    };
    return actionMap[action] || action;
}

function updateOutputs(action, result) {
    if (!result || !result.type) return;
    
    switch (result.type) {
        case 'news':
            updateResearchPanel({
                news: {
                    articles: result.num_articles,
                    sentiment: result.sentiment,
                    headlines: result.headlines || []
                }
            });
            break;
            
        case 'fundamentals':
            updateResearchPanel({
                fundamentals: {
                    pe_ratio: result.pe_ratio,
                    revenue_growth: result.revenue_growth,
                    profit_margin: result.profit_margin
                }
            });
            break;
            
        case 'sentiment':
            updateResearchPanel({
                sentiment: {
                    social_sentiment: result.social_sentiment,
                    analyst_rating: result.analyst_rating
                }
            });
            break;
            
        case 'ta_basic':
            updateTAPanel({
                rsi: result.rsi,
                ma20: result.ma20,
                current_price: result.current_price
            });
            break;
            
        case 'ta_advanced':
            updateTAPanel({
                trend: result.trend,
                macd_signal: result.macd_signal,
                ma50: result.ma50,
                ma200: result.ma200
            });
            break;
            
        case 'insights':
            updateInsightsPanel(result.insights || []);
            break;
            
        case 'recommendation':
            updateRecommendationPanel({
                recommendation: result.recommendation,
                confidence: result.confidence
            });
            break;
    }
}

function updateResearchPanel(data) {
    const panel = document.getElementById('research-content');
    let html = '';
    
    if (data.news) {
        html += `<div class="data-item">
            <span class="data-label">News Articles:</span>
            <span class="data-value">${data.news.articles}</span>
        </div>`;
        html += `<div class="data-item">
            <span class="data-label">Sentiment:</span>
            <span class="data-value">${(data.news.sentiment * 100).toFixed(1)}%</span>
        </div>`;
        if (data.news.headlines && data.news.headlines.length > 0) {
            html += `<div class="data-item">
                <span class="data-label">Sample Headlines:</span>
                <ul style="margin-top: 5px; padding-left: 20px;">
                    ${data.news.headlines.slice(0, 3).map(h => `<li>${h.substring(0, 80)}...</li>`).join('')}
                </ul>
            </div>`;
        }
    }
    
    if (data.fundamentals) {
        html += `<div class="data-item">
            <span class="data-label">P/E Ratio:</span>
            <span class="data-value">${data.fundamentals.pe_ratio?.toFixed(2) || 'N/A'}</span>
        </div>`;
        html += `<div class="data-item">
            <span class="data-label">Revenue Growth:</span>
            <span class="data-value">${data.fundamentals.revenue_growth ? (data.fundamentals.revenue_growth * 100).toFixed(1) + '%' : 'N/A'}</span>
        </div>`;
        html += `<div class="data-item">
            <span class="data-label">Profit Margin:</span>
            <span class="data-value">${data.fundamentals.profit_margin ? (data.fundamentals.profit_margin * 100).toFixed(1) + '%' : 'N/A'}</span>
        </div>`;
    }
    
    if (data.sentiment) {
        html += `<div class="data-item">
            <span class="data-label">Social Sentiment:</span>
            <span class="data-value">${(data.sentiment.social_sentiment * 100).toFixed(1)}%</span>
        </div>`;
        html += `<div class="data-item">
            <span class="data-label">Analyst Rating:</span>
            <span class="data-value">${data.sentiment.analyst_rating}</span>
        </div>`;
    }
    
    panel.innerHTML = html;
}

function updateTAPanel(data) {
    const panel = document.getElementById('ta-content');
    let html = '';
    
    if (data.rsi !== undefined) {
        html += `<div class="data-item">
            <span class="data-label">RSI:</span>
            <span class="data-value">${data.rsi.toFixed(2)}</span>
        </div>`;
    }
    
    if (data.ma20 !== undefined) {
        html += `<div class="data-item">
            <span class="data-label">MA20:</span>
            <span class="data-value">$${data.ma20.toFixed(2)}</span>
        </div>`;
    }
    
    if (data.ma50 !== undefined) {
        html += `<div class="data-item">
            <span class="data-label">MA50:</span>
            <span class="data-value">$${data.ma50.toFixed(2)}</span>
        </div>`;
    }
    
    if (data.ma200 !== undefined) {
        html += `<div class="data-item">
            <span class="data-label">MA200:</span>
            <span class="data-value">$${data.ma200.toFixed(2)}</span>
        </div>`;
    }
    
    if (data.trend) {
        html += `<div class="data-item">
            <span class="data-label">Trend:</span>
            <span class="data-value">${data.trend}</span>
        </div>`;
    }
    
    if (data.macd_signal !== undefined) {
        html += `<div class="data-item">
            <span class="data-label">MACD Signal:</span>
            <span class="data-value">${data.macd_signal.toFixed(2)}</span>
        </div>`;
    }
    
    panel.innerHTML = html;
}

function updateInsightsPanel(insights) {
    const panel = document.getElementById('insights-content');
    if (!insights || insights.length === 0) {
        panel.innerHTML = '<div class="data-item">No insights generated yet.</div>';
        return;
    }
    
    const html = insights.map(insight => 
        `<div class="insight-item">${insight}</div>`
    ).join('');
    
    panel.innerHTML = html;
}

function updateRecommendationPanel(data) {
    const panel = document.getElementById('recommendation-content');
    const recClass = data.recommendation?.toLowerCase() || 'hold';
    
    const html = `
        <div class="recommendation-badge ${recClass}">${data.recommendation || 'Hold'}</div>
        <div class="data-item">
            <span class="data-label">Confidence:</span>
            <span class="data-value">${((data.confidence || 0.5) * 100).toFixed(1)}%</span>
        </div>
    `;
    
    panel.innerHTML = html;
}

function handleComplete(results) {
    updateStatus('Analysis complete!');
    
    // Update all panels with final results
    if (results.insights) {
        updateInsightsPanel(results.insights);
    }
    
    if (results.recommendation) {
        updateRecommendationPanel({
            recommendation: results.recommendation,
            confidence: results.confidence
        });
    }
    
    // Reset button
    resetButton();
}

function updateStatus(message) {
    document.getElementById('status-message').textContent = message;
}

function resetUI() {
    currentStep = 0;
    document.getElementById('research-content').innerHTML = '';
    document.getElementById('ta-content').innerHTML = '';
    document.getElementById('insights-content').innerHTML = '';
    document.getElementById('recommendation-content').innerHTML = '';
}

function resetButton() {
    const analyzeBtn = document.getElementById('analyze-btn');
    analyzeBtn.disabled = false;
    analyzeBtn.textContent = 'Analyze';
}

function showError(message) {
    updateStatus('Error: ' + message);
    resetButton();
}

// Initialize network on page load
window.addEventListener('DOMContentLoaded', () => {
    initializeNetwork();
});

// Allow Enter key to trigger analysis
document.getElementById('company-input').addEventListener('keypress', function(e) {
    if (e.key === 'Enter') {
        startAnalysis();
    }
});
