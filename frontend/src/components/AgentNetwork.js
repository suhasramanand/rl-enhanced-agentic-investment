import React, { useEffect, useRef, useState } from 'react';
import { Paper, Box, Typography, Chip } from '@mui/material';
import { 
  AccountTree,
  Newspaper,
  BarChart,
  Settings,
  Lightbulb,
  TrackChanges,
  CheckCircle,
  RadioButtonUnchecked
} from '@mui/icons-material';
// Import vis-network for graph visualization
// Using ES6 import for React compatibility
import { Network } from 'vis-network/standalone/esm/index.js';

const AGENTS = {
  'ControllerAgent': { 
    id: 'controller', 
    label: 'Controller Agent', 
    icon: Settings,
    color: '#00D09C',
    description: 'RL Orchestrator'
  },
  'ResearchAgent': { 
    id: 'research', 
    label: 'Research Agent', 
    icon: Newspaper,
    color: '#3B82F6',
    description: 'News, Fundamentals, Sentiment'
  },
  'TechnicalAnalysisAgent': { 
    id: 'technical', 
    label: 'Technical Analysis', 
    icon: BarChart,
    color: '#8B5CF6',
    description: 'TA Indicators'
  },
  'InsightAgent': { 
    id: 'insight', 
    label: 'Insight Agent', 
    icon: Lightbulb,
    color: '#F59E0B',
    description: 'Generate Insights'
  },
  'RecommendationAgent': { 
    id: 'recommendation', 
    label: 'Recommendation Agent', 
    icon: TrackChanges,
    color: '#00D09C',
    description: 'Buy/Hold/Sell'
  },
  'EvaluatorAgent': { 
    id: 'evaluator', 
    label: 'Evaluator Agent', 
    icon: CheckCircle,
    color: '#10B981',
    description: 'Final Evaluation'
  }
};

function AgentNetwork({ agentCalls, llmSequence, isAnalyzing }) {
  const networkRef = useRef(null);
  const containerRef = useRef(null);
  const [activeAgent, setActiveAgent] = useState(null);
  const [completedAgents, setCompletedAgents] = useState(new Set());
  const [currentAction, setCurrentAction] = useState(null);

  useEffect(() => {
    if (agentCalls.length > 0) {
      if (isAnalyzing) {
        const latest = agentCalls[agentCalls.length - 1];
        setActiveAgent(latest.agent);
        setCurrentAction(latest.action);
        
        if (agentCalls.length > 1) {
          const prevAgent = agentCalls[agentCalls.length - 2].agent;
          setCompletedAgents(prev => new Set([...prev, prevAgent]));
        }
      } else {
        const allCalledAgents = new Set(agentCalls.map(call => call.agent));
        setCompletedAgents(allCalledAgents);
        setActiveAgent(null);
        setCurrentAction(null);
      }
    } else {
      setActiveAgent(null);
      setCurrentAction(null);
      setCompletedAgents(new Set());
    }
  }, [agentCalls, isAnalyzing]);

  useEffect(() => {
    if (!containerRef.current) return;

    // Define nodes with hierarchical positions
    const nodes = [
      {
        id: 'controller',
        label: 'Controller\nAgent',
        level: 0,
        x: 0,
        y: -150,
        fixed: { x: true, y: true },
        color: {
          background: activeAgent === 'ControllerAgent' ? '#FFF7ED' : completedAgents.has('ControllerAgent') ? '#ECFDF5' : '#FFFFFF',
          border: activeAgent === 'ControllerAgent' ? '#F59E0B' : completedAgents.has('ControllerAgent') ? '#00D09C' : '#E5E7EB',
          highlight: { background: '#FFF7ED', border: '#F59E0B' }
        },
        font: { size: 14, color: '#1F2937', face: 'Arial', bold: true },
        shape: 'box',
        widthConstraint: { maximum: 150 },
        heightConstraint: { maximum: 80 },
        borderWidth: activeAgent === 'ControllerAgent' ? 3 : completedAgents.has('ControllerAgent') ? 2 : 1,
        shadow: activeAgent === 'ControllerAgent' || completedAgents.has('ControllerAgent')
      },
      {
        id: 'research',
        label: 'Research\nAgent',
        level: 1,
        x: -300,
        y: 100,
        fixed: { x: true, y: true },
        color: {
          background: activeAgent === 'ResearchAgent' ? '#FFF7ED' : completedAgents.has('ResearchAgent') ? '#ECFDF5' : '#FFFFFF',
          border: activeAgent === 'ResearchAgent' ? '#F59E0B' : completedAgents.has('ResearchAgent') ? '#00D09C' : '#3B82F6',
          highlight: { background: '#FFF7ED', border: '#F59E0B' }
        },
        font: { size: 12, color: '#1F2937', face: 'Arial', bold: true },
        shape: 'box',
        widthConstraint: { maximum: 120 },
        heightConstraint: { maximum: 70 },
        borderWidth: activeAgent === 'ResearchAgent' ? 3 : completedAgents.has('ResearchAgent') ? 2 : 1,
        shadow: activeAgent === 'ResearchAgent' || completedAgents.has('ResearchAgent')
      },
      {
        id: 'technical',
        label: 'Technical\nAnalysis',
        level: 1,
        x: -150,
        y: 100,
        fixed: { x: true, y: true },
        color: {
          background: activeAgent === 'TechnicalAnalysisAgent' ? '#FFF7ED' : completedAgents.has('TechnicalAnalysisAgent') ? '#ECFDF5' : '#FFFFFF',
          border: activeAgent === 'TechnicalAnalysisAgent' ? '#F59E0B' : completedAgents.has('TechnicalAnalysisAgent') ? '#00D09C' : '#8B5CF6',
          highlight: { background: '#FFF7ED', border: '#F59E0B' }
        },
        font: { size: 12, color: '#1F2937', face: 'Arial', bold: true },
        shape: 'box',
        widthConstraint: { maximum: 120 },
        heightConstraint: { maximum: 70 },
        borderWidth: activeAgent === 'TechnicalAnalysisAgent' ? 3 : completedAgents.has('TechnicalAnalysisAgent') ? 2 : 1,
        shadow: activeAgent === 'TechnicalAnalysisAgent' || completedAgents.has('TechnicalAnalysisAgent')
      },
      {
        id: 'insight',
        label: 'Insight\nAgent',
        level: 1,
        x: 0,
        y: 100,
        fixed: { x: true, y: true },
        color: {
          background: activeAgent === 'InsightAgent' ? '#FFF7ED' : completedAgents.has('InsightAgent') ? '#ECFDF5' : '#FFFFFF',
          border: activeAgent === 'InsightAgent' ? '#F59E0B' : completedAgents.has('InsightAgent') ? '#00D09C' : '#F59E0B',
          highlight: { background: '#FFF7ED', border: '#F59E0B' }
        },
        font: { size: 12, color: '#1F2937', face: 'Arial', bold: true },
        shape: 'box',
        widthConstraint: { maximum: 120 },
        heightConstraint: { maximum: 70 },
        borderWidth: activeAgent === 'InsightAgent' ? 3 : completedAgents.has('InsightAgent') ? 2 : 1,
        shadow: activeAgent === 'InsightAgent' || completedAgents.has('InsightAgent')
      },
      {
        id: 'recommendation',
        label: 'Recommendation\nAgent',
        level: 1,
        x: 150,
        y: 100,
        fixed: { x: true, y: true },
        color: {
          background: activeAgent === 'RecommendationAgent' ? '#FFF7ED' : completedAgents.has('RecommendationAgent') ? '#ECFDF5' : '#FFFFFF',
          border: activeAgent === 'RecommendationAgent' ? '#F59E0B' : completedAgents.has('RecommendationAgent') ? '#00D09C' : '#00D09C',
          highlight: { background: '#FFF7ED', border: '#F59E0B' }
        },
        font: { size: 12, color: '#1F2937', face: 'Arial', bold: true },
        shape: 'box',
        widthConstraint: { maximum: 120 },
        heightConstraint: { maximum: 70 },
        borderWidth: activeAgent === 'RecommendationAgent' ? 3 : completedAgents.has('RecommendationAgent') ? 2 : 1,
        shadow: activeAgent === 'RecommendationAgent' || completedAgents.has('RecommendationAgent')
      },
      {
        id: 'evaluator',
        label: 'Evaluator\nAgent',
        level: 1,
        x: 300,
        y: 100,
        fixed: { x: true, y: true },
        color: {
          background: activeAgent === 'EvaluatorAgent' ? '#FFF7ED' : completedAgents.has('EvaluatorAgent') ? '#ECFDF5' : '#FFFFFF',
          border: activeAgent === 'EvaluatorAgent' ? '#F59E0B' : completedAgents.has('EvaluatorAgent') ? '#00D09C' : '#10B981',
          highlight: { background: '#FFF7ED', border: '#F59E0B' }
        },
        font: { size: 12, color: '#1F2937', face: 'Arial', bold: true },
        shape: 'box',
        widthConstraint: { maximum: 120 },
        heightConstraint: { maximum: 70 },
        borderWidth: activeAgent === 'EvaluatorAgent' ? 3 : completedAgents.has('EvaluatorAgent') ? 2 : 1,
        shadow: activeAgent === 'EvaluatorAgent' || completedAgents.has('EvaluatorAgent')
      }
    ];

    // Define edges (connections from Controller to each agent)
    const edges = [
      {
        from: 'controller',
        to: 'research',
        color: {
          color: activeAgent === 'ResearchAgent' ? '#F59E0B' : completedAgents.has('ResearchAgent') ? '#00D09C' : '#D1D5DB',
          highlight: '#F59E0B'
        },
        width: activeAgent === 'ResearchAgent' ? 3 : completedAgents.has('ResearchAgent') ? 2 : 1,
        arrows: { to: { enabled: true, scaleFactor: 1.2 } },
        smooth: { type: 'straightCross' }
      },
      {
        from: 'controller',
        to: 'technical',
        color: {
          color: activeAgent === 'TechnicalAnalysisAgent' ? '#F59E0B' : completedAgents.has('TechnicalAnalysisAgent') ? '#00D09C' : '#D1D5DB',
          highlight: '#F59E0B'
        },
        width: activeAgent === 'TechnicalAnalysisAgent' ? 3 : completedAgents.has('TechnicalAnalysisAgent') ? 2 : 1,
        arrows: { to: { enabled: true, scaleFactor: 1.2 } },
        smooth: { type: 'straightCross' }
      },
      {
        from: 'controller',
        to: 'insight',
        color: {
          color: activeAgent === 'InsightAgent' ? '#F59E0B' : completedAgents.has('InsightAgent') ? '#00D09C' : '#D1D5DB',
          highlight: '#F59E0B'
        },
        width: activeAgent === 'InsightAgent' ? 3 : completedAgents.has('InsightAgent') ? 2 : 1,
        arrows: { to: { enabled: true, scaleFactor: 1.2 } },
        smooth: { type: 'straightCross' }
      },
      {
        from: 'controller',
        to: 'recommendation',
        color: {
          color: activeAgent === 'RecommendationAgent' ? '#F59E0B' : completedAgents.has('RecommendationAgent') ? '#00D09C' : '#D1D5DB',
          highlight: '#F59E0B'
        },
        width: activeAgent === 'RecommendationAgent' ? 3 : completedAgents.has('RecommendationAgent') ? 2 : 1,
        arrows: { to: { enabled: true, scaleFactor: 1.2 } },
        smooth: { type: 'straightCross' }
      },
      {
        from: 'controller',
        to: 'evaluator',
        color: {
          color: activeAgent === 'EvaluatorAgent' ? '#F59E0B' : completedAgents.has('EvaluatorAgent') ? '#00D09C' : '#D1D5DB',
          highlight: '#F59E0B'
        },
        width: activeAgent === 'EvaluatorAgent' ? 3 : completedAgents.has('EvaluatorAgent') ? 2 : 1,
        arrows: { to: { enabled: true, scaleFactor: 1.2 } },
        smooth: { type: 'straightCross' }
      }
    ];

    const data = { nodes, edges };

    const options = {
      nodes: {
        shape: 'box',
        font: {
          size: 12,
          face: 'Arial'
        },
        borderWidth: 2,
        shadow: true
      },
      edges: {
        arrows: {
          to: {
            enabled: true,
            scaleFactor: 1.2
          }
        },
        smooth: {
          type: 'straightCross'
        },
        width: 2
      },
      physics: {
        enabled: false, // Disable physics since we're using fixed positions
        stabilization: false
      },
      interaction: {
        dragNodes: false,
        dragView: false,
        zoomView: false,
        selectConnectedEdges: false
      },
      layout: {
        hierarchical: {
          enabled: false // We're using fixed positions
        }
      }
    };

    // Create or update network
    if (!networkRef.current) {
      networkRef.current = new Network(containerRef.current, data, options);
    } else {
      networkRef.current.setData(data);
    }

    // Center the view
    if (networkRef.current) {
      networkRef.current.fit({
        animation: {
          duration: 300,
          easingFunction: 'easeInOutQuad'
        }
      });
    }

    return () => {
      if (networkRef.current) {
        networkRef.current.destroy();
        networkRef.current = null;
      }
    };
  }, [activeAgent, completedAgents, isAnalyzing]);

  return (
    <Paper
      elevation={0}
      sx={{
        p: { xs: 2, md: 3 },
        borderRadius: 2,
        border: '1px solid #E5E7EB',
        bgcolor: '#FFFFFF',
      }}
    >
      <Box sx={{ display: 'flex', alignItems: 'center', mb: 2.5, gap: 1 }}>
        <AccountTree sx={{ color: '#00D09C', fontSize: 22 }} />
        <Typography variant="h6" sx={{ fontWeight: 600, color: '#1F2937' }}>
          Agent Workflow Network
        </Typography>
      </Box>

      {/* Graph Visualization */}
      <Box
        ref={containerRef}
        sx={{
          width: '100%',
          height: 400,
          border: '1px solid #E5E7EB',
          borderRadius: 1,
          bgcolor: '#FAFBFC',
          position: 'relative',
          overflow: 'hidden'
        }}
      />

      {/* Current Action Status */}
      {isAnalyzing && currentAction && (
        <Box sx={{ mt: 3, pt: 2, borderTop: '1px solid #E5E7EB' }}>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, flexWrap: 'wrap' }}>
            <Typography variant="caption" sx={{ fontWeight: 600, color: '#6B7280', fontSize: '0.75rem' }}>
              Current Action:
            </Typography>
            <Chip
              label={currentAction}
              size="small"
              sx={{
                bgcolor: '#F59E0B',
                color: 'white',
                fontWeight: 600,
                fontSize: '0.75rem',
                height: 24,
              }}
            />
            {activeAgent && (
              <>
                <Typography variant="caption" sx={{ color: '#9CA3AF', fontSize: '0.75rem' }}>
                  →
                </Typography>
                <Chip
                  label={activeAgent}
                  size="small"
                  sx={{
                    bgcolor: '#ECFDF5',
                    color: '#00A67A',
                    fontWeight: 600,
                    fontSize: '0.75rem',
                    height: 24,
                    border: '1px solid #00D09C',
                  }}
                />
              </>
            )}
          </Box>
        </Box>
      )}

      {/* Completion Status */}
      {!isAnalyzing && agentCalls.length > 0 && (
        <Box sx={{ mt: 3, pt: 2, borderTop: '1px solid #E5E7EB' }}>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, flexWrap: 'wrap' }}>
            <CheckCircle sx={{ color: '#00D09C', fontSize: 20 }} />
            <Typography variant="body2" sx={{ fontWeight: 600, color: '#00A67A', fontSize: '0.875rem' }}>
              Analysis Complete - All agents finished execution
            </Typography>
          </Box>
        </Box>
      )}

      {/* Legend */}
      <Box sx={{ mt: 2, pt: 2, borderTop: '1px solid #E5E7EB' }}>
        <Typography variant="caption" sx={{ fontWeight: 600, color: '#6B7280', mb: 1.5, display: 'block' }}>
          Legend
        </Typography>
        <Box sx={{ display: 'flex', gap: 3, flexWrap: 'wrap' }}>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <Box sx={{ width: 12, height: 12, borderRadius: '50%', bgcolor: '#F59E0B', border: '2px solid white', boxShadow: '0 0 0 2px #F59E0B' }} />
            <Typography variant="caption" sx={{ color: '#6B7280', fontSize: '0.75rem' }}>
              Active
            </Typography>
          </Box>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <Box sx={{ width: 12, height: 12, borderRadius: '50%', bgcolor: '#00D09C', border: '2px solid white', boxShadow: '0 0 0 2px #00D09C' }} />
            <Typography variant="caption" sx={{ color: '#6B7280', fontSize: '0.75rem' }}>
              Completed
            </Typography>
          </Box>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <RadioButtonUnchecked sx={{ fontSize: 16, color: '#D1D5DB' }} />
            <Typography variant="caption" sx={{ color: '#6B7280', fontSize: '0.75rem' }}>
              Pending
            </Typography>
          </Box>
        </Box>
      </Box>
    </Paper>
  );
}

export default AgentNetwork;
