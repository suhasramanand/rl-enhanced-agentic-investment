import React from 'react';
import {
  Paper,
  Box,
  Typography,
  List,
  ListItem,
  ListItemText,
  Divider,
  Chip,
  IconButton,
  Tooltip,
} from '@mui/material';
import {
  Psychology,
  SmartToy,
  Close,
  History,
  CheckCircle,
  RadioButtonUnchecked,
} from '@mui/icons-material';

const ACTION_LABELS = {
  FETCH_NEWS: 'Fetch News',
  FETCH_FUNDAMENTALS: 'Fetch Fundamentals',
  FETCH_SENTIMENT: 'Fetch Sentiment',
  FETCH_MACRO: 'Fetch Macro',
  RUN_TA_BASIC: 'Run Basic TA',
  RUN_TA_ADVANCED: 'Run Advanced TA',
  GENERATE_INSIGHT: 'Generate Insight',
  GENERATE_RECOMMENDATION: 'Generate Recommendation',
  STOP: 'Stop',
};

const AGENT_LABELS = {
  ControllerAgent: 'Controller',
  ResearchAgent: 'Research',
  TechnicalAnalysisAgent: 'Technical',
  InsightAgent: 'Insight',
  RecommendationAgent: 'Recommendation',
  EvaluatorAgent: 'Evaluator',
};

function ActivityLog({ agentCalls, isOpen, onClose, navigationOpen }) {
  const formatTime = (timestamp) => {
    const date = new Date(timestamp);
    return date.toLocaleTimeString('en-US', { 
      hour12: false, 
      hour: '2-digit', 
      minute: '2-digit', 
      second: '2-digit',
      fractionalSecondDigits: 3
    });
  };

  const getAgentIcon = (source) => {
    return source === 'llm' ? <SmartToy sx={{ fontSize: 16 }} /> : <Psychology sx={{ fontSize: 16 }} />;
  };

  const getAgentColor = (source) => {
    return source === 'llm' ? '#00D09C' : '#F59E0B';
  };

  if (!isOpen) return null;

  return (
    <Paper
      elevation={8}
      sx={{
        position: 'fixed',
        right: navigationOpen ? { xs: 0, md: 280 } : 0,
        top: 0,
        bottom: 0,
        width: { xs: '100%', md: 400 },
        maxWidth: '90vw',
        zIndex: 1299,
        display: 'flex',
        flexDirection: 'column',
        borderRadius: 0,
        borderLeft: '1px solid #E5E7EB',
        bgcolor: '#FFFFFF',
      }}
    >
      {/* Header */}
      <Box
        sx={{
          p: 2,
          borderBottom: '1px solid #E5E7EB',
          bgcolor: '#F9FAFB',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
        }}
      >
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <History sx={{ color: '#00D09C', fontSize: 20 }} />
          <Typography variant="h6" sx={{ fontWeight: 600, color: '#1F2937', fontSize: '1rem' }}>
            Activity Log
          </Typography>
          <Chip
            label={agentCalls.length}
            size="small"
            sx={{
              bgcolor: '#ECFDF5',
              color: '#00A67A',
              fontWeight: 600,
              fontSize: '0.75rem',
              height: 20,
              border: '1px solid #00D09C',
            }}
          />
        </Box>
        <Tooltip title="Close">
          <IconButton
            onClick={onClose}
            size="small"
            sx={{
              color: '#6B7280',
              '&:hover': { bgcolor: '#F3F4F6' },
            }}
          >
            <Close sx={{ fontSize: 20 }} />
          </IconButton>
        </Tooltip>
      </Box>

      {/* Log List */}
      <Box
        sx={{
          flex: 1,
          overflowY: 'auto',
          bgcolor: '#FAFBFC',
          '&::-webkit-scrollbar': {
            width: 8,
          },
          '&::-webkit-scrollbar-track': {
            bgcolor: '#F3F4F6',
          },
          '&::-webkit-scrollbar-thumb': {
            bgcolor: '#D1D5DB',
            borderRadius: 4,
            '&:hover': {
              bgcolor: '#9CA3AF',
            },
          },
        }}
      >
        {agentCalls.length === 0 ? (
          <Box sx={{ textAlign: 'center', py: 8, px: 2 }}>
            <RadioButtonUnchecked sx={{ fontSize: 48, color: '#D1D5DB', mb: 2 }} />
            <Typography variant="body2" sx={{ color: '#9CA3AF', mb: 1 }}>
              No activity yet
            </Typography>
            <Typography variant="caption" sx={{ color: '#6B7280' }}>
              Activity will appear here as agents execute
            </Typography>
          </Box>
        ) : (
          <List sx={{ p: 0 }}>
            {agentCalls.map((call, index) => {
              const isLatest = index === agentCalls.length - 1;
              const timeStr = formatTime(call.timestamp);
              
              return (
                <React.Fragment key={index}>
                  <ListItem
                    sx={{
                      py: 1.5,
                      px: 2,
                      bgcolor: isLatest ? '#FFF7ED' : '#FFFFFF',
                      borderLeft: isLatest ? '3px solid #F59E0B' : '3px solid transparent',
                      transition: 'all 0.2s ease',
                      '&:hover': {
                        bgcolor: isLatest ? '#FFF7ED' : '#F9FAFB',
                      },
                    }}
                  >
                    <ListItemText
                      primary={
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, flexWrap: 'wrap', mb: 0.5 }}>
                          {/* Timestamp */}
                          <Typography
                            variant="caption"
                            sx={{
                              fontFamily: 'monospace',
                              color: '#6B7280',
                              fontSize: '0.7rem',
                              fontWeight: 500,
                              minWidth: 80,
                            }}
                          >
                            {timeStr}
                          </Typography>

                          {/* Step Number */}
                          <Chip
                            label={`#${call.step}`}
                            size="small"
                            sx={{
                              fontSize: '0.65rem',
                              height: 18,
                              bgcolor: '#F3F4F6',
                              color: '#374151',
                              fontWeight: 600,
                              minWidth: 40,
                            }}
                          />

                          {/* Source Badge */}
                          <Chip
                            icon={getAgentIcon(call.source)}
                            label={call.source.toUpperCase()}
                            size="small"
                            sx={{
                              fontSize: '0.65rem',
                              height: 18,
                              bgcolor: call.source === 'rl' ? '#FEF3C7' : '#ECFDF5',
                              color: call.source === 'rl' ? '#D97706' : '#00A67A',
                              fontWeight: 600,
                              border: `1px solid ${getAgentColor(call.source)}`,
                            }}
                          />

                          {/* Agent Name */}
                          <Chip
                            label={AGENT_LABELS[call.agent] || call.agent}
                            size="small"
                            sx={{
                              fontSize: '0.65rem',
                              height: 18,
                              bgcolor: '#ECFDF5',
                              color: '#00A67A',
                              fontWeight: 600,
                              border: '1px solid #00D09C',
                            }}
                          />
                        </Box>
                      }
                      secondary={
                        <Box sx={{ mt: 0.5 }}>
                          <Typography
                            variant="body2"
                            sx={{
                              color: '#1F2937',
                              fontWeight: 500,
                              fontSize: '0.8125rem',
                              fontFamily: 'monospace',
                            }}
                          >
                            → {ACTION_LABELS[call.action] || call.action}
                          </Typography>
                        </Box>
                      }
                    />
                    {isLatest && (
                      <Box
                        sx={{
                          ml: 1,
                          width: 8,
                          height: 8,
                          borderRadius: '50%',
                          bgcolor: '#F59E0B',
                          animation: 'pulse 2s infinite',
                          '@keyframes pulse': {
                            '0%, 100%': { transform: 'scale(1)', opacity: 1 },
                            '50%': { transform: 'scale(1.3)', opacity: 0.7 },
                          },
                        }}
                      />
                    )}
                  </ListItem>
                  {index < agentCalls.length - 1 && <Divider sx={{ mx: 2 }} />}
                </React.Fragment>
              );
            })}
          </List>
        )}
      </Box>

      {/* Footer */}
      <Box
        sx={{
          p: 1.5,
          borderTop: '1px solid #E5E7EB',
          bgcolor: '#F9FAFB',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
        }}
      >
        <Typography variant="caption" sx={{ color: '#6B7280', fontSize: '0.7rem' }}>
          {agentCalls.length > 0 ? `Total: ${agentCalls.length} actions` : 'Waiting for activity...'}
        </Typography>
        <Box sx={{ display: 'flex', gap: 1 }}>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
            <Box sx={{ width: 8, height: 8, borderRadius: '50%', bgcolor: '#F59E0B' }} />
            <Typography variant="caption" sx={{ color: '#6B7280', fontSize: '0.7rem' }}>
              Latest
            </Typography>
          </Box>
        </Box>
      </Box>
    </Paper>
  );
}

export default ActivityLog;

