import React from 'react';
import {
  Paper,
  Box,
  Typography,
  Chip,
  Stack,
  LinearProgress,
  Divider,
  Avatar,
  List,
  ListItem,
  ListItemAvatar,
  ListItemText,
  Fade,
  IconButton,
  Tooltip,
} from '@mui/material';
import { 
  SmartToy, 
  Psychology, 
  CheckCircle, 
  RadioButtonUnchecked, 
  PlayArrow,
  Newspaper,
  Assessment,
  SentimentSatisfiedAlt,
  Public,
  BarChart,
  TrendingUp,
  Lightbulb,
  TrackChanges,
  StopCircle,
  History
} from '@mui/icons-material';

const ACTION_LABELS = {
  FETCH_NEWS: 'Research',
  FETCH_FUNDAMENTALS: 'Fundamentals',
  FETCH_SENTIMENT: 'Sentiment',
  FETCH_MACRO: 'Macro',
  RUN_TA_BASIC: 'Basic TA',
  RUN_TA_ADVANCED: 'Advanced TA',
  GENERATE_INSIGHT: 'Insights',
  GENERATE_RECOMMENDATION: 'Recommendation',
  STOP: 'Stop',
};

const ACTION_ICONS = {
  FETCH_NEWS: Newspaper,
  FETCH_FUNDAMENTALS: Assessment,
  FETCH_SENTIMENT: SentimentSatisfiedAlt,
  FETCH_MACRO: Public,
  RUN_TA_BASIC: BarChart,
  RUN_TA_ADVANCED: TrendingUp,
  GENERATE_INSIGHT: Lightbulb,
  GENERATE_RECOMMENDATION: TrackChanges,
  STOP: StopCircle,
};

function StatusPanel({ status, agentCalls, isAnalyzing, onOpenActivityLog }) {
  const getAgentIcon = (source) => {
    return source === 'llm' ? <SmartToy /> : <Psychology />;
  };

  const getAgentColor = (source) => {
    return source === 'llm' ? 'success' : 'warning';
  };

  return (
    <Paper
      elevation={0}
      sx={{
        p: { xs: 2, md: 2.5 },
        borderRadius: 2,
        border: '1px solid #E5E7EB',
        bgcolor: '#FFFFFF',
      }}
    >
      <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 2, flexWrap: 'wrap', gap: 1 }}>
        <Typography variant="h6" sx={{ fontWeight: 600, color: '#1F2937', fontSize: '1rem' }}>
          Analysis Status
        </Typography>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          {agentCalls.length > 0 && (
            <Tooltip title="View Activity Log">
              <IconButton
                onClick={onOpenActivityLog}
                size="small"
                sx={{
                  bgcolor: '#ECFDF5',
                  color: '#00A67A',
                  border: '1px solid #00D09C',
                  '&:hover': {
                    bgcolor: '#D1FAE5',
                  },
                }}
              >
                <History sx={{ fontSize: 18 }} />
              </IconButton>
            </Tooltip>
          )}
          {status && (
            <Chip
              label={status}
              size="small"
              sx={{ 
                fontWeight: 500,
                bgcolor: '#ECFDF5',
                color: '#00A67A',
                fontSize: '0.75rem',
                height: 24,
                border: '1px solid #00D09C',
              }}
              icon={isAnalyzing ? <PlayArrow sx={{ fontSize: 14 }} /> : <CheckCircle sx={{ fontSize: 14 }} />}
            />
          )}
        </Box>
      </Box>

      {isAnalyzing && (
        <Box sx={{ mb: 2 }}>
          <LinearProgress
            sx={{
              height: 6,
              borderRadius: 3,
              bgcolor: '#F3F4F6',
              '& .MuiLinearProgress-bar': {
                borderRadius: 3,
                bgcolor: '#00D09C',
              },
            }}
          />
        </Box>
      )}

      {agentCalls.length > 0 && (
        <Box sx={{ mt: 2 }}>
          <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 1 }}>
            <Typography variant="caption" sx={{ fontWeight: 600, color: '#6B7280', fontSize: '0.75rem' }}>
              Activity Log
            </Typography>
            <Typography
              variant="caption"
              component="button"
              onClick={onOpenActivityLog}
              sx={{
                color: '#00D09C',
                fontSize: '0.75rem',
                fontWeight: 600,
                cursor: 'pointer',
                textDecoration: 'underline',
                border: 'none',
                bgcolor: 'transparent',
                '&:hover': {
                  color: '#00A67A',
                },
              }}
            >
              View Full Log ({agentCalls.length})
            </Typography>
          </Box>
        </Box>
      )}

      {!status && agentCalls.length === 0 && (
        <Box sx={{ textAlign: 'center', py: 2 }}>
          <RadioButtonUnchecked sx={{ fontSize: 32, color: '#D1D5DB', mb: 1 }} />
          <Typography variant="body2" sx={{ color: '#9CA3AF', fontSize: '0.875rem' }}>
            Ready to analyze. Enter a stock symbol above to begin.
          </Typography>
        </Box>
      )}
    </Paper>
  );
}

export default StatusPanel;
