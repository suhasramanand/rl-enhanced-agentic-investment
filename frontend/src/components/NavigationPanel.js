import React from 'react';
import {
  Paper,
  Box,
  Typography,
  List,
  ListItem,
  ListItemButton,
  ListItemIcon,
  ListItemText,
  Divider,
  Chip,
  IconButton,
} from '@mui/material';
import {
  Assessment,
  AccountTree,
  Newspaper,
  BarChart,
  Lightbulb,
  TrackChanges,
  ShowChart,
  Description,
  Close,
  Psychology,
  Warning,
} from '@mui/icons-material';

const MENU_ITEMS = [
  { id: 'status', label: 'Status', icon: Assessment, color: '#00D09C' },
  { id: 'workflow', label: 'Workflow', icon: AccountTree, color: '#3B82F6' },
  { id: 'research', label: 'Research', icon: Newspaper, color: '#8B5CF6' },
  { id: 'technical', label: 'Technical', icon: BarChart, color: '#F59E0B' },
  { id: 'insights', label: 'Insights', icon: Lightbulb, color: '#EC4899' },
  { id: 'recommendation', label: 'Recommendation', icon: TrackChanges, color: '#10B981' },
  { id: 'risks', label: 'Risks', icon: Warning, color: '#F59E0B' },
  { id: 'rl_optimizations', label: 'RL Optimizations', icon: Psychology, color: '#8B5CF6' },
  { id: 'charts', label: 'Charts', icon: ShowChart, color: '#6366F1' },
  { id: 'report', label: 'Report', icon: Description, color: '#00D09C' },
];

function NavigationPanel({ activeSection, onSectionChange, isOpen, onClose, hasData }) {
  if (!isOpen) return null;

  const handleClick = (sectionId) => {
    onSectionChange(sectionId);
    // Scroll to section
    const element = document.getElementById(`section-${sectionId}`);
    if (element) {
      element.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
  };

  return (
    <Paper
      elevation={8}
      sx={{
        position: 'fixed',
        right: 0,
        top: 0,
        bottom: 0,
        width: { xs: '100%', md: 280 },
        maxWidth: '90vw',
        zIndex: 1300,
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
        <Typography variant="h6" sx={{ fontWeight: 600, color: '#1F2937', fontSize: '1rem' }}>
          Navigation
        </Typography>
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
      </Box>

      {/* Menu List */}
      <Box
        sx={{
          flex: 1,
          overflowY: 'auto',
          '&::-webkit-scrollbar': {
            width: 6,
          },
          '&::-webkit-scrollbar-track': {
            bgcolor: '#F3F4F6',
          },
          '&::-webkit-scrollbar-thumb': {
            bgcolor: '#D1D5DB',
            borderRadius: 3,
            '&:hover': {
              bgcolor: '#9CA3AF',
            },
          },
        }}
      >
        <List sx={{ p: 0 }}>
          {MENU_ITEMS.map((item, index) => {
            const IconComponent = item.icon;
            const isActive = activeSection === item.id;
            const hasSectionData = hasData[item.id] || false;

            return (
              <React.Fragment key={item.id}>
                <ListItem disablePadding>
                  <ListItemButton
                    onClick={() => handleClick(item.id)}
                    sx={{
                      py: 1.5,
                      px: 2,
                      bgcolor: isActive ? '#ECFDF5' : 'transparent',
                      borderLeft: isActive ? `3px solid ${item.color}` : '3px solid transparent',
                      '&:hover': {
                        bgcolor: isActive ? '#ECFDF5' : '#F9FAFB',
                      },
                      transition: 'all 0.2s ease',
                    }}
                  >
                    <ListItemIcon sx={{ minWidth: 40 }}>
                      <IconComponent
                        sx={{
                          color: isActive ? item.color : '#6B7280',
                          fontSize: 22,
                        }}
                      />
                    </ListItemIcon>
                    <ListItemText
                      primary={
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                          <Typography
                            variant="body2"
                            sx={{
                              fontWeight: isActive ? 600 : 500,
                              color: isActive ? '#1F2937' : '#374151',
                              fontSize: '0.875rem',
                            }}
                          >
                            {item.label}
                          </Typography>
                          {hasSectionData && !isActive && (
                            <Box
                              sx={{
                                width: 6,
                                height: 6,
                                borderRadius: '50%',
                                bgcolor: item.color,
                              }}
                            />
                          )}
                        </Box>
                      }
                    />
                  </ListItemButton>
                </ListItem>
                {index < MENU_ITEMS.length - 1 && <Divider sx={{ mx: 2 }} />}
              </React.Fragment>
            );
          })}
        </List>
      </Box>

      {/* Footer */}
      <Box
        sx={{
          p: 1.5,
          borderTop: '1px solid #E5E7EB',
          bgcolor: '#F9FAFB',
        }}
      >
        <Typography variant="caption" sx={{ color: '#6B7280', fontSize: '0.7rem' }}>
          Click to navigate to sections
        </Typography>
      </Box>
    </Paper>
  );
}

export default NavigationPanel;

