import React from 'react';
import { AppBar, Toolbar, Typography, Box, Container } from '@mui/material';
import { TrendingUp } from '@mui/icons-material';

function Header() {
  return (
    <AppBar 
      position="static" 
      elevation={0}
      sx={{
        background: '#FFFFFF',
        borderBottom: '1px solid #E5E7EB',
        color: '#1F2937',
      }}
    >
      <Container maxWidth="xl">
        <Toolbar sx={{ py: 2, px: { xs: 2, md: 3 } }}>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, flexGrow: 1 }}>
            <Box
              sx={{
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                width: 40,
                height: 40,
                borderRadius: 2,
                bgcolor: '#00D09C',
              }}
            >
              <TrendingUp sx={{ fontSize: 24, color: 'white' }} />
            </Box>
            <Box>
              <Typography
                variant="h5"
                component="div"
                sx={{
                  fontWeight: 700,
                  letterSpacing: '-0.02em',
                  color: '#1F2937',
                  fontSize: { xs: '1.125rem', md: '1.5rem' },
                }}
              >
                RL-Enhanced Investment Research
              </Typography>
              <Typography
                variant="body2"
                sx={{
                  color: '#6B7280',
                  mt: 0.25,
                  fontSize: { xs: '0.75rem', md: '0.875rem' },
                }}
              >
                Q-Learning Agent Orchestration System
              </Typography>
            </Box>
          </Box>
        </Toolbar>
      </Container>
    </AppBar>
  );
}

export default Header;
