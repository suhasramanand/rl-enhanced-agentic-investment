import React from 'react';
import { Menu } from '@mui/icons-material';
import {
  Paper,
  TextField,
  Button,
  Box,
  Typography,
  Chip,
  Stack,
  InputAdornment,
  IconButton,
} from '@mui/material';
import { Search, TrendingUp } from '@mui/icons-material';

const SUPPORTED_STOCKS = [
  { symbol: 'NVDA', name: 'Nvidia', color: '#00D09C' },
  { symbol: 'AAPL', name: 'Apple', color: '#0C4A6E' },
  { symbol: 'TSLA', name: 'Tesla', color: '#F59E0B' },
  { symbol: 'JPM', name: 'JPMorgan', color: '#6366F1' },
  { symbol: 'XOM', name: 'ExxonMobil', color: '#EF4444' },
];

function InputSection({ company, setCompany, isAnalyzing, startAnalysis }) {
  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !isAnalyzing && company.trim()) {
      startAnalysis();
    }
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
      <Box sx={{ mb: 2 }}>
        <Typography variant="h6" sx={{ fontWeight: 600, mb: 0.5, color: '#1F2937', fontSize: '1.125rem' }}>
          Stock Analysis
        </Typography>
        <Typography variant="body2" sx={{ color: '#6B7280', fontSize: '0.875rem' }}>
          Enter a company name or stock symbol to begin analysis
        </Typography>
      </Box>

      <Box sx={{ display: 'flex', gap: 1.5, mb: 2, flexWrap: 'wrap', alignItems: 'flex-start' }}>
        <TextField
          fullWidth
          variant="outlined"
          placeholder="e.g., NVDA, Apple, Tesla..."
          value={company}
          onChange={(e) => setCompany(e.target.value.toUpperCase())}
          onKeyPress={handleKeyPress}
          disabled={isAnalyzing}
          InputProps={{
            startAdornment: (
              <InputAdornment position="start">
                <Search sx={{ color: '#9CA3AF', fontSize: 20 }} />
              </InputAdornment>
            ),
          }}
          sx={{
            flexGrow: 1,
            minWidth: { xs: '100%', sm: 280 },
            '& .MuiOutlinedInput-root': {
              bgcolor: '#FFFFFF',
              borderRadius: 1.5,
              fontSize: '0.875rem',
              height: 42,
              '& fieldset': {
                borderColor: '#D1D5DB',
              },
              '&:hover fieldset': {
                borderColor: '#00D09C',
              },
              '&.Mui-focused fieldset': {
                borderColor: '#00D09C',
                borderWidth: '1.5px',
              },
            },
          }}
        />
        <Button
          variant="contained"
          onClick={startAnalysis}
          disabled={isAnalyzing || !company.trim()}
          startIcon={isAnalyzing ? null : <Search />}
          sx={{
            minWidth: { xs: '100%', sm: 120 },
            height: 42,
            fontSize: '0.875rem',
            fontWeight: 600,
            bgcolor: '#00D09C',
            color: '#FFFFFF',
            borderRadius: 1.5,
            px: 2,
            '&:hover': {
              bgcolor: '#00A67A',
            },
            '&:disabled': {
              bgcolor: '#D1D5DB',
              color: '#9CA3AF',
            },
          }}
        >
          {isAnalyzing ? 'Analyzing...' : 'Analyze'}
        </Button>
      </Box>

      <Box>
        <Stack direction="row" spacing={0.75} flexWrap="wrap" useFlexGap>
          {SUPPORTED_STOCKS.map((stock) => (
            <Chip
              key={stock.symbol}
              label={`${stock.symbol}`}
              onClick={() => !isAnalyzing && setCompany(stock.symbol)}
              sx={{
                bgcolor: '#F9FAFB',
                color: '#374151',
                fontWeight: 500,
                fontSize: '0.75rem',
                height: 28,
                cursor: isAnalyzing ? 'default' : 'pointer',
                border: '1px solid #E5E7EB',
                '&:hover': {
                  bgcolor: isAnalyzing ? '#F9FAFB' : '#F3F4F6',
                  borderColor: isAnalyzing ? '#E5E7EB' : '#00D09C',
                },
                transition: 'all 0.15s ease-in-out',
              }}
            />
          ))}
        </Stack>
      </Box>
    </Paper>
  );
}

export default InputSection;
