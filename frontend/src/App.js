import React, { useState, useEffect, useRef } from 'react';
import { ThemeProvider, createTheme, CssBaseline } from '@mui/material';
import { Container, Box, Fade, Grow } from '@mui/material';
import AgentNetwork from './components/AgentNetwork';
import OutputPanels from './components/OutputPanels';
import StatusPanel from './components/StatusPanel';
import InputSection from './components/InputSection';
import Header from './components/Header';
import ActivityLog from './components/ActivityLog';
import NavigationPanel from './components/NavigationPanel';

// In development, React dev server proxies to Flask (see package.json proxy)
// In production, Flask serves React build, so use same origin
const API_BASE_URL = process.env.NODE_ENV === 'development' 
  ? 'http://localhost:5001' 
  : window.location.origin;

// Groww-inspired theme - Clean, professional, financial app aesthetic
const theme = createTheme({
  palette: {
    mode: 'light',
    primary: {
      main: '#00D09C', // Groww's signature teal/green
      light: '#33D9B0',
      dark: '#00A67A',
      contrastText: '#ffffff',
    },
    secondary: {
      main: '#0C4A6E',
      light: '#1E6B94',
      dark: '#082F47',
    },
    background: {
      default: '#F5F7FA',
      paper: '#FFFFFF',
    },
    success: {
      main: '#00D09C',
      light: '#33D9B0',
      dark: '#00A67A',
    },
    warning: {
      main: '#F59E0B',
      light: '#FBBF24',
      dark: '#D97706',
    },
    error: {
      main: '#EF4444',
      light: '#F87171',
      dark: '#DC2626',
    },
    text: {
      primary: '#1F2937',
      secondary: '#6B7280',
    },
    grey: {
      50: '#F9FAFB',
      100: '#F3F4F6',
      200: '#E5E7EB',
      300: '#D1D5DB',
      400: '#9CA3AF',
      500: '#6B7280',
      600: '#4B5563',
      700: '#374151',
      800: '#1F2937',
      900: '#111827',
    },
  },
  typography: {
    fontFamily: '"Inter", "Roboto", "Helvetica", "Arial", sans-serif',
    h1: {
      fontWeight: 700,
      fontSize: '2rem',
      letterSpacing: '-0.02em',
      color: '#1F2937',
    },
    h2: {
      fontWeight: 600,
      fontSize: '1.5rem',
      letterSpacing: '-0.01em',
      color: '#1F2937',
    },
    h3: {
      fontWeight: 600,
      fontSize: '1.25rem',
      color: '#1F2937',
    },
    h4: {
      fontWeight: 600,
      fontSize: '1.125rem',
      color: '#1F2937',
    },
    h5: {
      fontWeight: 600,
      fontSize: '1rem',
      color: '#1F2937',
    },
    h6: {
      fontWeight: 600,
      fontSize: '0.875rem',
      color: '#1F2937',
    },
    body1: {
      fontSize: '0.9375rem',
      lineHeight: 1.6,
      color: '#374151',
    },
    body2: {
      fontSize: '0.875rem',
      lineHeight: 1.5,
      color: '#6B7280',
    },
    button: {
      textTransform: 'none',
      fontWeight: 600,
      fontSize: '0.9375rem',
    },
  },
  shape: {
    borderRadius: 12,
  },
  components: {
    MuiCard: {
      styleOverrides: {
        root: {
          borderRadius: 12,
          boxShadow: '0 1px 3px 0 rgba(0, 0, 0, 0.1), 0 1px 2px 0 rgba(0, 0, 0, 0.06)',
          border: '1px solid #E5E7EB',
          transition: 'all 0.2s ease-in-out',
          '&:hover': {
            boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06)',
          },
        },
      },
    },
    MuiButton: {
      styleOverrides: {
        root: {
          textTransform: 'none',
          fontWeight: 600,
          borderRadius: 8,
          padding: '10px 24px',
          fontSize: '0.9375rem',
          boxShadow: 'none',
          '&:hover': {
            boxShadow: '0 2px 4px rgba(0, 208, 156, 0.2)',
          },
        },
        contained: {
          background: '#00D09C',
          color: '#FFFFFF',
          '&:hover': {
            background: '#00A67A',
          },
          '&:disabled': {
            background: '#D1D5DB',
            color: '#9CA3AF',
          },
        },
      },
    },
    MuiPaper: {
      styleOverrides: {
        root: {
          backgroundImage: 'none',
          borderRadius: 12,
        },
      },
    },
    MuiTextField: {
      styleOverrides: {
        root: {
          '& .MuiOutlinedInput-root': {
            borderRadius: 8,
            backgroundColor: '#FFFFFF',
            '&:hover fieldset': {
              borderColor: '#00D09C',
            },
            '&.Mui-focused fieldset': {
              borderColor: '#00D09C',
              borderWidth: '2px',
            },
          },
        },
      },
    },
  },
});

function App() {
  const [company, setCompany] = useState('NVDA');
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [status, setStatus] = useState('');
  const [llmSequence, setLlmSequence] = useState([]);
  const [agentCalls, setAgentCalls] = useState([]);
  const [outputs, setOutputs] = useState({
    research: {},
    technical: {},
    insights: [],
    recommendation: null,
    confidence: null
  });
  const [formattedReport, setFormattedReport] = useState(null);
  const [activityLogOpen, setActivityLogOpen] = useState(false);
  const [navigationOpen, setNavigationOpen] = useState(true);
  const [activeSection, setActiveSection] = useState('status');

  const startAnalysis = async () => {
    if (!company.trim()) {
      return;
    }

    setIsAnalyzing(true);
    setStatus('Starting analysis...');
    setAgentCalls([]);
    setOutputs({ research: {}, technical: {}, insights: [], recommendation: null, confidence: null });
    setFormattedReport(null);
    setLlmSequence([]);

    try {
      // Use fetch with cache: 'no-store' to prevent service worker from caching streaming endpoint
      const response = await fetch(`${API_BASE_URL}/api/analyze-stream`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ company: company.trim() }),
        cache: 'no-store' // Prevent service worker from caching streaming endpoint
      });

      if (!response.ok) {
        throw new Error(`Failed to start analysis: ${response.status} ${response.statusText}`);
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();

      const readStream = () => {
        reader.read().then(({ done, value }) => {
          if (done) {
            setIsAnalyzing(false);
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
                console.error('Error parsing data:', e);
              }
            }
          }

          readStream();
        }).catch(error => {
          console.error('Stream read error:', error);
          setStatus(`Error during stream: ${error.message}`);
          setIsAnalyzing(false);
        });
      };

      readStream();
    } catch (error) {
      console.error('Error starting analysis:', error);
      setStatus(`Error: ${error.message}`);
      setIsAnalyzing(false);
    }
  };

  const handleUpdate = (data) => {
    console.log('Received update:', data);
    switch (data.type) {
      case 'status':
        setStatus(data.message);
        break;

      case 'llm_interpretation':
        setStatus(data.message);
        // LLM only extracts ticker, RL will select actions
        break;

      case 'agent_call':
        // Use the source from backend, or default to 'controller_orchestration'
        const source = data.source || 'controller_orchestration';
        setAgentCalls(prev => [...prev, {
          agent: data.agent,
          action: data.action,
          step: data.step,
          source: source, // Use source from backend (controller_orchestration)
          timestamp: Date.now()
        }]);
        // Update sequence for visualization (RL-selected actions)
        setLlmSequence(prev => [...(prev || []), data.action]);
        setStatus(`RL-Managed → ${data.agent} executing: ${data.action}`);
        break;

      case 'action_result':
        updateOutputs(data.action, data.result);
        break;

      case 'complete':
        setIsAnalyzing(false);
        if (data.results) {
          // Merge with existing outputs to preserve articles and other data
          setOutputs(prev => {
            const newResearch = data.results.raw_outputs?.research || {};
            const prevResearch = prev.research || {};
            
            return {
              research: {
                ...prevResearch,
                ...newResearch,
                // Preserve articles - prioritize new data over old
                news: {
                  ...(newResearch.news || {}),
                  // Use new data first, fallback to previous if new is empty
                  articles: (newResearch.news?.articles && newResearch.news.articles.length > 0) 
                    ? newResearch.news.articles 
                    : (prevResearch.news?.articles || []),
                  headlines: (newResearch.news?.headlines && newResearch.news.headlines.length > 0)
                    ? newResearch.news.headlines
                    : (prevResearch.news?.headlines || []),
                  num_articles: newResearch.news?.num_articles ?? prevResearch.news?.num_articles ?? 0,
                  sentiment: newResearch.news?.sentiment ?? prevResearch.news?.sentiment ?? 0.5,
                }
              },
              technical: {
                ...prev.technical,
                ...(data.results.raw_outputs?.technical || {})
              },
              insights: data.results.insights || prev.insights || [],
              recommendation: data.results.recommendation || prev.recommendation,
              confidence: data.results.confidence || prev.confidence,
              price_levels: data.results.raw_outputs?.price_levels || prev.price_levels,
              rl_optimizations: data.results.raw_outputs?.rl_optimizations || prev.rl_optimizations,
              risks: data.results.raw_outputs?.risks || prev.risks
            };
          });
          if (data.results.formatted_report) {
            setFormattedReport(data.results.formatted_report);
          }
        }
        setStatus('Analysis complete!');
        break;

      case 'error':
        setIsAnalyzing(false);
        setStatus('Error: ' + data.message);
        break;
    }
  };

  const updateOutputs = (action, result) => {
    if (!result || !result.type) return;

    switch (result.type) {
      case 'news':
        setOutputs(prev => ({
          ...prev,
          research: {
            ...prev.research,
            news: {
              num_articles: result.num_articles,
              sentiment: result.sentiment,
              headlines: result.headlines || [],
              articles: result.articles || []  // Include full article data with links
            }
          }
        }));
        break;

      case 'fundamentals':
        setOutputs(prev => ({
          ...prev,
          research: {
            ...prev.research,
            fundamentals: {
              pe_ratio: result.pe_ratio,
              revenue_growth: result.revenue_growth,
              profit_margin: result.profit_margin
            }
          }
        }));
        break;

      case 'sentiment':
        setOutputs(prev => ({
          ...prev,
          research: {
            ...prev.research,
            sentiment: {
              social_sentiment: result.social_sentiment,
              analyst_rating: result.analyst_rating
            }
          }
        }));
        break;

      case 'ta_basic':
        setOutputs(prev => ({
          ...prev,
          technical: {
            ...prev.technical,
            rsi: result.rsi,
            ma20: result.ma20,
            current_price: result.current_price,
            price_history: result.price_history,
            ohlcv: result.ohlcv
          }
        }));
        break;

      case 'ta_advanced':
        setOutputs(prev => ({
          ...prev,
          technical: {
            ...prev.technical,
            trend: result.trend,
            macd_signal: result.macd_signal,
            ma50: result.ma50,
            ma200: result.ma200,
            price_history: result.price_history,
            ohlcv: result.ohlcv
          }
        }));
        break;

      case 'insights':
        setOutputs(prev => ({
          ...prev,
          insights: result.insights || []
        }));
        break;

      case 'recommendation':
        setOutputs(prev => ({
          ...prev,
          recommendation: result.recommendation,
          confidence: result.confidence
        }));
        break;
    }
  };

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Box sx={{ minHeight: '100vh', bgcolor: 'background.default' }}>
        <Header />
        
        <Container 
          maxWidth="xl" 
          sx={{ 
            py: { xs: 2, md: 3 }, 
            px: { xs: 2, md: 3 },
            pr: { 
              xs: 2, 
              md: navigationOpen 
                ? (activityLogOpen ? 'calc(400px + 280px + 24px)' : 'calc(280px + 24px)')
                : (activityLogOpen ? 'calc(400px + 24px)' : 3)
            },
            transition: 'padding-right 0.3s ease',
          }}
        >
          <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2.5 }}>
            {/* Input Section - Fixed at Top */}
            <Box id="section-input" sx={{ position: 'sticky', top: 0, zIndex: 100, bgcolor: 'background.default', pb: 2 }}>
              <InputSection
                company={company}
                setCompany={setCompany}
                isAnalyzing={isAnalyzing}
                startAnalysis={startAnalysis}
                onToggleNavigation={() => setNavigationOpen(!navigationOpen)}
              />
            </Box>

            {/* Status Panel */}
            <Box id="section-status">
              <StatusPanel
                status={status}
                agentCalls={agentCalls}
                isAnalyzing={isAnalyzing}
                onOpenActivityLog={() => setActivityLogOpen(true)}
              />
            </Box>

            {/* Agent Network */}
            <Box id="section-workflow">
              <AgentNetwork
                agentCalls={agentCalls}
                llmSequence={llmSequence}
                isAnalyzing={isAnalyzing}
              />
            </Box>

            {/* Output Panels */}
            {(Object.keys(outputs.research).length > 0 || Object.keys(outputs.technical).length > 0) && (
              <Box id="section-outputs">
                <OutputPanels
                  outputs={outputs}
                  formattedReport={formattedReport}
                  activeSection={activeSection}
                />
              </Box>
            )}
          </Box>
        </Container>

        {/* Navigation Panel (Fixed Right Side) */}
        <NavigationPanel
          activeSection={activeSection}
          onSectionChange={setActiveSection}
          isOpen={navigationOpen}
          onClose={() => setNavigationOpen(false)}
          hasData={{
            status: true,
            workflow: agentCalls.length > 0,
            research: Object.keys(outputs.research).length > 0,
            technical: Object.keys(outputs.technical).length > 0,
            insights: outputs.insights && outputs.insights.length > 0,
            recommendation: outputs.recommendation !== null,
            charts: outputs.technical && (outputs.technical.ohlcv || outputs.technical.price_history),
            report: formattedReport !== null,
          }}
        />

        {/* Activity Log Panel (Fixed Right Side, below Navigation) */}
        <ActivityLog
          agentCalls={agentCalls}
          isOpen={activityLogOpen}
          onClose={() => setActivityLogOpen(false)}
          navigationOpen={navigationOpen}
        />
      </Box>
    </ThemeProvider>
  );
}

export default App;
