import React, { useState } from 'react';
import {
  Grid,
  Paper,
  Typography,
  Box,
  Chip,
  List,
  ListItem,
  ListItemText,
  Divider,
  Card,
  CardContent,
  Fade,
  Stack,
  Tabs,
  Tab,
} from '@mui/material';
import {
  Article,
  Assessment,
  SentimentSatisfied,
  Lightbulb,
  TrendingUp,
  Description,
  AttachMoney,
  TrendingDown,
  Remove,
  ShowChart,
  Warning,
  Psychology,
  AutoAwesome,
} from '@mui/icons-material';
import TechnicalCharts from './TechnicalCharts';
import ReactMarkdown from 'react-markdown';

function TabPanel({ children, value, index, ...other }) {
  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`output-tabpanel-${index}`}
      aria-labelledby={`output-tab-${index}`}
      {...other}
    >
      {value === index && <Box sx={{ pt: 3 }}>{children}</Box>}
    </div>
  );
}

function OutputPanels({ outputs, formattedReport, activeSection }) {
  const [tabValue, setTabValue] = useState(0);

  const handleTabChange = (event, newValue) => {
    setTabValue(newValue);
  };

  // Map activeSection to tab index
  React.useEffect(() => {
    if (activeSection === 'research') setTabValue(0);
    else if (activeSection === 'technical') setTabValue(1);
    else if (activeSection === 'insights') setTabValue(2);
    else if (activeSection === 'recommendation') setTabValue(3);
    else if (activeSection === 'risks') setTabValue(4);
    else if (activeSection === 'rl_optimizations') setTabValue(5);
    else if (activeSection === 'charts') setTabValue(6);
    else if (activeSection === 'report') setTabValue(7);
  }, [activeSection]);

  const renderResearch = (research) => {
    if (!research || Object.keys(research).length === 0) {
      return (
        <Box sx={{ textAlign: 'center', py: 4 }}>
          <Article sx={{ fontSize: 40, color: '#D1D5DB', mb: 1 }} />
          <Typography variant="body2" sx={{ color: '#9CA3AF' }}>
            Research data will appear here...
          </Typography>
        </Box>
      );
    }

    return (
      <Box>
                    {research.news && (
                      <Fade in timeout={400}>
                        <Box sx={{ mb: 3 }}>
                          <Box sx={{ display: 'flex', alignItems: 'center', mb: 2, gap: 1 }}>
                            <Article sx={{ color: '#00D09C', fontSize: 20 }} />
                            <Typography variant="h6" sx={{ fontWeight: 600, color: '#1F2937' }}>
                              News Analysis
                            </Typography>
                          </Box>
                          <Stack direction="row" spacing={1} sx={{ mb: 2 }} flexWrap="wrap" useFlexGap>
                            <Chip
                              label={`${research.news.num_articles || research.news.articles || 0} Articles`}
                              size="small"
                              sx={{
                                fontWeight: 600,
                                bgcolor: '#F3F4F6',
                                color: '#374151',
                                fontSize: '0.75rem',
                                height: 28,
                              }}
                            />
                            <Chip
                              label={`Sentiment: ${(research.news.sentiment * 100).toFixed(1)}%`}
                              size="small"
                              sx={{
                                fontWeight: 600,
                                bgcolor: research.news.sentiment > 0.6 ? '#ECFDF5' : research.news.sentiment < 0.4 ? '#FEF2F2' : '#F3F4F6',
                                color: research.news.sentiment > 0.6 ? '#00A67A' : research.news.sentiment < 0.4 ? '#DC2626' : '#374151',
                                fontSize: '0.75rem',
                                height: 28,
                              }}
                            />
                          </Stack>
                          {research.news.articles && research.news.articles.length > 0 ? (
                            <List dense sx={{ bgcolor: '#F9FAFB', borderRadius: 2, p: 1, border: '1px solid #E5E7EB' }}>
                              {research.news.articles.slice(0, 10).map((article, i) => (
                                <ListItem 
                                  key={i} 
                                  sx={{ 
                                    py: 1,
                                    borderBottom: i < research.news.articles.slice(0, 10).length - 1 ? '1px solid #E5E7EB' : 'none',
                                    '&:hover': { bgcolor: '#F3F4F6' }
                                  }}
                                >
                                  <ListItemText
                                    primary={
                                      <Box>
                                        <Typography
                                          variant="body2"
                                          component="a"
                                          href={article.link || '#'}
                                          target="_blank"
                                          rel="noopener noreferrer"
                                          sx={{
                                            fontWeight: 600,
                                            color: '#1F2937',
                                            textDecoration: 'none',
                                            '&:hover': { 
                                              color: '#00D09C',
                                              textDecoration: 'underline'
                                            },
                                            lineHeight: 1.4,
                                            display: 'block',
                                            mb: 0.5
                                          }}
                                        >
                                          {article.title || article.headline || 'Untitled Article'}
                                        </Typography>
                                        {article.summary && (
                                          <Typography
                                            variant="caption"
                                            sx={{
                                              color: '#6B7280',
                                              display: 'block',
                                              lineHeight: 1.5,
                                              mb: 0.5
                                            }}
                                          >
                                            {article.summary.substring(0, 150)}...
                                          </Typography>
                                        )}
                                        <Box sx={{ display: 'flex', gap: 1, alignItems: 'center', flexWrap: 'wrap' }}>
                                          {article.link && (
                                            <Chip
                                              label="Read Article"
                                              size="small"
                                              component="a"
                                              href={article.link}
                                              target="_blank"
                                              rel="noopener noreferrer"
                                              clickable
                                              sx={{
                                                fontSize: '0.65rem',
                                                height: 20,
                                                bgcolor: '#ECFDF5',
                                                color: '#00A67A',
                                                '&:hover': { bgcolor: '#D1FAE5' },
                                                textDecoration: 'none'
                                              }}
                                            />
                                          )}
                                          {article.sentiment !== undefined && (
                                            <Chip
                                              label={`Sentiment: ${(article.sentiment * 100 + 100) / 2 > 50 ? '+' : ''}${((article.sentiment * 100 + 100) / 2 - 50).toFixed(0)}%`}
                                              size="small"
                                              sx={{
                                                fontSize: '0.65rem',
                                                height: 20,
                                                bgcolor: article.sentiment > 0 ? '#ECFDF5' : '#FEF2F2',
                                                color: article.sentiment > 0 ? '#00A67A' : '#DC2626',
                                              }}
                                            />
                                          )}
                                          {article.published && (
                                            <Typography variant="caption" sx={{ color: '#9CA3AF', fontSize: '0.65rem' }}>
                                              {new Date(article.published).toLocaleDateString()}
                                            </Typography>
                                          )}
                                        </Box>
                                      </Box>
                                    }
                                  />
                                </ListItem>
                              ))}
                            </List>
                          ) : research.news.headlines && research.news.headlines.length > 0 ? (
                            <List dense sx={{ bgcolor: '#F9FAFB', borderRadius: 2, p: 1, border: '1px solid #E5E7EB' }}>
                              {research.news.headlines.slice(0, 10).map((headline, i) => (
                                <ListItem key={i} sx={{ py: 0.75 }}>
                                  <ListItemText
                                    primary={headline.substring(0, 150) + (headline.length > 150 ? '...' : '')}
                                    primaryTypographyProps={{
                                      variant: 'body2',
                                      sx: { fontWeight: 500, color: '#374151', lineHeight: 1.5 }
                                    }}
                                  />
                                </ListItem>
                              ))}
                            </List>
                          ) : (
                            <Typography variant="body2" sx={{ color: '#9CA3AF', textAlign: 'center', py: 2 }}>
                              No news articles found
                            </Typography>
                          )}
                        </Box>
                      </Fade>
                    )}

        {research.fundamentals && (
          <Fade in timeout={600}>
            <Box sx={{ mb: 3 }}>
              {research.news && <Divider sx={{ my: 2.5 }} />}
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 2, gap: 1 }}>
                <Assessment sx={{ color: '#00D09C', fontSize: 20 }} />
                <Typography variant="h6" sx={{ fontWeight: 600, color: '#1F2937' }}>
                  Fundamentals
                </Typography>
              </Box>
              <Grid container spacing={2}>
                <Grid item xs={12} sm={4}>
                  <Card variant="outlined" sx={{ bgcolor: '#F9FAFB', borderColor: '#E5E7EB' }}>
                    <CardContent sx={{ p: 2 }}>
                      <Typography variant="caption" sx={{ fontWeight: 600, color: '#6B7280' }}>
                        P/E Ratio
                      </Typography>
                      <Typography variant="h5" sx={{ fontWeight: 700, mt: 0.5, color: '#1F2937' }}>
                        {research.fundamentals.pe_ratio?.toFixed(2) || 'N/A'}
                      </Typography>
                    </CardContent>
                  </Card>
                </Grid>
                <Grid item xs={12} sm={4}>
                  <Card variant="outlined" sx={{ bgcolor: '#F9FAFB', borderColor: '#E5E7EB' }}>
                    <CardContent sx={{ p: 2 }}>
                      <Typography variant="caption" sx={{ fontWeight: 600, color: '#6B7280' }}>
                        Revenue Growth
                      </Typography>
                      <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5, mt: 0.5 }}>
                        {research.fundamentals.revenue_growth > 0 ? (
                          <TrendingUp sx={{ color: '#00D09C', fontSize: 18 }} />
                        ) : (
                          <TrendingDown sx={{ color: '#EF4444', fontSize: 18 }} />
                        )}
                        <Typography
                          variant="h5"
                          sx={{
                            fontWeight: 700,
                            color: research.fundamentals.revenue_growth > 0 ? '#00D09C' : '#EF4444',
                          }}
                        >
                          {research.fundamentals.revenue_growth
                            ? (research.fundamentals.revenue_growth * 100).toFixed(1) + '%'
                            : 'N/A'}
                        </Typography>
                      </Box>
                    </CardContent>
                  </Card>
                </Grid>
                <Grid item xs={12} sm={4}>
                  <Card variant="outlined" sx={{ bgcolor: '#F9FAFB', borderColor: '#E5E7EB' }}>
                    <CardContent sx={{ p: 2 }}>
                      <Typography variant="caption" sx={{ fontWeight: 600, color: '#6B7280' }}>
                        Profit Margin
                      </Typography>
                      <Typography
                        variant="h5"
                        sx={{
                          fontWeight: 700,
                          mt: 0.5,
                          color: research.fundamentals.profit_margin > 0.1 ? '#00D09C' : '#1F2937',
                        }}
                      >
                        {research.fundamentals.profit_margin
                          ? (research.fundamentals.profit_margin * 100).toFixed(1) + '%'
                          : 'N/A'}
                      </Typography>
                    </CardContent>
                  </Card>
                </Grid>
              </Grid>
            </Box>
          </Fade>
        )}

        {research.sentiment && (
          <Fade in timeout={800}>
            <Box>
              {(research.news || research.fundamentals) && <Divider sx={{ my: 2.5 }} />}
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 2, gap: 1 }}>
                <SentimentSatisfied sx={{ color: '#00D09C', fontSize: 20 }} />
                <Typography variant="h6" sx={{ fontWeight: 600, color: '#1F2937' }}>
                  Sentiment Analysis
                </Typography>
              </Box>
              <Stack direction="row" spacing={1} flexWrap="wrap" useFlexGap>
                <Chip
                  label={`Social: ${(research.sentiment.social_sentiment * 100).toFixed(1)}%`}
                  size="small"
                  sx={{
                    fontWeight: 600,
                    bgcolor: research.sentiment.social_sentiment > 0.6 ? '#ECFDF5' : '#F3F4F6',
                    color: research.sentiment.social_sentiment > 0.6 ? '#00A67A' : '#374151',
                    fontSize: '0.75rem',
                    height: 28,
                  }}
                />
                <Chip
                  label={`Analyst: ${research.sentiment.analyst_rating}`}
                  size="small"
                  sx={{
                    fontWeight: 600,
                    bgcolor:
                      research.sentiment.analyst_rating === 'Buy'
                        ? '#ECFDF5'
                        : research.sentiment.analyst_rating === 'Sell'
                        ? '#FEF2F2'
                        : '#F3F4F6',
                    color:
                      research.sentiment.analyst_rating === 'Buy'
                        ? '#00A67A'
                        : research.sentiment.analyst_rating === 'Sell'
                        ? '#DC2626'
                        : '#374151',
                    fontSize: '0.75rem',
                    height: 28,
                  }}
                />
              </Stack>
            </Box>
          </Fade>
        )}
      </Box>
    );
  };

  const renderTechnical = (technical) => {
    if (!technical || Object.keys(technical).length === 0) {
      return (
        <Box sx={{ textAlign: 'center', py: 4 }}>
          <Assessment sx={{ fontSize: 40, color: '#D1D5DB', mb: 1 }} />
          <Typography variant="body2" sx={{ color: '#9CA3AF' }}>
            Technical analysis data will appear here...
          </Typography>
        </Box>
      );
    }

    return (
      <Box>
        <Grid container spacing={2} sx={{ mb: 3 }}>
          {technical.current_price !== undefined && (
            <Grid item xs={6} sm={4}>
              <Card variant="outlined" sx={{ bgcolor: '#ECFDF5', borderColor: '#00D09C' }}>
                <CardContent sx={{ p: 2 }}>
                  <Typography variant="caption" sx={{ fontWeight: 600, color: '#6B7280' }}>
                    Current Price
                  </Typography>
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5, mt: 0.5 }}>
                    <AttachMoney sx={{ color: '#00D09C', fontSize: 18 }} />
                    <Typography variant="h5" sx={{ fontWeight: 700, color: '#00D09C' }}>
                      {technical.current_price.toFixed(2)}
                    </Typography>
                  </Box>
                </CardContent>
              </Card>
            </Grid>
          )}
          {technical.rsi !== undefined && (
            <Grid item xs={6} sm={4}>
              <Card
                variant="outlined"
                sx={{
                  bgcolor:
                    technical.rsi > 70
                      ? '#FEF2F2'
                      : technical.rsi < 30
                      ? '#ECFDF5'
                      : '#F9FAFB',
                  borderColor:
                    technical.rsi > 70
                      ? '#EF4444'
                      : technical.rsi < 30
                      ? '#00D09C'
                      : '#E5E7EB',
                }}
              >
                <CardContent sx={{ p: 2 }}>
                  <Typography variant="caption" sx={{ fontWeight: 600, color: '#6B7280' }}>
                    RSI
                  </Typography>
                  <Typography
                    variant="h5"
                    sx={{
                      fontWeight: 700,
                      mt: 0.5,
                      color:
                        technical.rsi > 70
                          ? '#EF4444'
                          : technical.rsi < 30
                          ? '#00D09C'
                          : '#1F2937',
                    }}
                  >
                    {technical.rsi.toFixed(2)}
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
          )}
          {technical.ma20 !== undefined && (
            <Grid item xs={6} sm={4}>
              <Card variant="outlined" sx={{ bgcolor: '#F9FAFB', borderColor: '#E5E7EB' }}>
                <CardContent sx={{ p: 2 }}>
                  <Typography variant="caption" sx={{ fontWeight: 600, color: '#6B7280' }}>
                    MA20
                  </Typography>
                  <Typography variant="h5" sx={{ fontWeight: 700, mt: 0.5, color: '#1F2937' }}>
                    ${technical.ma20.toFixed(2)}
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
          )}
          {technical.ma50 !== undefined && (
            <Grid item xs={6} sm={4}>
              <Card variant="outlined" sx={{ bgcolor: '#F9FAFB', borderColor: '#E5E7EB' }}>
                <CardContent sx={{ p: 2 }}>
                  <Typography variant="caption" sx={{ fontWeight: 600, color: '#6B7280' }}>
                    MA50
                  </Typography>
                  <Typography variant="h5" sx={{ fontWeight: 700, mt: 0.5, color: '#1F2937' }}>
                    ${technical.ma50.toFixed(2)}
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
          )}
          {technical.ma200 !== undefined && (
            <Grid item xs={6} sm={4}>
              <Card variant="outlined" sx={{ bgcolor: '#F9FAFB', borderColor: '#E5E7EB' }}>
                <CardContent sx={{ p: 2 }}>
                  <Typography variant="caption" sx={{ fontWeight: 600, color: '#6B7280' }}>
                    MA200
                  </Typography>
                  <Typography variant="h5" sx={{ fontWeight: 700, mt: 0.5, color: '#1F2937' }}>
                    ${technical.ma200.toFixed(2)}
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
          )}
          {technical.trend && (
            <Grid item xs={6} sm={4}>
              <Card variant="outlined" sx={{ bgcolor: '#F9FAFB', borderColor: '#E5E7EB' }}>
                <CardContent sx={{ p: 2 }}>
                  <Typography variant="caption" sx={{ fontWeight: 600, color: '#6B7280' }}>
                    Trend
                  </Typography>
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5, mt: 0.5 }}>
                    {technical.trend.toLowerCase() === 'bullish' || technical.trend.toLowerCase() === 'uptrend' ? (
                      <TrendingUp sx={{ color: '#00D09C', fontSize: 18 }} />
                    ) : technical.trend.toLowerCase() === 'bearish' || technical.trend.toLowerCase() === 'downtrend' ? (
                      <TrendingDown sx={{ color: '#EF4444', fontSize: 18 }} />
                    ) : (
                      <Remove sx={{ color: '#9CA3AF', fontSize: 18 }} />
                    )}
                    <Typography variant="h5" sx={{ fontWeight: 700, textTransform: 'capitalize', color: '#1F2937' }}>
                      {technical.trend}
                    </Typography>
                  </Box>
                </CardContent>
              </Card>
            </Grid>
          )}
          {technical.macd_signal !== undefined && (
            <Grid item xs={6} sm={4}>
              <Card variant="outlined" sx={{ bgcolor: '#F9FAFB', borderColor: '#E5E7EB' }}>
                <CardContent sx={{ p: 2 }}>
                  <Typography variant="caption" sx={{ fontWeight: 600, color: '#6B7280' }}>
                    MACD Signal
                  </Typography>
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5, mt: 0.5 }}>
                    {technical.macd_signal > 0 ? (
                      <TrendingUp sx={{ color: '#00D09C', fontSize: 18 }} />
                    ) : (
                      <TrendingDown sx={{ color: '#EF4444', fontSize: 18 }} />
                    )}
                    <Typography variant="h5" sx={{ fontWeight: 700, color: technical.macd_signal > 0 ? '#00D09C' : '#EF4444' }}>
                      {technical.macd_signal > 0 ? '+' : ''}{technical.macd_signal.toFixed(3)}
                    </Typography>
                  </Box>
                  <Typography variant="caption" sx={{ color: '#9CA3AF', mt: 0.5, display: 'block' }}>
                    {technical.macd_signal > 0 ? 'Bullish' : 'Bearish'}
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
          )}
        </Grid>
      </Box>
    );
  };

  const renderInsights = (insights) => {
    if (!insights || insights.length === 0) {
      return (
        <Box sx={{ textAlign: 'center', py: 4 }}>
          <Lightbulb sx={{ fontSize: 40, color: '#D1D5DB', mb: 1 }} />
          <Typography variant="body2" sx={{ color: '#9CA3AF' }}>
            AI-generated insights will appear here...
          </Typography>
        </Box>
      );
    }

    return (
      <List sx={{ p: 0 }}>
        {insights.map((insight, i) => (
          <Fade in timeout={200 * (i + 1)} key={i}>
            <ListItem
              sx={{
                bgcolor: '#F9FAFB',
                borderRadius: 2,
                mb: 1,
                border: '1px solid #E5E7EB',
              }}
            >
              <ListItemText
                primary={insight}
                primaryTypographyProps={{
                  variant: 'body2',
                  sx: { fontWeight: 500, lineHeight: 1.6, color: '#374151' },
                }}
              />
            </ListItem>
          </Fade>
        ))}
      </List>
    );
  };

  const renderRLOptimizations = (rlOpts) => {
    if (!rlOpts || Object.keys(rlOpts).length === 0) {
      return (
        <Box sx={{ textAlign: 'center', py: 4 }}>
          <Psychology sx={{ fontSize: 40, color: '#D1D5DB', mb: 1 }} />
          <Typography variant="body2" sx={{ color: '#9CA3AF' }}>
            RL optimizations will appear here...
          </Typography>
        </Box>
      );
    }

    return (
      <Box>
        <Typography variant="h6" sx={{ fontWeight: 600, mb: 3, color: '#1F2937', display: 'flex', alignItems: 'center', gap: 1 }}>
          <AutoAwesome sx={{ color: '#00D09C', fontSize: 24 }} />
          Reinforcement Learning Optimizations
        </Typography>

        <Grid container spacing={2}>
          {/* Entry/Exit Timing */}
          {rlOpts.entry_exit_timing && (
            <Grid item xs={12} sm={6} md={4}>
              <Card sx={{ bgcolor: '#F9FAFB', border: '1px solid #E5E7EB', height: '100%' }}>
                <CardContent>
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1 }}>
                    <TrendingUp sx={{ color: '#00D09C', fontSize: 20 }} />
                    <Typography variant="subtitle2" sx={{ fontWeight: 600, color: '#6B7280' }}>
                      Entry/Exit Timing
                    </Typography>
                  </Box>
                  <Chip
                    label={rlOpts.entry_exit_timing}
                    size="small"
                    sx={{
                      bgcolor: '#ECFDF5',
                      color: '#00A67A',
                      fontWeight: 600,
                      fontSize: '0.75rem',
                    }}
                  />
                </CardContent>
              </Card>
            </Grid>
          )}

          {/* Position Sizing */}
          {rlOpts.position_size !== undefined && (
            <Grid item xs={12} sm={6} md={4}>
              <Card sx={{ bgcolor: '#F9FAFB', border: '1px solid #E5E7EB', height: '100%' }}>
                <CardContent>
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1 }}>
                    <AttachMoney sx={{ color: '#00D09C', fontSize: 20 }} />
                    <Typography variant="subtitle2" sx={{ fontWeight: 600, color: '#6B7280' }}>
                      Position Size
                    </Typography>
                  </Box>
                  <Typography variant="h5" sx={{ fontWeight: 700, color: '#00D09C' }}>
                    {(rlOpts.position_size * 100).toFixed(1)}%
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
          )}

          {/* Risk Management */}
          {rlOpts.risk_management && (
            <Grid item xs={12} sm={6} md={4}>
              <Card sx={{ bgcolor: '#F9FAFB', border: '1px solid #E5E7EB', height: '100%' }}>
                <CardContent>
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1 }}>
                    <Warning sx={{ color: '#F59E0B', fontSize: 20 }} />
                    <Typography variant="subtitle2" sx={{ fontWeight: 600, color: '#6B7280' }}>
                      Risk Management
                    </Typography>
                  </Box>
                  <Chip
                    label={rlOpts.risk_management}
                    size="small"
                    sx={{
                      bgcolor: '#FEF3C7',
                      color: '#D97706',
                      fontWeight: 600,
                      fontSize: '0.75rem',
                    }}
                  />
                </CardContent>
              </Card>
            </Grid>
          )}

          {/* Calibrated Confidence */}
          {rlOpts.calibrated_confidence !== undefined && (
            <Grid item xs={12} sm={6} md={4}>
              <Card sx={{ bgcolor: '#F9FAFB', border: '1px solid #E5E7EB', height: '100%' }}>
                <CardContent>
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1 }}>
                    <Assessment sx={{ color: '#00D09C', fontSize: 20 }} />
                    <Typography variant="subtitle2" sx={{ fontWeight: 600, color: '#6B7280' }}>
                      Calibrated Confidence
                    </Typography>
                  </Box>
                  <Typography variant="h5" sx={{ fontWeight: 700, color: '#00D09C' }}>
                    {(rlOpts.calibrated_confidence * 100).toFixed(1)}%
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
          )}

          {/* Stop Loss & Take Profit */}
          {rlOpts.stop_loss_tp && (
            <Grid item xs={12} sm={6} md={4}>
              <Card sx={{ bgcolor: '#F9FAFB', border: '1px solid #E5E7EB', height: '100%' }}>
                <CardContent>
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 2 }}>
                    <ShowChart sx={{ color: '#00D09C', fontSize: 20 }} />
                    <Typography variant="subtitle2" sx={{ fontWeight: 600, color: '#6B7280' }}>
                      Stop Loss / Take Profit
                    </Typography>
                  </Box>
                  <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
                    <Box>
                      <Typography variant="caption" sx={{ color: '#6B7280' }}>Stop Loss</Typography>
                      <Typography variant="body1" sx={{ fontWeight: 600, color: '#EF4444' }}>
                        ${rlOpts.stop_loss_tp.stop_loss?.toFixed(2)} ({(rlOpts.stop_loss_tp.stop_loss_pct * 100).toFixed(1)}%)
                      </Typography>
                    </Box>
                    <Box>
                      <Typography variant="caption" sx={{ color: '#6B7280' }}>Take Profit</Typography>
                      <Typography variant="body1" sx={{ fontWeight: 600, color: '#00D09C' }}>
                        ${rlOpts.stop_loss_tp.take_profit?.toFixed(2)} ({(rlOpts.stop_loss_tp.take_profit_pct * 100).toFixed(1)}%)
                      </Typography>
                    </Box>
                  </Box>
                </CardContent>
              </Card>
            </Grid>
          )}

          {/* Feature Weights */}
          {rlOpts.feature_weights && (
            <Grid item xs={12} md={6}>
              <Card sx={{ bgcolor: '#F9FAFB', border: '1px solid #E5E7EB' }}>
                <CardContent>
                  <Typography variant="subtitle2" sx={{ fontWeight: 600, mb: 2, color: '#1F2937' }}>
                    Top Feature Weights
                  </Typography>
                  <List dense>
                    {Object.entries(rlOpts.feature_weights)
                      .sort((a, b) => b[1] - a[1])
                      .slice(0, 5)
                      .map(([feature, weight], i) => (
                        <ListItem key={i} sx={{ py: 0.5, px: 1 }}>
                          <ListItemText
                            primary={feature.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}
                            secondary={`${(weight * 100).toFixed(1)}%`}
                            primaryTypographyProps={{ variant: 'body2', fontWeight: 500 }}
                            secondaryTypographyProps={{ variant: 'body2', color: '#00D09C', fontWeight: 600 }}
                          />
                        </ListItem>
                      ))}
                  </List>
                </CardContent>
              </Card>
            </Grid>
          )}

          {/* Timeframe Weights */}
          {rlOpts.timeframe_weights && (
            <Grid item xs={12} md={6}>
              <Card sx={{ bgcolor: '#F9FAFB', border: '1px solid #E5E7EB' }}>
                <CardContent>
                  <Typography variant="subtitle2" sx={{ fontWeight: 600, mb: 2, color: '#1F2937' }}>
                    Timeframe Weights
                  </Typography>
                  <List dense>
                    {Object.entries(rlOpts.timeframe_weights)
                      .sort((a, b) => b[1] - a[1])
                      .map(([timeframe, weight], i) => (
                        <ListItem key={i} sx={{ py: 0.5, px: 1 }}>
                          <ListItemText
                            primary={timeframe}
                            secondary={`${(weight * 100).toFixed(1)}%`}
                            primaryTypographyProps={{ variant: 'body2', fontWeight: 500 }}
                            secondaryTypographyProps={{ variant: 'body2', color: '#00D09C', fontWeight: 600 }}
                          />
                        </ListItem>
                      ))}
                  </List>
                </CardContent>
              </Card>
            </Grid>
          )}

          {/* Sentiment Weights */}
          {rlOpts.sentiment_weights && (
            <Grid item xs={12}>
              <Card sx={{ bgcolor: '#F9FAFB', border: '1px solid #E5E7EB' }}>
                <CardContent>
                  <Typography variant="subtitle2" sx={{ fontWeight: 600, mb: 2, color: '#1F2937' }}>
                    News Source Weights
                  </Typography>
                  <Stack direction="row" spacing={1} flexWrap="wrap" useFlexGap>
                    {Object.entries(rlOpts.sentiment_weights)
                      .sort((a, b) => b[1] - a[1])
                      .map(([source, weight], i) => (
                        <Chip
                          key={i}
                          label={`${source.replace(/_/g, ' ')}: ${(weight * 100).toFixed(1)}%`}
                          size="small"
                          sx={{
                            bgcolor: '#ECFDF5',
                            color: '#00A67A',
                            fontWeight: 500,
                            fontSize: '0.75rem',
                          }}
                        />
                      ))}
                  </Stack>
                </CardContent>
              </Card>
            </Grid>
          )}
        </Grid>
      </Box>
    );
  };

  const renderRisks = (risks) => {
    if (!risks || risks.length === 0) {
      return (
        <Box sx={{ textAlign: 'center', py: 4 }}>
          <Warning sx={{ fontSize: 40, color: '#D1D5DB', mb: 1 }} />
          <Typography variant="body2" sx={{ color: '#9CA3AF' }}>
            Risk analysis will appear here...
          </Typography>
        </Box>
      );
    }

    const getSeverityColor = (severity) => {
      if (severity === 'High') return { bg: '#FEF2F2', text: '#DC2626', border: '#EF4444' };
      if (severity === 'Medium') return { bg: '#FEF3C7', text: '#D97706', border: '#F59E0B' };
      return { bg: '#F3F4F6', text: '#6B7280', border: '#9CA3AF' };
    };

    return (
      <List sx={{ p: 0 }}>
        {risks.map((risk, i) => {
          const colors = getSeverityColor(risk.severity);
          return (
            <Fade in timeout={200 * (i + 1)} key={i}>
              <ListItem
                sx={{
                  bgcolor: colors.bg,
                  borderRadius: 2,
                  mb: 1.5,
                  border: `1px solid ${colors.border}`,
                  p: 2,
                }}
              >
                <Box sx={{ width: '100%' }}>
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1 }}>
                    <Warning sx={{ color: colors.text, fontSize: 20 }} />
                    <Chip
                      label={risk.severity}
                      size="small"
                      sx={{
                        bgcolor: colors.bg,
                        color: colors.text,
                        border: `1px solid ${colors.border}`,
                        fontWeight: 600,
                        fontSize: '0.75rem',
                      }}
                    />
                    <Chip
                      label={risk.category}
                      size="small"
                      sx={{
                        bgcolor: '#F3F4F6',
                        color: '#374151',
                        fontWeight: 500,
                        fontSize: '0.75rem',
                      }}
                    />
                  </Box>
                  <Typography
                    variant="body2"
                    sx={{
                      color: '#374151',
                      lineHeight: 1.6,
                      fontWeight: 400,
                    }}
                  >
                    {risk.description}
                  </Typography>
                </Box>
              </ListItem>
            </Fade>
          );
        })}
      </List>
    );
  };

  const renderRecommendation = (recommendation, confidence, priceLevels) => {
    if (!recommendation) {
      return (
        <Box sx={{ textAlign: 'center', py: 4 }}>
          <TrendingUp sx={{ fontSize: 40, color: '#D1D5DB', mb: 1 }} />
          <Typography variant="body2" sx={{ color: '#9CA3AF' }}>
            Investment recommendation will appear here...
          </Typography>
        </Box>
      );
    }

    const getRecommendationColor = (rec) => {
      if (rec === 'BUY' || rec === 'Buy') return { bg: '#ECFDF5', text: '#00A67A', border: '#00D09C' };
      if (rec === 'SELL' || rec === 'Sell') return { bg: '#FEF2F2', text: '#DC2626', border: '#EF4444' };
      return { bg: '#FEF3C7', text: '#D97706', border: '#F59E0B' };
    };

    const getRecommendationIcon = (rec) => {
      if (rec === 'BUY' || rec === 'Buy') return <TrendingUp />;
      if (rec === 'SELL' || rec === 'Sell') return <TrendingDown />;
      return <Remove />;
    };

    const colors = getRecommendationColor(recommendation);

    return (
      <Box>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, mb: 3, flexWrap: 'wrap' }}>
          <Chip
            icon={getRecommendationIcon(recommendation)}
            label={recommendation}
            sx={{
              fontSize: '1.125rem',
              fontWeight: 700,
              height: 44,
              px: 2,
              bgcolor: colors.bg,
              color: colors.text,
              border: `2px solid ${colors.border}`,
              '& .MuiChip-icon': {
                color: colors.text,
              },
            }}
          />
          {confidence !== null && (
            <Box>
              <Typography variant="caption" sx={{ fontWeight: 600, color: '#6B7280' }}>
                Confidence
              </Typography>
              <Typography variant="h4" sx={{ fontWeight: 700, color: '#00D09C' }}>
                {(confidence * 100).toFixed(1)}%
              </Typography>
            </Box>
          )}
        </Box>

        {/* Price Levels */}
        {priceLevels && (
          <Box sx={{ mt: 3 }}>
            <Typography variant="h6" sx={{ fontWeight: 600, mb: 2, color: '#1F2937' }}>
              Trading Levels
            </Typography>
            <Grid container spacing={2}>
              {priceLevels.entry_price && (
                <Grid item xs={6} sm={3}>
                  <Card sx={{ bgcolor: '#F3F4F6', border: '1px solid #E5E7EB' }}>
                    <CardContent>
                      <Typography variant="caption" sx={{ color: '#6B7280', fontWeight: 600 }}>
                        Entry Price
                      </Typography>
                      <Typography variant="h6" sx={{ fontWeight: 700, color: '#00D09C' }}>
                        ${priceLevels.entry_price.toFixed(2)}
                      </Typography>
                    </CardContent>
                  </Card>
                </Grid>
              )}
              {priceLevels.stop_loss && (
                <Grid item xs={6} sm={3}>
                  <Card sx={{ bgcolor: '#FEF2F2', border: '1px solid #EF4444' }}>
                    <CardContent>
                      <Typography variant="caption" sx={{ color: '#6B7280', fontWeight: 600 }}>
                        Stop Loss
                      </Typography>
                      <Typography variant="h6" sx={{ fontWeight: 700, color: '#EF4444' }}>
                        ${priceLevels.stop_loss.toFixed(2)}
                      </Typography>
                    </CardContent>
                  </Card>
                </Grid>
              )}
              {priceLevels.exit_price && (
                <Grid item xs={6} sm={3}>
                  <Card sx={{ bgcolor: '#ECFDF5', border: '1px solid #00D09C' }}>
                    <CardContent>
                      <Typography variant="caption" sx={{ color: '#6B7280', fontWeight: 600 }}>
                        Exit Price
                      </Typography>
                      <Typography variant="h6" sx={{ fontWeight: 700, color: '#00A67A' }}>
                        ${priceLevels.exit_price.toFixed(2)}
                      </Typography>
                    </CardContent>
                  </Card>
                </Grid>
              )}
              {priceLevels.resistance_levels && priceLevels.resistance_levels.length > 0 && (
                <Grid item xs={6} sm={3}>
                  <Card sx={{ bgcolor: '#FEF3C7', border: '1px solid #F59E0B' }}>
                    <CardContent>
                      <Typography variant="caption" sx={{ color: '#6B7280', fontWeight: 600 }}>
                        Resistance
                      </Typography>
                      <Typography variant="h6" sx={{ fontWeight: 700, color: '#D97706' }}>
                        ${priceLevels.resistance_levels[0].toFixed(2)}
                      </Typography>
                      {priceLevels.resistance_levels.length > 1 && (
                        <Typography variant="caption" sx={{ color: '#9CA3AF' }}>
                          +{priceLevels.resistance_levels.length - 1} more
                        </Typography>
                      )}
                    </CardContent>
                  </Card>
                </Grid>
              )}
            </Grid>

            {/* Detailed Levels */}
            {(priceLevels.resistance_levels?.length > 0 || priceLevels.support_levels?.length > 0) && (
              <Box sx={{ mt: 3 }}>
                <Grid container spacing={2}>
                  {priceLevels.resistance_levels && priceLevels.resistance_levels.length > 0 && (
                    <Grid item xs={12} sm={6}>
                      <Typography variant="subtitle2" sx={{ fontWeight: 600, mb: 1, color: '#374151' }}>
                        Resistance Levels
                      </Typography>
                      <List dense>
                        {priceLevels.resistance_levels.map((level, i) => (
                          <ListItem key={i} sx={{ py: 0.5, px: 1, bgcolor: '#FEF3C7', borderRadius: 1, mb: 0.5 }}>
                            <ListItemText
                              primary={`R${i + 1}: $${level.toFixed(2)}`}
                              primaryTypographyProps={{ variant: 'body2', fontWeight: 500 }}
                            />
                          </ListItem>
                        ))}
                      </List>
                    </Grid>
                  )}
                  {priceLevels.support_levels && priceLevels.support_levels.length > 0 && (
                    <Grid item xs={12} sm={6}>
                      <Typography variant="subtitle2" sx={{ fontWeight: 600, mb: 1, color: '#374151' }}>
                        Support Levels
                      </Typography>
                      <List dense>
                        {priceLevels.support_levels.map((level, i) => (
                          <ListItem key={i} sx={{ py: 0.5, px: 1, bgcolor: '#ECFDF5', borderRadius: 1, mb: 0.5 }}>
                            <ListItemText
                              primary={`S${i + 1}: $${level.toFixed(2)}`}
                              primaryTypographyProps={{ variant: 'body2', fontWeight: 500 }}
                            />
                          </ListItem>
                        ))}
                      </List>
                    </Grid>
                  )}
                </Grid>
              </Box>
            )}
          </Box>
        )}
      </Box>
    );
  };

  const hasData = Object.keys(outputs.research).length > 0 || Object.keys(outputs.technical).length > 0;
  const hasCharts = outputs.technical?.ohlcv && outputs.technical.ohlcv.length > 0;

  if (!hasData && !formattedReport) {
    return null;
  }

  return (
    <Paper
      elevation={0}
      sx={{
        borderRadius: 2,
        border: '1px solid #E5E7EB',
        bgcolor: '#FFFFFF',
        overflow: 'hidden',
      }}
    >
      <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
        <Tabs
          value={tabValue}
          onChange={handleTabChange}
          aria-label="output tabs"
          sx={{
            '& .MuiTab-root': {
              textTransform: 'none',
              fontWeight: 600,
              fontSize: '0.875rem',
              minHeight: 64,
              '&.Mui-selected': {
                color: '#00D09C',
              },
            },
            '& .MuiTabs-indicator': {
              bgcolor: '#00D09C',
              height: 3,
            },
          }}
        >
          <Tab
            icon={<Article sx={{ fontSize: 20 }} />}
            iconPosition="start"
            label="Research"
            id="output-tab-0"
            aria-controls="output-tabpanel-0"
          />
          <Tab
            icon={<Assessment sx={{ fontSize: 20 }} />}
            iconPosition="start"
            label="Technical"
            id="output-tab-1"
            aria-controls="output-tabpanel-1"
          />
          <Tab
            icon={<Lightbulb sx={{ fontSize: 20 }} />}
            iconPosition="start"
            label="Insights"
            id="output-tab-2"
            aria-controls="output-tabpanel-2"
          />
          <Tab
            icon={<TrendingUp sx={{ fontSize: 20 }} />}
            iconPosition="start"
            label="Recommendation"
            id="output-tab-3"
            aria-controls="output-tabpanel-3"
          />
          <Tab
            icon={<Warning sx={{ fontSize: 20 }} />}
            iconPosition="start"
            label="Risks"
            id="output-tab-4"
            aria-controls="output-tabpanel-4"
          />
          {outputs.rl_optimizations && (
            <Tab
              icon={<Psychology sx={{ fontSize: 20 }} />}
              iconPosition="start"
              label="RL Optimizations"
              id="output-tab-5"
              aria-controls="output-tabpanel-5"
            />
          )}
          {hasCharts && (
            <Tab
              icon={<ShowChart sx={{ fontSize: 20 }} />}
              iconPosition="start"
              label="Charts"
              id="output-tab-6"
              aria-controls="output-tabpanel-6"
            />
          )}
          {formattedReport && (
            <Tab
              icon={<Description sx={{ fontSize: 20 }} />}
              iconPosition="start"
              label="Report"
              id="output-tab-6"
              aria-controls="output-tabpanel-6"
            />
          )}
        </Tabs>
      </Box>

      <Box sx={{ p: { xs: 2, md: 3 } }}>
        <TabPanel value={tabValue} index={0}>
          {renderResearch(outputs.research)}
        </TabPanel>

        <TabPanel value={tabValue} index={1}>
          {renderTechnical(outputs.technical)}
        </TabPanel>

        <TabPanel value={tabValue} index={2}>
          {renderInsights(outputs.insights)}
        </TabPanel>

        <TabPanel value={tabValue} index={3}>
          {renderRecommendation(outputs.recommendation, outputs.confidence, outputs.price_levels)}
        </TabPanel>

          <TabPanel value={tabValue} index={4}>
          {renderRisks(outputs.risks)}
        </TabPanel>

        {outputs.rl_optimizations && (
          <TabPanel value={tabValue} index={5}>
            {renderRLOptimizations(outputs.rl_optimizations)}
          </TabPanel>
        )}
        {hasCharts && (
          <TabPanel value={tabValue} index={outputs.rl_optimizations ? 6 : 5}>
            <TechnicalCharts ohlcv={outputs.technical.ohlcv} rsi={outputs.technical.rsi} priceLevels={outputs.price_levels} />
          </TabPanel>
        )}

        {formattedReport && (
          <TabPanel value={tabValue} index={outputs.rl_optimizations ? (hasCharts ? 7 : 6) : (hasCharts ? 6 : 5)}>
            <Box
              sx={{
                bgcolor: '#F9FAFB',
                p: 2.5,
                borderRadius: 1.5,
                maxHeight: 600,
                overflow: 'auto',
                border: '1px solid #E5E7EB',
                '& h1, & h2, & h3': {
                  fontWeight: 600,
                  mt: 2,
                  mb: 1,
                  color: '#1F2937',
                  fontSize: '1rem',
                },
                '& p': {
                  mb: 1,
                  lineHeight: 1.6,
                  color: '#374151',
                  fontSize: '0.875rem',
                },
                '& ul, & ol': {
                  pl: 2.5,
                  mb: 1,
                },
                '& li': {
                  mb: 0.5,
                  color: '#374151',
                  fontSize: '0.875rem',
                },
                '& strong': {
                  fontWeight: 600,
                  color: '#1F2937',
                },
              }}
            >
              <ReactMarkdown>{formattedReport}</ReactMarkdown>
            </Box>
          </TabPanel>
        )}
      </Box>
    </Paper>
  );
}

export default OutputPanels;
