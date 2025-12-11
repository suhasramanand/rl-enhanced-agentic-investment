import React from 'react';
import { Paper, Typography, Box, Chip } from '@mui/material';
import {
  LineChart,
  Line,
  AreaChart,
  Area,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  ReferenceLine,
} from 'recharts';

const CustomTooltip = ({ active, payload, label }) => {
  if (active && payload && payload.length) {
    return (
      <Box
        sx={{
          bgcolor: '#FFFFFF',
          p: 1.5,
          border: '1px solid #E5E7EB',
          borderRadius: 2,
          boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)',
        }}
      >
        <Typography variant="body2" sx={{ fontWeight: 700, mb: 0.5, color: '#1F2937' }}>
          {label}
        </Typography>
        {payload.map((p, i) => (
          <Typography key={i} variant="body2" sx={{ color: p.color, fontWeight: 500 }}>
            {`${p.name}: ${p.value?.toFixed(2) || 'N/A'}`}
          </Typography>
        ))}
      </Box>
    );
  }
  return null;
};

function TechnicalCharts({ ohlcv, rsi, priceLevels }) {
  if (!ohlcv || ohlcv.length === 0) {
    return null;
  }

  const chartData = ohlcv.map((data) => ({
    date: new Date(data.date).toLocaleDateString('en-US', { month: 'short', day: 'numeric' }),
    open: data.open,
    high: data.high,
    low: data.low,
    close: data.close,
    volume: data.volume,
    ma20: data.ma20,
    ma50: data.ma50,
    ma200: data.ma200,
    rsi: data.rsi || (rsi !== undefined && ohlcv.length > 0 ? rsi : undefined),
  }));

  const rsiData = chartData.map((d) => ({
    date: d.date,
    rsi: d.rsi || 50,
  }));

  const getRSIColor = (rsiValue) => {
    if (rsiValue > 70) return { bg: '#FEF2F2', text: '#DC2626', border: '#EF4444' };
    if (rsiValue < 30) return { bg: '#ECFDF5', text: '#00A67A', border: '#00D09C' };
    return { bg: '#F3F4F6', text: '#374151', border: '#D1D5DB' };
  };

  const rsiColors = getRSIColor(rsi);

  return (
    <Box>
      {/* Price & Moving Averages */}
      <Paper
        variant="outlined"
        sx={{
          p: 3,
          mb: 3,
          borderRadius: 2,
          borderColor: '#E5E7EB',
          bgcolor: '#FFFFFF',
        }}
      >
        <Typography variant="h6" sx={{ fontWeight: 600, mb: 2, color: '#1F2937' }}>
          Price & Moving Averages
        </Typography>
        <ResponsiveContainer width="100%" height={400}>
          <AreaChart data={chartData} margin={{ top: 10, right: 30, left: 0, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#E5E7EB" />
            <XAxis
              dataKey="date"
              tick={{ fontSize: 11, fill: '#6B7280' }}
              angle={-45}
              textAnchor="end"
              height={80}
            />
            <YAxis
              domain={['dataMin - 10', 'dataMax + 10']}
              tick={{ fontSize: 11, fill: '#6B7280' }}
            />
            <Tooltip content={<CustomTooltip />} />
            <Legend />
            <Area
              type="monotone"
              dataKey="close"
              stroke="#00D09C"
              fill="#00D09C"
              fillOpacity={0.15}
              name="Close Price"
              strokeWidth={2}
            />
            {chartData[0]?.ma20 !== undefined && (
              <Line
                type="monotone"
                dataKey="ma20"
                stroke="#F59E0B"
                dot={false}
                strokeWidth={2}
                name="MA20"
              />
            )}
            {chartData[0]?.ma50 !== undefined && (
              <Line
                type="monotone"
                dataKey="ma50"
                stroke="#6366F1"
                dot={false}
                strokeWidth={2}
                name="MA50"
              />
            )}
            {chartData[0]?.ma200 !== undefined && (
              <Line
                type="monotone"
                dataKey="ma200"
                stroke="#8B5CF6"
                dot={false}
                strokeWidth={2}
                name="MA200"
              />
            )}
            {/* Entry Price Line */}
            {priceLevels?.entry_price && (
              <ReferenceLine
                y={priceLevels.entry_price}
                stroke="#00D09C"
                strokeWidth={2.5}
                strokeDasharray="8 4"
                label={{ value: `Entry: $${priceLevels.entry_price.toFixed(2)}`, position: 'right', fill: '#00D09C', fontSize: 12, fontWeight: 600 }}
              />
            )}
            {/* Stop Loss Line */}
            {priceLevels?.stop_loss && (
              <ReferenceLine
                y={priceLevels.stop_loss}
                stroke="#EF4444"
                strokeWidth={2.5}
                strokeDasharray="8 4"
                label={{ value: `Stop Loss: $${priceLevels.stop_loss.toFixed(2)}`, position: 'right', fill: '#EF4444', fontSize: 12, fontWeight: 600 }}
              />
            )}
            {/* Exit Price Line */}
            {priceLevels?.exit_price && (
              <ReferenceLine
                y={priceLevels.exit_price}
                stroke="#10B981"
                strokeWidth={2.5}
                strokeDasharray="5 5"
                label={{ value: `Exit: $${priceLevels.exit_price.toFixed(2)}`, position: 'right', fill: '#10B981', fontSize: 12, fontWeight: 600 }}
              />
            )}
            {/* Resistance Levels */}
            {priceLevels?.resistance_levels?.map((level, i) => (
              <ReferenceLine
                key={`resistance-${i}`}
                y={level}
                stroke="#F59E0B"
                strokeWidth={1.5}
                strokeDasharray="3 3"
                label={{ value: `R${i + 1}: $${level.toFixed(2)}`, position: 'right', fill: '#F59E0B', fontSize: 10 }}
              />
            ))}
            {/* Support Levels */}
            {priceLevels?.support_levels?.map((level, i) => (
              <ReferenceLine
                key={`support-${i}`}
                y={level}
                stroke="#6366F1"
                strokeWidth={1.5}
                strokeDasharray="3 3"
                label={{ value: `S${i + 1}: $${level.toFixed(2)}`, position: 'right', fill: '#6366F1', fontSize: 10 }}
              />
            ))}
          </AreaChart>
        </ResponsiveContainer>
      </Paper>

      {/* RSI */}
      {rsi !== undefined && (
        <Paper
          variant="outlined"
          sx={{
            p: 3,
            mb: 3,
            borderRadius: 2,
            borderColor: rsiColors.border,
            bgcolor: rsiColors.bg,
          }}
        >
          <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
            <Typography variant="h6" sx={{ fontWeight: 600, color: '#1F2937' }}>
              RSI (Relative Strength Index)
            </Typography>
            <Chip
              label={`Current: ${rsi.toFixed(2)}`}
              size="small"
              sx={{
                fontWeight: 700,
                bgcolor: rsiColors.bg,
                color: rsiColors.text,
                border: `1px solid ${rsiColors.border}`,
                fontSize: '0.75rem',
              }}
            />
          </Box>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={rsiData} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#E5E7EB" />
              <XAxis
                dataKey="date"
                tick={{ fontSize: 11, fill: '#6B7280' }}
                angle={-45}
                textAnchor="end"
                height={80}
              />
              <YAxis domain={[0, 100]} tick={{ fontSize: 11, fill: '#6B7280' }} />
              <Tooltip content={<CustomTooltip />} />
              <Legend />
              <Line
                type="monotone"
                dataKey="rsi"
                stroke="#EF4444"
                strokeWidth={2.5}
                name="RSI"
                dot={{ fill: '#EF4444', r: 3 }}
              />
              <Line
                type="monotone"
                dataKey={() => 70}
                stroke="#9CA3AF"
                dot={false}
                strokeDasharray="5 5"
                strokeWidth={1.5}
                name="Overbought (70)"
              />
              <Line
                type="monotone"
                dataKey={() => 30}
                stroke="#9CA3AF"
                dot={false}
                strokeDasharray="5 5"
                strokeWidth={1.5}
                name="Oversold (30)"
              />
            </LineChart>
          </ResponsiveContainer>
        </Paper>
      )}

      {/* Volume */}
      {chartData[0]?.volume !== undefined && (
        <Paper
          variant="outlined"
          sx={{
            p: 3,
            borderRadius: 2,
            borderColor: '#E5E7EB',
            bgcolor: '#FFFFFF',
          }}
        >
          <Typography variant="h6" sx={{ fontWeight: 600, mb: 2, color: '#1F2937' }}>
            Volume
          </Typography>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={chartData} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#E5E7EB" />
              <XAxis
                dataKey="date"
                tick={{ fontSize: 11, fill: '#6B7280' }}
                angle={-45}
                textAnchor="end"
                height={80}
              />
              <YAxis tick={{ fontSize: 11, fill: '#6B7280' }} />
              <Tooltip content={<CustomTooltip />} />
              <Legend />
              <Bar dataKey="volume" fill="#00D09C" name="Volume" radius={[6, 6, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </Paper>
      )}
    </Box>
  );
}

export default TechnicalCharts;
