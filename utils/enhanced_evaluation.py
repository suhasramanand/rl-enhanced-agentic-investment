"""
Enhanced Evaluation Metrics
Includes Sharpe ratio, maximum drawdown, risk-adjusted returns, and walk-forward analysis.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta


def calculate_sharpe_ratio(returns: List[float], risk_free_rate: float = 0.02) -> float:
    """
    Calculate Sharpe ratio for a series of returns.
    
    Args:
        returns: List of returns (as decimals, e.g., 0.05 for 5%)
        risk_free_rate: Annual risk-free rate (default 2%)
    
    Returns:
        Sharpe ratio
    """
    if len(returns) == 0:
        return 0.0
    
    returns_array = np.array(returns)
    
    # Annualize if needed (assuming daily returns)
    mean_return = np.mean(returns_array) * 252  # Annualize daily returns
    std_return = np.std(returns_array) * np.sqrt(252)  # Annualize volatility
    
    if std_return == 0:
        return 0.0
    
    sharpe = (mean_return - risk_free_rate) / std_return
    return float(sharpe)


def calculate_maximum_drawdown(returns: List[float]) -> Dict[str, float]:
    """
    Calculate maximum drawdown and related metrics.
    
    Args:
        returns: List of returns (as decimals)
    
    Returns:
        Dictionary with max_drawdown, max_drawdown_period, and recovery_time
    """
    if len(returns) == 0:
        return {
            'max_drawdown': 0.0,
            'max_drawdown_period': 0,
            'recovery_time': 0,
            'drawdown_duration': 0
        }
    
    # Calculate cumulative returns
    cumulative = np.cumprod(1 + np.array(returns))
    
    # Calculate running maximum
    running_max = np.maximum.accumulate(cumulative)
    
    # Calculate drawdown
    drawdown = (cumulative - running_max) / running_max
    
    # Find maximum drawdown
    max_dd = float(np.min(drawdown))
    max_dd_idx = int(np.argmin(drawdown))
    
    # Find peak before drawdown
    peak_idx = int(np.argmax(cumulative[:max_dd_idx + 1]))
    
    # Find recovery point (where cumulative returns exceed previous peak)
    if max_dd_idx < len(cumulative) - 1:
        recovery_idx = None
        peak_value = cumulative[peak_idx]
        for i in range(max_dd_idx + 1, len(cumulative)):
            if cumulative[i] >= peak_value:
                recovery_idx = i
                break
        recovery_time = (recovery_idx - max_dd_idx) if recovery_idx else None
    else:
        recovery_time = None
    
    drawdown_duration = max_dd_idx - peak_idx if max_dd_idx > peak_idx else 0
    
    return {
        'max_drawdown': abs(max_dd),
        'max_drawdown_period': max_dd_idx,
        'recovery_time': recovery_time,
        'drawdown_duration': drawdown_duration,
        'peak_index': peak_idx,
        'trough_index': max_dd_idx
    }


def calculate_risk_adjusted_returns(returns: List[float]) -> Dict[str, float]:
    """
    Calculate various risk-adjusted return metrics.
    
    Args:
        returns: List of returns (as decimals)
    
    Returns:
        Dictionary with various risk-adjusted metrics
    """
    if len(returns) == 0:
        return {
            'sharpe_ratio': 0.0,
            'sortino_ratio': 0.0,
            'calmar_ratio': 0.0,
            'information_ratio': 0.0,
            'total_return': 0.0,
            'volatility': 0.0,
            'downside_deviation': 0.0
        }
    
    returns_array = np.array(returns)
    
    # Basic metrics
    total_return = float(np.prod(1 + returns_array) - 1)
    mean_return = float(np.mean(returns_array))
    volatility = float(np.std(returns_array))
    
    # Sharpe ratio
    sharpe = calculate_sharpe_ratio(returns)
    
    # Sortino ratio (only penalizes downside volatility)
    downside_returns = returns_array[returns_array < 0]
    downside_deviation = float(np.std(downside_returns)) if len(downside_returns) > 0 else 0.0
    sortino = (mean_return * 252 - 0.02) / (downside_deviation * np.sqrt(252)) if downside_deviation > 0 else 0.0
    
    # Calmar ratio (return / max drawdown)
    dd_metrics = calculate_maximum_drawdown(returns)
    calmar = abs(total_return / dd_metrics['max_drawdown']) if dd_metrics['max_drawdown'] > 0 else 0.0
    
    # Information ratio (excess return / tracking error)
    # Assuming benchmark return of 0 for simplicity
    excess_returns = returns_array - 0.0
    tracking_error = float(np.std(excess_returns))
    information_ratio = float(np.mean(excess_returns) / tracking_error) if tracking_error > 0 else 0.0
    
    return {
        'sharpe_ratio': sharpe,
        'sortino_ratio': sortino,
        'calmar_ratio': calmar,
        'information_ratio': information_ratio,
        'total_return': total_return,
        'volatility': volatility,
        'downside_deviation': downside_deviation,
        'mean_return': mean_return
    }


def walk_forward_analysis(
    predictions: List[Dict[str, Any]],
    actual_returns: List[float],
    train_window: int = 100,
    test_window: int = 20,
    step_size: int = 10
) -> Dict[str, Any]:
    """
    Perform walk-forward analysis for time-series validation.
    
    Args:
        predictions: List of prediction dictionaries with 'recommendation', 'confidence', etc.
        actual_returns: List of actual future returns
        train_window: Size of training window
        test_window: Size of test window
        step_size: Step size for moving window
    
    Returns:
        Dictionary with walk-forward analysis results
    """
    if len(predictions) < train_window + test_window:
        return {
            'error': 'Insufficient data for walk-forward analysis',
            'min_required': train_window + test_window,
            'available': len(predictions)
        }
    
    results = []
    total_folds = (len(predictions) - train_window - test_window) // step_size + 1
    
    for fold in range(total_folds):
        train_start = fold * step_size
        train_end = train_start + train_window
        test_start = train_end
        test_end = min(test_start + test_window, len(predictions))
        
        if test_end <= test_start:
            break
        
        # Get predictions and actuals for this fold
        fold_predictions = predictions[test_start:test_end]
        fold_actuals = actual_returns[test_start:test_end]
        
        # Calculate metrics for this fold
        correct = 0
        total = len(fold_predictions)
        returns = []
        
        for pred, actual in zip(fold_predictions, fold_actuals):
            recommendation = pred.get('recommendation', 'Hold')
            
            # Determine if prediction was correct
            if recommendation == 'Buy' and actual > 0.02:
                correct += 1
                returns.append(actual)
            elif recommendation == 'Sell' and actual < -0.02:
                correct += 1
                returns.append(abs(actual))
            elif recommendation == 'Hold' and -0.02 <= actual <= 0.02:
                correct += 1
                returns.append(0.0)
            else:
                # Wrong prediction - negative return
                returns.append(-abs(actual))
        
        accuracy = correct / total if total > 0 else 0.0
        avg_return = np.mean(returns) if returns else 0.0
        
        # Calculate risk metrics
        risk_metrics = calculate_risk_adjusted_returns(returns)
        
        results.append({
            'fold': fold + 1,
            'train_start': train_start,
            'train_end': train_end,
            'test_start': test_start,
            'test_end': test_end,
            'accuracy': accuracy,
            'avg_return': avg_return,
            'total_return': risk_metrics['total_return'],
            'sharpe_ratio': risk_metrics['sharpe_ratio'],
            'max_drawdown': calculate_maximum_drawdown(returns)['max_drawdown']
        })
    
    # Aggregate results
    if not results:
        return {'error': 'No valid folds generated'}
    
    avg_accuracy = np.mean([r['accuracy'] for r in results])
    avg_sharpe = np.mean([r['sharpe_ratio'] for r in results])
    avg_return = np.mean([r['avg_return'] for r in results])
    avg_drawdown = np.mean([r['max_drawdown'] for r in results])
    
    return {
        'total_folds': len(results),
        'avg_accuracy': float(avg_accuracy),
        'avg_sharpe_ratio': float(avg_sharpe),
        'avg_return': float(avg_return),
        'avg_max_drawdown': float(avg_drawdown),
        'fold_results': results,
        'consistency': float(np.std([r['accuracy'] for r in results]))  # Lower is better
    }


def calculate_portfolio_metrics(
    recommendations: List[Dict[str, Any]],
    actual_returns: List[float],
    initial_capital: float = 10000.0
) -> Dict[str, Any]:
    """
    Calculate comprehensive portfolio performance metrics.
    
    Args:
        recommendations: List of recommendation dictionaries
        actual_returns: List of actual returns
        initial_capital: Initial portfolio value
    
    Returns:
        Dictionary with comprehensive portfolio metrics
    """
    if len(recommendations) != len(actual_returns):
        return {'error': 'Mismatch between recommendations and returns'}
    
    portfolio_value = initial_capital
    portfolio_values = [initial_capital]
    returns = []
    positions = []
    
    for rec, actual_return in zip(recommendations, actual_returns):
        recommendation = rec.get('recommendation', 'Hold')
        confidence = rec.get('confidence', 0.5)
        
        # Calculate position size based on confidence
        position_size = confidence  # Use confidence as position size multiplier
        
        if recommendation == 'Buy':
            # Positive return scaled by position size
            return_pct = actual_return * position_size
            returns.append(return_pct)
            portfolio_value *= (1 + return_pct)
        elif recommendation == 'Sell':
            # Short position - inverse return
            return_pct = -actual_return * position_size
            returns.append(return_pct)
            portfolio_value *= (1 + return_pct)
        else:  # Hold
            returns.append(0.0)
            # No change in portfolio value
        
        portfolio_values.append(portfolio_value)
        positions.append({
            'recommendation': recommendation,
            'confidence': confidence,
            'return': return_pct,
            'portfolio_value': portfolio_value
        })
    
    # Calculate metrics
    total_return = (portfolio_value - initial_capital) / initial_capital
    risk_metrics = calculate_risk_adjusted_returns(returns)
    dd_metrics = calculate_maximum_drawdown(returns)
    
    # Win rate
    winning_trades = sum(1 for r in returns if r > 0)
    win_rate = winning_trades / len(returns) if returns else 0.0
    
    # Average win/loss
    wins = [r for r in returns if r > 0]
    losses = [r for r in returns if r < 0]
    avg_win = np.mean(wins) if wins else 0.0
    avg_loss = abs(np.mean(losses)) if losses else 0.0
    win_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 0.0
    
    return {
        'initial_capital': initial_capital,
        'final_capital': portfolio_value,
        'total_return': float(total_return),
        'total_return_pct': float(total_return * 100),
        'sharpe_ratio': risk_metrics['sharpe_ratio'],
        'sortino_ratio': risk_metrics['sortino_ratio'],
        'calmar_ratio': risk_metrics['calmar_ratio'],
        'max_drawdown': dd_metrics['max_drawdown'],
        'volatility': risk_metrics['volatility'],
        'win_rate': float(win_rate),
        'avg_win': float(avg_win),
        'avg_loss': float(avg_loss),
        'win_loss_ratio': float(win_loss_ratio),
        'portfolio_values': portfolio_values,
        'returns': returns,
        'positions': positions
    }

