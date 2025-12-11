"""
Confidence Calibration Module
Implements explicit confidence calibration mechanisms to improve reliability of confidence scores.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore')


class ConfidenceCalibrator:
    """
    Calibrates confidence scores to match actual accuracy.
    Uses Platt scaling (logistic regression) and isotonic regression.
    """
    
    def __init__(self, method: str = 'isotonic'):
        """
        Initialize calibrator.
        
        Args:
            method: Calibration method ('platt', 'isotonic', or 'temperature')
        """
        self.method = method
        self.calibrator = None
        self.temperature = 1.0
        self.is_fitted = False
        
        if method == 'isotonic':
            self.calibrator = IsotonicRegression(out_of_bounds='clip')
        elif method == 'platt':
            self.calibrator = LogisticRegression()
        elif method == 'temperature':
            # Temperature scaling - just store temperature parameter
            self.calibrator = None
        else:
            raise ValueError(f"Unknown calibration method: {method}")
    
    def fit(self, confidences: List[float], actual_correct: List[bool]) -> None:
        """
        Fit calibrator on historical data.
        
        Args:
            confidences: List of confidence scores [0, 1]
            actual_correct: List of boolean values indicating if prediction was correct
        """
        if len(confidences) != len(actual_correct):
            raise ValueError("Confidences and actual_correct must have same length")
        
        confidences_array = np.array(confidences)
        actual_correct_array = np.array(actual_correct).astype(int)
        
        if self.method == 'temperature':
            # Find optimal temperature using validation set
            # Temperature scaling: calibrated = sigmoid(logit(confidence) / T)
            # We optimize T to minimize calibration error
            from scipy.optimize import minimize_scalar
            
            def calibration_error(T):
                calibrated = self._temperature_scale(confidences_array, T)
                # Calculate ECE (Expected Calibration Error)
                return self._calculate_ece(calibrated, actual_correct_array)
            
            result = minimize_scalar(calibration_error, bounds=(0.1, 10.0), method='bounded')
            self.temperature = result.x
        else:
            # Reshape for sklearn
            X = confidences_array.reshape(-1, 1)
            y = actual_correct_array
            
            self.calibrator.fit(X, y)
        
        self.is_fitted = True
    
    def calibrate(self, confidence: float) -> float:
        """
        Calibrate a single confidence score.
        
        Args:
            confidence: Raw confidence score [0, 1]
        
        Returns:
            Calibrated confidence score [0, 1]
        """
        if not self.is_fitted:
            return confidence  # Return uncalibrated if not fitted
        
        if self.method == 'temperature':
            return self._temperature_scale(np.array([confidence]), self.temperature)[0]
        else:
            calibrated = self.calibrator.predict_proba(np.array([[confidence]]))[0]
            # Return probability of being correct
            return float(calibrated[1]) if len(calibrated) > 1 else float(calibrated[0])
    
    def calibrate_batch(self, confidences: List[float]) -> List[float]:
        """
        Calibrate a batch of confidence scores.
        
        Args:
            confidences: List of confidence scores
        
        Returns:
            List of calibrated confidence scores
        """
        if not self.is_fitted:
            return confidences
        
        confidences_array = np.array(confidences)
        
        if self.method == 'temperature':
            calibrated = self._temperature_scale(confidences_array, self.temperature)
        else:
            X = confidences_array.reshape(-1, 1)
            calibrated_probs = self.calibrator.predict_proba(X)
            calibrated = calibrated_probs[:, 1] if calibrated_probs.shape[1] > 1 else calibrated_probs[:, 0]
        
        return [float(c) for c in calibrated]
    
    def _temperature_scale(self, confidences: np.ndarray, T: float) -> np.ndarray:
        """Apply temperature scaling to confidences."""
        # Convert confidence to logit space
        eps = 1e-8
        confidences_clipped = np.clip(confidences, eps, 1 - eps)
        logits = np.log(confidences_clipped / (1 - confidences_clipped))
        
        # Scale by temperature
        scaled_logits = logits / T
        
        # Convert back to probability
        calibrated = 1 / (1 + np.exp(-scaled_logits))
        return calibrated
    
    def _calculate_ece(self, confidences: np.ndarray, actual_correct: np.ndarray, n_bins: int = 10) -> float:
        """Calculate Expected Calibration Error."""
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0.0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = actual_correct[in_bin].mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        return float(ece)
    
    def evaluate_calibration(self, confidences: List[float], actual_correct: List[bool]) -> Dict[str, float]:
        """
        Evaluate calibration quality.
        
        Args:
            confidences: List of confidence scores
            actual_correct: List of boolean values
        
        Returns:
            Dictionary with calibration metrics
        """
        confidences_array = np.array(confidences)
        actual_correct_array = np.array(actual_correct).astype(int)
        
        # Calculate ECE
        ece = self._calculate_ece(confidences_array, actual_correct_array)
        
        # Calculate Brier score
        brier_score = np.mean((confidences_array - actual_correct_array) ** 2)
        
        # Calculate reliability (calibration curve)
        try:
            fraction_of_positives, mean_predicted_value = calibration_curve(
                actual_correct_array, confidences_array, n_bins=10
            )
            # Reliability is the difference between predicted and actual
            reliability = np.mean(np.abs(fraction_of_positives - mean_predicted_value))
        except:
            reliability = 0.0
        
        return {
            'ece': float(ece),
            'brier_score': float(brier_score),
            'reliability': float(reliability)
        }


class ConfidenceTracker:
    """
    Tracks confidence scores and their accuracy over time for calibration.
    """
    
    def __init__(self):
        """Initialize tracker."""
        self.confidences = []
        self.actual_correct = []
        self.recommendations = []
        self.actual_returns = []
    
    def add_prediction(self, confidence: float, recommendation: str, actual_return: float) -> None:
        """
        Add a prediction with its outcome.
        
        Args:
            confidence: Confidence score
            recommendation: Buy/Hold/Sell
            actual_return: Actual future return
        """
        self.confidences.append(confidence)
        self.recommendations.append(recommendation)
        self.actual_returns.append(actual_return)
        
        # Determine if prediction was correct
        is_correct = False
        if recommendation == 'Buy' and actual_return > 0.02:
            is_correct = True
        elif recommendation == 'Sell' and actual_return < -0.02:
            is_correct = True
        elif recommendation == 'Hold' and -0.02 <= actual_return <= 0.02:
            is_correct = True
        
        self.actual_correct.append(is_correct)
    
    def get_calibration_data(self) -> Tuple[List[float], List[bool]]:
        """Get data for calibration."""
        return self.confidences.copy(), self.actual_correct.copy()
    
    def get_confidence_bins(self, n_bins: int = 10) -> Dict[str, Any]:
        """
        Analyze confidence by bins.
        
        Args:
            n_bins: Number of bins
        
        Returns:
            Dictionary with bin analysis
        """
        if len(self.confidences) == 0:
            return {}
        
        confidences_array = np.array(self.confidences)
        actual_correct_array = np.array(self.actual_correct)
        
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bins = []
        
        for i in range(n_bins):
            bin_lower = bin_boundaries[i]
            bin_upper = bin_boundaries[i + 1]
            
            in_bin = (confidences_array >= bin_lower) & (confidences_array < bin_upper)
            if i == n_bins - 1:  # Include upper bound for last bin
                in_bin = (confidences_array >= bin_lower) & (confidences_array <= bin_upper)
            
            if in_bin.sum() > 0:
                avg_confidence = confidences_array[in_bin].mean()
                accuracy = actual_correct_array[in_bin].mean()
                count = in_bin.sum()
                
                bins.append({
                    'bin_range': (bin_lower, bin_upper),
                    'avg_confidence': float(avg_confidence),
                    'actual_accuracy': float(accuracy),
                    'calibration_error': float(abs(avg_confidence - accuracy)),
                    'count': int(count)
                })
        
        return {
            'bins': bins,
            'total_predictions': len(self.confidences),
            'overall_accuracy': float(actual_correct_array.mean())
        }

