"""
Regression Metrics for Model Evaluation.
"""
import numpy as np
from typing import Dict, Optional
from dataclasses import dataclass
import warnings


@dataclass
class MetricsResult:
    """Container for regression metrics."""
    mae: float          # Mean Absolute Error
    mse: float          # Mean Squared Error
    rmse: float         # Root Mean Squared Error
    mape: float         # Mean Absolute Percentage Error
    smape: float        # Symmetric Mean Absolute Percentage Error
    r2: float           # R-squared (Coefficient of Determination)
    explained_var: float  # Explained Variance Score
    max_error: float    # Maximum Error
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            'MAE': self.mae,
            'MSE': self.mse,
            'RMSE': self.rmse,
            'MAPE': self.mape,
            'SMAPE': self.smape,
            'R2': self.r2,
            'Explained_Variance': self.explained_var,
            'Max_Error': self.max_error
        }
    
    def __str__(self) -> str:
        """String representation."""
        return (
            f"MAE: {self.mae:.2f} | RMSE: {self.rmse:.2f} | "
            f"MAPE: {self.mape:.2f}% | R²: {self.r2:.4f}"
        )


class RegressionMetrics:
    """
    Calculate comprehensive regression metrics for model evaluation.
    
    Metrics included:
    - MAE: Mean Absolute Error
    - MSE: Mean Squared Error  
    - RMSE: Root Mean Squared Error
    - MAPE: Mean Absolute Percentage Error
    - SMAPE: Symmetric Mean Absolute Percentage Error
    - R²: Coefficient of Determination
    - Explained Variance
    - Max Error
    """
    
    @staticmethod
    def calculate(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        epsilon: float = 1e-10
    ) -> MetricsResult:
        """
        Calculate all regression metrics.
        
        Args:
            y_true: Ground truth values
            y_pred: Predicted values
            epsilon: Small value to avoid division by zero
            
        Returns:
            MetricsResult with all metrics
        """
        y_true = np.asarray(y_true).flatten()
        y_pred = np.asarray(y_pred).flatten()
        
        # Basic error metrics
        errors = y_true - y_pred
        abs_errors = np.abs(errors)
        
        mae = np.mean(abs_errors)
        mse = np.mean(errors ** 2)
        rmse = np.sqrt(mse)
        max_error = np.max(abs_errors)
        
        # Percentage errors (handle zeros)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            # MAPE - Mean Absolute Percentage Error
            non_zero_mask = np.abs(y_true) > epsilon
            if np.any(non_zero_mask):
                mape = np.mean(np.abs(errors[non_zero_mask] / y_true[non_zero_mask])) * 100
            else:
                mape = np.inf
            
            # SMAPE - Symmetric Mean Absolute Percentage Error
            denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
            denominator = np.where(denominator < epsilon, epsilon, denominator)
            smape = np.mean(abs_errors / denominator) * 100
        
        # R² Score
        ss_res = np.sum(errors ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1 - (ss_res / (ss_tot + epsilon))
        
        # Explained Variance
        var_res = np.var(errors)
        var_y = np.var(y_true)
        explained_var = 1 - (var_res / (var_y + epsilon))
        
        return MetricsResult(
            mae=float(mae),
            mse=float(mse),
            rmse=float(rmse),
            mape=float(mape),
            smape=float(smape),
            r2=float(r2),
            explained_var=float(explained_var),
            max_error=float(max_error)
        )
    
    @staticmethod
    def calculate_fold_metrics(
        fold_results: list
    ) -> Dict[str, Dict[str, float]]:
        """
        Aggregate metrics across multiple folds.
        
        Args:
            fold_results: List of MetricsResult from each fold
            
        Returns:
            Dictionary with mean and std for each metric
        """
        if not fold_results:
            return {}
        
        metrics_dict = {
            'MAE': [],
            'MSE': [],
            'RMSE': [],
            'MAPE': [],
            'SMAPE': [],
            'R2': [],
            'Explained_Variance': [],
            'Max_Error': []
        }
        
        for result in fold_results:
            d = result.to_dict()
            for key in metrics_dict:
                metrics_dict[key].append(d[key])
        
        aggregated = {}
        for metric, values in metrics_dict.items():
            values = np.array(values)
            aggregated[metric] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values))
            }
        
        return aggregated

