"""
Forecaster for Sales Prediction with Confidence Intervals.

Loads trained models and generates forecasts for N days.
Supports confidence intervals using:
- Native SD-based intervals for Prophet/SARIMA
- MAD-based intervals for ML models (XGBoost, RandomForest, LinearTrend)
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional, List
import pickle

from ..models.models import BaseSimpleModel, MODEL_REGISTRY


class Forecaster:
    """
    Forecaster that loads trained models and generates predictions with confidence intervals.
    
    Usage:
        forecaster = Forecaster()
        forecaster.load_model('prophet', 'models/saved/prophet.pkl')
        predictions = forecaster.forecast('prophet', n_days=30, include_ci=True)
    """
    
    def __init__(self):
        """Initialize forecaster."""
        self.models: Dict[str, BaseSimpleModel] = {}
        self.last_data_date: Optional[pd.Timestamp] = None
    
    def set_last_data_date(self, date: pd.Timestamp) -> None:
        """
        Set the last date in the actual data.
        
        This is used to ensure forecasts start from the correct date,
        even if models were trained on older data.
        
        Args:
            date: The last date in the current dataset
        """
        self.last_data_date = pd.to_datetime(date)
    
    def load_model(self, model_name: str, path: str) -> None:
        """Load a trained model from file."""
        with open(path, 'rb') as f:
            model = pickle.load(f)
        self.models[model_name] = model
        print(f"Loaded {model_name} from {path}")
    
    def load_all_models(self, models_dir: str = "models/saved") -> None:
        """Load all available models from directory."""
        models_dir = Path(models_dir)
        
        for model_name in MODEL_REGISTRY.keys():
            model_path = models_dir / f"{model_name}.pkl"
            if model_path.exists():
                self.load_model(model_name, str(model_path))
    
    def _update_model_last_date(self, model: BaseSimpleModel) -> None:
        """
        Update model's last_date if we have a more recent data date.
        
        This ensures forecasts start from the actual last data point,
        not from when the model was trained.
        """
        if self.last_data_date is not None:
            model.last_date = self.last_data_date
    
    def forecast(
        self,
        model_name: str,
        n_days: int,
        include_ci: bool = True
    ) -> pd.DataFrame:
        """
        Generate forecast for n_days ahead with optional confidence intervals.
        
        Args:
            model_name: Name of model to use
            n_days: Number of days to forecast
            include_ci: Whether to include confidence intervals (default True)
            
        Returns:
            DataFrame with columns:
            - date: Forecast date
            - predicted_sales: Point prediction
            - lower_bound: Lower CI bound (if include_ci=True)
            - upper_bound: Upper CI bound (if include_ci=True)
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not loaded")
        
        model = self.models[model_name]
        
        # Update model's last_date to match current data
        self._update_model_last_date(model)
        
        # Call predict with include_ci parameter
        predictions = model.predict(n_days, include_ci=include_ci)
        
        return predictions
    
    def forecast_all(
        self,
        n_days: int,
        include_ci: bool = True
    ) -> Dict[str, pd.DataFrame]:
        """
        Generate forecasts from all loaded models.
        
        Args:
            n_days: Number of days to forecast
            include_ci: Whether to include confidence intervals
            
        Returns:
            Dictionary of model_name -> predictions DataFrame
        """
        results = {}
        for model_name in self.models:
            try:
                results[model_name] = self.forecast(model_name, n_days, include_ci=include_ci)
            except Exception as e:
                print(f"Error forecasting with {model_name}: {e}")
        
        return results
    
    def get_ensemble_forecast(
        self,
        n_days: int,
        weights: Optional[Dict[str, float]] = None,
        include_ci: bool = True
    ) -> pd.DataFrame:
        """
        Generate ensemble forecast (weighted average of all models).
        
        For confidence intervals, we combine the individual model CIs
        using weighted average (approximation).
        
        Args:
            n_days: Number of days to forecast
            weights: Optional weights for each model (default: equal weights)
            include_ci: Whether to include confidence intervals
            
        Returns:
            DataFrame with ensemble predictions and optional CI
        """
        all_forecasts = self.forecast_all(n_days, include_ci=include_ci)
        
        if not all_forecasts:
            raise ValueError("No models loaded")
        
        # Default to equal weights
        if weights is None:
            weights = {name: 1.0 / len(all_forecasts) for name in all_forecasts}
        
        # Normalize weights
        total_weight = sum(weights.get(name, 0) for name in all_forecasts)
        weights = {name: weights.get(name, 0) / total_weight for name in all_forecasts}
        
        # Get dates from first model
        dates = list(all_forecasts.values())[0]['date']
        
        # Weighted average of predictions
        ensemble_pred = np.zeros(n_days)
        ensemble_lower = np.zeros(n_days) if include_ci else None
        ensemble_upper = np.zeros(n_days) if include_ci else None
        
        for model_name, forecast_df in all_forecasts.items():
            weight = weights.get(model_name, 0)
            ensemble_pred += weight * forecast_df['predicted_sales'].values
            
            if include_ci and 'lower_bound' in forecast_df.columns:
                ensemble_lower += weight * forecast_df['lower_bound'].values
                ensemble_upper += weight * forecast_df['upper_bound'].values
        
        result = pd.DataFrame({
            'date': dates,
            'predicted_sales': ensemble_pred
        })
        
        if include_ci and ensemble_lower is not None:
            result['lower_bound'] = ensemble_lower
            result['upper_bound'] = ensemble_upper
        
        return result
    
    def compare_forecasts(
        self,
        n_days: int,
        include_ci: bool = False
    ) -> pd.DataFrame:
        """
        Compare forecasts from all models side by side.
        
        Args:
            n_days: Number of days to forecast
            include_ci: Whether to include CI columns for each model
            
        Returns:
            DataFrame with all model predictions
        """
        all_forecasts = self.forecast_all(n_days, include_ci=include_ci)
        
        if not all_forecasts:
            raise ValueError("No models loaded")
        
        # Start with dates
        result = list(all_forecasts.values())[0][['date']].copy()
        
        # Add each model's predictions
        for model_name, forecast_df in all_forecasts.items():
            result[model_name] = forecast_df['predicted_sales'].values
            
            if include_ci and 'lower_bound' in forecast_df.columns:
                result[f'{model_name}_lower'] = forecast_df['lower_bound'].values
                result[f'{model_name}_upper'] = forecast_df['upper_bound'].values
        
        return result
