"""
Simple Forecaster for Sales Prediction.

Loads trained models and generates forecasts for N days.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional, List
import pickle

from ..models.models import BaseSimpleModel, MODEL_REGISTRY


class Forecaster:
    """
    Simple forecaster that loads trained models and generates predictions.
    
    Usage:
        forecaster = SimpleForecaster()
        forecaster.load_model('prophet', 'models/saved/prophet.pkl')
        predictions = forecaster.forecast('prophet', n_days=30)
    """
    
    def __init__(self):
        """Initialize forecaster."""
        self.models: Dict[str, BaseSimpleModel] = {}
    
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
    
    def forecast(
        self,
        model_name: str,
        n_days: int
    ) -> pd.DataFrame:
        """
        Generate forecast for n_days ahead.
        
        Args:
            model_name: Name of model to use
            n_days: Number of days to forecast
            
        Returns:
            DataFrame with date and predicted_sales columns
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not loaded")
        
        model = self.models[model_name]
        predictions = model.predict(n_days)
        
        return predictions
    
    def forecast_all(self, n_days: int) -> Dict[str, pd.DataFrame]:
        """
        Generate forecasts from all loaded models.
        
        Args:
            n_days: Number of days to forecast
            
        Returns:
            Dictionary of model_name -> predictions DataFrame
        """
        results = {}
        for model_name in self.models:
            try:
                results[model_name] = self.forecast(model_name, n_days)
            except Exception as e:
                print(f"Error forecasting with {model_name}: {e}")
        
        return results
    
    def get_ensemble_forecast(
        self,
        n_days: int,
        weights: Optional[Dict[str, float]] = None
    ) -> pd.DataFrame:
        """
        Generate ensemble forecast (weighted average of all models).
        
        Args:
            n_days: Number of days to forecast
            weights: Optional weights for each model (default: equal weights)
            
        Returns:
            DataFrame with ensemble predictions
        """
        all_forecasts = self.forecast_all(n_days)
        
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
        
        # Weighted average
        ensemble_pred = np.zeros(n_days)
        for model_name, forecast_df in all_forecasts.items():
            weight = weights.get(model_name, 0)
            ensemble_pred += weight * forecast_df['predicted_sales'].values
        
        return pd.DataFrame({
            'date': dates,
            'predicted_sales': ensemble_pred
        })
    
    def compare_forecasts(self, n_days: int) -> pd.DataFrame:
        """
        Compare forecasts from all models side by side.
        
        Args:
            n_days: Number of days to forecast
            
        Returns:
            DataFrame with all model predictions
        """
        all_forecasts = self.forecast_all(n_days)
        
        if not all_forecasts:
            raise ValueError("No models loaded")
        
        # Start with dates
        result = list(all_forecasts.values())[0][['date']].copy()
        
        # Add each model's predictions
        for model_name, forecast_df in all_forecasts.items():
            result[model_name] = forecast_df['predicted_sales'].values
        
        return result

