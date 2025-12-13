"""
Simple Time Series Models for Sales Forecasting.

Models:
1. LinearTrendModel - Captures overall trend
2. ProphetModel - Facebook's forecasting (trend + seasonality)
3. XGBoostModel - Gradient boosting with lag features (with learning curves)
4. RandomForestModel - Ensemble of decision trees with lag features
5. SARIMAModel - Statistical time series model

All models follow a simple interface:
- fit(df, target_col) -> trains the model
- predict(n_days) -> returns predictions for n_days ahead with confidence intervals

Confidence Intervals:
- Prophet/SARIMA: Native SD-based intervals (from model)
- XGBoost/RandomForest/LinearTrend: MAD-based intervals (robust to outliers)
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
import pickle
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
import xgboost as xgb
import matplotlib.pyplot as plt


# ==================== CONFIDENCE INTERVAL UTILITIES ====================

def calculate_mad(residuals: np.ndarray) -> float:
    """
    Calculate Median Absolute Deviation (MAD).
    
    MAD is more robust to outliers than standard deviation.
    
    Args:
        residuals: Array of residuals (actual - predicted)
        
    Returns:
        MAD value
    """
    median = np.median(residuals)
    mad = np.median(np.abs(residuals - median))
    return mad


def mad_to_std_equivalent(mad: float) -> float:
    """
    Convert MAD to standard deviation equivalent.
    
    For normal distribution: SD ≈ 1.4826 × MAD
    
    Args:
        mad: Median Absolute Deviation
        
    Returns:
        Standard deviation equivalent
    """
    return 1.4826 * mad


def calculate_confidence_bounds(
    predictions: np.ndarray,
    residual_std: float,
    confidence_level: float = 0.95
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate confidence interval bounds.
    
    Args:
        predictions: Point predictions
        residual_std: Standard deviation (or MAD-equivalent) of residuals
        confidence_level: Confidence level (default 0.95 for 95% CI)
        
    Returns:
        Tuple of (lower_bounds, upper_bounds)
    """
    from scipy import stats
    
    # Z-score for confidence level (1.96 for 95%)
    z_score = stats.norm.ppf((1 + confidence_level) / 2)
    
    margin = z_score * residual_std
    lower = predictions - margin
    upper = predictions + margin
    
    # Ensure non-negative for sales
    lower = np.maximum(lower, 0)
    
    return lower, upper


class BaseSimpleModel:
    """Base class for simple forecasting models."""
    
    def __init__(self, name: str):
        self.name = name
        self.is_fitted = False
        self.last_date = None
        self.training_data = None
        # Confidence interval parameters
        self.residual_std = None  # For MAD-based CI
        self.confidence_level = 0.95  # 95% CI
    
    def fit(self, df: pd.DataFrame, target_col: str = 'daily_sales') -> Dict[str, Any]:
        """Fit the model. Returns training info."""
        raise NotImplementedError
    
    def predict(self, n_days: int, include_ci: bool = True) -> pd.DataFrame:
        """
        Predict n_days ahead.
        
        Args:
            n_days: Number of days to forecast
            include_ci: Whether to include confidence intervals
            
        Returns:
            DataFrame with columns:
            - date: Forecast date
            - predicted_sales: Point prediction
            - lower_bound: Lower CI bound (if include_ci=True)
            - upper_bound: Upper CI bound (if include_ci=True)
        """
        raise NotImplementedError
    
    def _calculate_residual_std(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate MAD-based standard deviation equivalent for confidence intervals.
        
        Uses MAD (Median Absolute Deviation) for robustness against outliers.
        
        Args:
            y_true: Actual values
            y_pred: Predicted values
            
        Returns:
            Standard deviation equivalent from MAD
        """
        residuals = y_true - y_pred
        mad = calculate_mad(residuals)
        return mad_to_std_equivalent(mad)
    
    def save(self, path: str) -> str:
        """Save model to file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self, f)
        return str(path)
    
    @classmethod
    def load(cls, path: str) -> 'BaseSimpleModel':
        """Load model from file."""
        with open(path, 'rb') as f:
            return pickle.load(f)
    
    def get_hyperparameters(self) -> Dict[str, Any]:
        """Return current hyperparameters."""
        return {}
    
    @staticmethod
    def get_optuna_params(trial) -> Dict[str, Any]:
        """Get hyperparameters from Optuna trial."""
        return {}


class LinearTrendModel(BaseSimpleModel):
    """
    Simple Linear Trend Model.
    
    Captures the overall trend using linear regression on day number.
    Can also include day-of-week effect.
    """
    
    def __init__(self, include_dow: bool = True, regularization: float = 0.0):
        super().__init__('linear_trend')
        self.include_dow = include_dow
        self.regularization = regularization
        self.model = None
        self.scaler = StandardScaler()
    
    def _prepare_features(self, df: pd.DataFrame, start_day: int = 0) -> np.ndarray:
        """Prepare features for linear model."""
        n = len(df)
        day_num = np.arange(start_day, start_day + n).reshape(-1, 1)
        
        if self.include_dow and 'date' in df.columns:
            dow = df['date'].dt.dayofweek.values.reshape(-1, 1)
            # One-hot encode day of week
            dow_onehot = np.zeros((n, 7))
            for i, d in enumerate(dow.flatten()):
                dow_onehot[i, d] = 1
            features = np.hstack([day_num, dow_onehot])
        else:
            features = day_num
        
        return features
    
    def fit(self, df: pd.DataFrame, target_col: str = 'daily_sales') -> Dict[str, Any]:
        """Fit linear trend model."""
        self.training_data = df.copy()
        self.last_date = df['date'].max()
        self.target_col = target_col
        
        X = self._prepare_features(df)
        y = df[target_col].values
        
        X_scaled = self.scaler.fit_transform(X)
        
        if self.regularization > 0:
            self.model = Ridge(alpha=self.regularization)
        else:
            self.model = LinearRegression()
        
        self.model.fit(X_scaled, y)
        
        # Training metrics
        y_pred = self.model.predict(X_scaled)
        mape = mean_absolute_percentage_error(y, y_pred) * 100
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        
        # Calculate MAD-based residual std for confidence intervals
        self.residual_std = self._calculate_residual_std(y, y_pred)
        
        self.is_fitted = True
        self.n_train = len(df)
        
        return {
            'train_mape': mape,
            'train_rmse': rmse,
            'n_samples': len(df),
            'residual_std': self.residual_std
        }
    
    def predict(self, n_days: int, include_ci: bool = True) -> pd.DataFrame:
        """
        Predict n_days ahead with confidence intervals.
        
        Args:
            n_days: Number of days to forecast
            include_ci: Whether to include confidence intervals (MAD-based)
            
        Returns:
            DataFrame with date, predicted_sales, lower_bound, upper_bound
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        
        # Generate future dates
        future_dates = pd.date_range(
            start=self.last_date + pd.Timedelta(days=1),
            periods=n_days,
            freq='D'
        )
        
        future_df = pd.DataFrame({'date': future_dates})
        X = self._prepare_features(future_df, start_day=self.n_train)
        X_scaled = self.scaler.transform(X)
        
        predictions = self.model.predict(X_scaled)
        predictions = np.maximum(predictions, 0)  # No negative sales
        
        result = pd.DataFrame({
            'date': future_dates,
            'predicted_sales': predictions
        })
        
        # Add confidence intervals
        if include_ci:
            residual_std = getattr(self, 'residual_std', None)
            confidence_level = getattr(self, 'confidence_level', 0.95)
            
            if residual_std is not None:
                lower, upper = calculate_confidence_bounds(
                    predictions, residual_std, confidence_level
                )
            else:
                # Fallback: estimate CI from prediction variance (~10%)
                estimated_std = np.mean(predictions) * 0.1
                lower, upper = calculate_confidence_bounds(
                    predictions, estimated_std, confidence_level
                )
            
            result['lower_bound'] = lower
            result['upper_bound'] = upper
        
        return result
    
    def get_hyperparameters(self) -> Dict[str, Any]:
        return {
            'include_dow': self.include_dow,
            'regularization': self.regularization
        }
    
    @staticmethod
    def get_optuna_params(trial) -> Dict[str, Any]:
        return {
            'include_dow': trial.suggest_categorical('lt_include_dow', [True, False]),
            'regularization': trial.suggest_float('lt_regularization', 0.0, 100.0)
        }


class XGBoostModel(BaseSimpleModel):
    """
    XGBoost model with lag features for time series.
    
    Features:
    - Lag features (1, 7, 14, 28 days)
    - Rolling mean (7, 14 days)
    - Day of week
    - Month
    - Trend (day number)
    - Iteration-wise loss tracking for learning curves
    """
    
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 5,
        learning_rate: float = 0.1,
        min_child_weight: int = 1,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        early_stopping_rounds: int = 20
    ):
        super().__init__('xgboost')
        self.params = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'learning_rate': learning_rate,
            'min_child_weight': min_child_weight,
            'subsample': subsample,
            'colsample_bytree': colsample_bytree,
            'random_state': 42,
            'n_jobs': -1
        }
        self.early_stopping_rounds = early_stopping_rounds
        self.model = None
        self.feature_names = []
        self.lag_days = [1, 7, 14, 28]
        self.rolling_windows = [7, 14]
        # Learning curve tracking
        self.train_losses = []
        self.val_losses = []
        self.learning_curve_path = None
    
    def _create_features(self, df: pd.DataFrame, target_col: str) -> pd.DataFrame:
        """Create features for XGBoost."""
        df = df.copy()
        
        # Lag features
        for lag in self.lag_days:
            df[f'lag_{lag}'] = df[target_col].shift(lag)
        
        # Rolling features
        for window in self.rolling_windows:
            df[f'rolling_mean_{window}'] = df[target_col].shift(1).rolling(window).mean()
            df[f'rolling_std_{window}'] = df[target_col].shift(1).rolling(window).std()
        
        # Date features
        df['day_of_week'] = df['date'].dt.dayofweek
        df['month'] = df['date'].dt.month
        df['day_of_year'] = df['date'].dt.dayofyear
        
        # Trend
        df['day_num'] = range(len(df))
        
        return df
    
    def _get_feature_columns(self) -> List[str]:
        """Get list of feature column names."""
        features = []
        features.extend([f'lag_{lag}' for lag in self.lag_days])
        features.extend([f'rolling_mean_{w}' for w in self.rolling_windows])
        features.extend([f'rolling_std_{w}' for w in self.rolling_windows])
        features.extend(['day_of_week', 'month', 'day_of_year', 'day_num'])
        return features
    
    def fit(self, df: pd.DataFrame, target_col: str = 'daily_sales', 
            val_df: pd.DataFrame = None) -> Dict[str, Any]:
        """
        Fit XGBoost model with iteration-wise loss tracking.
        
        Args:
            df: Training data
            target_col: Target column name
            val_df: Optional validation data for learning curves
        """
        self.training_data = df.copy()
        self.last_date = df['date'].max()
        self.target_col = target_col
        
        # Create features
        df_features = self._create_features(df, target_col)
        
        # Drop NaN rows (from lag features)
        df_features = df_features.dropna()
        
        self.feature_names = self._get_feature_columns()
        X_train = df_features[self.feature_names].values
        y_train = df_features[target_col].values
        
        # Prepare validation set if provided
        eval_set = [(X_train, y_train)]
        if val_df is not None and len(val_df) > 0:
            val_features = self._create_features(val_df, target_col).dropna()
            if len(val_features) > 0:
                X_val = val_features[self.feature_names].values
                y_val = val_features[target_col].values
                eval_set.append((X_val, y_val))
        else:
            # Use last 20% of training data as validation for learning curves
            split_idx = int(len(X_train) * 0.8)
            X_val = X_train[split_idx:]
            y_val = y_train[split_idx:]
            eval_set = [(X_train[:split_idx], y_train[:split_idx]), (X_val, y_val)]
        
        # Train model with eval_set for tracking
        self.model = xgb.XGBRegressor(
            **self.params,
            early_stopping_rounds=self.early_stopping_rounds
        )
        
        self.model.fit(
            X_train, y_train,
            eval_set=eval_set,
            verbose=False
        )
        
        # Extract learning curves from eval results
        self.train_losses = []
        self.val_losses = []
        
        if hasattr(self.model, 'evals_result'):
            evals_result = self.model.evals_result()
            if evals_result:
                if 'validation_0' in evals_result:
                    self.train_losses = evals_result['validation_0']['rmse']
                if 'validation_1' in evals_result:
                    self.val_losses = evals_result['validation_1']['rmse']
        
        # Training metrics
        y_pred = self.model.predict(X_train)
        mape = mean_absolute_percentage_error(y_train, y_pred) * 100
        rmse = np.sqrt(mean_squared_error(y_train, y_pred))
        
        # Calculate MAD-based residual std for confidence intervals
        self.residual_std = self._calculate_residual_std(y_train, y_pred)
        
        self.is_fitted = True
        self.n_train = len(df)
        
        # Store recent data for prediction
        self.recent_data = df.tail(max(self.lag_days) + max(self.rolling_windows)).copy()
        
        return {
            'train_mape': mape,
            'train_rmse': rmse,
            'n_samples': len(df_features),
            'n_features': len(self.feature_names),
            'n_iterations': len(self.train_losses),
            'best_iteration': self.model.best_iteration if hasattr(self.model, 'best_iteration') else None,
            'residual_std': self.residual_std
        }
    
    def plot_learning_curve(self, save_path: str = None) -> str:
        """
        Plot and save the learning curve showing train vs validation loss.
        
        Args:
            save_path: Path to save the plot
            
        Returns:
            Path to saved plot
        """
        if not self.train_losses:
            print("No learning curve data available")
            return None
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        iterations = range(1, len(self.train_losses) + 1)
        ax.plot(iterations, self.train_losses, label='Train RMSE', color='blue', linewidth=2)
        
        if self.val_losses:
            ax.plot(iterations, self.val_losses, label='Validation RMSE', color='red', linewidth=2)
            
            # Mark best iteration
            if hasattr(self.model, 'best_iteration') and self.model.best_iteration:
                best_iter = self.model.best_iteration
                ax.axvline(x=best_iter, color='green', linestyle='--', 
                          label=f'Best Iteration ({best_iter})')
        
        ax.set_xlabel('Boosting Round (Iteration)', fontsize=12)
        ax.set_ylabel('RMSE Loss', fontsize=12)
        ax.set_title('XGBoost Learning Curve - Train vs Validation Loss', fontsize=14)
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Add overfitting/underfitting analysis
        if self.train_losses and self.val_losses:
            final_train = self.train_losses[-1]
            final_val = self.val_losses[-1]
            gap = final_val - final_train
            
            if gap > final_train * 0.3:
                status = "OVERFITTING (large gap)"
                color = 'red'
            elif final_train > 1000:
                status = "UNDERFITTING (high train loss)"
                color = 'orange'
            else:
                status = "GOOD FIT"
                color = 'green'
            
            ax.text(0.02, 0.98, f"Status: {status}\nTrain RMSE: {final_train:.2f}\nVal RMSE: {final_val:.2f}",
                   transform=ax.transAxes, fontsize=10, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor=color, alpha=0.2))
        
        plt.tight_layout()
        
        # Save plot
        if save_path is None:
            save_path = "models/learning_curves/xgboost_learning_curve.png"
        
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        self.learning_curve_path = save_path
        print(f"  Learning curve saved to: {save_path}")
        
        return save_path
    
    def get_learning_curve_data(self) -> Dict[str, List[float]]:
        """Get learning curve data for database storage."""
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_iteration': self.model.best_iteration if hasattr(self.model, 'best_iteration') else None
        }
    
    def predict(self, n_days: int, include_ci: bool = True) -> pd.DataFrame:
        """
        Predict n_days ahead using recursive forecasting with confidence intervals.
        
        Args:
            n_days: Number of days to forecast
            include_ci: Whether to include confidence intervals (MAD-based)
            
        Returns:
            DataFrame with date, predicted_sales, lower_bound, upper_bound
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        
        # Start with recent historical data
        history = self.recent_data.copy()
        predictions = []
        
        for i in range(n_days):
            # Generate next date
            next_date = self.last_date + pd.Timedelta(days=i+1)
            
            # Create a temporary dataframe with history + placeholder for prediction
            temp_df = history.copy()
            new_row = pd.DataFrame({
                'date': [next_date],
                self.target_col: [np.nan]
            })
            temp_df = pd.concat([temp_df, new_row], ignore_index=True)
            
            # Create features
            temp_features = self._create_features(temp_df, self.target_col)
            
            # Get features for the last row (the one we want to predict)
            X_pred = temp_features[self.feature_names].iloc[-1:].values
            
            # Predict
            pred = self.model.predict(X_pred)[0]
            pred = max(pred, 0)  # No negative sales
            
            predictions.append({
                'date': next_date,
                'predicted_sales': pred
            })
            
            # Update history with prediction
            new_row = pd.DataFrame({
                'date': [next_date],
                self.target_col: [pred]
            })
            history = pd.concat([history, new_row], ignore_index=True)
            history = history.tail(max(self.lag_days) + max(self.rolling_windows) + 1)
        
        result = pd.DataFrame(predictions)
        
        # Add confidence intervals (MAD-based)
        if include_ci:
            pred_values = result['predicted_sales'].values
            residual_std = getattr(self, 'residual_std', None)
            confidence_level = getattr(self, 'confidence_level', 0.95)
            
            if residual_std is not None:
                lower, upper = calculate_confidence_bounds(
                    pred_values, residual_std, confidence_level
                )
            else:
                # Fallback: estimate CI from prediction variance (rough estimate ~10%)
                estimated_std = np.mean(pred_values) * 0.1
                lower, upper = calculate_confidence_bounds(
                    pred_values, estimated_std, confidence_level
                )
            
            result['lower_bound'] = lower
            result['upper_bound'] = upper
        
        return result
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance."""
        if not self.is_fitted:
            return {}
        importance = self.model.feature_importances_
        return dict(zip(self.feature_names, importance))
    
    def get_hyperparameters(self) -> Dict[str, Any]:
        return self.params.copy()
    
    @staticmethod
    def get_optuna_params(trial) -> Dict[str, Any]:
        return {
            'n_estimators': trial.suggest_int('xgb_n_estimators', 50, 300),
            'max_depth': trial.suggest_int('xgb_max_depth', 3, 10),
            'learning_rate': trial.suggest_float('xgb_learning_rate', 0.01, 0.3),
            'min_child_weight': trial.suggest_int('xgb_min_child_weight', 1, 10),
            'subsample': trial.suggest_float('xgb_subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('xgb_colsample_bytree', 0.6, 1.0)
        }


class ProphetModel(BaseSimpleModel):
    """
    Facebook Prophet model for time series forecasting.
    
    Good for:
    - Trend detection
    - Seasonality (weekly, yearly)
    - Holiday effects
    """
    
    def __init__(
        self,
        changepoint: float = 0.05,
        seasonality: float = 10.0,
        mode: str = 'multiplicative',
        yearly: bool = True,
        weekly: bool = True
    ):
        super().__init__('prophet')
        self.changepoint_prior_scale = changepoint
        self.seasonality_prior_scale = seasonality
        self.seasonality_mode = mode
        self.yearly_seasonality = yearly
        self.weekly_seasonality = weekly
        self.model = None
    
    def fit(self, df: pd.DataFrame, target_col: str = 'daily_sales') -> Dict[str, Any]:
        """Fit Prophet model."""
        from prophet import Prophet
        
        self.training_data = df.copy()
        self.last_date = df['date'].max()
        self.target_col = target_col
        
        # Prophet requires specific column names
        prophet_df = df[['date', target_col]].copy()
        prophet_df.columns = ['ds', 'y']
        
        # Initialize and fit Prophet
        self.model = Prophet(
            changepoint_prior_scale=self.changepoint_prior_scale,
            seasonality_prior_scale=self.seasonality_prior_scale,
            seasonality_mode=self.seasonality_mode,
            yearly_seasonality=self.yearly_seasonality,
            weekly_seasonality=self.weekly_seasonality,
            daily_seasonality=False
        )
        
        # Add holiday effect if available
        if 'is_holiday' in df.columns:
            holidays = df[df['is_holiday'] == 1][['date']].copy()
            holidays.columns = ['ds']
            holidays['holiday'] = 'holiday'
            self.model = Prophet(
                changepoint_prior_scale=self.changepoint_prior_scale,
                seasonality_prior_scale=self.seasonality_prior_scale,
                seasonality_mode=self.seasonality_mode,
                yearly_seasonality=self.yearly_seasonality,
                weekly_seasonality=self.weekly_seasonality,
                daily_seasonality=False,
                holidays=holidays
            )
        
        self.model.fit(prophet_df)
        
        # Training metrics
        y_pred = self.model.predict(prophet_df)['yhat'].values
        y_true = prophet_df['y'].values
        mape = mean_absolute_percentage_error(y_true, y_pred) * 100
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        
        self.is_fitted = True
        
        return {
            'train_mape': mape,
            'train_rmse': rmse,
            'n_samples': len(df)
        }
    
    def predict(self, n_days: int, include_ci: bool = True) -> pd.DataFrame:
        """
        Predict n_days ahead with native Prophet confidence intervals.
        
        Prophet provides native uncertainty intervals based on:
        - Trend uncertainty (changepoint posterior)
        - Seasonality uncertainty
        - Observation noise
        
        Args:
            n_days: Number of days to forecast
            include_ci: Whether to include confidence intervals
            
        Returns:
            DataFrame with date, predicted_sales, lower_bound, upper_bound
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        
        # Create future dataframe
        future = self.model.make_future_dataframe(periods=n_days)
        
        # Prophet predict returns yhat, yhat_lower, yhat_upper by default
        # The interval_width can be set during model init (default 0.8)
        # We'll use the native intervals which are ~80% CI
        forecast = self.model.predict(future)
        
        # Get only future predictions
        future_forecast = forecast.tail(n_days)[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
        future_forecast.columns = ['date', 'predicted_sales', 'lower_bound', 'upper_bound']
        
        # Ensure non-negative values
        future_forecast['predicted_sales'] = future_forecast['predicted_sales'].clip(lower=0)
        future_forecast['lower_bound'] = future_forecast['lower_bound'].clip(lower=0)
        future_forecast['upper_bound'] = future_forecast['upper_bound'].clip(lower=0)
        
        result = future_forecast.reset_index(drop=True)
        
        # If CI not requested, drop the bounds
        if not include_ci:
            result = result[['date', 'predicted_sales']]
        
        return result
    
    def get_hyperparameters(self) -> Dict[str, Any]:
        return {
            'changepoint_prior_scale': self.changepoint_prior_scale,
            'seasonality_prior_scale': self.seasonality_prior_scale,
            'seasonality_mode': self.seasonality_mode,
            'yearly_seasonality': self.yearly_seasonality,
            'weekly_seasonality': self.weekly_seasonality
        }
    
    @staticmethod
    def get_optuna_params(trial) -> Dict[str, Any]:
        return {
            'changepoint_prior_scale': trial.suggest_float('prophet_changepoint', 0.001, 0.5, log=True),
            'seasonality_prior_scale': trial.suggest_float('prophet_seasonality', 0.1, 20.0),
            'seasonality_mode': trial.suggest_categorical('prophet_mode', ['additive', 'multiplicative']),
            'yearly_seasonality': trial.suggest_categorical('prophet_yearly', [True, False]),
            'weekly_seasonality': trial.suggest_categorical('prophet_weekly', [True, False])
        }


class SARIMAModel(BaseSimpleModel):
    """
    SARIMA model for time series forecasting.
    
    Seasonal ARIMA with configurable (p,d,q)(P,D,Q,s) parameters.
    Uses weekly seasonality (s=7) by default.
    """
    
    def __init__(
        self,
        p: int = 1,
        d: int = 1,
        q: int = 1,
        P: int = 0,
        D: int = 0,
        Q: int = 0,
        trend: str = 't'
    ):
        super().__init__('sarima')
        self.p = p
        self.d = d
        self.q = q
        self.P = P
        self.D = D
        self.Q = Q
        self.order = (p, d, q)
        self.seasonal_order = (P, D, Q, 7) if (P > 0 or D > 0 or Q > 0) else (0, 0, 0, 0)
        self.trend = trend
        self.model = None
        self.fitted_model = None
    
    def fit(self, df: pd.DataFrame, target_col: str = 'daily_sales') -> Dict[str, Any]:
        """Fit SARIMA model."""
        from statsmodels.tsa.statespace.sarimax import SARIMAX
        
        self.training_data = df.copy()
        self.last_date = pd.to_datetime(df['date']).max()
        self.target_col = target_col
        
        # Get target values
        y = df[target_col].values.astype(float)
        
        try:
            # Try with specified parameters
            self.model = SARIMAX(
                y,
                order=self.order,
                seasonal_order=self.seasonal_order,
                trend=self.trend,
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            self.fitted_model = self.model.fit(disp=False, maxiter=500)
            
        except Exception as e:
            # Fallback to simpler ARIMA if SARIMA fails
            print(f"    SARIMA failed, using simpler ARIMA: {str(e)[:50]}")
            self.order = (1, 1, 1)
            self.seasonal_order = (0, 0, 0, 0)
            self.model = SARIMAX(
                y,
                order=self.order,
                seasonal_order=self.seasonal_order,
                trend='t',
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            self.fitted_model = self.model.fit(disp=False, maxiter=500)
        
        # Calculate training metrics
        y_pred = self.fitted_model.fittedvalues
        if hasattr(y_pred, 'values'):
            y_pred = y_pred.values
        
        # Handle NaN in fitted values (common at start of series)
        valid_idx = ~np.isnan(y_pred) & ~np.isnan(y)
        if valid_idx.sum() > 0:
            mape = mean_absolute_percentage_error(y[valid_idx], y_pred[valid_idx]) * 100
            rmse = np.sqrt(mean_squared_error(y[valid_idx], y_pred[valid_idx]))
        else:
            mape = 100.0
            rmse = np.std(y)
        
        self.is_fitted = True
        
        return {
            'train_mape': mape,
            'train_rmse': rmse,
            'n_samples': len(df),
            'order': self.order,
            'seasonal_order': self.seasonal_order,
            'aic': self.fitted_model.aic if hasattr(self.fitted_model, 'aic') else None
        }
    
    def predict(self, n_days: int, include_ci: bool = True) -> pd.DataFrame:
        """
        Predict n_days ahead with native SARIMA confidence intervals.
        
        SARIMA provides analytical confidence intervals based on:
        - Forecast error variance
        - Model parameter uncertainty
        
        Args:
            n_days: Number of days to forecast
            include_ci: Whether to include confidence intervals
            
        Returns:
            DataFrame with date, predicted_sales, lower_bound, upper_bound
        """
        if not self.is_fitted or self.fitted_model is None:
            raise ValueError("Model not fitted")
        
        # Generate future dates
        future_dates = pd.date_range(
            start=self.last_date + pd.Timedelta(days=1),
            periods=n_days,
            freq='D'
        )
        
        # Use get_forecast for confidence intervals
        forecast_result = self.fitted_model.get_forecast(steps=n_days)
        predictions = forecast_result.predicted_mean
        
        # Convert to numpy array
        if hasattr(predictions, 'values'):
            predictions = predictions.values
        elif hasattr(predictions, '__iter__'):
            predictions = np.array(list(predictions))
        else:
            predictions = np.array([predictions])
        
        # Ensure no negative predictions
        predictions = np.maximum(predictions.flatten(), 0)
        
        result = pd.DataFrame({
            'date': future_dates,
            'predicted_sales': predictions
        })
        
        # Add native confidence intervals (95% CI by default)
        if include_ci:
            try:
                # Get confidence intervals (alpha=0.05 for 95% CI)
                conf_int = forecast_result.conf_int(alpha=0.05)
                
                if hasattr(conf_int, 'values'):
                    ci_values = conf_int.values
                else:
                    ci_values = np.array(conf_int)
                
                lower = np.maximum(ci_values[:, 0].flatten(), 0)
                upper = np.maximum(ci_values[:, 1].flatten(), 0)
                
                result['lower_bound'] = lower
                result['upper_bound'] = upper
            except Exception as e:
                # Fallback: use simple std-based CI if native fails
                print(f"    Warning: SARIMA native CI failed, using fallback: {str(e)[:50]}")
                std_estimate = np.std(predictions) * 0.3  # Rough estimate
                result['lower_bound'] = np.maximum(predictions - 1.96 * std_estimate, 0)
                result['upper_bound'] = predictions + 1.96 * std_estimate
        
        return result
    
    def get_hyperparameters(self) -> Dict[str, Any]:
        return {
            'order': self.order,
            'seasonal_order': self.seasonal_order,
            'trend': self.trend
        }
    
    @staticmethod
    def get_optuna_params(trial) -> Dict[str, Any]:
        """Get SARIMA hyperparameters from Optuna trial."""
        return {
            'p': trial.suggest_int('sarima_p', 0, 2),
            'd': trial.suggest_int('sarima_d', 0, 1),
            'q': trial.suggest_int('sarima_q', 0, 2),
            'P': trial.suggest_int('sarima_P', 0, 1),
            'D': trial.suggest_int('sarima_D', 0, 1),
            'Q': trial.suggest_int('sarima_Q', 0, 1),
            'trend': trial.suggest_categorical('sarima_trend', ['n', 'c', 't'])
        }


class RandomForestModel(BaseSimpleModel):
    """
    Random Forest model with lag features for time series.
    
    Features:
    - Lag features (1, 7, 14, 28 days)
    - Rolling mean (7, 14 days)
    - Day of week
    - Month
    - Trend (day number)
    - Out-of-bag (OOB) error tracking for overfitting detection
    """
    
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 10,
        min_samples_split: int = 5,
        min_samples_leaf: int = 2,
        max_features: str = 'sqrt',
        bootstrap: bool = True
    ):
        super().__init__('random_forest')
        self.params = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'min_samples_split': min_samples_split,
            'min_samples_leaf': min_samples_leaf,
            'max_features': max_features,
            'bootstrap': bootstrap,
            'oob_score': bootstrap,  # Enable OOB score for overfitting detection
            'random_state': 42,
            'n_jobs': -1
        }
        self.model = None
        self.feature_names = []
        self.lag_days = [1, 7, 14, 28]
        self.rolling_windows = [7, 14]
        # Learning curve tracking
        self.train_scores = []
        self.oob_scores = []
        self.learning_curve_path = None
    
    def _create_features(self, df: pd.DataFrame, target_col: str) -> pd.DataFrame:
        """Create features for Random Forest."""
        df = df.copy()
        
        # Lag features
        for lag in self.lag_days:
            df[f'lag_{lag}'] = df[target_col].shift(lag)
        
        # Rolling features
        for window in self.rolling_windows:
            df[f'rolling_mean_{window}'] = df[target_col].shift(1).rolling(window).mean()
            df[f'rolling_std_{window}'] = df[target_col].shift(1).rolling(window).std()
        
        # Date features
        df['day_of_week'] = df['date'].dt.dayofweek
        df['month'] = df['date'].dt.month
        df['day_of_year'] = df['date'].dt.dayofyear
        
        # Trend
        df['day_num'] = range(len(df))
        
        return df
    
    def _get_feature_columns(self) -> List[str]:
        """Get list of feature column names."""
        features = []
        features.extend([f'lag_{lag}' for lag in self.lag_days])
        features.extend([f'rolling_mean_{w}' for w in self.rolling_windows])
        features.extend([f'rolling_std_{w}' for w in self.rolling_windows])
        features.extend(['day_of_week', 'month', 'day_of_year', 'day_num'])
        return features
    
    def fit(self, df: pd.DataFrame, target_col: str = 'daily_sales',
            val_df: pd.DataFrame = None) -> Dict[str, Any]:
        """
        Fit Random Forest model with incremental tree tracking.
        
        Args:
            df: Training data
            target_col: Target column name
            val_df: Optional validation data for learning curves
        """
        self.training_data = df.copy()
        self.last_date = df['date'].max()
        self.target_col = target_col
        
        # Create features
        df_features = self._create_features(df, target_col)
        
        # Drop NaN rows (from lag features)
        df_features = df_features.dropna()
        
        self.feature_names = self._get_feature_columns()
        X_train = df_features[self.feature_names].values
        y_train = df_features[target_col].values
        
        # Prepare validation set
        X_val, y_val = None, None
        if val_df is not None and len(val_df) > 0:
            val_features = self._create_features(val_df, target_col).dropna()
            if len(val_features) > 0:
                X_val = val_features[self.feature_names].values
                y_val = val_features[target_col].values
        else:
            # Use last 20% of training data as validation
            split_idx = int(len(X_train) * 0.8)
            X_val = X_train[split_idx:]
            y_val = y_train[split_idx:]
            X_train = X_train[:split_idx]
            y_train = y_train[:split_idx]
        
        # Track learning curves by training incrementally
        self.train_scores = []
        self.oob_scores = []
        
        n_trees_steps = [10, 25, 50, 75, 100, 150, 200]
        n_trees_steps = [n for n in n_trees_steps if n <= self.params['n_estimators']]
        if self.params['n_estimators'] not in n_trees_steps:
            n_trees_steps.append(self.params['n_estimators'])
        
        for n_trees in n_trees_steps:
            params = self.params.copy()
            params['n_estimators'] = n_trees
            
            model = RandomForestRegressor(**params)
            model.fit(X_train, y_train)
            
            # Train score
            y_pred_train = model.predict(X_train)
            train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
            self.train_scores.append({'n_trees': n_trees, 'rmse': train_rmse})
            
            # Validation/OOB score
            if X_val is not None:
                y_pred_val = model.predict(X_val)
                val_rmse = np.sqrt(mean_squared_error(y_val, y_pred_val))
                self.oob_scores.append({'n_trees': n_trees, 'rmse': val_rmse})
            elif hasattr(model, 'oob_score_') and model.oob_score_:
                # Use OOB score as validation proxy
                oob_rmse = np.sqrt(1 - model.oob_score_) * np.std(y_train)
                self.oob_scores.append({'n_trees': n_trees, 'rmse': oob_rmse})
        
        # Train final model with all estimators
        self.model = RandomForestRegressor(**self.params)
        
        # Combine train and val back for final training
        if val_df is None:
            df_features_full = self._create_features(df, target_col).dropna()
            X_full = df_features_full[self.feature_names].values
            y_full = df_features_full[target_col].values
            self.model.fit(X_full, y_full)
        else:
            self.model.fit(X_train, y_train)
        
        # Training metrics on full data
        df_features_full = self._create_features(df, target_col).dropna()
        X_full = df_features_full[self.feature_names].values
        y_full = df_features_full[target_col].values
        
        y_pred = self.model.predict(X_full)
        mape = mean_absolute_percentage_error(y_full, y_pred) * 100
        rmse = np.sqrt(mean_squared_error(y_full, y_pred))
        
        # Calculate MAD-based residual std for confidence intervals
        self.residual_std = self._calculate_residual_std(y_full, y_pred)
        
        self.is_fitted = True
        self.n_train = len(df)
        
        # Store recent data for prediction
        self.recent_data = df.tail(max(self.lag_days) + max(self.rolling_windows)).copy()
        
        return {
            'train_mape': mape,
            'train_rmse': rmse,
            'n_samples': len(df_features_full),
            'n_features': len(self.feature_names),
            'oob_score': self.model.oob_score_ if hasattr(self.model, 'oob_score_') and self.model.oob_score_ else None,
            'residual_std': self.residual_std
        }
    
    def predict(self, n_days: int, include_ci: bool = True) -> pd.DataFrame:
        """
        Predict n_days ahead using recursive forecasting with confidence intervals.
        
        Args:
            n_days: Number of days to forecast
            include_ci: Whether to include confidence intervals (MAD-based)
            
        Returns:
            DataFrame with date, predicted_sales, lower_bound, upper_bound
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        
        # Start with recent historical data
        history = self.recent_data.copy()
        predictions = []
        
        for i in range(n_days):
            # Generate next date
            next_date = self.last_date + pd.Timedelta(days=i+1)
            
            # Create a temporary dataframe with history + placeholder for prediction
            temp_df = history.copy()
            new_row = pd.DataFrame({
                'date': [next_date],
                self.target_col: [np.nan]
            })
            temp_df = pd.concat([temp_df, new_row], ignore_index=True)
            
            # Create features
            temp_features = self._create_features(temp_df, self.target_col)
            
            # Get features for the last row (the one we want to predict)
            X_pred = temp_features[self.feature_names].iloc[-1:].values
            
            # Predict
            pred = self.model.predict(X_pred)[0]
            pred = max(pred, 0)  # No negative sales
            
            predictions.append({
                'date': next_date,
                'predicted_sales': pred
            })
            
            # Update history with prediction
            new_row = pd.DataFrame({
                'date': [next_date],
                self.target_col: [pred]
            })
            history = pd.concat([history, new_row], ignore_index=True)
            history = history.tail(max(self.lag_days) + max(self.rolling_windows) + 1)
        
        result = pd.DataFrame(predictions)
        
        # Add confidence intervals (MAD-based)
        if include_ci:
            pred_values = result['predicted_sales'].values
            residual_std = getattr(self, 'residual_std', None)
            confidence_level = getattr(self, 'confidence_level', 0.95)
            
            if residual_std is not None:
                lower, upper = calculate_confidence_bounds(
                    pred_values, residual_std, confidence_level
                )
            else:
                # Fallback: estimate CI from prediction variance (~10%)
                estimated_std = np.mean(pred_values) * 0.1
                lower, upper = calculate_confidence_bounds(
                    pred_values, estimated_std, confidence_level
                )
            
            result['lower_bound'] = lower
            result['upper_bound'] = upper
        
        return result
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance."""
        if not self.is_fitted:
            return {}
        importance = self.model.feature_importances_
        return dict(zip(self.feature_names, importance))
    
    def plot_learning_curve(self, save_path: str = None) -> str:
        """
        Plot and save the learning curve showing train vs validation loss.
        
        Args:
            save_path: Path to save the plot
            
        Returns:
            Path to saved plot
        """
        if not self.train_scores:
            print("No learning curve data available")
            return None
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        n_trees = [s['n_trees'] for s in self.train_scores]
        train_rmse = [s['rmse'] for s in self.train_scores]
        
        ax.plot(n_trees, train_rmse, label='Train RMSE', color='blue', linewidth=2, marker='o')
        
        if self.oob_scores:
            val_rmse = [s['rmse'] for s in self.oob_scores]
            ax.plot(n_trees, val_rmse, label='Validation RMSE', color='red', linewidth=2, marker='s')
        
        ax.set_xlabel('Number of Trees', fontsize=12)
        ax.set_ylabel('RMSE Loss', fontsize=12)
        ax.set_title('Random Forest Learning Curve - Train vs Validation Loss', fontsize=14)
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Add overfitting/underfitting analysis
        if self.train_scores and self.oob_scores:
            final_train = self.train_scores[-1]['rmse']
            final_val = self.oob_scores[-1]['rmse']
            gap = final_val - final_train
            
            if gap > final_train * 0.3:
                status = "OVERFITTING (large gap)"
                color = 'red'
            elif final_train > 1000:
                status = "UNDERFITTING (high train loss)"
                color = 'orange'
            else:
                status = "GOOD FIT"
                color = 'green'
            
            ax.text(0.02, 0.98, f"Status: {status}\nTrain RMSE: {final_train:.2f}\nVal RMSE: {final_val:.2f}",
                   transform=ax.transAxes, fontsize=10, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor=color, alpha=0.2))
        
        plt.tight_layout()
        
        # Save plot
        if save_path is None:
            save_path = "models/learning_curves/random_forest_learning_curve.png"
        
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        self.learning_curve_path = save_path
        print(f"  Learning curve saved to: {save_path}")
        
        return save_path
    
    def get_learning_curve_data(self) -> Dict[str, Any]:
        """Get learning curve data for database storage."""
        return {
            'train_scores': self.train_scores,
            'oob_scores': self.oob_scores,
            'oob_score': self.model.oob_score_ if hasattr(self.model, 'oob_score_') and self.model.oob_score_ else None
        }
    
    def get_hyperparameters(self) -> Dict[str, Any]:
        return self.params.copy()
    
    @staticmethod
    def get_optuna_params(trial) -> Dict[str, Any]:
        return {
            'n_estimators': trial.suggest_int('rf_n_estimators', 50, 300),
            'max_depth': trial.suggest_int('rf_max_depth', 5, 20),
            'min_samples_split': trial.suggest_int('rf_min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('rf_min_samples_leaf', 1, 10),
            'max_features': trial.suggest_categorical('rf_max_features', ['sqrt', 'log2', None]),
            'bootstrap': True  # Always True for OOB score
        }


# Model registry for easy access
MODEL_REGISTRY = {
    'linear_trend': LinearTrendModel,
    'xgboost': XGBoostModel,
    'random_forest': RandomForestModel,
    'prophet': ProphetModel,
    'sarima': SARIMAModel
}

