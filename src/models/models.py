"""
Simple Time Series Models for Sales Forecasting.

Models:
1. LinearTrendModel - Captures overall trend
2. ProphetModel - Facebook's forecasting (trend + seasonality)
3. XGBoostModel - Gradient boosting with lag features
4. SARIMAModel - Statistical time series model

All models follow a simple interface:
- fit(df, target_col) -> trains the model
- predict(n_days) -> returns predictions for n_days ahead
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
import pickle
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
import xgboost as xgb


class BaseSimpleModel:
    """Base class for simple forecasting models."""
    
    def __init__(self, name: str):
        self.name = name
        self.is_fitted = False
        self.last_date = None
        self.training_data = None
    
    def fit(self, df: pd.DataFrame, target_col: str = 'daily_sales') -> Dict[str, Any]:
        """Fit the model. Returns training info."""
        raise NotImplementedError
    
    def predict(self, n_days: int) -> pd.DataFrame:
        """Predict n_days ahead. Returns DataFrame with date and predicted_sales."""
        raise NotImplementedError
    
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
        
        self.is_fitted = True
        self.n_train = len(df)
        
        return {
            'train_mape': mape,
            'train_rmse': rmse,
            'n_samples': len(df)
        }
    
    def predict(self, n_days: int) -> pd.DataFrame:
        """Predict n_days ahead."""
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
        
        return pd.DataFrame({
            'date': future_dates,
            'predicted_sales': predictions
        })
    
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
    """
    
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 5,
        learning_rate: float = 0.1,
        min_child_weight: int = 1,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8
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
        self.model = None
        self.feature_names = []
        self.lag_days = [1, 7, 14, 28]
        self.rolling_windows = [7, 14]
    
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
    
    def fit(self, df: pd.DataFrame, target_col: str = 'daily_sales') -> Dict[str, Any]:
        """Fit XGBoost model."""
        self.training_data = df.copy()
        self.last_date = df['date'].max()
        self.target_col = target_col
        
        # Create features
        df_features = self._create_features(df, target_col)
        
        # Drop NaN rows (from lag features)
        df_features = df_features.dropna()
        
        self.feature_names = self._get_feature_columns()
        X = df_features[self.feature_names].values
        y = df_features[target_col].values
        
        # Train model
        self.model = xgb.XGBRegressor(**self.params)
        self.model.fit(X, y)
        
        # Training metrics
        y_pred = self.model.predict(X)
        mape = mean_absolute_percentage_error(y, y_pred) * 100
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        
        self.is_fitted = True
        self.n_train = len(df)
        
        # Store recent data for prediction
        self.recent_data = df.tail(max(self.lag_days) + max(self.rolling_windows)).copy()
        
        return {
            'train_mape': mape,
            'train_rmse': rmse,
            'n_samples': len(df_features),
            'n_features': len(self.feature_names)
        }
    
    def predict(self, n_days: int) -> pd.DataFrame:
        """Predict n_days ahead using recursive forecasting."""
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
        
        return pd.DataFrame(predictions)
    
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
    
    def predict(self, n_days: int) -> pd.DataFrame:
        """Predict n_days ahead."""
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        
        # Create future dataframe
        future = self.model.make_future_dataframe(periods=n_days)
        forecast = self.model.predict(future)
        
        # Get only future predictions
        future_forecast = forecast.tail(n_days)[['ds', 'yhat']].copy()
        future_forecast.columns = ['date', 'predicted_sales']
        future_forecast['predicted_sales'] = future_forecast['predicted_sales'].clip(lower=0)
        
        return future_forecast.reset_index(drop=True)
    
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
    
    def predict(self, n_days: int) -> pd.DataFrame:
        """Predict n_days ahead."""
        if not self.is_fitted or self.fitted_model is None:
            raise ValueError("Model not fitted")
        
        # Forecast
        forecast = self.fitted_model.forecast(steps=n_days)
        
        # Generate future dates
        future_dates = pd.date_range(
            start=self.last_date + pd.Timedelta(days=1),
            periods=n_days,
            freq='D'
        )
        
        # Convert forecast to numpy array
        if hasattr(forecast, 'values'):
            predictions = forecast.values
        elif hasattr(forecast, '__iter__'):
            predictions = np.array(list(forecast))
        else:
            predictions = np.array([forecast])
        
        # Ensure no negative predictions
        predictions = np.maximum(predictions.flatten(), 0)
        
        return pd.DataFrame({
            'date': future_dates,
            'predicted_sales': predictions
        })
    
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


# Model registry for easy access
MODEL_REGISTRY = {
    'linear_trend': LinearTrendModel,
    'xgboost': XGBoostModel,
    'prophet': ProphetModel,
    'sarima': SARIMAModel
}

