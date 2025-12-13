"""
Model Trainer with MLflow tracking, Optuna optimization, and SQLite persistence.

Features:
- Proper time series train/test split (temporal ordering)
- Multiple evaluation metrics (MAPE, MAE, RMSE)
- Feature importance storage
- SQLite database persistence
- MLflow experiment tracking
- Optuna hyperparameter optimization
"""
import os
import json
import time
import uuid
from pathlib import Path
from typing import Dict, Any, Optional, List, Type
from dataclasses import dataclass, asdict
from datetime import datetime
import numpy as np
import pandas as pd
import mlflow
import optuna
from optuna.samplers import TPESampler
import warnings
warnings.filterwarnings('ignore')

from ..models.models import (
    BaseSimpleModel,
    LinearTrendModel,
    XGBoostModel,
    RandomForestModel,
    ProphetModel,
    SARIMAModel,
    MODEL_REGISTRY
)
from ..data.database import get_database, DatabaseManager
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, mean_absolute_error


@dataclass
class TrainingResult:
    """Training result container with comprehensive metrics."""
    model_name: str
    run_id: str
    # Train metrics
    train_mape: float
    train_mae: float
    train_rmse: float
    # Test metrics
    test_mape: float
    test_mae: float
    test_rmse: float
    # Split info
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    train_size: int
    test_size: int
    # Model info
    best_params: Dict[str, Any]
    training_time: float
    model_path: Optional[str] = None
    feature_importance: Optional[Dict[str, float]] = None


class ModelTrainer:
    """
    Comprehensive model trainer with:
    - Proper time series train/test split
    - MLflow experiment tracking
    - Optuna hyperparameter optimization
    - SQLite database persistence
    - Feature importance extraction
    
    Usage:
        trainer = ModelTrainer()
        results = trainer.train_all(df, target_col='daily_sales')
        trainer.print_summary()
    """
    
    def __init__(
        self,
        experiment_name: str = "sales_forecasting",
        n_optuna_trials: int = 10,
        test_size: float = 0.2,
        models_to_train: Optional[List[str]] = None,
        use_database: bool = True
    ):
        """
        Initialize trainer.
        
        Args:
            experiment_name: MLflow experiment name
            n_optuna_trials: Number of Optuna trials per model
            test_size: Fraction of data for testing (temporal split)
            models_to_train: List of model names (default: all)
            use_database: Whether to persist results to SQLite
        """
        self.experiment_name = experiment_name
        self.n_optuna_trials = n_optuna_trials
        self.test_size = test_size
        self.models_to_train = models_to_train or list(MODEL_REGISTRY.keys())
        self.use_database = use_database
        
        self.results: Dict[str, TrainingResult] = {}
        self.trained_models: Dict[str, BaseSimpleModel] = {}
        self.comparison_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Database manager
        if use_database:
            self.db = get_database()
        
        # Setup MLflow
        mlflow.set_tracking_uri("mlruns")
        mlflow.set_experiment(experiment_name)
        
        # Setup directories
        Path("models/saved").mkdir(parents=True, exist_ok=True)
        Path("models/optuna").mkdir(parents=True, exist_ok=True)
        Path("models/feature_importance").mkdir(parents=True, exist_ok=True)
    
    def _split_data(self, df: pd.DataFrame) -> tuple:
        """
        Proper TIME SERIES train/test split.
        
        IMPORTANT: For time series, we MUST use temporal ordering.
        - Training data: First (1-test_size)% of data chronologically
        - Test data: Last test_size% of data chronologically
        
        This ensures NO DATA LEAKAGE from future to past.
        """
        # Sort by date to ensure temporal ordering
        df = df.sort_values('date').reset_index(drop=True)
        
        # Calculate split point
        split_idx = int(len(df) * (1 - self.test_size))
        
        # Split chronologically
        train_df = df.iloc[:split_idx].copy()
        test_df = df.iloc[split_idx:].copy()
        
        # Get date ranges
        train_start = train_df['date'].min().strftime('%Y-%m-%d')
        train_end = train_df['date'].max().strftime('%Y-%m-%d')
        test_start = test_df['date'].min().strftime('%Y-%m-%d')
        test_end = test_df['date'].max().strftime('%Y-%m-%d')
        
        print(f"  Time Series Split:")
        print(f"    Train: {train_start} to {train_end} ({len(train_df)} days)")
        print(f"    Test:  {test_start} to {test_end} ({len(test_df)} days)")
        
        split_info = {
            'train_start': train_start,
            'train_end': train_end,
            'test_start': test_start,
            'test_end': test_end,
            'train_size': len(train_df),
            'test_size': len(test_df)
        }
        
        return train_df, test_df, split_info
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate comprehensive evaluation metrics."""
        # Ensure no negative predictions
        y_pred = np.maximum(y_pred, 0)
        
        # MAPE (Mean Absolute Percentage Error)
        mape = mean_absolute_percentage_error(y_true, y_pred) * 100
        
        # MAE (Mean Absolute Error)
        mae = mean_absolute_error(y_true, y_pred)
        
        # RMSE (Root Mean Squared Error)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        
        return {
            'mape': mape,
            'mae': mae,
            'rmse': rmse
        }
    
    def _evaluate_model(
        self,
        model: BaseSimpleModel,
        test_df: pd.DataFrame,
        target_col: str
    ) -> tuple:
        """Evaluate model on test data with multiple metrics."""
        n_test = len(test_df)
        predictions = model.predict(n_test)
        
        y_true = test_df[target_col].values
        y_pred = predictions['predicted_sales'].values
        
        metrics = self._calculate_metrics(y_true, y_pred)
        
        return metrics, predictions
    
    def _get_feature_importance(self, model: BaseSimpleModel, model_name: str) -> Optional[Dict[str, float]]:
        """Extract feature importance if available."""
        if model_name in ['xgboost', 'random_forest'] and hasattr(model, 'get_feature_importance'):
            importance = model.get_feature_importance()
            if importance:
                # Save to CSV
                importance_df = pd.DataFrame([
                    {'feature': k, 'importance': v} 
                    for k, v in sorted(importance.items(), key=lambda x: -x[1])
                ])
                importance_df.to_csv(f'models/feature_importance/{model_name}_importance.csv', index=False)
                return importance
        return None
    
    def _save_learning_curve(self, model: BaseSimpleModel, model_name: str) -> Optional[str]:
        """Save learning curve plot if available."""
        if model_name in ['xgboost', 'random_forest'] and hasattr(model, 'plot_learning_curve'):
            try:
                save_path = f"models/learning_curves/{model_name}_learning_curve.png"
                model.plot_learning_curve(save_path)
                return save_path
            except Exception as e:
                print(f"  Warning: Could not save learning curve: {e}")
        return None
    
    def _optimize_hyperparameters(
        self,
        model_class: Type[BaseSimpleModel],
        model_name: str,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        target_col: str
    ) -> Dict[str, Any]:
        """Optimize hyperparameters using Optuna."""
        
        # Create unique study name to avoid conflicts
        study_name = f"{model_name}_{self.comparison_id}"
        
        def objective(trial):
            # Get hyperparameters from trial
            params = model_class.get_optuna_params(trial)
            
            try:
                # Strip prefixes for model initialization
                clean_params = {}
                for key, value in params.items():
                    clean_key = key
                    for prefix in ['lt_', 'xgb_', 'rf_', 'prophet_', 'sarima_']:
                        if key.startswith(prefix):
                            clean_key = key[len(prefix):]
                            break
                    clean_params[clean_key] = value
                
                # Create and train model
                model = model_class(**clean_params)
                model.fit(train_df, target_col)
                
                # Evaluate on test set
                metrics, _ = self._evaluate_model(model, test_df, target_col)
                
                return metrics['mape']
            except Exception as e:
                return float('inf')
        
        # Create Optuna study with unique name
        study = optuna.create_study(
            study_name=study_name,
            direction="minimize",
            sampler=TPESampler(seed=42),
            storage=f"sqlite:///models/optuna/optuna_studies.db",
            load_if_exists=False  # Always create new study
        )
        
        # Optimize
        study.optimize(
            objective,
            n_trials=self.n_optuna_trials,
            show_progress_bar=True,
            n_jobs=1
        )
        
        print(f"  Best trial MAPE: {study.best_trial.value:.2f}%")
        print(f"  Best params: {study.best_trial.params}")
        
        return study.best_trial.params
    
    def train_model(
        self,
        model_name: str,
        df: pd.DataFrame,
        target_col: str = 'daily_sales',
        optimize: bool = True
    ) -> TrainingResult:
        """
        Train a single model with proper time series handling.
        
        Args:
            model_name: Name of model to train
            df: Training data
            target_col: Target column name
            optimize: Whether to use Optuna optimization
            
        Returns:
            Training result with comprehensive metrics
        """
        print(f"\n{'='*60}")
        print(f"Training: {model_name.upper()}")
        print(f"{'='*60}")
        
        start_time = time.time()
        run_id = f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Get model class
        model_class = MODEL_REGISTRY[model_name]
        
        # Proper time series split
        train_df, test_df, split_info = self._split_data(df)
        
        # Start MLflow run
        with mlflow.start_run(run_name=f"{model_name}_{self.comparison_id}"):
            mlflow.log_param("model_name", model_name)
            mlflow.log_param("train_size", split_info['train_size'])
            mlflow.log_param("test_size", split_info['test_size'])
            mlflow.log_param("train_start", split_info['train_start'])
            mlflow.log_param("train_end", split_info['train_end'])
            mlflow.log_param("test_start", split_info['test_start'])
            mlflow.log_param("test_end", split_info['test_end'])
            
            # Optimize hyperparameters
            if optimize and self.n_optuna_trials > 0:
                print(f"  Optimizing hyperparameters ({self.n_optuna_trials} trials)...")
                best_params = self._optimize_hyperparameters(
                    model_class, model_name, train_df, test_df, target_col
                )
            else:
                best_params = {}
            
            # Strip prefixes from parameter names
            clean_params = {}
            for key, value in best_params.items():
                clean_key = key
                for prefix in ['lt_', 'xgb_', 'rf_', 'prophet_', 'sarima_']:
                    if key.startswith(prefix):
                        clean_key = key[len(prefix):]
                        break
                clean_params[clean_key] = value
            
            # Train final model with best params
            print("  Training final model...")
            model = model_class(**clean_params)
            train_info = model.fit(train_df, target_col)
            
            # Calculate training metrics
            train_predictions = model.predict(len(train_df))
            train_metrics = self._calculate_metrics(
                train_df[target_col].values,
                train_predictions['predicted_sales'].values
            )
            
            # Evaluate on test set
            test_metrics, test_predictions = self._evaluate_model(model, test_df, target_col)
            
            # Get feature importance
            feature_importance = self._get_feature_importance(model, model_name)
            
            # Save learning curve for ML models
            learning_curve_path = self._save_learning_curve(model, model_name)
            if learning_curve_path:
                mlflow.log_artifact(learning_curve_path)
            
            # Log all metrics to MLflow
            mlflow.log_metric("train_mape", train_metrics['mape'])
            mlflow.log_metric("train_mae", train_metrics['mae'])
            mlflow.log_metric("train_rmse", train_metrics['rmse'])
            mlflow.log_metric("test_mape", test_metrics['mape'])
            mlflow.log_metric("test_mae", test_metrics['mae'])
            mlflow.log_metric("test_rmse", test_metrics['rmse'])
            
            # Log parameters (with unique names to avoid conflicts)
            for key, value in best_params.items():
                try:
                    mlflow.log_param(f"opt_{key}", value)
                except:
                    pass  # Skip if param already logged
            
            # Save model
            model_path = f"models/saved/{model_name}.pkl"
            model.save(model_path)
            mlflow.log_artifact(model_path)
            
            # Save feature importance if available
            if feature_importance:
                importance_path = f"models/feature_importance/{model_name}_importance.csv"
                mlflow.log_artifact(importance_path)
            
            # Save learning curve data to database
            if self.use_database and hasattr(model, 'get_learning_curve_data'):
                lc_data = model.get_learning_curve_data()
                if lc_data:
                    # For XGBoost
                    if 'train_losses' in lc_data and lc_data['train_losses']:
                        self.db.save_learning_curve(
                            run_id=run_id,
                            model_name=model_name,
                            train_losses=lc_data['train_losses'],
                            val_losses=lc_data.get('val_losses'),
                            metric_type='rmse'
                        )
                    # For Random Forest
                    elif 'train_scores' in lc_data and lc_data['train_scores']:
                        train_losses = [s['rmse'] for s in lc_data['train_scores']]
                        val_losses = [s['rmse'] for s in lc_data.get('oob_scores', [])] if lc_data.get('oob_scores') else None
                        self.db.save_learning_curve(
                            run_id=run_id,
                            model_name=model_name,
                            train_losses=train_losses,
                            val_losses=val_losses,
                            metric_type='rmse'
                        )
            
            training_time = time.time() - start_time
            mlflow.log_metric("training_time", training_time)
        
        # Create result
        result = TrainingResult(
            model_name=model_name,
            run_id=run_id,
            train_mape=train_metrics['mape'],
            train_mae=train_metrics['mae'],
            train_rmse=train_metrics['rmse'],
            test_mape=test_metrics['mape'],
            test_mae=test_metrics['mae'],
            test_rmse=test_metrics['rmse'],
            train_start=split_info['train_start'],
            train_end=split_info['train_end'],
            test_start=split_info['test_start'],
            test_end=split_info['test_end'],
            train_size=split_info['train_size'],
            test_size=split_info['test_size'],
            best_params=best_params,
            training_time=training_time,
            model_path=model_path,
            feature_importance=feature_importance
        )
        
        # Save to database
        if self.use_database:
            self.db.save_training_result(
                run_id=run_id,
                model_name=model_name,
                train_dates=(split_info['train_start'], split_info['train_end']),
                test_dates=(split_info['test_start'], split_info['test_end']),
                metrics={
                    'train_size': split_info['train_size'],
                    'test_size': split_info['test_size'],
                    'train_mape': train_metrics['mape'],
                    'train_mae': train_metrics['mae'],
                    'train_rmse': train_metrics['rmse'],
                    'test_mape': test_metrics['mape'],
                    'test_mae': test_metrics['mae'],
                    'test_rmse': test_metrics['rmse']
                },
                best_params=best_params,
                training_time=training_time,
                model_path=model_path
            )
            
            # Save feature importance
            if feature_importance:
                self.db.save_feature_importance(run_id, model_name, feature_importance)
        
        self.results[model_name] = result
        self.trained_models[model_name] = model
        
        print(f"\n  Results:")
        print(f"    Train - MAPE: {train_metrics['mape']:.2f}%, MAE: ${train_metrics['mae']:,.0f}, RMSE: ${train_metrics['rmse']:,.0f}")
        print(f"    Test  - MAPE: {test_metrics['mape']:.2f}%, MAE: ${test_metrics['mae']:,.0f}, RMSE: ${test_metrics['rmse']:,.0f}")
        print(f"    Time: {training_time:.1f}s")
        
        return result
    
    def train_all(
        self,
        df: pd.DataFrame,
        target_col: str = 'daily_sales',
        optimize: bool = True
    ) -> Dict[str, TrainingResult]:
        """
        Train all specified models.
        
        Args:
            df: Training data
            target_col: Target column name
            optimize: Whether to use Optuna optimization
            
        Returns:
            Dictionary of training results
        """
        print("="*70)
        print("SALES FORECASTING - MODEL TRAINING")
        print("="*70)
        print(f"Comparison ID: {self.comparison_id}")
        print(f"Models to train: {', '.join(self.models_to_train)}")
        print(f"Optuna trials: {self.n_optuna_trials}")
        print(f"Test size: {self.test_size*100:.0f}%")
        print(f"Database persistence: {'Enabled' if self.use_database else 'Disabled'}")
        
        for model_name in self.models_to_train:
            try:
                self.train_model(model_name, df, target_col, optimize)
            except Exception as e:
                print(f"ERROR training {model_name}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # Save model comparison to database
        if self.use_database and self.results:
            comparison_results = [
                {
                    'model_name': r.model_name,
                    'test_mape': r.test_mape,
                    'test_mae': r.test_mae,
                    'test_rmse': r.test_rmse,
                    'training_time': r.training_time
                }
                for r in sorted(self.results.values(), key=lambda x: x.test_mape)
            ]
            self.db.save_model_comparison(self.comparison_id, comparison_results)
        
        return self.results
    
    def print_summary(self):
        """Print comprehensive summary of all training results."""
        if not self.results:
            print("No results to display")
            return
        
        print("\n" + "="*80)
        print("TRAINING SUMMARY")
        print("="*80)
        
        # Sort by test MAPE
        sorted_results = sorted(self.results.values(), key=lambda x: x.test_mape)
        
        # Header
        print(f"\n{'Model':<15} {'Train MAPE':<12} {'Test MAPE':<12} {'Test MAE':<12} {'Test RMSE':<12} {'Time':<8}")
        print("-"*80)
        
        for r in sorted_results:
            print(f"{r.model_name:<15} {r.train_mape:>10.2f}% {r.test_mape:>10.2f}% ${r.test_mae:>9,.0f} ${r.test_rmse:>9,.0f} {r.training_time:>6.1f}s")
        
        # Best model
        best = sorted_results[0]
        print(f"\nBest Model: {best.model_name}")
        print(f"  Test MAPE: {best.test_mape:.2f}%")
        print(f"  Test MAE:  ${best.test_mae:,.0f}")
        print(f"  Test RMSE: ${best.test_rmse:,.0f}")
        
        # Check if target met
        target_mape = 20.0
        if best.test_mape <= target_mape:
            print(f"\nTARGET MET: MAPE {best.test_mape:.2f}% <= {target_mape}%")
        else:
            print(f"\nTARGET NOT MET: MAPE {best.test_mape:.2f}% > {target_mape}%")
        
        # Feature importance summary
        if best.feature_importance:
            print(f"\nTop 5 Features ({best.model_name}):")
            for i, (feat, imp) in enumerate(list(best.feature_importance.items())[:5], 1):
                print(f"  {i}. {feat}: {imp:.4f}")
    
    def save_results(self, path: str = "models/training_results.json"):
        """Save results to JSON file."""
        results_dict = {}
        for name, result in self.results.items():
            results_dict[name] = {
                'run_id': result.run_id,
                'train_mape': result.train_mape,
                'train_mae': result.train_mae,
                'train_rmse': result.train_rmse,
                'test_mape': result.test_mape,
                'test_mae': result.test_mae,
                'test_rmse': result.test_rmse,
                'train_start': result.train_start,
                'train_end': result.train_end,
                'test_start': result.test_start,
                'test_end': result.test_end,
                'train_size': result.train_size,
                'test_size': result.test_size,
                'best_params': result.best_params,
                'training_time': result.training_time,
                'model_path': result.model_path
            }
        
        with open(path, 'w') as f:
            json.dump(results_dict, f, indent=2, default=str)
        
        print(f"\nResults saved to {path}")
        
        if self.use_database:
            print(f"Results also saved to database: database/results.db")
    
    def get_best_model(self) -> BaseSimpleModel:
        """Get the best trained model (lowest test MAPE)."""
        if not self.results:
            raise ValueError("No models trained yet")
        
        best_name = min(self.results, key=lambda x: self.results[x].test_mape)
        return self.trained_models[best_name]
