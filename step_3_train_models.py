"""
Step 3: Train Models for Sales Forecasting

This script:
1. Loads sales data from SQLite database (or CSV fallback)
2. Trains 4 models: Linear Trend, XGBoost, Prophet, SARIMA
3. Uses Optuna for hyperparameter optimization
4. Tracks experiments with MLflow
5. Saves results to SQLite database
6. Stores feature importance for XGBoost

Models:
- Linear Trend: Simple but captures the upward trend
- XGBoost: Gradient boosting with lag features
- Prophet: Facebook's forecasting (trend + seasonality)
- SARIMA: Statistical time series model

Usage:
    python step_3_train_models.py
    python step_3_train_models.py --trials 20
    python step_3_train_models.py --no-optimize
    python step_3_train_models.py --source csv

MLflow UI:
    mlflow ui --port 5000

Optuna Dashboard:
    optuna-dashboard sqlite:///models/optuna/optuna_studies.db --port 8080
"""
import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from src.training import ModelTrainer
from src.data.database import get_database
from src.utils.terminal_logger import TerminalLogger


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train sales forecasting models")
    parser.add_argument(
        '--trials', '-t',
        type=int,
        default=10,
        help='Number of Optuna hyperparameter optimization trials (default: 10)'
    )
    parser.add_argument(
        '--no-optimize',
        action='store_true',
        help='Skip hyperparameter optimization (use defaults)'
    )
    parser.add_argument(
        '--models', '-m',
        type=str,
        nargs='+',
        default=['linear_trend', 'xgboost', 'prophet', 'sarima'],
        help='Models to train (default: all)'
    )
    parser.add_argument(
        '--test-size',
        type=float,
        default=0.2,
        help='Fraction of data for testing (default: 0.2)'
    )
    parser.add_argument(
        '--source',
        type=str,
        choices=['database', 'csv'],
        default='database',
        help='Data source: database (SQLite) or csv (default: database)'
    )
    return parser.parse_args()


def load_data(source: str) -> pd.DataFrame:
    """Load data from specified source."""
    if source == 'database':
        db = get_database()
        
        # Check if database has data
        summary = db.get_data_summary()
        if summary['total_records'] == 0:
            print("Database is empty. Importing from CSV...")
            csv_path = "ecommerce_sales_data (1).csv"
            if Path(csv_path).exists():
                n_imported = db.import_sales_data(csv_path)
                print(f"Imported {n_imported} records to database")
            else:
                raise FileNotFoundError(f"CSV file not found: {csv_path}")
        
        df = db.load_sales_data()
        print(f"Loaded {len(df)} records from database")
        
    else:
        csv_path = "ecommerce_sales_data (1).csv"
        df = pd.read_csv(csv_path, parse_dates=['date'])
        print(f"Loaded {len(df)} records from CSV")
    
    return df


def main():
    """Main training pipeline."""
    args = parse_args()
    
    # Initialize terminal logging
    terminal_logger = TerminalLogger("step_3_training", logs_dir="logs")
    terminal_logger.start()
    
    try:
        return _run_training(args, terminal_logger)
    finally:
        terminal_logger.stop()
        print(f"\nLog saved to: {terminal_logger.get_log_path()}")


def _run_training(args, terminal_logger):
    """Internal training execution."""
    
    print("="*70)
    print("SALES FORECASTING - MODEL TRAINING")
    print("="*70)
    print(f"Log file: {terminal_logger.get_log_path()}")
    print(f"Data source: {args.source}")
    
    # Load data
    print(f"\nLoading data...")
    df = load_data(args.source)
    df = df.sort_values('date').reset_index(drop=True)
    
    print(f"\nData Summary:")
    print(f"  Records: {len(df)}")
    print(f"  Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    print(f"  Target: daily_sales")
    print(f"    Mean: ${df['daily_sales'].mean():,.0f}")
    print(f"    Std:  ${df['daily_sales'].std():,.0f}")
    print(f"    Min:  ${df['daily_sales'].min():,.0f}")
    print(f"    Max:  ${df['daily_sales'].max():,.0f}")
    
    # Initialize trainer
    trainer = ModelTrainer(
        experiment_name="sales_forecasting",
        n_optuna_trials=args.trials if not args.no_optimize else 0,
        test_size=args.test_size,
        models_to_train=args.models,
        use_database=True  # Enable SQLite persistence
    )
    
    # Train all models
    results = trainer.train_all(
        df=df,
        target_col='daily_sales',
        optimize=not args.no_optimize
    )
    
    # Print summary
    trainer.print_summary()
    
    # Save results
    trainer.save_results("models/training_results.json")
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    print("\nResults saved to:")
    print("  - models/training_results.json")
    print("  - database/results.db (SQLite)")
    print("  - mlruns/ (MLflow)")
    print("  - models/feature_importance/ (XGBoost)")
    print("\nNext steps:")
    print("1. View MLflow UI:        mlflow ui --port 5000")
    print("2. View Optuna Dashboard: optuna-dashboard sqlite:///models/optuna/optuna_studies.db --port 8080")
    print("3. Run forecasting:       python step_4_forecast.py --days 30")
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
