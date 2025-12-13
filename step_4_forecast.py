"""
Step 4: Sales Forecasting

This script generates forecasts for N days using trained models.
Results are saved to both CSV and SQLite database.

Usage:
    python step_4_forecast.py --days 30
    python step_4_forecast.py --days 60 --model prophet
    python step_4_forecast.py --days 30 --model all

Arguments:
    --days: Number of days to forecast (required)
    --model: Model to use (linear_trend, xgboost, prophet, sarima, all, ensemble)
"""
import sys
import argparse
from pathlib import Path
from datetime import datetime
import uuid

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from src.forecasting import Forecaster
from src.data.database import get_database
from src.utils.terminal_logger import TerminalLogger


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Forecast sales for N days")
    parser.add_argument(
        '--days', '-d',
        type=int,
        required=True,
        help='Number of days to forecast'
    )
    parser.add_argument(
        '--model', '-m',
        type=str,
        default='all',
        choices=['linear_trend', 'xgboost', 'prophet', 'sarima', 'all', 'ensemble'],
        help='Model to use (default: all)'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='forecasts/forecast_results.csv',
        help='Output file path'
    )
    parser.add_argument(
        '--no-database',
        action='store_true',
        help='Skip saving to database'
    )
    return parser.parse_args()


def print_forecast_summary(forecasts: dict, n_days: int):
    """Print forecast summary."""
    print("\n" + "="*70)
    print(f"FORECAST SUMMARY - {n_days} Days Ahead")
    print("="*70)
    
    for model_name, df in forecasts.items():
        print(f"\n{model_name.upper()}")
        print("-"*40)
        
        preds = df['predicted_sales'].values
        print(f"Date Range: {df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}")
        print(f"Total Predicted Sales: ${preds.sum():,.0f}")
        print(f"Average Daily Sales:   ${preds.mean():,.0f}")
        print(f"Min Daily Sales:       ${preds.min():,.0f}")
        print(f"Max Daily Sales:       ${preds.max():,.0f}")
        
        print(f"\nFirst 5 days:")
        for _, row in df.head(5).iterrows():
            print(f"  {row['date'].strftime('%Y-%m-%d')}: ${row['predicted_sales']:,.0f}")
        
        if n_days > 5:
            print(f"\nLast 3 days:")
            for _, row in df.tail(3).iterrows():
                print(f"  {row['date'].strftime('%Y-%m-%d')}: ${row['predicted_sales']:,.0f}")


def main():
    """Main forecasting function."""
    args = parse_args()
    
    # Initialize terminal logging
    terminal_logger = TerminalLogger("step_4_forecasting", logs_dir="logs")
    terminal_logger.start()
    
    try:
        return _run_forecasting(args, terminal_logger)
    finally:
        terminal_logger.stop()
        print(f"\nLog saved to: {terminal_logger.get_log_path()}")


def _run_forecasting(args, terminal_logger):
    """Internal forecasting execution."""
    
    print("="*70)
    print("SALES FORECASTING - PREDICTION")
    print("="*70)
    print(f"Log file: {terminal_logger.get_log_path()}")
    print(f"Days to forecast: {args.days}")
    print(f"Model: {args.model}")
    
    # Initialize forecaster
    forecaster = Forecaster()
    
    # Load models
    print("\nLoading models...")
    if args.model == 'all' or args.model == 'ensemble':
        forecaster.load_all_models("models/saved")
    else:
        model_path = f"models/saved/{args.model}.pkl"
        if not Path(model_path).exists():
            print(f"ERROR: Model not found: {model_path}")
            print("Please run step_3_train_models.py first.")
            return 1
        forecaster.load_model(args.model, model_path)
    
    print(f"Loaded models: {list(forecaster.models.keys())}")
    
    # Generate forecasts
    print(f"\nGenerating {args.days}-day forecast...")
    
    if args.model == 'all':
        forecasts = forecaster.forecast_all(args.days)
    elif args.model == 'ensemble':
        ensemble_df = forecaster.get_ensemble_forecast(args.days)
        forecasts = {'ensemble': ensemble_df}
        # Also get individual forecasts
        forecasts.update(forecaster.forecast_all(args.days))
    else:
        forecast_df = forecaster.forecast(args.model, args.days)
        forecasts = {args.model: forecast_df}
    
    # Print summary
    print_forecast_summary(forecasts, args.days)
    
    # Save results to CSV
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Combine all forecasts
    all_results = []
    for model_name, df in forecasts.items():
        df_copy = df.copy()
        df_copy['model'] = model_name
        all_results.append(df_copy)
    
    combined_df = pd.concat(all_results, ignore_index=True)
    combined_df.to_csv(output_path, index=False)
    
    print(f"\n{'='*70}")
    print(f"Forecasts saved to: {output_path}")
    
    # Save comparison view
    if len(forecasts) > 1:
        comparison = forecaster.compare_forecasts(args.days)
        comparison_path = output_path.parent / "forecast_comparison.csv"
        comparison.to_csv(comparison_path, index=False)
        print(f"Comparison saved to: {comparison_path}")
    
    # Save to database
    if not args.no_database:
        db = get_database()
        forecast_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        for model_name, df in forecasts.items():
            db.save_forecast(forecast_id, model_name, df)
        
        print(f"Forecasts saved to database: database/results.db")
    
    print(f"{'='*70}")
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
