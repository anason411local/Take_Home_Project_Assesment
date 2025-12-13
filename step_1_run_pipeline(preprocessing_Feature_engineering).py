"""
Step 1: Data Processing Pipeline for Sales Forecasting

This script:
1. Prompts user for CSV file path (or uses default/command-line arg)
2. Imports raw data into SQLite database
3. Validates data quality
4. Cleans and preprocesses the data
5. Engineers features for ML modeling
6. Saves processed data to CSV and database

Usage:
    python "step_1_run_pipeline(preprocessing_Feature_engineering).py"
    python "step_1_run_pipeline(preprocessing_Feature_engineering).py" --input path/to/file.csv

All terminal output is automatically logged to logs/ directory.
"""
import sys
import argparse
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from src.data.pipeline import DataProcessingPipeline
from src.data.database import get_database
from src.utils.config import Config
from src.utils.terminal_logger import TerminalLogger


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Data Processing Pipeline for Sales Forecasting")
    parser.add_argument(
        '--input', '-i',
        type=str,
        default=None,
        help='Path to input CSV file (skips interactive prompt if provided)'
    )
    return parser.parse_args()


def get_csv_file_path(input_arg: str = None) -> str:
    """Get CSV file path from argument or prompt user."""
    
    # If input argument provided, use it directly
    if input_arg:
        csv_path = input_arg
        if not Path(csv_path).exists():
            print(f"ERROR: File not found: {csv_path}")
            sys.exit(1)
        if not csv_path.lower().endswith('.csv'):
            print(f"ERROR: File must be a CSV file: {csv_path}")
            sys.exit(1)
        return csv_path
    
    # Interactive mode
    print("\n" + "="*60)
    print("DATA INPUT")
    print("="*60)
    
    # Default file
    default_file = "ecommerce_sales_data (1).csv"
    
    # Check if default exists
    if Path(default_file).exists():
        print(f"Default file found: {default_file}")
        user_input = input(f"Press Enter to use default, or enter path to different CSV file: ").strip()
        
        if user_input == "":
            return default_file
        else:
            csv_path = user_input
    else:
        csv_path = input("Enter path to CSV file: ").strip()
    
    # Validate file exists
    if not Path(csv_path).exists():
        print(f"ERROR: File not found: {csv_path}")
        sys.exit(1)
    
    # Validate it's a CSV
    if not csv_path.lower().endswith('.csv'):
        print(f"ERROR: File must be a CSV file: {csv_path}")
        sys.exit(1)
    
    return csv_path


def main():
    """Run the complete data processing pipeline."""
    args = parse_args()
    
    # Initialize terminal logging
    terminal_logger = TerminalLogger("step_1_preprocessing", logs_dir="logs")
    terminal_logger.start()
    
    try:
        return _run_pipeline(terminal_logger, input_file_arg=args.input)
    finally:
        terminal_logger.stop()
        print(f"\nLog saved to: {terminal_logger.get_log_path()}")


def _run_pipeline(terminal_logger, input_file_arg=None):
    """Internal pipeline execution."""
    
    print("="*70)
    print("STEP 1: DATA PROCESSING PIPELINE")
    print("="*70)
    print(f"Log file: {terminal_logger.get_log_path()}")
    
    # Get CSV file path from argument or user prompt
    input_file = get_csv_file_path(input_file_arg)
    print(f"\nUsing file: {input_file}")
    
    # Initialize database
    print("\n" + "="*60)
    print("IMPORTING DATA TO DATABASE")
    print("="*60)
    
    db = get_database()
    
    try:
        # Import CSV to database
        n_imported = db.import_sales_data(input_file)
        print(f"Imported {n_imported} rows to database")
        
        # Show data summary
        summary = db.get_data_summary()
        print(f"\nData Summary:")
        print(f"  Total records: {summary['total_records']}")
        print(f"  Date range: {summary['date_start']} to {summary['date_end']}")
        print(f"  Avg sales: ${summary['avg_sales']:,.0f}")
        print(f"  Min sales: ${summary['min_sales']:,.0f}")
        print(f"  Max sales: ${summary['max_sales']:,.0f}")
        
    except Exception as e:
        print(f"ERROR: Failed to import data: {e}")
        return 1
    
    # Run data processing pipeline
    print("\n" + "="*60)
    print("RUNNING FEATURE ENGINEERING PIPELINE")
    print("="*60)
    
    # Configuration
    config = Config(
        raw_data_path=Path("data/raw"),
        processed_data_path=Path("data/processed"),
        lag_days=[1, 7, 14, 28],
        rolling_windows=[7, 14, 28]
    )
    
    # Initialize pipeline
    pipeline = DataProcessingPipeline(config)
    
    # Run pipeline
    output_file = "data/processed/sales_features.csv"
    
    result = pipeline.run(
        input_path=input_file,
        output_path=output_file,
        validate=True,
        create_features=True,
        feature_mode='minimal',  # 'minimal' or 'full'
        drop_na=True,
        save_report=True
    )
    
    # Print results
    if result.success:
        print("\n" + "="*60)
        print("PIPELINE EXECUTION SUMMARY")
        print("="*60)
        print(f"Status: SUCCESS")
        print(f"Rows processed: {result.rows_processed}")
        print(f"Rows dropped (NaN): {result.rows_dropped}")
        print(f"Final rows: {len(result.data)}")
        print(f"Final columns: {len(result.data.columns)}")
        
        if result.feature_summary:
            print(f"\nFeatures created: {result.feature_summary['total_features']}")
            if 'feature_names' in result.feature_summary:
                print("Features:", ', '.join(result.feature_summary['feature_names'][:10]))
                if len(result.feature_summary['feature_names']) > 10:
                    print(f"  ... and {len(result.feature_summary['feature_names']) - 10} more")
        
        if result.validation_result:
            print(f"\nValidation: {'PASSED' if result.validation_result.is_valid else 'FAILED'}")
            print(f"Errors: {len(result.validation_result.errors)}")
            print(f"Warnings: {len(result.validation_result.warnings)}")
        
        print(f"\nData saved to:")
        print(f"  - CSV: {output_file}")
        print(f"  - Database: database/sales_data.db")
        
        # Show sample of processed data
        print("\n" + "="*60)
        print("SAMPLE DATA (first 5 rows)")
        print("="*60)
        display_cols = ['date', 'daily_sales', 'daily_sales_lag_1', 'daily_sales_lag_7', 
                       'daily_sales_rolling_mean_7', 'day_of_week_num', 'is_weekend', 'is_holiday']
        available_cols = [c for c in display_cols if c in result.data.columns]
        print(result.data[available_cols].head().to_string())
        
        print("\n" + "="*60)
        print("NEXT STEPS")
        print("="*60)
        print("1. Run EDA:           python step_2_eda_analysis.py")
        print("2. Train models:      python step_3_train_models.py")
        print("3. Generate forecast: python step_4_forecast.py --days 30")
        
    else:
        print(f"\nPipeline failed: {result.error_message}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
