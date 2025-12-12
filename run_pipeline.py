"""
Main script to run the sales forecasting data processing pipeline.

Usage:
    python run_pipeline.py

This script:
1. Loads the raw ecommerce sales data
2. Validates data quality
3. Cleans and preprocesses the data
4. Engineers features for ML modeling
5. Saves processed data and reports
"""
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from src.data.pipeline import DataProcessingPipeline
from src.utils.config import Config


def main():
    """Run the complete data processing pipeline."""
    
    # Configuration
    config = Config(
        raw_data_path=Path("data/raw"),
        processed_data_path=Path("data/processed"),
        lag_days=[1, 7, 14, 28],
        rolling_windows=[7, 14, 28]
    )
    
    # Initialize pipeline
    pipeline = DataProcessingPipeline(config)
    
    # Input and output paths
    input_file = "ecommerce_sales_data (1).csv"
    output_file = "data/processed/sales_features.csv"
    
    # Run pipeline
    # Use 'minimal' mode for small datasets (~15 features)
    # Use 'full' mode for larger datasets (~68 features)
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
        
        print(f"\nOutput saved to: {output_file}")
        print("="*60)
        
        # Show sample of processed data
        print("\nSample of processed data (first 5 rows, selected columns):")
        display_cols = ['date', 'daily_sales', 'daily_sales_lag_1', 'daily_sales_lag_7', 
                       'daily_sales_rolling_mean_7', 'day_of_week_num', 'is_weekend', 'is_holiday']
        available_cols = [c for c in display_cols if c in result.data.columns]
        print(result.data[available_cols].head().to_string())
        
    else:
        print(f"\nPipeline failed: {result.error_message}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

