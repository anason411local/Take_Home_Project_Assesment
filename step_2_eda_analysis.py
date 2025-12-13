"""
Step 2: Exploratory Data Analysis (EDA)

This script performs comprehensive EDA on the processed sales data:
- Statistical analysis
- Trend detection
- Seasonality patterns
- Correlation analysis
- Data quality checks
- Visualization generation

Run this after Step 1 (Preprocessing) and before Step 3 (Model Training).

Usage:
    python step_2_eda_analysis.py
    python step_2_eda_analysis.py --input data/processed/sales_features.csv
"""
import sys
import argparse
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
from src.eda import SalesDataAnalyzer, SalesVisualizer
from src.utils.terminal_logger import TerminalLogger


def main():
    """Run EDA analysis and generate visualizations."""
    parser = argparse.ArgumentParser(description='Run Exploratory Data Analysis')
    parser.add_argument('--input', type=str, default='data/processed/sales_features.csv',
                       help='Path to processed data file')
    parser.add_argument('--output', type=str, default='reports/eda',
                       help='Output directory for EDA reports')
    parser.add_argument('--raw', action='store_true',
                       help='Use raw data instead of processed')
    args = parser.parse_args()
    
    # Initialize terminal logger
    logger = TerminalLogger('step_2_eda')
    logger.start()
    
    try:
        print("=" * 70)
        print("STEP 2: EXPLORATORY DATA ANALYSIS (EDA)")
        print("=" * 70)
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        # Determine input file
        if args.raw:
            input_path = Path('data/raw/sales_data.csv')
        else:
            input_path = Path(args.input)
        
        # Check if file exists
        if not input_path.exists():
            print(f"Error: Input file not found: {input_path}")
            print("\nPlease run Step 1 (Preprocessing) first:")
            print("  python step_1_run_pipeline(preprocessing_Feature_engineering).py")
            return 1
        
        print(f"Loading data from: {input_path}")
        df = pd.read_csv(input_path)
        print(f"Loaded {len(df):,} rows, {len(df.columns)} columns")
        print()
        
        # Determine target column
        if 'daily_sales' in df.columns:
            target_col = 'daily_sales'
        elif 'sales' in df.columns:
            target_col = 'sales'
        else:
            # Find first numeric column that looks like sales
            for col in df.columns:
                if 'sales' in col.lower():
                    target_col = col
                    break
            else:
                target_col = df.select_dtypes(include=['number']).columns[0]
        
        print(f"Target column: {target_col}")
        print()
        
        # Create output directory
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # ========================================
        # 1. STATISTICAL ANALYSIS
        # ========================================
        print("-" * 70)
        print("1. STATISTICAL ANALYSIS")
        print("-" * 70)
        
        analyzer = SalesDataAnalyzer(df, date_col='date', target_col=target_col)
        result = analyzer.analyze_all()
        
        # Print summary report
        report = analyzer.get_summary_report()
        print(report)
        
        # Save report to file
        report_path = output_dir / 'eda_report.txt'
        with open(report_path, 'w') as f:
            f.write(report)
        print(f"\nReport saved to: {report_path}")
        
        # ========================================
        # 2. VISUALIZATIONS
        # ========================================
        print()
        print("-" * 70)
        print("2. GENERATING VISUALIZATIONS")
        print("-" * 70)
        
        visualizer = SalesVisualizer(df, date_col='date', target_col=target_col, output_dir=str(output_dir))
        
        print("\nGenerating plots...")
        
        # Time series plot
        print("  - Time series plot...", end=" ")
        path = visualizer.plot_time_series()
        print(f"Done: {path}")
        
        # Distribution plot
        print("  - Distribution plot...", end=" ")
        path = visualizer.plot_distribution()
        print(f"Done: {path}")
        
        # Seasonality plot
        print("  - Seasonality plot...", end=" ")
        path = visualizer.plot_seasonality()
        print(f"Done: {path}")
        
        # Trend analysis plot
        print("  - Trend analysis plot...", end=" ")
        path = visualizer.plot_trend_analysis()
        print(f"Done: {path}")
        
        # Correlation heatmap
        print("  - Correlation heatmap...", end=" ")
        path = visualizer.plot_correlation_heatmap()
        print(f"Done: {path}")
        
        # Boxplots
        print("  - Boxplots...", end=" ")
        path = visualizer.plot_boxplots()
        print(f"Done: {path}")
        
        # Summary dashboard
        print("  - Summary dashboard...", end=" ")
        path = visualizer.plot_summary_dashboard()
        print(f"Done: {path}")
        
        # ========================================
        # 3. KEY INSIGHTS
        # ========================================
        print()
        print("-" * 70)
        print("3. KEY INSIGHTS")
        print("-" * 70)
        
        insights = []
        
        # Trend insight
        trend = result.trend_analysis
        if trend['r_squared'] > 0.5:
            insights.append(f"Strong {trend['trend_direction']} trend detected (R²={trend['r_squared']:.3f})")
        elif trend['r_squared'] > 0.3:
            insights.append(f"Moderate {trend['trend_direction']} trend detected (R²={trend['r_squared']:.3f})")
        else:
            insights.append("No strong linear trend detected")
        
        # Growth insight
        if abs(trend['growth_rate_percent']) > 50:
            insights.append(f"Significant growth: {trend['growth_rate_percent']:.1f}% from first to second half")
        
        # Seasonality insight
        season = result.seasonality
        if 'day_of_week' in season:
            best = season['day_of_week']['best_day']
            worst = season['day_of_week']['worst_day']
            insights.append(f"Day-of-week effect: {best} is best, {worst} is worst")
        
        # Correlation insight
        if result.correlations:
            top_corr = list(result.correlations.items())[0]
            if abs(top_corr[1]) > 0.7:
                insights.append(f"Strong correlation: {top_corr[0]} (r={top_corr[1]:.3f})")
        
        # Autocorrelation insight
        if 'autocorrelation' in season:
            lag1 = season['autocorrelation']['lag_1']
            if lag1 > 0.5:
                insights.append(f"High autocorrelation at lag-1: {lag1:.3f} (good for time series models)")
        
        # Data quality insight
        quality = result.data_quality
        if quality['duplicates'] > 0:
            insights.append(f"Warning: {quality['duplicates']} duplicate rows found")
        if len(quality['missing_values']) > 0:
            insights.append(f"Warning: Missing values in {len(quality['missing_values'])} columns")
        if len(quality['date_gaps']) > 0:
            insights.append(f"Warning: {len(quality['date_gaps'])} date gaps found")
        
        for i, insight in enumerate(insights, 1):
            print(f"  {i}. {insight}")
        
        # Save insights
        insights_path = output_dir / 'key_insights.txt'
        with open(insights_path, 'w') as f:
            f.write("KEY INSIGHTS FROM EDA\n")
            f.write("=" * 50 + "\n\n")
            for i, insight in enumerate(insights, 1):
                f.write(f"{i}. {insight}\n")
        
        # ========================================
        # 4. SUMMARY
        # ========================================
        print()
        print("-" * 70)
        print("4. EDA SUMMARY")
        print("-" * 70)
        print(f"\n  Output Directory: {output_dir}")
        print("\n  Generated Files:")
        for f in sorted(output_dir.glob('*')):
            size = f.stat().st_size / 1024
            print(f"    - {f.name} ({size:.1f} KB)")
        
        print()
        print("=" * 70)
        print("EDA COMPLETED SUCCESSFULLY")
        print("=" * 70)
        print(f"\nNext Step: Run model training:")
        print("  python step_3_train_models.py")
        print()
        
        return 0
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    finally:
        logger.stop()


if __name__ == '__main__':
    sys.exit(main())

