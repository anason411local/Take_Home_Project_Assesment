# Sales Forecasting System - Take Home Assessment

A modular, production-ready data processing pipeline for e-commerce sales forecasting.

## Project Structure

```
├── src/
│   ├── data/                    # Data processing modules
│   │   ├── loader.py            # Data loading utilities
│   │   ├── validator.py         # Data validation and quality checks
│   │   ├── cleaner.py           # Data cleaning and preprocessing
│   │   └── pipeline.py          # End-to-end processing pipeline
│   ├── features/                # Feature engineering
│   │   └── engineer.py          # Feature creation (lag, rolling, cyclical)
│   └── utils/                   # Utilities
│       ├── config.py            # Configuration management
│       └── logger.py            # Logging setup
├── tests/                       # Unit tests
├── data/
│   ├── raw/                     # Raw data files
│   └── processed/               # Processed output files
├── run_pipeline.py              # Main execution script
└── requirements.txt             # Dependencies
```

## Part 1: Backend & ML Pipeline - Data Processing

### Feature Engineering Modes

The pipeline supports two feature engineering modes:

#### Minimal Mode (Recommended for this dataset) - 15 features
Best for small datasets (~700 rows). Only loses 7 rows to NaN values.

| Category | Features |
|----------|----------|
| **Date (5)** | `day_of_week_num`, `month`, `quarter`, `day_of_year`, `is_weekend` |
| **Cyclical (2)** | `day_of_week_sin`, `day_of_week_cos` |
| **Lag (3)** | `daily_sales_lag_1`, `daily_sales_lag_7`, `daily_sales_diff_1` |
| **Rolling (3)** | `daily_sales_rolling_mean_7`, `daily_sales_rolling_std_7`, `marketing_spend_rolling_mean_7` |
| **Trend (2)** | `days_since_start`, `time_normalized` |

#### Full Mode - 68 features
For larger datasets. Uses 28-day windows, loses more rows to NaN.

- 15 date features (year, month, day, quarter, week, etc.)
- 8 cyclical features (sin/cos encodings)
- 12 lag features (1, 7, 14, 28 day lags)
- 24 rolling features (7, 14, 28 day windows)
- 4 trend features
- 5 interaction features

### Data Quality Checks

The validator performs comprehensive checks:
- **Schema validation**: Required columns, data types
- **Missing value detection**: Critical vs. non-critical columns
- **Value range validation**: Sales and marketing spend bounds
- **Date continuity**: Gap and duplicate detection
- **Outlier detection**: IQR-based statistical outliers

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Run the Complete Pipeline

```bash
python run_pipeline.py
```

### Use Individual Components

```python
from src.data.pipeline import DataProcessingPipeline
from src.utils.config import Config

# Configure pipeline
config = Config(
    lag_days=[1, 7, 14, 28],
    rolling_windows=[7, 14, 28]
)

# Run pipeline
pipeline = DataProcessingPipeline(config)
result = pipeline.run(
    input_path="ecommerce_sales_data (1).csv",
    output_path="data/processed/sales_features.csv",
    validate=True,
    create_features=True
)

# Access results
if result.success:
    df = result.data
    print(f"Processed {len(df)} rows with {len(df.columns)} columns")
```

### Use Components Individually

```python
from src.data.loader import DataLoader
from src.data.validator import DataValidator
from src.data.cleaner import DataCleaner
from src.features.engineer import FeatureEngineer

# Load data
loader = DataLoader()
df = loader.load_csv("ecommerce_sales_data (1).csv")

# Validate
validator = DataValidator()
result = validator.validate(df)
print(result.summary())

# Clean
cleaner = DataCleaner()
df_clean = cleaner.clean(df)

# Engineer features (choose mode: 'minimal' or 'full')
engineer = FeatureEngineer()
df_features = engineer.create_features(df_clean, mode='minimal')
```

## Running Tests

```bash
python -m pytest tests/ -v
```

## Output Files

After running the pipeline:
- `data/processed/sales_features.csv`: Processed data with all features
- `data/processed/sales_features.report.json`: Processing report with statistics

## Dataset Summary

- **Source**: `ecommerce_sales_data (1).csv`
- **Period**: 2023-01-01 to 2024-10-30 (669 days)
- **Columns**: date, daily_sales, product_category, marketing_spend, day_of_week, is_holiday
- **Category**: Electronics only
- **Key insight**: Strong upward trend (~5x growth), 0.9999 correlation between sales and marketing spend

## Pipeline Output (Minimal Mode)

- **Input**: 669 rows, 6 columns
- **Output**: 662 rows, 21 columns (15 new features)
- **Rows lost**: 7 (due to 7-day rolling window initialization)
