# Sales Forecasting System

A complete machine learning pipeline for sales forecasting with MLflow experiment tracking, Optuna hyperparameter optimization, and SQLite data persistence.

## Overview

This project implements a 4-step pipeline for sales forecasting:

1. **Data Preprocessing & Feature Engineering** - Clean and prepare data
2. **Exploratory Data Analysis (EDA)** - Understand data patterns
3. **Model Training** - Train and optimize models with MLflow + Optuna
4. **Forecasting** - Generate predictions for N days ahead

## Key Features

- **Proper Time Series Split** - Temporal ordering preserved (no data leakage)
- **Multiple Metrics** - MAPE, MAE, RMSE for comprehensive evaluation
- **SQLite Persistence** - Input data, results, and forecasts stored in database
- **Feature Importance** - XGBoost feature importance tracking
- **Experiment Tracking** - MLflow for metrics and model versioning
- **Hyperparameter Optimization** - Optuna with progress visualization

## Project Structure

```
├── data/
│   ├── raw/                    # Raw input data
│   └── processed/              # Processed features
├── database/
│   ├── sales_data.db           # Input data (SQLite)
│   └── results.db              # Training results, forecasts (SQLite)
├── forecasts/                  # Forecast outputs (CSV)
├── logs/                       # Terminal logs for each step
├── models/
│   ├── saved/                  # Trained model files (.pkl)
│   ├── optuna/                 # Optuna study database
│   └── feature_importance/     # Feature importance CSVs
├── mlruns/                     # MLflow experiment tracking
├── reports/
│   └── eda/                    # EDA visualizations and reports
├── src/
│   ├── data/                   # Data loading, validation, cleaning, database
│   ├── eda/                    # Exploratory data analysis
│   ├── features/               # Feature engineering
│   ├── models/                 # Model implementations
│   ├── training/               # Training with Optuna + MLflow
│   ├── forecasting/            # Prediction generation
│   └── utils/                  # Utilities (logging, config)
├── tests/                      # Unit tests
├── step_1_run_pipeline.py      # Data preprocessing
├── step_2_eda_analysis.py      # EDA and visualizations
├── step_3_train_models.py      # Model training
├── step_4_forecast.py          # Generate forecasts
├── requirements.txt            # Python dependencies
└── README.md
```

## Models

| Model | Description | Best For |
|-------|-------------|----------|
| **Linear Trend** | Linear regression with day-of-week effects | Simple trends |
| **XGBoost** | Gradient boosting with lag features | Complex patterns |
| **Prophet** | Facebook's forecasting library | Trend + seasonality |
| **SARIMA** | Statistical time series model | Seasonal data |

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **MAPE** | Mean Absolute Percentage Error - Primary metric |
| **MAE** | Mean Absolute Error - Dollar amount error |
| **RMSE** | Root Mean Squared Error - Penalizes large errors |

## Installation

### 1. Create Environment

```bash
conda create -n sales_forecast python=3.10
conda activate sales_forecast
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

## Quick Start

### Step 1: Preprocess Data

```bash
python "step_1_run_pipeline(preprocessing_Feature_engineering).py"
```

### Step 2: Run EDA

```bash
python step_2_eda_analysis.py
```

### Step 3: Train Models

```bash
# Train all models with 10 Optuna trials (uses SQLite database)
python step_3_train_models.py

# Train with more optimization trials
python step_3_train_models.py --trials 20

# Use CSV instead of database
python step_3_train_models.py --source csv

# Train specific models only
python step_3_train_models.py --models prophet xgboost
```

### Step 4: Generate Forecasts

```bash
# Forecast 30 days with all models
python step_4_forecast.py --days 30

# Forecast with specific model
python step_4_forecast.py --days 60 --model prophet

# Use ensemble (average of all models)
python step_4_forecast.py --days 30 --model ensemble
```

## Data Persistence (SQLite)

All data is stored in SQLite databases for easy access:

### Input Data (`database/sales_data.db`)
- Raw sales data imported from CSV
- Automatically imported on first training run

### Results (`database/results.db`)
- **training_runs** - All training metrics and parameters
- **model_comparisons** - Model ranking by performance
- **feature_importance** - XGBoost feature importance
- **forecasts** - All generated forecasts
- **eda_insights** - EDA analysis results

### Query Results

```python
from src.data.database import get_database

db = get_database()

# Get data summary
print(db.get_data_summary())

# Get training results
print(db.get_training_results())

# Get best model
print(db.get_best_model())

# Get feature importance
print(db.get_feature_importance('xgboost'))

# Get forecasts
print(db.get_forecasts(model_name='prophet'))
```

## Monitoring & Visualization

### MLflow UI

View experiment metrics, parameters, and compare runs:

```bash
mlflow ui --port 5000
```

Open http://localhost:5000

### Optuna Dashboard

View hyperparameter optimization progress:

```bash
optuna-dashboard sqlite:///models/optuna/optuna_studies.db --port 8080
```

Open http://localhost:8080

## Time Series Train/Test Split

**IMPORTANT**: This system uses proper temporal splitting to prevent data leakage:

```
Training Data: First 80% chronologically
Test Data:     Last 20% chronologically

Example:
  Train: 2023-01-01 to 2024-06-18 (535 days)
  Test:  2024-06-19 to 2024-10-30 (134 days)
```

This ensures:
- No future data leaks into training
- Realistic evaluation of forecasting ability
- Proper time series cross-validation

## Performance Target

| Metric | Target | Typical Result |
|--------|--------|----------------|
| MAPE | < 20% | 1.9% - 6.3% |

All models consistently achieve MAPE well under the 20% target.

## Example Results

```
Model           Train MAPE   Test MAPE    Test MAE     Test RMSE    Time    
--------------------------------------------------------------------------------
sarima              158.26%       1.90% $    1,910 $    2,610    1.8s
linear_trend        154.61%       1.90% $    1,889 $    2,027    1.2s
prophet             158.11%       2.68% $    2,645 $    3,374    1.2s
xgboost             102.73%       6.34% $    6,497 $    8,302    6.5s

Best Model: sarima (Test MAPE: 1.90%)
TARGET MET: MAPE 1.90% <= 20%
```

## Feature Importance (XGBoost)

Top features for XGBoost model:
1. `lag_7` - Sales 7 days ago
2. `lag_14` - Sales 14 days ago
3. `lag_28` - Sales 28 days ago
4. `rolling_mean_7` - 7-day rolling average
5. `lag_1` - Yesterday's sales

## Logs

All terminal output is automatically saved to the `logs/` folder:

- `logs/step_1_preprocessing_YYYYMMDD_HHMMSS.log`
- `logs/step_2_eda_YYYYMMDD_HHMMSS.log`
- `logs/step_3_training_YYYYMMDD_HHMMSS.log`
- `logs/step_4_forecasting_YYYYMMDD_HHMMSS.log`

## Data Requirements

Input CSV should have:

| Column | Type | Description |
|--------|------|-------------|
| date | datetime | Date of observation |
| daily_sales | float | Target variable |
| marketing_spend | float | (Optional) Marketing spend |
| is_holiday | int | (Optional) Holiday flag (0/1) |

## Testing

```bash
pytest tests/ -v
```

## License

MIT License

## Author

AI Engineer Assessment Project
