# Sales Forecasting System

## Table of Contents
- [Overview](#overview)
- [Tech Stack](#tech-stack)
- [Key Features](#key-features)
- [Project Structure](#project-structure)
- [ML Model Workflow & MLOps](#ml-model-workflow--mlops)
- [Models](#models)
- [Evaluation Metrics](#evaluation-metrics)
[Getting Started](#getting-started)
  - [Method 1: Docker Hub (Recommended)](#method-1-docker-hub-recommended)
  - [Method 2: Git Clone & Local Setup](#method-2-git-clone--local-setup)
- [Installation](#installation)
- [Quick Start (4-Step Pipeline)](#quick-start-4-step-pipeline)
  - [Step 1: Preprocess Data](#step-1-preprocess-data)
  - [Feature Engineering Details](#feature-engineering-details)
  - [Step 2: Run EDA (Exploratory Data Analysis)](#step-2-run-eda-exploratory-data-analysis)
  - [Step 3: Train Models](#step-3-train-models)
  - [Step 4: Generate Forecasts](#step-4-generate-forecasts)
- [Run the Web Application (API + Frontend)](#run-the-web-application-api--frontend)
- [API Documentation](#api-documentation)
- [REST API Endpoints](#rest-api-endpoints)
- [WebSocket Endpoints](#websocket-endpoints)
- [Frontend Web Application](#frontend-web-application)
- [Data Persistence (SQLite)](#data-persistence-sqlite)
  - [Database Schemas](#database-schemas)
- [Monitoring & Visualization](#monitoring--visualization)
- [Performance Target](#performance-target)
- [Example Results (After Full Data Retraining)](#example-results-after-full-data-retraining)
- [Feature Importance (XGBoost, Random Forest)](#feature-importance-xgboost-random-forest)
- [Logs](#logs)
- [Data Requirements](#data-requirements)
- [Testing](#testing)
- [License](#license)
- [Author](#author)

## Overview

A complete sales forecasting system featuring a machine learning pipeline, real-time API, and interactive web dashboard. It incorporates MLflow for experiment tracking, Optuna for hyperparameter optimization, and SQLite for data persistence.

## Tech Stack

This project leverages a comprehensive set of technologies across different layers:

### 1. Backend & Machine Learning
-   **Python**: Primary programming language.
-   **Pandas & NumPy**: For data manipulation and numerical operations.
-   **Scikit-learn**: Base for various machine learning utilities.
-   **XGBoost**: Gradient Boosting machine learning model.
-   **Random Forest**: Ensemble machine learning model.
-   **Prophet (Meta)**: Time series forecasting library.
-   **Statsmodels**: For statistical modeling, including SARIMA.
-   **FastAPI**: High-performance Python web framework for the API.
-   **Uvicorn**: ASGI server for running FastAPI.
-   **Pydantic**: Data validation and settings management.
-   **SQLAlchemy**: ORM for database interactions (indirectly via custom `database.py`).
-   **SQLite**: Lightweight, file-based database for data persistence (`sales_data.db`, `results.db`).

### 2. MLOps & Experimentation
-   **MLflow**: For tracking experiments, logging parameters, metrics, models, and artifacts.
-   **Optuna**: For automated hyperparameter optimization with pruning.

### 3. Frontend
-   **HTML5**: Structure of the web application.
-   **CSS3**: Styling, including custom themes and responsive design.
-   **Bootstrap 5**: Responsive CSS framework for UI components.
-   **JavaScript (ES6+)**: Frontend logic and interactivity.
-   **jQuery**: Simplified DOM manipulation and AJAX requests.
-   **Plotly.js**: Interactive charting library for data visualization.

### 4. Communication
-   **WebSockets**: Real-time bidirectional communication between frontend and backend for live updates.

### 5. Development & Operations
-   **Conda/UV**: Environment management.
-   **Pytest**: For unit and integration testing.
-   **Docker (Containerization)**: Designed for easy containerization and deployment.
-   **Git**: Version control.

## Key Features

-   **End-to-End Pipeline**: From raw data ingestion to interactive forecasts.
-   **Robust Data Handling**: Data validation, cleaning, and feature engineering.
-   **SQLite Persistence**: Input data, processed features, training results, and forecasts are stored in a local SQLite database.
-   **Multiple Forecasting Models**:
    *   **Linear Trend**: Simple, interpretable, captures linear trends.
    *   **XGBoost**: Powerful gradient boosting model leveraging rich lag and rolling features.
    *   **Random Forest**: Ensemble tree-based model for robust predictions.
    *   **Prophet**: Facebook's model for time series with strong trend and seasonality components.
    *   **SARIMA**: Statistical model for seasonal ARIMA processes.
    *   **Ensemble**: Combines predictions from multiple models for improved accuracy.
-   **Confidence Intervals**:
    *   **Native CI**: Prophet and SARIMA leverage their built-in confidence interval mechanisms using based on Standrd Deviations .
    *   **MAD-based CI**: ML models (XGBoost, Random Forest, Linear Trend) use Median Absolute Deviation-based intervals for robustness against outliers.
-   **Hyperparameter Optimization**: Optuna for efficient tuning of model parameters, with pruning for faster optimization.
-   **Experiment Tracking**: MLflow for logging parameters, metrics (MAPE, MAE, RMSE), artifacts (models, plots), and easy comparison of runs.
-   **Comprehensive EDA**: Dedicated step for statistical analysis, trend, seasonality, and data quality, generating insightful reports and visualizations.
-   **Real-time REST API (FastAPI)**:
    *   Endpoints for forecasting, historical data, model metrics, feature importance, and data upload.
    *   Session management for unique user interactions.
    *   Dynamic dashboards management for MLflow and Optuna UI.
    *   Serving of EDA reports and visualizations.
-   **Real-time WebSocket**: Stream forecast progress updates to the frontend for interactive user experience.
-   **Interactive Web Frontend**:
    *   Multi-page application (Overview, Forecast, Data View, Real-time Training).
    *   Built with HTML, CSS (Bootstrap 5), JavaScript (jQuery), and Plotly.js for interactive charts.
    *   Light theme for a clean, modern aesthetic.
    *   Seamless forecast continuity between historical and forecasted data points.
    *   Visual representation of confidence intervals on forecast charts.
    *   MLOps and Optuna dashboard access buttons.
    *   Structured display of EDA reports and images.
-   **Real-time Model Training Pipeline**:
    *   Upload new CSV data and trigger full ML pipeline from the web interface.
    *   Confirmation step with important requirements and warnings before upload.
    *   Live terminal log streaming via WebSocket during training.
    *   Chained execution: Preprocessing → EDA → Model Training.
    *   Configurable options: model selection, Optuna trials, test size, feature mode.
    *   Newly trained models automatically become the default for forecasting.
-   **Detailed Logging**: All pipeline steps log terminal output to timestamped files in the `logs/` directory.
-   **Docker Ready**: API and dashboard commands are configured for containerized deployment.

## Project Structure

```
├── 📄 app.py                          # Main application entry point
├── 📄 requirements.txt                # Python dependencies
├── 📄 README.md                       # This file
│
├── 🔧 Pipeline Scripts
│   ├── step_1_run_pipeline(...).py    # Data preprocessing & feature engineering
│   ├── step_2_eda_analysis.py         # Exploratory data analysis
│   ├── step_3_train_models.py         # Model training with Optuna/MLflow
│   └── step_4_forecast(testing).py    # Generate predictions
│
├── 📂 src/                            # Source code modules
│   ├── 📂 api/                        # FastAPI application
│   │   ├── main.py                    # API routes and endpoints
│   │   ├── schemas.py                 # Pydantic models
│   │   ├── session.py                 # Session management
│   │   ├── websocket.py               # WebSocket connection manager
│   │   └── pipeline_manager.py        # Real-time training orchestrator
│   │
│   ├── 📂 data/                       # Data handling
│   │   ├── loader.py                  # CSV data loading
│   │   ├── validator.py               # Data validation rules
│   │   ├── cleaner.py                 # Data cleaning utilities
│   │   ├── pipeline.py                # Data pipeline orchestrator
│   │   └── database.py                # SQLite database manager
│   │
│   ├── 📂 eda/                        # Exploratory Data Analysis
│   │   ├── analyzer.py                # Statistical analysis
│   │   └── visualizer.py              # EDA visualizations
│   │
│   ├── 📂 features/                   # Feature engineering
│   │   └── engineer.py                # Feature creation logic
│   │
│   ├── 📂 models/                     # ML model implementations
│   │   └── models.py                  # All model classes
│   │
│   ├── 📂 training/                   # Model training
│   │   ├── trainer.py                 # Training orchestrator
│   │   ├── data_splitter.py           # Time series splitting
│   │   └── metrics.py                 # Evaluation metrics
│   │
│   ├── 📂 forecasting/                # Prediction generation
│   │   └── forecaster.py              # Multi-day forecasting
│   │
│   └── 📂 utils/                      # Utilities
│       ├── config.py                  # Configuration settings
│       ├── logger.py                  # Logging utilities
│       └── terminal_logger.py         # Terminal output capture
│
├── 📂 frontend/                       # Web interface
│   ├── 📂 css/
│   │   └── styles.css                 # Custom styles (light theme)
│   ├── 📂 js/
│   │   ├── config.js                  # Frontend configuration
│   │   ├── api.js                     # API client
│   │   ├── charts.js                  # Plotly chart utilities
│   │   ├── overview.js                # Overview page logic
│   │   ├── forecast.js                # Forecast page logic
│   │   ├── data.js                    # Data view page logic
│   │   └── retraining.js              # Real-time training page logic
│   ├── index.html                     # Overview/Dashboard page
│   ├── forecast.html                  # Forecast generation page
│   ├── data.html                      # Data view & metrics page
│   └── retraining.html                # Real-time model training page
│
├── 📂 data/                           # Data files
│   ├── 📂 raw/                        # Raw input data
│   ├── 📂 processed/                  # Processed features
│   └── 📂 uploads/                    # Uploaded files (runtime)
│
├── 📂 database/                       # SQLite databases
│   ├── sales_data.db                  # Input data & processed features
│   └── results.db                     # Training results & forecasts
│
├── 📂 models/                         # Model artifacts
│   ├── 📂 saved/                      # Trained model files (.pkl)
│   │   ├── linear_trend.pkl
│   │   ├── xgboost.pkl
│   │   ├── random_forest.pkl
│   │   ├── prophet.pkl
│   │   └── sarima.pkl
│   ├── 📂 optuna/                     # Optuna studies
│   │   └── optuna_studies.db
│   ├── 📂 feature_importance/         # Feature importance CSVs
│   │   ├── xgboost_importance.csv
│   │   └── random_forest_importance.csv
│   └── 📂 learning_curves/            # Learning curve plots
│       ├── xgboost_learning_curve.png
│       └── random_forest_learning_curve.png
│
├── 📂 reports/                        # Generated reports
│   └── 📂 eda/                        # EDA outputs
│       ├── eda_report.txt             # Full statistical report
│       ├── key_insights.txt           # Key findings summary
│       ├── time_series.png            # Time series plot
│       ├── distribution.png           # Distribution analysis
│       ├── seasonality.png            # Seasonality patterns
│       ├── trend_analysis.png         # Trend decomposition
│       ├── correlation_heatmap.png    # Feature correlations
│       ├── boxplots.png               # Outlier analysis
│       └── summary_dashboard.png      # Summary visualization
│
├── 📂 mlruns/                         # MLflow experiment tracking
│
├── 📂 logs/                           # Pipeline execution logs
│   ├── step_1_preprocessing_*.log
│   ├── step_2_eda_*.log
│   ├── step_3_training_*.log
│   └── step_4_forecasting_*.log
│
└── 📂 tests/                          # Unit tests
    └── test_data_pipeline.py
```
## Docker Architecture

The Sales Forecasting System runs as a containerized application with the following architecture:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Docker Container                                   │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                      FastAPI (Port 8000)                               │ │
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────────┐ │ │
│  │  │   REST API      │  │   WebSocket     │  │   Static Files          │ │ │
│  │  │   Endpoints     │  │   Streaming     │  │   (Frontend)            │ │ │
│  │  └─────────────────┘  └─────────────────┘  └─────────────────────────┘ │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                    │                                         │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                   ML Pipeline & Services                               │ │
│  │  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌──────────────────┐ │ │
│  │  │  Training  │  │ Forecaster │  │   MLflow   │  │      Optuna      │ │ │
│  │  │  Pipeline  │  │            │  │   (5000)   │  │      (8080)      │ │ │
│  │  └────────────┘  └────────────┘  └────────────┘  └──────────────────┘ │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                    │                                         │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                Persistent Storage (Volumes)                            │ │
│  │  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌──────────────────┐ │ │
│  │  │ Databases  │  │   Models   │  │   MLruns   │  │      Logs        │ │ │
│  │  │ (SQLite)   │  │   (.pkl)   │  │            │  │                  │ │ │
│  │  └────────────┘  └────────────┘  └────────────┘  └──────────────────┘ │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
```


## ML Model Workflow & MLOps

This section details the machine learning model training and operational workflow, covering model selection, hyperparameter optimization, experiment tracking, and the robust time series splitting strategy.
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Raw CSV Data  │───▶│  Preprocessing  │───▶│     Features    │
│                 │    │  & Validation   │    │   Engineering   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                       │
                                                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Forecasting   │◀───│  Best Model     │◀───│  Model Training │
│   & Serving     │    │  Selection      │    │  (Optuna+MLflow)│
└─────────────────┘    └─────────────────┘    └─────────────────┘
```


### 1. Overall ML Workflow

The sales forecasting pipeline follows a structured workflow:

1.  **Data Ingestion & Preprocessing**: Raw sales data is loaded from CSV, validated, cleaned, and stored in `sales_data.db`. Irrelevant columns are dropped, and `daily_sales` is set as the target.
2.  **Feature Engineering**: A comprehensive set of time-series specific features (lag values, rolling statistics, date-based components, and a linear trend) are generated to capture temporal dependencies and patterns. These processed features are also saved to `sales_data.db`.
3.  **Exploratory Data Analysis (EDA)**: Before modeling, an in-depth EDA is performed to understand data distributions, trends, seasonality, and correlations. This step generates key insights and visualizations saved in `reports/eda/`.
4.  **Model Training & Optimization**: Multiple forecasting models (Linear Trend, XGBoost, Random Forest, Prophet, SARIMA) are trained. Optuna is used to find optimal hyperparameters for each model, and MLflow tracks all experiments.
5.  **Model Evaluation**: Models are rigorously evaluated using appropriate time series metrics (MAPE, MAE, RMSE) on a dedicated test set, respecting temporal order. Learning curves, residual plots, and feature importance (for tree-based models) are generated.
6.  **Model Persistence**: Trained models are saved as `.pkl` files in `models/saved/` and their performance metrics, parameters, and other metadata are logged to `results.db` via MLflow.
7.  **Forecasting**: The best-performing model (or an ensemble) is used to generate multi-day forecasts with confidence intervals.
8.  **API & Frontend**: Forecasts and historical data are served via a FastAPI backend to an interactive web frontend, providing real-time predictions and visualizations.

### 2. Model Selection Rationale

We employ a diverse set of models to capture various aspects of time series data:

-   **Linear Trend**: Simple and fast, effective for datasets with a clear linear progression over time. It provides a strong baseline.
-   **XGBoost & Random Forest**: Tree-based ensemble models are powerful for capturing complex, non-linear relationships between engineered features (lags, rolling means, date features) and the target variable. They are robust to outliers and provide valuable feature importance insights.
-   **Prophet**: Specifically designed by Meta for forecasting business time series. It excels at handling trends, multiple seasonality patterns (daily, weekly, yearly), and holidays, making it very suitable for sales data.
-   **SARIMA**: A classical statistical time series model (`Seasonal AutoRegressive Integrated Moving Average`). It's highly effective for data exhibiting clear seasonality and autocorrelation, making it a robust choice for regular sales patterns.

#### Why LSTMs/GRUs (Deep Learning) Were Not Selected:

While LSTMs and GRUs are powerful for sequence modeling, they were intentionally **not included** in this project due to the following reasons:

-   **Data Size Constraints**: Deep learning models, especially LSTMs and GRUs, typically require a *very large amount* of sequential data (thousands to tens of thousands of data points or more) to learn complex temporal patterns effectively. Your dataset, with approximately 669 daily records (roughly 22 months), is **insufficient** for a GRU to generalize well and avoid severe overfitting.
-   **Complexity vs. Benefit**: For the given data size and characteristics, the simpler, interpretable models (SARIMA, Prophet, XGBoost) offer excellent performance (as demonstrated by the low MAPE scores) with significantly less computational overhead and complexity in training, tuning, and deployment. The marginal (or non-existent) performance gains from deep learning models would not justify the added complexity.
-   **Training Time & Resources**: Training deep learning models is resource-intensive and time-consuming, especially without dedicated GPU hardware ( the primary constraint here is data volume). The current models provide quick and accurate forecasts.

### 3. Hyperparameter Optimization with Optuna

**Optuna** is used to efficiently search for the best hyperparameters for each model, minimizing the Mean Absolute Percentage Error (MAPE) on the validation set. Key aspects include:

-   **Objective Function**: For each model, an Optuna `study` is run where an objective function trains and evaluates the model with different parameter combinations, returning the validation MAPE.
-   **Trial & Error Studies**: Optuna performs multiple `trials`, iteratively sampling hyperparameters (e.g., `n_estimators`, `max_depth` for XGBoost; `changepoint_prior_scale` for Prophet).
-   **Pruning**: Optuna's pruning mechanism is enabled, which automatically stops unpromising trials early based on intermediate evaluation metrics. This significantly speeds up the optimization process by avoiding the full training of poorly performing hyperparameter sets.
-   **Dashboard**: The `optuna-dashboard` provides real-time visualization of the optimization process, showing hyperparameter importance, optimization history, and parallel coordinate plots.

### 4. Experiment Tracking with MLflow

**MLflow** is integrated to manage the entire machine learning lifecycle:

-   **Tracking**: All training runs are tracked, logging:
    *   **Parameters**: Hyperparameters used for each model.
    *   **Metrics**: Training and testing MAPE, MAE, RMSE.
    *   **Artifacts**: Trained model `.pkl` files, feature importance plots, learning curve plots, and other diagnostic visualizations.
-   **Reproducibility**: Each run is assigned a unique ID, ensuring full reproducibility of experiments.
-   **Comparison**: The MLflow UI allows easy comparison of different model runs, identifying the best-performing models and understanding the impact of parameter changes.

### 5. Time Series Train/Test Split Strategy

To ensure realistic model evaluation and prevent data leakage (a common pitfall in time series modeling), a rigorous temporal splitting strategy is employed:

```
Training Data: First 80% chronologically
Test Data:     Last 20% chronologically
```

This means that models are always trained only on historical data, and evaluated on future, unseen data. The split is performed once on the entire dataset.

During model training (`step_3_train_models.py`), two distinct options are available for the final production model:

-   **Option 1 (--holdout - Traditional Train/Test Split)**:
    *   The model is trained **only on the initial 80% training portion** of the data.
    *   Validation metrics (MAPE, MAE, RMSE) are reported based on the remaining 20% test set.
    *   This is useful for assessing model generalization *without* using future data in the final training.

-   **Option 2 (Default - Full Data Training)**:
    *   **Hyperparameter Optimization**: Optuna first uses the 80/20 train/test split (with walk-forward cross-validation) to determine the best hyperparameters based on validation performance.
    *   **Final Model Retraining**: After selecting the optimal hyperparameters, the final production model is automatically **retrained on the entire 100% of the available dataset**.
    *   **Rationale**: This approach ensures that the final model deployed for forecasting leverages all historical information to maximize accuracy, while still benefiting from a robust validation process for hyperparameter tuning. The reported metrics (MAPE, MAE, RMSE) still reflect the performance on the initial 20% holdout set, giving a realistic expectation of performance on unseen data, even though the final model uses all data.

We prioritize Option 2 as the default because, for production forecasting, using all available data for the final model often leads to the most accurate predictions, assuming the underlying data distribution remains stable.

## Models

| Model          | Description                                                                 | Best For                           | Confidence Interval Method |
|:---------------|:----------------------------------------------------------------------------|:-----------------------------------|:---------------------------|
| **Linear Trend** | Simple linear regression on day index, with optional day-of-week effects.   | Capturing overall linear trends.   | MAD-based                  |
| **XGBoost**      | Gradient Boosting Regressor using engineered lag and rolling features.      | Complex non-linear patterns, high accuracy. | MAD-based                  |
| **Random Forest**| Ensemble of decision trees, robust to outliers and non-linear relationships. | Capturing non-linear patterns, feature importance. | MAD-based                  |
| **Prophet**      | Decomposes time series into trend, seasonality (yearly, weekly), and holidays. | Data with strong seasonality and trend. | Native (SD-based)          |
| **SARIMA**       | Seasonal AutoRegressive Integrated Moving Average model.                    | Data with clear seasonality and autocorrelation. | Native (SD-based)          |

## Evaluation Metrics

| Metric    | Description                                                 | Target   |
|:----------|:------------------------------------------------------------|:---------|
| **MAPE**  | Mean Absolute Percentage Error (Primary Metric). Lower is better. | < 20%    |
| **MAE**   | Mean Absolute Error. Average dollar amount error.           | As low as possible |
| **RMSE**  | Root Mean Squared Error. Penalizes larger errors more.      | As low as possible |

## Installation

## Getting Started

There are two methods to run the Sales Forecasting System.

### Method 1: Docker Hub (Recommended)

The fastest way to get started. Pull the pre-built Docker image and run instantly.

#### Prerequisites

- **Docker** (version 20.10 or higher)
- **Docker Compose** (version 2.0 or higher)

```bash
docker --version
docker-compose --version
```

#### Step 1: Create Docker Compose File

Create a file named `docker-compose.yml` (or use the provided `docker_compose_up.yml`):

```yaml
version: '3.8'

services:
  sales-forecast:
    image: anasaloor/sales-forecast:latest
    container_name: sales-forecast-app
    restart: unless-stopped
    ports:
      - "8000:8000"   # FastAPI Application
      - "5000:5000"   # MLflow UI
      - "8080:8080"   # Optuna Dashboard
    environment:
      - APP_HOST=0.0.0.0
      - APP_PORT=8000
      - PYTHONUNBUFFERED=1
      - MLFLOW_TRACKING_URI=file:///app/mlruns
      - AUTO_START_DASHBOARDS=true
      - TZ=UTC
    volumes:
      - sales_databases:/app/database
      - sales_models:/app/models
      - sales_mlruns:/app/mlruns
      - sales_reports:/app/reports
      - sales_logs:/app/logs
      - sales_forecasts:/app/forecasts
      - sales_data:/app/data
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 10s
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 2G
        reservations:
          cpus: '1'
          memory: 1G

volumes:
  sales_databases:
    name: sales_forecast_databases
  sales_models:
    name: sales_forecast_models
  sales_mlruns:
    name: sales_forecast_mlruns
  sales_reports:
    name: sales_forecast_reports
  sales_logs:
    name: sales_forecast_logs
  sales_forecasts:
    name: sales_forecast_forecasts
  sales_data:
    name: sales_forecast_data
```

#### Step 2: Pull and Run

```bash
# Pull the latest image
docker-compose pull

# Run in background (detached mode)
docker-compose up -d

# Or run in foreground to see logs
docker-compose up
```

#### Step 3: Access the Application

| Service | URL | Description |
|---------|-----|-------------|
| **Web Dashboard** | http://localhost:8000/app/ | Main application interface |
| **API Docs (Swagger)** | http://localhost:8000/docs | Interactive API documentation |
| **API Docs (ReDoc)** | http://localhost:8000/redoc | Alternative API documentation |
| **MLflow UI** | http://localhost:5000 | Experiment tracking |
| **Optuna Dashboard** | http://localhost:8080 | Hyperparameter optimization |

#### Docker Management Commands

```bash
# View logs
docker-compose logs -f

# Stop the application
docker-compose stop

# Stop and remove containers
docker-compose down

# Stop and remove containers + volumes (resets all data!)
docker-compose down -v

# Restart the application
docker-compose restart

# Enter container shell
docker-compose exec sales-forecast bash
```

---

### Method 2: Git Clone & Local Setup

For development or customization, clone the repository and run locally.

#### Prerequisites

- **Python 3.12.9** (recommended)
- **Conda** or **UV** for environment management
- **Git**

#### Step 1: Create Environment

Using Conda:
```bash
conda create -name sales_forecast python=3.12.9
conda activate sales_forecast
```

Or using UV:
```bash
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

#### Step 2: Clone Repository

```bash
git clone https://github.com/anason411local/Take_Home_Project_Assesment.git
cd Take_Home_Project_Assesment
```

#### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

#### Step 4: Run the Pipeline

Follow the Quick Start (4-Step Pipeline) section below.

#### Step 5: Start the Web Application

```bash
python app.py
```

---

## Quick Start (4-Step Pipeline)

### Step 1: Preprocess Data

```bash
python step_1_run_pipeline.py
```

### Step 2: Run EDA (Exploratory Data Analysis)

```bash
python step_2_eda_analysis.py
```

### Step 3: Train Models

```bash
# Default - Train final model on ALL data after validation
python step_3_train_models.py

# Train with more optimization trials
python step_3_train_models.py --trials 20

# Train specific models only
python step_3_train_models.py --models prophet xgboost
```

### Step 4: Generate Forecasts

```bash
# Forecast 30 days with all models
python step_4_forecast.py --days 30

# Forecast 60 days with a specific model
python step_4_forecast.py --days 60 --model prophet
```

## Run the Web Application (API + Frontend)

```bash
python app.py
```

### Command Line Options

-   `--host <ip_address>`: Specify host (default: 127.0.0.1)
-   `--port <port_number>`: Specify port (default: 8000)
-   `--reload`: Enable auto-reload for development
-   `--no-browser`: Prevent automatic browser opening

## API Documentation

-   **Swagger UI**: http://127.0.0.1:8000/docs
-   **ReDoc**: http://127.0.0.1:8000/redoc


### Feature Engineering Details

Effective feature engineering is crucial for time series forecasting. This pipeline automatically generates a minimal yet powerful set of features to capture trend, seasonality, and past behavior from the `daily_sales` and `date` columns:

-   **Lag Features**: Past sales values at specific time intervals. These are vital for capturing autocorrelation and predicting future values based on recent history.
    *   `lag_1`: Sales from the previous day.
    *   `lag_7`: Sales from exactly one week ago (captures weekly seasonality).
    *   `lag_14`: Sales from two weeks ago.
    *   `lag_28`: Sales from four weeks ago.

-   **Rolling Window Features**: Statistical aggregates over a sliding window of past sales. These help in smoothing out noise and capturing underlying trends or levels.
    *   `rolling_mean_7`: The average sales over the past 7 days (captures weekly trend/level).
    *   `rolling_std_7`: The standard deviation of sales over the past 7 days (captures weekly volatility).
    *   `rolling_mean_14`: The average sales over the past 14 days.
    *   `rolling_std_14`: The standard deviation of sales over the past 14 days.

-   **Date-based Features**: Components extracted from the `date` column to capture various periodic effects.
    *   `day_of_week_num`: Numeric representation of the day of the week (e.g., 0 for Monday, 6 for Sunday).
    *   `is_weekend`: Binary flag (0 or 1) indicating if the day is a weekend.
    *   `day_of_year`: Day number within the year (1-365/366).
    *   `month`: Month number (1-12).
    *   `week_of_year`: Week number within the year.
    *   `quarter`: Quarter of the year.
    *   `is_month_start`, `is_month_end`, `is_quarter_start`, `is_quarter_end`, `is_year_start`, `is_year_end`: Binary flags for specific period boundaries.

-   **Trend Feature**: A simple numerical counter to represent the linear progression of time.
    *   `day_num`: A sequential number assigned to each day, starting from 0 or 1, to model a global linear trend.

This minimal feature set is carefully selected to provide strong predictive signals without introducing excessive complexity or multicollinearity, which is particularly beneficial given the dataset size.

### Step 2: Run EDA (Exploratory Data Analysis)

Generates statistical summaries, key insights, and various visualizations (e.g., time series plots, seasonality, distribution, correlations) and saves them to `reports/eda/`.

```bash
python step_2_eda_analysis.py
```

### Step 3: Train Models

Trains all configured models (Linear Trend, XGBoost, Random Forest, Prophet, SARIMA) using Optuna for hyperparameter optimization and MLflow for experiment tracking. Models are saved as `.pkl` files and results are persisted to `database/results.db`.

By default, the final model for production is trained on **all available data** after optimizing hyperparameters using a train/test split for validation (Option 2). You can use `--holdout` for a traditional train/test split (Option 1).

```bash
# Option 2 (Default - Train final model on ALL data after validation)
python step_3_train_models.py

# Option 1 (--holdout - Train final model only on the train portion)
python step_3_train_models.py --holdout

# Train with more optimization trials
python step_3_train_models.py --trials 20

# Train specific models only
python step_3_train_models.py --models prophet xgboost
```

### Step 4: Generate Forecasts

Generates future sales predictions using the trained models. Forecasts are saved to `database/results.db` and can optionally be saved to CSV. Confidence intervals are included by default.

```bash
# Forecast 30 days with all models (includes confidence intervals)
python step_4_forecast.py --days 30

# Forecast 60 days with a specific model
python step_4_forecast.py --days 60 --model prophet

# Use ensemble (average of all models)
python step_4_forecast.py --days 30 --model ensemble

# Also save results to CSV files
python step_4_forecast.py --days 30 --save-csv
```

## Run the Web Application (API + Frontend)

After running the pipeline steps (especially Step 1 and Step 3), you can start the FastAPI server to access the API and the interactive web frontend:

```bash
python app.py
```

This will start the server and automatically open your default web browser to the dashboard URL (http://127.0.0.1:8000/app/).

### Command Line Options for `app.py`

-   `--host <ip_address>`: Specify host (default: `127.0.0.1`)
-   `--port <port_number>`: Specify port (default: `8000`)
-   `--reload`: Enable auto-reload for development (restarts server on code changes)
-   `--no-browser`: Prevent automatic browser opening

## API Documentation

Access the interactive API documentation (Swagger UI and ReDoc):

-   **Swagger UI**: http://127.0.0.1:8000/docs
-   **ReDoc**: http://127.0.0.1:8000/redoc

## REST API Endpoints

The FastAPI backend provides the following endpoints:

| Method | Endpoint                    | Description                                       | Tags        |
|:-------|:----------------------------|:--------------------------------------------------|:------------|
| `POST` | `/api/session`              | Create a new unique session.                      | `Session`   |
| `GET`  | `/api/session/{session_id}` | Get details of a specific session.                | `Session`   |
| `PUT`  | `/api/session/{session_id}` | Update session preferences (e.g., preferred model). | `Session`   |
| `DELETE`| `/api/session/{session_id}`| Delete an active session.                         | `Session`   |
| `GET`  | `/api/sessions`             | List all active sessions (admin/monitoring).      | `Session`   |
| `POST` | `/api/forecast`             | Generate sales forecast with confidence intervals. | `Forecasting` |
| `GET`  | `/api/forecasts`            | Get stored forecasts from the database.           | `Forecasting` |
| `GET`  | `/api/historical`           | Get historical sales data.                        | `Data`      |
| `POST` | `/api/upload`               | Upload new sales data from a CSV file.            | `Data`      |
| `GET`  | `/api/metrics`              | Get model performance metrics (MAPE, MAE, RMSE). | `Metrics`   |
| `GET`  | `/api/feature-importance`   | Get feature importance for tree-based models (XGBoost, Random Forest). | `Metrics` |
| `GET`  | `/api/models`               | List all available trained models and their status. | `Models`    |
| `GET`  | `/api/eda/report`           | Get the EDA report text and key insights.         | `EDA`       |
| `GET`  | `/api/eda/images`           | List all available EDA visualization images.      | `EDA`       |
| `GET`  | `/api/eda/image/{filename}` | Get a specific EDA visualization image.           | `EDA`       |
| `GET`  | `/api/dashboards/status`    | Get status of MLflow and Optuna dashboards.       | `Dashboards`|
| `POST` | `/api/dashboards/mlflow/start` | Start the MLflow UI server.                     | `Dashboards`|
| `POST` | `/api/dashboards/optuna/start` | Start the Optuna Dashboard server.              | `Dashboards`|
| `POST` | `/api/dashboards/mlflow/stop`  | Stop the MLflow UI server.                      | `Dashboards`|
| `POST` | `/api/dashboards/optuna/stop`  | Stop the Optuna Dashboard server.               | `Dashboards`|
| `POST` | `/api/training/upload`      | Upload a CSV file for real-time model training.   | `Training`  |
| `POST` | `/api/training/start`       | Start the training pipeline with uploaded data.   | `Training`  |
| `GET`  | `/api/training/status`      | Get the current status of the training pipeline.  | `Training`  |
| `POST` | `/api/training/cancel`      | Cancel the currently running training pipeline.   | `Training`  |
| `POST` | `/api/training/reset`       | Reset the training pipeline state.                | `Training`  |
| `GET`  | `/`                         | API health check.                                 | `Health`    |
| `GET`  | `/health`                   | Detailed API health check.                        | `Health`    |

## WebSocket Endpoints

### Forecast Streaming

For real-time forecast streaming and progress updates:

-   **Endpoint**: `ws://127.0.0.1:8000/ws/{session_id}`
-   Connect using a valid `session_id` obtained from `/api/session`.
-   Send `{"action": "forecast", "payload": {"horizon": 30, "model": "sarima"}}` to initiate a real-time forecast.

### Training Pipeline Logs

For real-time training pipeline log streaming:

-   **Endpoint**: `ws://127.0.0.1:8000/ws/training/{session_id}`
-   Connect to receive live terminal output during model training.
-   Messages include: `log` (terminal output), `step_start`, `step_complete`, `pipeline_complete`, `pipeline_error`.

## Frontend Web Application

The interactive web dashboard provides a user-friendly interface to the forecasting system:

### Pages

1.  **Overview (`/app/index.html`)**:
    *   Key Performance Indicators (KPIs) like total records, average sales, best model, and MAPE target status.
    *   Interactive historical sales trend chart.
    *   Quick forecast for the next 7 days with confidence intervals.
    *   Model performance comparison chart.
    *   System information and usage guide.

2.  **Forecast (`/app/forecast.html`)**:
    *   Form to configure and generate custom forecasts (horizon, model selection, real-time streaming toggle).
    *   Interactive forecast chart displaying predictions and 95% confidence intervals.
    *   Seamless continuity between historical and forecasted data points.
    *   Detailed table of daily predictions.

3.  **Data View (`/app/data.html`)**:
    *   **Historical Data Tab**: Paginated table of raw historical sales data.
    *   **EDA Analysis Tab**: Structured display of EDA key insights, statistical summaries, and all generated visualization images (categorized and filterable). Full EDA report view.
    *   **Model Metrics Tab**: Detailed table and bar chart comparing model performance (MAPE, MAE, RMSE).
    *   **Feature Importance Tab**: Bar chart and table displaying feature importance for tree-based models (XGBoost, Random Forest).
    *   **Upload Data Tab**: User interface to upload new CSV sales data to the system.

4.  **Real-time Model Training (`/app/retraining.html`)**:
    *   **Confirmation Required**: Before uploading, users must read important requirements and type "confirmed" to enable the upload section.
    *   **Requirements & Warnings Displayed**:
        1.  CSV format is required.
        2.  Required column order: `date, daily_sales, product_category, marketing_spend, day_of_week, is_holiday`.
        3.  Warning about computational resource limits (Medium/Tiny hosted instances).
        4.  Notification that newly trained models will automatically become the default.
        5.  Guidance to use MLflow and Optuna dashboards for detailed tracking.
    *   **File Upload**: Drag & drop or browse to upload a new CSV dataset.
    *   **Pipeline Configuration**: Options to select models, number of Optuna trials, test size, feature mode, and training mode.
    *   **Real-time Terminal Logs**: Live streaming of terminal output from each pipeline step via WebSocket.
    *   **Pipeline Steps Executed**:
        1.  `step_1_run_pipeline.py` (Preprocessing & Feature Engineering)
        2.  `step_2_eda_analysis.py` (Exploratory Data Analysis)
        3.  `step_3_train_models.py` (Model Training with Optuna & MLflow)
    *   **Progress Tracking**: Visual progress bar and step indicators.
    *   **Download Logs**: Option to download the complete training log.

### Dashboards Integration

The frontend includes buttons to directly open the MLflow UI and Optuna Dashboard in new tabs.

-   **MLOps Tracking**: Opens MLflow UI (default: http://localhost:5000)
-   **Optuna Studies**: Opens Optuna Dashboard (default: http://localhost:8080)

**Important**: A warning message will prompt users to enable pop-ups if these dashboards do not open automatically.

## Data Persistence (SQLite)

All system data is robustly stored in SQLite databases:

### Input Data (`database/sales_data.db`)

This database stores the raw and processed sales data, serving as the input for model training.

-   **`raw_sales`**: Stores the original data imported from the CSV file.
    | Column          | Type    | Description                                  |
    |:----------------|:--------|:---------------------------------------------|
    | `date`          | `TEXT`  | Date of observation (YYYY-MM-DD)             |
    | `daily_sales`   | `REAL`  | Daily sales value                            |
    | `marketing_spend`| `REAL`  | Optional: Daily marketing expenditure        |
    | `is_holiday`    | `INTEGER`| Optional: Binary flag for holidays (0 or 1)  |

-   **`processed_features`**: Stores the data after preprocessing and feature engineering.
    | Column                   | Type    | Description                                            |
    |:-------------------------|:--------|:-------------------------------------------------------|
    | `date`                   | `TEXT`  | Date of observation (YYYY-MM-DD)                       |
    | `daily_sales`            | `REAL`  | Daily sales value (target variable)                    |
    | `marketing_spend`        | `REAL`  | Marketing spend (if available)                         |
    | `is_holiday`             | `INTEGER`| Holiday flag (if available)                            |
    | `lag_X`                  | `REAL`  | Lagged daily sales (e.g., `lag_1`, `lag_7`)            |
    | `rolling_mean_X`         | `REAL`  | Rolling mean of daily sales (e.g., `rolling_mean_7`)   |
    | `rolling_std_X`          | `REAL`  | Rolling standard deviation of daily sales (e.g., `rolling_std_7`) |
    | `day_of_week_num`        | `INTEGER`| Numeric day of the week (0=Mon, 6=Sun)                 |
    | `is_weekend`             | `INTEGER`| Binary flag for weekend (0 or 1)                       |
    | `day_of_year`            | `INTEGER`| Day number of the year                                 |
    | `month`                  | `INTEGER`| Month number                                           |
    | `week_of_year`           | `INTEGER`| Week number of the year                                |
    | `quarter`                | `INTEGER`| Quarter of the year                                    |
    | `is_month_start`/`end`   | `INTEGER`| Binary flag for month start/end                        |
    | `is_quarter_start`/`end` | `INTEGER`| Binary flag for quarter start/end                      |
    | `is_year_start`/`end`    | `INTEGER`| Binary flag for year start/end                         |
    | `day_num`                | `INTEGER`| Sequential day number (trend feature)                  |

### Results (`database/results.db`)

This database stores all training results, model comparisons, feature importances, forecasts, and EDA insights.

-   **`training_runs`**: Stores metadata and metrics for each model training run.
    | Column                 | Type    | Description                                  |
    |:-----------------------|:--------|:---------------------------------------------|
    | `run_id`               | `TEXT`  | Unique MLflow run ID                         |
    | `model_name`           | `TEXT`  | Name of the model trained                    |
    | `comparison_id`        | `TEXT`  | ID for comparing multiple models in one batch |
    | `train_mape`           | `REAL`  | Mean Absolute Percentage Error on training set |
    | `test_mape`            | `REAL`  | Mean Absolute Percentage Error on test set   |
    | `train_mae`            | `REAL`  | Mean Absolute Error on training set          |
    | `test_mae`             | `REAL`  | Mean Absolute Error on test set              |
    | `train_rmse`           | `REAL`  | Root Mean Squared Error on training set      |
    | `test_rmse`            | `REAL`  | Root Mean Squared Error on test set          |
    | `training_time`        | `REAL`  | Time taken for model training (seconds)      |
    | `best_params`          | `TEXT`  | JSON string of best hyperparameters (Optuna) |
    | `n_samples`            | `INTEGER`| Number of samples used for training          |
    | `n_features`           | `INTEGER`| Number of features used                      |
    | `start_date`           | `TEXT`  | Start date of training data                  |
    | `end_date`             | `TEXT`  | End date of training data                    |
    | `train_size`           | `INTEGER`| Number of samples in training set            |
    | `test_size`            | `INTEGER`| Number of samples in test set                |
    | `train_on_full_data`   | `INTEGER`| 1 if final model trained on full data, 0 otherwise |
    | `created_at`           | `TEXT`  | Timestamp of run creation                    |

-   **`model_comparisons`**: Stores the ranking of models from the latest training batch.
    | Column          | Type    | Description                                  |
    |:----------------|:--------|:---------------------------------------------|
    | `comparison_id` | `TEXT`  | ID of the comparison batch                   |
    | `model_name`    | `TEXT`  | Name of the model                            |
    | `rank_by_mape`  | `INTEGER`| Rank based on test MAPE (1 being best)       |
    | `test_mape`     | `REAL`  | Test MAPE                                    |
    | `created_at`    | `TEXT`  | Timestamp of comparison                      |

-   **`feature_importance`**: Stores feature importance scores for tree-based models.
    | Column            | Type    | Description                                  |
    |:------------------|:--------|:---------------------------------------------|
    | `run_id`          | `TEXT`  | MLflow run ID associated with the model      |
    | `model_name`      | `TEXT`  | Name of the model (e.g., 'xgboost', 'random_forest') |
    | `feature_name`    | `TEXT`  | Name of the feature                          |
    | `importance_score`| `REAL`  | Importance score of the feature              |
    | `importance_rank` | `INTEGER`| Rank of the feature by importance            |
    | `created_at`      | `TEXT`  | Timestamp of feature importance logging      |

-   **`forecasts`**: Stores generated forecast predictions.
    | Column             | Type    | Description                                  |
    |:-------------------|:--------|:---------------------------------------------|
    | `forecast_id`      | `TEXT`  | Unique ID for a batch of forecasts           |
    | `model_name`       | `TEXT`  | Model used for this forecast                 |
    | `forecast_date`    | `TEXT`  | Date of the forecasted period (YYYY-MM-DD)   |
    | `predicted_sales`  | `REAL`  | Predicted sales value for the date           |
    | `lower_bound`      | `REAL`  | Lower bound of the confidence interval       |
    | `upper_bound`      | `REAL`  | Upper bound of the confidence interval       |
    | `confidence_level` | `REAL`  | Confidence level (e.g., 0.95 for 95% CI)     |
    | `ci_method`        | `TEXT`  | Method used for CI ('native' or 'mad')       |
    | `day_number`       | `INTEGER`| Day number within the forecast horizon       |
    | `created_at`       | `TEXT`  | Timestamp of forecast generation             |

-   **`ed-insights`**: Stores key insights and statistical summaries from EDA.
    | Column          | Type    | Description                                  |
    |:----------------|:--------|:---------------------------------------------|
    | `insight_id`    | `TEXT`  | Unique ID for the insight                    |
    | `type`          | `TEXT`  | Type of insight (e.g., 'trend', 'seasonality') |
    | `description`   | `TEXT`  | Detailed description of the insight          |
    | `value`         | `REAL`  | Numeric value related to the insight (optional)|
    | `created_at`    | `TEXT`  | Timestamp of insight generation              |

-   **`learning_curves`**: Stores iteration-wise loss data for models like XGBoost and Random Forest.
    | Column          | Type    | Description                                  |
    |:----------------|:--------|:---------------------------------------------|
    | `run_id`        | `TEXT`  | MLflow run ID                                |
    | `model_name`    | `TEXT`  | Name of the model                            |
    | `iteration`     | `INTEGER`| Boosting round or number of trees            |
    | `train_loss`    | `REAL`  | Training loss (e.g., RMSE)                   |
    | `val_loss`      | `REAL`  | Validation loss (e.g., RMSE or OOB score)    |
    | `created_at`    | `TEXT`  | Timestamp                                    |

### Example: Querying Results (Python)

```python
from src.data.database import get_database

db = get_database()

# Get data summary
print(db.get_data_summary())

# Get latest training results
print(db.get_training_results())

# Get best model (based on test MAPE)
print(db.get_best_model())

# Get feature importance for XGBoost
print(db.get_feature_importance('xgboost'))

# Get all forecasts for the latest run
print(db.get_forecasts())

# Get a specific forecast by ID
latest_forecast_id = db.get_latest_forecast_id()
if latest_forecast_id:
    print(db.get_forecasts(forecast_id=latest_forecast_id, model_name='prophet'))
```

## Monitoring & Visualization

### MLflow UI

Access the MLflow UI to view, compare, and manage your machine learning experiments:

```bash
mlflow ui --port 5000
```

Open your browser to [http://localhost:5000](http://localhost:5000)

### Optuna Dashboard

Visualize the hyperparameter optimization process and analyze trial results:

```bash
optuna-dashboard sqlite:///models/optuna/optuna_studies.db --port 8080
```

Open your browser to [http://localhost:8080](http://localhost:8080)

## Performance Target

The system aims for a Mean Absolute Percentage Error (MAPE) of less than 20% for all models.

| Metric | Target | Latest Results (SARIMA) |
|:-------|:-------|:------------------------|
| MAPE   | < 20%  | **0.44%**               |

All models consistently achieve MAPE well under the 20% target.

## Example Results (After Full Data Retraining)

```
Model           Train MAPE   Test MAPE    Test MAE     Test RMSE    Time    
--------------------------------------------------------------------------------
sarima              161.67%       0.44% $      433 $      623    2.7s
linear_trend        154.61%       1.90% $    1,889 $    2,027    1.3s
prophet             158.11%       2.68% $    2,645 $    3,374    1.4s
xgboost             102.72%       6.34% $    6,498 $    8,303   10.6s
random_forest       100.83%       8.82% $    8,956 $   10,611   52.9s

Best Model: sarima
  Test MAPE: 0.44%
  Test MAE:  $433
  Test RMSE: $623

TARGET MET: MAPE 0.44% <= 20.0%
```

## Feature Importance (XGBoost, Random Forest)

For tree-based models, feature importance provides insight into which factors drive predictions.

### XGBoost Top 5 Features

1.  `lag_7` - Sales 7 days ago
2.  `lag_14` - Sales 14 days ago
3.  `lag_28` - Sales 28 days ago
4.  `rolling_mean_7` - 7-day rolling average
5.  `lag_1` - Yesterday's sales

### Random Forest Top 5 Features

(Similar features as XGBoost, typically highlighting recent sales and rolling averages)

1.  `lag_7` - Sales 7 days ago
2.  `lag_14` - Sales 14 days ago
3.  `rolling_mean_7` - 7-day rolling average
4.  `lag_1` - Yesterday's sales
5.  `rolling_mean_14` - 14-day rolling average

## Logs

All terminal output for each step of the pipeline is automatically saved to timestamped log files in the `logs/` directory:

-   `logs/step_1_preprocessing_YYYYMMDD_HHMMSS.log`
-   `logs/step_2_eda_YYYYMMDD_HHMMSS.log`
-   `logs/step_3_training_YYYYMMDD_HHMMSS.log`
-   `logs/step_4_forecasting_YYYYMMDD_HHMMSS.log`

## Data Requirements

The input CSV file for `step_1_run_pipeline.py` should contain at least the following columns:

| Column          | Type       | Description                                  |
|:----------------|:-----------|:---------------------------------------------|
| `date`          | `datetime` | Date of observation (YYYY-MM-DD format)      |
| `daily_sales`   | `float`    | Daily sales value (target variable)          |
| `marketing_spend`| `float`    | (Optional) Daily marketing expenditure       |
| `is_holiday`    | `int`      | (Optional) Binary flag for holidays (0 or 1) |

## Testing

Unit tests are available in the `tests/` directory:

```bash
pytest tests/ -v
```

## License

MIT License

## Author

Anas Aloor
