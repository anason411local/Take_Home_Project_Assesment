# Sales Forecasting System API Documentation

This document provides comprehensive documentation for the RESTful API and WebSocket endpoints of the Sales Forecasting System, built with FastAPI.

## 1. Base URL

The base URL for the API is typically `http://127.0.0.1:8000` for local development or the public IP/domain of your deployed instance.

## 2. API Documentation Interfaces

FastAPI automatically generates interactive API documentation:

*   **Swagger UI**: `/docs` (e.g., `http://127.0.0.1:8000/docs`)
*   **ReDoc**: `/redoc` (e.g., `http://127.0.0.1:8000/redoc`)

These interfaces allow you to explore endpoints, view request/response schemas, and make test API calls directly from your browser.

## 3. REST API Endpoints

### Health Endpoints

| Method | Endpoint    | Description                          | Tags     |
|:-------|:------------|:-------------------------------------|:---------|
| `GET`  | `/`         | Root endpoint - API health check.    | `Health` |
| `GET`  | `/health`   | Detailed API health check.           | `Health` |

**`/` (Root Endpoint)**
*   **Description**: Simple health check, returns basic service information and main endpoints.
*   **Response**: `{"status": "healthy", "service": "Sales Forecasting API", "version": "1.0.0", "endpoints": {...}}`

**`/health` (Detailed Health Check)**
*   **Description**: Provides a more detailed health status including database connection and total records.
*   **Response**: `{"status": "healthy", "database": "connected", "records": <int>, "timestamp": "<ISO_DATE_TIME>"}`

### Session Management Endpoints

| Method   | Endpoint                    | Description                                       | Tags      |
|:---------|:----------------------------|:--------------------------------------------------|:----------|
| `POST`   | `/api/session`              | Create a new unique session.                      | `Session` |
| `GET`    | `/api/session/{session_id}` | Get details of a specific session.                | `Session` |
| `PUT`    | `/api/session/{session_id}` | Update session preferences (e.g., preferred model). | `Session` |
| `DELETE` | `/api/session/{session_id}` | Delete an active session.                         | `Session` |
| `GET`    | `/api/sessions`             | List all active sessions (admin/monitoring).      | `Session` |

**`/api/session` (Create Session)**
*   **Method**: `POST`
*   **Description**: Creates a new session, returning a unique `session_id` and a WebSocket URL for real-time communication.
*   **Request Headers**: `User-Agent` (Optional)
*   **Response**: `{"success": true, "session_id": "<UUID>", "message": "Session created successfully", "websocket_url": "/ws/<UUID>", "expires_in": 3600}`

**`/api/session/{session_id}` (Get Session Info)**
*   **Method**: `GET`
*   **Description**: Retrieves details for a specific session, including whether its WebSocket is connected.
*   **Path Parameters**: `session_id` (string, required)
*   **Response**: `{"success": true, "session": {...}, "websocket_connected": <boolean>}`

**`/api/session/{session_id}` (Update Session)**
*   **Method**: `PUT`
*   **Description**: Updates preferences for a given session, such as the preferred forecasting model or default horizon.
*   **Path Parameters**: `session_id` (string, required)
*   **Query Parameters**: 
    *   `preferred_model` (string, optional): Preferred forecasting model.
    *   `default_horizon` (integer, optional, >=1, <=365): Default forecast horizon.
*   **Response**: `{"success": true, "message": "Session updated successfully", "session": {...}}`

**`/api/session/{session_id}` (Delete Session)**
*   **Method**: `DELETE`
*   **Description**: Deletes a session and disconnects any associated WebSocket connections.
*   **Path Parameters**: `session_id` (string, required)
*   **Response**: `{"success": true, "message": "Session deleted successfully"}`

**`/api/sessions` (List Sessions)**
*   **Method**: `GET`
*   **Description**: Lists all currently active sessions (primarily for administrative or monitoring purposes).
*   **Response**: `{"success": true, "active_sessions": <int>, "websocket_connections": <int>, "sessions": [...]}`

### Forecasting Endpoints

| Method | Endpoint                    | Description                                       | Tags        |
|:-------|:----------------------------|:--------------------------------------------------|:------------|
| `POST` | `/api/forecast`             | Generate sales forecast with confidence intervals. | `Forecasting` |
| `GET`  | `/api/forecasts`            | Get stored forecasts from the database.           | `Forecasting` |

**`/api/forecast` (Generate Forecast)**
*   **Method**: `POST`
*   **Description**: Generates sales forecasts for a specified number of days using a chosen model, with optional confidence intervals.
*   **Request Body**: `ForecastRequest` schema
    ```json
    {
      "horizon": 30,         // Number of days to forecast (1-365)
      "model": "sarima",     // Model to use (e.g., "sarima", "xgboost", "best", "ensemble")
      "include_ci": true     // Whether to include 95% confidence intervals
    }
    ```
*   **Response**: `ForecastResponse` schema
    ```json
    {
      "success": true,
      "model_used": "sarima",
      "horizon": 30,
      "predictions": [
        { "date": "2025-01-01", "value": 1234.56, "lower_bound": 1100.00, "upper_bound": 1350.00 },
        // ... more predictions
      ],
      "summary": { "total": ..., "mean": ..., "min": ..., "max": ..., "ci_mean_width": ... },
      "confidence_level": 0.95,
      "ci_method": "native" // or "mad", "ensemble", "none"
    }
    ```

**`/api/forecasts` (Get Stored Forecasts)**
*   **Method**: `GET`
*   **Description**: Retrieves past generated forecasts from the database. Can filter by a specific `forecast_id` or `model`.
*   **Query Parameters**: 
    *   `forecast_id` (string, optional): ID of a specific forecast batch.
    *   `model` (string, optional): Filter by model name.
*   **Response**: `{"success": true, "forecast_id": "<UUID>", "total_predictions": <int>, "forecasts": [...]}`

### Data Endpoints

| Method | Endpoint            | Description                                       | Tags   |
|:-------|:--------------------|:--------------------------------------------------|:-------|
| `GET`  | `/api/historical`   | Get historical sales data.                        | `Data` |
| `POST` | `/api/upload`       | Upload new sales data from a CSV file.            | `Data` |

**`/api/historical` (Get Historical Data)**
*   **Method**: `GET`
*   **Description**: Retrieves historical sales data from the database with pagination and date filtering options.
*   **Query Parameters**: 
    *   `limit` (integer, optional, >=1, <=10000): Max records to return.
    *   `offset` (integer, optional, >=0): Records to skip (for pagination).
    *   `start_date` (string, optional, YYYY-MM-DD): Filter from this date.
    *   `end_date` (string, optional, YYYY-MM-DD): Filter until this date.
*   **Response**: `HistoricalDataResponse` schema
    ```json
    {
      "success": true,
      "total_records": <int>,
      "date_range": { "start": "YYYY-MM-DD", "end": "YYYY-MM-DD" },
      "data": [
        { "date": "YYYY-MM-DD", "daily_sales": 1234.56, "marketing_spend": 100.00, "is_holiday": 0 },
        // ... more data
      ],
      "summary": { "mean": ..., "min": ..., "max": ..., "std": ... }
    }
    ```

**`/api/upload` (Upload New Data)**
*   **Method**: `POST`
*   **Description**: Uploads a new CSV file containing sales data. The data is validated and imported into the `sales_data.db` database.
*   **Request Body**: `file` (UploadFile, CSV file)
*   **CSV Requirements**:
    *   **Required**: `date` (YYYY-MM-DD), `daily_sales` (numeric)
    *   **Optional**: `marketing_spend`, `is_holiday`, `product_category`, `day_of_week`
*   **Response**: `UploadResponse` schema
    ```json
    {
      "success": true,
      "message": "Data uploaded successfully",
      "records_imported": <int>,
      "date_range": { "start": "YYYY-MM-DD", "end": "YYYY-MM-DD" }
    }
    ```

### Model & Metrics Endpoints

| Method | Endpoint                    | Description                                       | Tags      |
|:-------|:----------------------------|:--------------------------------------------------|:----------|
| `GET`  | `/api/metrics`              | Get model performance metrics (MAPE, MAE, RMSE). | `Metrics` |
| `GET`  | `/api/feature-importance`   | Get feature importance for tree-based models.     | `Metrics` |
| `GET`  | `/api/models`               | List all available trained models and their status. | `Models`  |

**`/api/metrics` (Get Model Metrics)**
*   **Method**: `GET`
*   **Description**: Retrieves performance metrics (MAPE, MAE, RMSE) for all trained models, or a specific model.
*   **Query Parameters**: `model` (string, optional): Filter by model name.
*   **Response**: `MetricsResponse` schema
    ```json
    {
      "success": true,
      "best_model": "sarima",
      "target_mape": 20.0,
      "target_met": true,
      "models": [
        { "model_name": "sarima", "test_mape": 0.44, "test_mae": 433.0, "test_rmse": 623.0, "training_time": 2.7, "best_params": {...} },
        // ... more models
      ]
    }
    ```

**`/api/feature-importance` (Get Feature Importance)**
*   **Method**: `GET`
*   **Description**: Returns feature importance scores for tree-based models (XGBoost, Random Forest).
*   **Query Parameters**: `model` (string, optional, default: `xgboost`): Model name (`xgboost` or `random_forest`).
*   **Response**: `FeatureImportanceResponse` schema
    ```json
    {
      "success": true,
      "model_name": "xgboost",
      "features": [
        { "feature": "lag_7", "importance": 0.5, "rank": 1 },
        // ... more features
      ]
    }
    ```

**`/api/models` (List Models)**
*   **Method**: `GET`
*   **Description**: Lists all configured models and their current status (loaded, trained, test MAPE).
*   **Response**: `{"success": true, "models": [...], "loaded_count": <int>, "trained_count": <int>}`

### EDA Endpoints

| Method | Endpoint                    | Description                                       | Tags    |
|:-------|:----------------------------|:--------------------------------------------------|:--------|
| `GET`  | `/api/eda/report`           | Get the EDA report text and key insights.         | `EDA`   |
| `GET`  | `/api/eda/images`           | List all available EDA visualization images.      | `EDA`   |
| `GET`  | `/api/eda/image/{filename}` | Get a specific EDA visualization image.           | `EDA`   |

**`/api/eda/report` (Get EDA Report)**
*   **Method**: `GET`
*   **Description**: Retrieves the comprehensive EDA report, including key insights and structured sections.
*   **Response**: `{"success": true, "report": "<full_text_report>", "insights": [...], "sections": {...}}`

**`/api/eda/images` (List EDA Images)**
*   **Method**: `GET`
*   **Description**: Lists all available EDA visualization images with their titles, descriptions, and categories.
*   **Response**: `{"success": true, "total": <int>, "images": [...]}`

**`/api/eda/image/{filename}` (Get EDA Image)**
*   **Method**: `GET`
*   **Description**: Serves a specific EDA visualization image file.
*   **Path Parameters**: `filename` (string, required): Name of the image file (e.g., `time_series.png`).
*   **Response**: `FileResponse` (image/png)

### MLOps Dashboard Endpoints

| Method | Endpoint                           | Description                               | Tags         |
|:-------|:-----------------------------------|:------------------------------------------|:-------------|
| `GET`  | `/api/dashboards/status`           | Get status of MLflow and Optuna dashboards. | `Dashboards` |
| `POST` | `/api/dashboards/mlflow/start`     | Start the MLflow UI server.               | `Dashboards` |
| `POST` | `/api/dashboards/optuna/start`     | Start the Optuna Dashboard server.        | `Dashboards` |
| `POST` | `/api/dashboards/mlflow/stop`      | Stop the MLflow UI server.                | `Dashboards` |
| `POST` | `/api/dashboards/optuna/stop`      | Stop the Optuna Dashboard server.         | `Dashboards` |

**`/api/dashboards/status` (Get Dashboard Status)**
*   **Method**: `GET`
*   **Description**: Returns the running status and URLs for the MLflow and Optuna dashboards.
*   **Response**: `{"success": true, "mlflow": {"running": <bool>, "port": <int>, "url": "<url>"}, "optuna": {...}}`

**`/api/dashboards/mlflow/start` (Start MLflow UI)**
*   **Method**: `POST`
*   **Description**: Starts the MLflow UI server as a background process.
*   **Response**: `{"success": true, "status": "started"|"running"|"error", "port": 5000, "url": "<url>", "message": "..."}`

**`/api/dashboards/optuna/start` (Start Optuna Dashboard)**
*   **Method**: `POST`
*   **Description**: Starts the Optuna Dashboard server as a background process.
*   **Response**: `{"success": true, "status": "started"|"running"|"error", "port": 8080, "url": "<url>", "message": "..."}`

**`/api/dashboards/mlflow/stop` (Stop MLflow UI)**
*   **Method**: `POST`
*   **Description**: Stops the MLflow UI server.
*   **Response**: `{"success": true, "status": "stopped"|"not_running", "message": "..."}`

**`/api/dashboards/optuna/stop` (Stop Optuna Dashboard)**
*   **Method**: `POST`
*   **Description**: Stops the Optuna Dashboard server.
*   **Response**: `{"success": true, "status": "stopped"|"not_running", "message": "..."}`

### Training Pipeline Endpoints (Real-time Model Training)

| Method | Endpoint                      | Description                                       | Tags       |
|:-------|:------------------------------|:--------------------------------------------------|:-----------|
| `POST` | `/api/training/upload`        | Upload a CSV file for real-time model training.   | `Training` |
| `POST` | `/api/training/start`         | Start the training pipeline with uploaded data.   | `Training` |
| `GET`  | `/api/training/status`        | Get the current status of the training pipeline.  | `Training` |
| `POST` | `/api/training/cancel`        | Cancel the currently running training pipeline.   | `Training` |
| `POST` | `/api/training/reset`         | Reset the training pipeline state.                | `Training` |

**`/api/training/upload` (Upload CSV for Training)**
*   **Method**: `POST`
*   **Description**: Uploads a CSV file to be used as input for the real-time model training pipeline.
*   **Request Body**: `file` (UploadFile, CSV file)
*   **CSV Requirements**: Must contain `date` (YYYY-MM-DD) and `daily_sales` (numeric).
*   **Response**: `{"success": true, "message": "File uploaded successfully", "filename": "<filename>", "records_uploaded": <int>, "date_range": {...}}`

**`/api/training/start` (Start Training Pipeline)**
*   **Method**: `POST`
*   **Description**: Initiates the full ML training pipeline (Preprocessing → EDA → Training) using the previously uploaded CSV data.
*   **Query Parameters**: 
    *   `optuna_trials` (integer, optional, default: 10, >=1, <=100): Number of Optuna optimization trials.
    *   `models` (string, optional, default: "linear_trend,xgboost,random_forest,prophet,sarima"): Comma-separated list of models to train.
    *   `test_size` (float, optional, default: 0.2, >=0.1, <=0.4): Fraction of data to use for the test set.
    *   `use_holdout` (boolean, optional, default: `false`): If true, trains the final model only on the train split.
    *   `skip_eda` (boolean, optional, default: `false`): If true, skips the EDA step.
    *   `skip_training` (boolean, optional, default: `false`): If true, skips the training step (only preprocessing runs).
*   **Response**: `{"success": true, "message": "Pipeline started", "config": {...}}`

**`/api/training/status` (Get Training Status)**
*   **Method**: `GET`
*   **Description**: Returns the current status of the real-time training pipeline, including its state, current step, and recent logs.
*   **Response**: `{"success": true, "is_running": <bool>, "current_step": "<step_name>", "progress": <float>, "log_file": "<filename>", "recent_logs": [...]}`

**`/api/training/cancel` (Cancel Training)**
*   **Method**: `POST`
*   **Description**: Requests cancellation of the currently running training pipeline.
*   **Response**: `{"success": true, "message": "Pipeline cancellation requested"}` or `{"success": false, "message": "No pipeline is currently running"}`

**`/api/training/reset` (Reset Training State)**
*   **Method**: `POST`
*   **Description**: Resets the state of the training pipeline, clearing any uploaded files and status.
*   **Response**: `{"success": true, "message": "Training state reset"}`

## 4. WebSocket Endpoints

### A. Forecast Streaming

*   **Endpoint**: `ws://<your_api_host>:8000/ws/{session_id}`
*   **Description**: Provides real-time progress updates and final forecast results.
*   **Connection**: Establish a WebSocket connection using a valid `session_id` obtained from `POST /api/session`.
*   **Client Messages (JSON)**:
    *   `{"action": "forecast", "payload": {"horizon": <int>, "model": "<model_name>"}}`: Request a new forecast. `horizon` and `model` are optional and will use session defaults if not provided.
    *   `{"action": "heartbeat"}`: Send to keep the connection alive.
    *   `{"action": "ping"}`: Simple ping, expects a "pong" response.
*   **Server Messages (JSON)**:
    *   `{"type": "forecast_start", "data": {"horizon": ..., "model": ..., "message": "Starting forecast generation..."}, "session_id": "<UUID>"}`
    *   `{"type": "forecast_progress", "data": {"progress": <int>, "message": "...", "details": {...}}, "session_id": "<UUID>"}`
    *   `{"type": "forecast_complete", "data": {"predictions": [...], "model_used": "...", "summary": {...}}, "session_id": "<UUID>"}`
    *   `{"type": "forecast_error", "data": {"error": "<error_type>", "message": "<error_message>"}, "session_id": "<UUID>"}`
    *   `{"type": "notification", "data": {"message": "pong"}, "session_id": "<UUID>"}` (response to "ping")

### B. Training Pipeline Log Streaming

*   **Endpoint**: `ws://<your_api_host>:8000/ws/training/{session_id}`
*   **Description**: Streams real-time terminal output and status updates from the model training pipeline.
*   **Connection**: Connect using a valid `session_id`. No specific client messages are expected other than optional `ping` for keep-alive.
*   **Server Messages (JSON)**:
    *   `{"type": "connected", "session_id": "<UUID>", "status": {...}}`: Sent on successful connection with current pipeline status.
    *   `{"type": "log", "message": "<log_line>", "log_type": "info"|"warning"|"error"|"success", "timestamp": "<ISO_DATE_TIME>"}`: Real-time terminal output.
    *   `{"type": "status", "status": {"is_running": <bool>, "current_step": "<step_name>", "progress": <float>, "log_file": "<filename>"}}`: Updates on pipeline state.
    *   `{"type": "complete", "success": <bool>, "results": {...}, "error": "..."}`: Notification upon pipeline completion or error.
    *   `{"type": "heartbeat"}`: Sent periodically to keep the connection alive.
