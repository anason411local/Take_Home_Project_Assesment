# System Architecture and Design Decisions

This document outlines the architectural design and key design decisions made during the development of the Sales Forecasting System.

## 1. Overall System Architecture

The Sales Forecasting System is built as a modular, containerized application designed for scalability, maintainability, and real-time performance. It comprises a FastAPI backend, a set of machine learning pipeline scripts, and an interactive web frontend, all orchestrated within a Docker environment.

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

**Key Components:**

*   **FastAPI Backend**: Serves RESTful APIs for data interaction, forecasting, and managing MLflow/Optuna dashboards. It also hosts the static frontend files and manages WebSocket connections for real-time updates and log streaming.
*   **Machine Learning Pipeline Scripts**: A series of Python scripts (`step_1_run_pipeline.py`, `step_2_eda_analysis.py`, `step_3_train_models.py`, `step_4_forecast.py`) that handle data preprocessing, feature engineering, EDA, model training, and forecasting. These can be run independently or triggered via the FastAPI backend for real-time training.
*   **Frontend Web Application**: An interactive multi-page web interface (HTML, CSS, JavaScript) that provides dashboards, forecast generation, data visualization, and a real-time model retraining portal.
*   **MLflow**: An MLOps platform for tracking experiments, managing models, and visualizing training runs.
*   **Optuna**: A hyperparameter optimization framework with a web dashboard for visualizing optimization studies.
*   **SQLite Databases**: Lightweight, file-based databases (`sales_data.db`, `results.db`, `optuna_studies.db`) used for data persistence, including raw/processed data, training results, forecasts, and Optuna studies.
*   **Docker**: Containerization technology to package the entire application and its dependencies into a portable unit, ensuring consistent environments across development, testing, and production.
*   **Docker Compose**: Used for defining and running multi-container Docker applications, simplifying the deployment of the FastAPI app, MLflow, and Optuna dashboards.

## 2. Design Decisions

### A. Data Management and Persistence

1.  **SQLite for Data Storage**:
    *   **Decision**: Utilize SQLite as the primary database for storing raw sales data, processed features, training results, and forecasts.
    *   **Rationale**: SQLite is a lightweight, file-based database, ideal for this project's scope, especially for deployment in Docker containers on potentially resource-constrained environments (e.g., small GCP instances). It eliminates the need for a separate database server, simplifying deployment and management. It's sufficient for the volume of data expected and provides adequate performance for read/write operations.
    *   **Impact**: Simplifies local development, Dockerization, and deployment. Ensures data persistence across container restarts (when coupled with Docker volumes).

2.  **Modular `src/data/database.py`**:
    *   **Decision**: Encapsulate all database interaction logic within `src/data/database.py` with clear `get`, `add`, `update` methods for different data entities.
    *   **Rationale**: Promotes a clean separation of concerns, making the database layer easily maintainable and testable. It abstracts away the SQLite specifics from other parts of the application.
    *   **Impact**: Improves code organization, reduces coupling, and facilitates future changes to the database backend if needed.

3.  **Docker Volumes for Persistence**:
    *   **Decision**: Map Docker volumes to `database/`, `models/`, `mlruns/`, and `logs/` directories.
    *   **Rationale**: Ensures that critical application data (databases, trained models, MLflow artifacts, Optuna studies, logs) persists even if the Docker container is removed or recreated. This is crucial for maintaining state and avoiding data loss.
    *   **Impact**: Enables robust deployment where application state is preserved, and allows for easy backup/restoration of data.

### B. Machine Learning Pipeline

```mermaid
graph TD
    A[Raw CSV Data] --> B{Data Ingestion & Preprocessing};
    B --> C{Feature Engineering};
    C --> D{Exploratory Data Analysis (EDA)};
    D --> E[Split Data (Train/Test)];
    E --> F{Model Training & Hyperparameter Optimization};
    F --> G[MLflow Experiment Tracking];
    F --> H[Optuna Study Management];
    F --> I[Trained Models (.pkl)];
    G --> J[MLflow UI];
    H --> K[Optuna Dashboard];
    I --> L{Model Evaluation & Selection};
    L --> M[Best Model];
    M --> N{Forecasting & Serving};
    N --> O[FastAPI Frontend];

    subgraph ML Workflow
        B; C; D; E; F; G; H; I; L; M; N;
    end

    style A fill:#ECEFF1,stroke:#607D8B,stroke-width:2px,color:#212121;
    style B fill:#E3F2FD,stroke:#2196F3,stroke-width:2px,color:#212121;
    style C fill:#E3F2FD,stroke:#2196F3,stroke-width:2px,color:#212121;
    style D fill:#E3F2FD,stroke:#2196F3,stroke-width:2px,color:#212121;
    style E fill:#E3F2FD,stroke:#2196F3,stroke-width:2px,color:#212121;
    style F fill:#FFF3E0,stroke:#FF9800,stroke-width:2px,color:#212121;
    style G fill:#FCE4EC,stroke:#E91E63,stroke-width:2px,color:#212121;
    style H fill:#E8F5E9,stroke:#4CAF50,stroke-width:2px,color:#212121;
    style I fill:#DCEDC8,stroke:#8BC34A,stroke-width:2px,color:#212121;
    style L fill:#BBDEFB,stroke:#42A5F5,stroke-width:2px,color:#212121;
    style M fill:#C8E6C9,stroke:#66BB6A,stroke-width:2px,color:#212121;
    style N fill:#F0F4C3,stroke:#C0CA33,stroke-width:2px,color:#212121;
    style O fill:#FFEBEE,stroke:#EF5350,stroke-width:2px,color:#212121;
```

**Figure 1: Detailed ML Model Workflow**

This diagram illustrates the comprehensive flow of data and processes within the machine learning pipeline, from raw data ingestion to serving forecasts through the frontend.

1.  **Raw CSV Data**: The starting point of the pipeline, where raw sales data is provided in CSV format.
2.  **Data Ingestion & Preprocessing**: This module (`src/data/loader.py`, `src/data/validator.py`, `src/data/cleaner.py`) handles loading the raw data, validating its format, and performing initial cleaning steps.
3.  **Feature Engineering**: The cleaned data is then passed to the feature engineering module (`src/features/engineer.py`) to create a rich set of time-series specific features (lags, rolling statistics, date-based features).
4.  **Exploratory Data Analysis (EDA)**: An optional but recommended step (`step_2_eda_analysis.py`, `src/eda/analyzer.py`, `src/eda/visualizer.py`) to understand data characteristics, trends, and seasonality before modeling.
5.  **Split Data (Train/Test)**: The engineered features are split into training and testing sets chronologically to simulate real-world forecasting scenarios and prevent data leakage (`src/training/data_splitter.py`).
6.  **Model Training & Hyperparameter Optimization**: This core module (`step_3_train_models.py`, `src/training/trainer.py`, `src/models/models.py`) trains multiple forecasting models and uses Optuna for automated hyperparameter tuning.
7.  **MLflow Experiment Tracking**: During training, all experiment details (parameters, metrics, models, artifacts) are logged to MLflow for tracking and comparison.
8.  **Optuna Study Management**: Optuna manages the hyperparameter optimization studies, storing trial results and metadata.
9.  **Trained Models (.pkl)**: The best-performing models from training are serialized and saved as `.pkl` files.
10. **MLflow UI**: Provides a web-based interface for visualizing and comparing MLflow experiments.
11. **Optuna Dashboard**: Offers a web-based interface for visualizing Optuna studies, hyperparameter importance, and optimization history.
12. **Model Evaluation & Selection**: Models are evaluated based on defined metrics (MAPE, MAE, RMSE) on the test set, and the best model is identified.
13. **Best Model**: The selected best model is then used for production forecasting.
14. **Forecasting & Serving**: The `src/forecasting/forecaster.py` module generates multi-day forecasts using the best model, which are then served via the FastAPI backend.
15. **FastAPI Frontend**: The interactive web application consumes the forecasts and historical data from the FastAPI backend, displaying them to the user.

---

### B.1. Training Pipeline Architecture

```mermaid
graph TD
    A[Start Training Pipeline (API Request)] --> B{PipelineManager};
    B --> C{Log Stream (WebSocket)};
    B --> D{Execute Step 1: Preprocessing & Feature Engineering};
    D --> E{Log to File (logs/)} & F{Update DB (sales_data.db)};
    D --> G{Check for Cancellation};
    G -- No --> H{Execute Step 2: EDA};
    H --> I{Log to File (logs/)} & J{Save Reports (reports/eda/)};
    H --> K{Check for Cancellation};
    K -- No --> L{Execute Step 3: Model Training & Optuna};
    L --> M{Log to File (logs/)} & N{Update DB (results.db)};
    L --> O{MLflow Tracking};
    L --> P{Optuna Studies (optuna_studies.db)};
    L --> Q{Save Models (models/saved/)};
    L --> R{Check for Cancellation};
    R -- No --> S[Pipeline Complete];
    B -- Status Updates --> T[Frontend (Real-time Logs & Progress)];
    T --> C;

    style A fill:#FFFDE7,stroke:#FFC107,stroke-width:2px,color:#212121;
    style B fill:#E0F7FA,stroke:#00BCD4,stroke-width:2px,color:#212121;
    style C fill:#E0F2F7,stroke:#4DD0E1,stroke-width:1px,color:#212121;
    style D fill:#C8E6C9,stroke:#4CAF50,stroke-width:2px,color:#212121;
    style E fill:#F8F8F8,stroke:#BDBDBD,stroke-width:1px,color:#212121;
    style F fill:#F8F8F8,stroke:#BDBDBD,stroke-width:1px,color:#212121;
    style G fill:#FFEBEE,stroke:#F44336,stroke-width:1px,color:#212121;
    style H fill:#E8F5E9,stroke:#8BC34A,stroke-width:2px,color:#212121;
    style I fill:#F8F8F8,stroke:#BDBDBD,stroke-width:1px,color:#212121;
    style J fill:#F8F8F8,stroke:#BDBDBD,stroke-width:1px,color:#212121;
    style K fill:#FFEBEE,stroke:#F44336,stroke-width:1px,color:#212121;
    style L fill:#FFECB3,stroke:#FF9800,stroke-width:2px,color:#212121;
    style M fill:#F8F8F8,stroke:#BDBDBD,stroke-width:1px,color:#212121;
    style N fill:#F8F8F8,stroke:#BDBDBD,stroke-width:1px,color:#212121;
    style O fill:#FCE4EC,stroke:#E91E63,stroke-width:1px,color:#212121;
    style P fill:#E8F5E9,stroke:#4CAF50,stroke-width:1px,color:#212121;
    style Q fill:#DCEDC8,stroke:#8BC34A,stroke-width:1px,color:#212121;
    style R fill:#FFEBEE,stroke:#F44336,stroke-width:1px,color:#212121;
    style S fill:#DCEDC8,stroke:#8BC34A,stroke-width:2px,color:#212121;
    style T fill:#E0F7FA,stroke:#00BCD4,stroke-width:2px,color:#212121;

    linkStyle 0 stroke-width:2px,fill:none,stroke:#00BCD4;
    linkStyle 1 stroke-width:1px,fill:none,stroke:#00BCD4;
    linkStyle 2 stroke-width:1px,fill:none,stroke:#4CAF50;
    linkStyle 3 stroke-width:1px,fill:none,stroke:#4CAF50;
    linkStyle 4 stroke-width:1px,fill:none,stroke:#4CAF50;
    linkStyle 5 stroke-width:1px,fill:none,stroke:#F44336;
    linkStyle 6 stroke-width:1px,fill:none,stroke:#F44336;
    linkStyle 7 stroke-width:1px,fill:none,stroke:#8BC34A;
    linkStyle 8 stroke-width:1px,fill:none,stroke:#8BC34A;
    linkStyle 9 stroke-width:1px,fill:none,stroke:#8BC34A;
    linkStyle 10 stroke-width:1px,fill:none,stroke:#F44336;
    linkStyle 11 stroke-width:1px,fill:none,stroke:#F44336;
    linkStyle 12 stroke-width:1px,fill:none,stroke:#FF9800;
    linkStyle 13 stroke-width:1px,fill:none,stroke:#FF9800;
    linkStyle 14 stroke-width:1px,fill:none,stroke:#FF9800;
    linkStyle 15 stroke-width:1px,fill:none,stroke:#FF9800;
    linkStyle 16 stroke-width:1px,fill:none,stroke:#FF9800;
    linkStyle 17 stroke-width:1px,fill:none,stroke:#FF9800;
    linkStyle 18 stroke-width:1px,fill:none,stroke:#F44336;
    linkStyle 19 stroke-width:1px,fill:none,stroke:#F44336;
    linkStyle 20 stroke-width:2px,fill:none,stroke:#00BCD4;
```

**Figure 2: Real-time Training Pipeline Architecture**

This diagram illustrates the detailed architecture of the real-time model training pipeline, managed by the `PipelineManager` in `src/api/pipeline_manager.py`.

1.  **Start Training Pipeline (API Request)**: The process is initiated by a POST request to `/api/training/start` from the frontend, triggering the `PipelineManager`.
2.  **PipelineManager**: Orchestrates the execution of the various pipeline steps. It manages the state, progress, and cancellation logic for the entire training workflow.
3.  **Log Stream (WebSocket)**: The `PipelineManager` streams real-time terminal logs and status updates to connected WebSocket clients in the frontend, providing live feedback.
4.  **Execute Step 1: Preprocessing & Feature Engineering**: The first stage involves data validation, cleaning, and the creation of time-series features. This step utilizes `step_1_run_pipeline.py`.
5.  **Log to File (logs/) & Update DB (sales_data.db)**: Output from Step 1 is logged to a file in the `logs/` directory, and the processed data is stored in `sales_data.db`.
6.  **Check for Cancellation**: After each major step, the `PipelineManager` checks for a cancellation request from the user. If cancelled, the pipeline stops gracefully.
7.  **Execute Step 2: EDA**: If not cancelled, the Exploratory Data Analysis step (`step_2_eda_analysis.py`) is executed, generating statistical summaries and visualizations.
8.  **Log to File (logs/) & Save Reports (reports/eda/)**: Output from EDA is logged, and generated reports/images are saved.
9.  **Check for Cancellation**: Another cancellation check is performed.
10. **Execute Step 3: Model Training & Optuna**: The core training phase (`step_3_train_models.py`, `src/training/trainer.py`) is run, training multiple models and utilizing Optuna for hyperparameter optimization.
11. **Log to File (logs/) & Update DB (results.db)**: Training logs are saved, and model performance metrics, parameters, and other results are stored in `results.db`.
12. **MLflow Tracking**: All training runs are tracked by MLflow, logging essential experiment details.
13. **Optuna Studies (optuna_studies.db)**: Optuna manages its studies and stores results in its dedicated SQLite database.
14. **Save Models (models/saved/)**: Trained model artifacts are serialized and saved to the `models/saved/` directory.
15. **Check for Cancellation**: The final cancellation check before marking the pipeline as complete.
16. **Pipeline Complete**: The pipeline successfully finishes, and a completion notification is sent.
17. **Frontend (Real-time Logs & Progress)**: The frontend displays the streamed logs and updates a progress bar, offering the user a transparent view of the training process.

---

1.  **Separation of Pipeline Steps**:
    *   **Decision**: Design the ML pipeline as a series of independent Python scripts (`step_1_run_pipeline.py`, `step_2_eda_analysis.py`, `step_3_train_models.py`, `step_4_forecast.py`).
    *   **Rationale**:
        *   **Modularity**: Each script focuses on a single responsibility (preprocessing, EDA, training, forecasting), making the pipeline easier to understand, develop, debug, and maintain.
        *   **Flexibility**: Allows individual steps to be run independently for development, testing, or specific analyses.
        *   **Orchestration**: Enables the `PipelineManager` to orchestrate these steps in a chained fashion for real-time training, providing granular control and logging.
    *   **Impact**: Enhanced reusability, testability, and clarity of the ML workflow.

2.  **Comprehensive Feature Engineering**:
    *   **Decision**: Implement a rich set of time-series features (lags, rolling statistics, date-based components) in `src/features/engineer.py`.
    *   **Rationale**: Provides robust input for diverse ML models (especially tree-based ones like XGBoost and Random Forest) to capture complex temporal patterns, seasonality, and trends effectively.
    *   **Impact**: Improves model accuracy and generalization capabilities.

3.  **Multiple Forecasting Models**:
    *   **Decision**: Incorporate a diverse range of models: Linear Trend, XGBoost, Random Forest, Prophet, and SARIMA.
    *   **Rationale**: Different models excel at different types of time series patterns. A diverse set increases the likelihood of finding an optimal model for varying data characteristics and provides robustness. It also offers a comparative benchmark.
    *   **Impact**: Higher forecasting accuracy and flexibility in model selection.

4.  **Optuna for Hyperparameter Optimization**:
    *   **Decision**: Use Optuna for automated hyperparameter tuning with pruning.
    *   **Rationale**: Automates the often-tedious process of finding optimal model parameters, leading to better-performing models. Pruning intelligently stops unpromising trials, saving computational resources and time.
    *   **Impact**: Improved model performance, faster optimization, and more efficient resource utilization.

5.  **MLflow for Experiment Tracking**:
    *   **Decision**: Integrate MLflow to log parameters, metrics, models, and artifacts for all training runs.
    *   **Rationale**: Provides a centralized system for tracking and comparing experiments, ensuring reproducibility and facilitating model management. The MLflow UI offers a powerful interface for analysis.
    *   **Impact**: Enhanced reproducibility, better experiment management, and clear insights into model performance across runs.

6.  **Time Series Train/Test Split**:
    *   **Decision**: Employ a chronological 80/20 train/test split. For the final production model, retrain on 100% of the data after hyperparameters are optimized on the split data (default).
    *   **Rationale**: Prevents data leakage by evaluating models only on future, unseen data. Retraining on full data maximizes the information used by the final model, typically leading to better accuracy in production, while the split ensures robust validation during development.
    *   **Impact**: More reliable model evaluation and higher confidence in production forecasts.

### C. Backend API (FastAPI)

1.  **FastAPI for Backend Development**:
    *   **Decision**: Use FastAPI as the web framework for the backend.
    *   **Rationale**: FastAPI offers high performance (comparable to Node.js and Go), automatic interactive API documentation (Swagger UI, ReDoc), and excellent developer experience with Pydantic for data validation. Its asynchronous capabilities (async/await) are well-suited for I/O-bound tasks like database operations and external API calls.
    *   **Impact**: Fast API development, robust data validation, clear API documentation, and high-performance backend.

2.  **WebSocket for Real-time Communication**:
    *   **Decision**: Implement WebSocket endpoints for streaming forecast progress and real-time training logs.
    *   **Rationale**: Provides a persistent, bidirectional communication channel between the frontend and backend, enabling live updates without constant polling. This is crucial for a responsive and interactive user experience, especially during long-running tasks like model training.
    *   **Impact**: Enhanced user experience with real-time feedback and dynamic content updates.

3.  **`src/api/pipeline_manager.py` for Real-time Training Orchestration**:
    *   **Decision**: Create a dedicated `PipelineManager` class to manage the execution of the ML pipeline scripts for real-time training.
    *   **Rationale**: Centralizes the logic for triggering, monitoring, and canceling pipeline steps. It integrates with WebSockets to stream logs and status updates to the frontend, providing transparency during long-running operations.
    *   **Impact**: Enables the "Real-time Model Training" feature, providing a robust and user-friendly mechanism for retraining models.

4.  **Auto-start MLflow/Optuna Dashboards**:
    *   **Decision**: Automatically start MLflow and Optuna dashboards as background processes during FastAPI application startup, controlled by an environment variable.
    *   **Rationale**: Improves user convenience by ensuring these monitoring tools are available as soon as the main application starts, without manual intervention.
    *   **Impact**: Streamlined setup and immediate access to MLOps dashboards.

### D. Frontend (HTML, CSS, JavaScript)

1.  **Standard Web Technologies**:
    *   **Decision**: Build the frontend using HTML, CSS (Bootstrap 5), and JavaScript (jQuery, Plotly.js).
    *   **Rationale**: These are universally supported web standards, ensuring broad compatibility and ease of development. Bootstrap provides a responsive and aesthetically pleasing UI framework, while Plotly.js offers powerful interactive charting capabilities. jQuery simplifies DOM manipulation and AJAX calls.
    *   **Impact**: Robust, interactive, and responsive web interface without the overhead of a complex frontend framework.

2.  **Modular JavaScript (`frontend/js/`)**:
    *   **Decision**: Organize JavaScript logic into separate files for each page (`overview.js`, `forecast.js`, `data.js`, `retraining.js`) and common utilities (`api.js`, `charts.js`, `config.js`).
    *   **Rationale**: Enhances code organization, readability, and maintainability. Avoids a monolithic JavaScript file, making it easier to manage and debug specific page functionalities.
    *   **Impact**: Improved frontend code quality and development efficiency.

3.  **Real-time Training Confirmation**:
    *   **Decision**: Implement a confirmation step with explicit warnings and requirements before enabling data upload for real-time training.
    *   **Rationale**: Guides users on proper data format and resource considerations, preventing common errors and managing expectations, especially regarding computational limits on hosted instances.
    *   **Impact**: Improved user experience, reduced errors, and clear communication of system constraints.

### E. Containerization and Deployment

1.  **Docker for Application Isolation**:
    *   **Decision**: Containerize the entire application using Docker.
    *   **Rationale**: Provides a consistent and isolated environment for the application, bundling all dependencies (Python, packages, system libraries) into a single image. This eliminates "it works on my machine" problems and simplifies deployment across different environments (local, cloud VMs).
    *   **Impact**: Ensures environment consistency, simplifies dependency management, and streamlines deployment.

2.  **Multi-stage Dockerfile**:
    *   **Decision**: Use a multi-stage `Dockerfile` for building the image.
    *   **Rationale**: Separates build-time dependencies from runtime dependencies, resulting in smaller, more secure production images.
    *   **Impact**: Optimized image size, faster deployments, and reduced attack surface.

3.  **`docker-compose.yml` for Orchestration**:
    *   **Decision**: Define application services (FastAPI, MLflow, Optuna) in `docker-compose.yml`.
    *   **Rationale**: Simplifies the process of defining, running, and managing multi-container Docker applications. It allows defining network configurations, environment variables, and volumes in a single file.
    *   **Impact**: Easy local development and deployment of the entire system as a cohesive unit.

4.  **Non-root User in Docker**:
    *   **Decision**: Run the application inside the Docker container as a non-root user.
    *   **Rationale**: A crucial security best practice that minimizes the potential impact of vulnerabilities within the container, reducing the attack surface.
    *   **Impact**: Enhanced security of the deployed application.

5.  **`uv` for Dependency Management in Docker**:
    *   **Decision**: Use `uv` for installing Python dependencies within the Docker image.
    *   **Rationale**: `uv` is a modern, extremely fast Python package installer and resolver. It significantly speeds up Docker image builds compared to `pip`, leading to more efficient CI/CD pipelines.
    *   **Impact**: Faster Docker builds and more efficient resource usage during image creation.

### F. Logging and Monitoring

1.  **Centralized Logging (`src/utils/logger.py`)**:
    *   **Decision**: Implement a custom logging utility to capture and format application logs, saving them to timestamped files in `logs/`.
    *   **Rationale**: Provides a structured way to monitor application behavior, debug issues, and track pipeline execution over time.
    *   **Impact**: Improved observability and easier troubleshooting.

2.  **Terminal Log Streaming (WebSockets)**:
    *   **Decision**: Stream real-time terminal output from backend pipeline execution to the frontend via WebSockets.
    *   **Rationale**: Gives users immediate feedback on the progress and status of long-running operations like model training, enhancing transparency and user experience.
    *   **Impact**: Interactive user feedback and reduced perceived waiting times.

This comprehensive architectural overview and rationale for design decisions ensure a robust, maintainable, and efficient Sales Forecasting System.
