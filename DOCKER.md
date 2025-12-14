# 🐳 Docker Deployment Guide

This guide explains how to run the Sales Forecasting System using Docker.

---

## 📋 Table of Contents

- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Option 1: Pull from Docker Hub](#option-1-pull-from-docker-hub-recommended)
- [Option 2: Build Locally](#option-2-build-locally)
- [Accessing the Application](#accessing-the-application)
- [Managing the Container](#managing-the-container)
- [Data Persistence](#data-persistence)
- [Publishing to Docker Hub](#publishing-to-docker-hub)
- [Environment Variables](#environment-variables)
- [Troubleshooting](#troubleshooting)

---

## Prerequisites

- **Docker** (version 20.10 or higher)
- **Docker Compose** (version 2.0 or higher)

Verify installation:
```bash
docker --version
docker-compose --version
```

---

## Quick Start

### Option 1: Pull from Docker Hub (Recommended)

```bash
# 1. Create a docker-compose.yml file with:
cat > docker-compose.yml << 'EOF'
version: '3.8'
services:
  sales-forecast:
    image: YOUR_DOCKERHUB_USERNAME/sales-forecast:latest
    container_name: sales-forecast-app
    restart: unless-stopped
    ports:
      - "8000:8000"
      - "5000:5000"
      - "8080:8080"
    volumes:
      - sales_data:/app/database
      - sales_models:/app/models
      - sales_mlruns:/app/mlruns

volumes:
  sales_data:
  sales_models:
  sales_mlruns:
EOF

# 2. Pull and run
docker-compose pull
docker-compose up -d

# 3. Open browser
# http://localhost:8000/app/
```

### Option 2: Build Locally

```bash
# 1. Clone repository
git clone https://github.com/YOUR_USERNAME/sales-forecasting-system.git
cd sales-forecasting-system

# 2. Build and run
docker-compose up --build

# 3. Open browser
# http://localhost:8000/app/
```

---

## Accessing the Application

Once the container is running, access these URLs:

| Service | URL | Description |
|---------|-----|-------------|
| **Web Dashboard** | http://localhost:8000/app/ | Main application interface |
| **API Docs (Swagger)** | http://localhost:8000/docs | Interactive API documentation |
| **API Docs (ReDoc)** | http://localhost:8000/redoc | Alternative API documentation |
| **MLflow UI** | http://localhost:5000 | Experiment tracking (start from dashboard) |
| **Optuna Dashboard** | http://localhost:8080 | Hyperparameter optimization (start from dashboard) |

---

## Managing the Container

### Start the application
```bash
# Foreground (see logs)
docker-compose up

# Background (detached)
docker-compose up -d
```

### Stop the application
```bash
# Stop containers
docker-compose stop

# Stop and remove containers
docker-compose down

# Stop and remove containers + volumes (resets all data!)
docker-compose down -v
```

### View logs
```bash
# All logs
docker-compose logs

# Follow logs in real-time
docker-compose logs -f

# Last 100 lines
docker-compose logs --tail=100
```

### Check status
```bash
docker-compose ps
```

### Enter container shell
```bash
docker-compose exec sales-forecast bash
```

### Restart the application
```bash
docker-compose restart
```

---

## Data Persistence

Data is stored in Docker volumes and persists across container restarts:

| Volume | Container Path | Description |
|--------|----------------|-------------|
| `sales_forecast_databases` | `/app/database` | SQLite databases |
| `sales_forecast_models` | `/app/models` | Trained ML models |
| `sales_forecast_mlruns` | `/app/mlruns` | MLflow experiment data |
| `sales_forecast_reports` | `/app/reports` | EDA reports and visualizations |
| `sales_forecast_logs` | `/app/logs` | Application logs |
| `sales_forecast_data` | `/app/data` | Raw and processed data |

### Backup volumes
```bash
# Backup databases
docker run --rm -v sales_forecast_databases:/data -v $(pwd):/backup alpine \
    tar cvf /backup/databases_backup.tar /data

# Backup models
docker run --rm -v sales_forecast_models:/data -v $(pwd):/backup alpine \
    tar cvf /backup/models_backup.tar /data
```

### Restore volumes
```bash
# Restore databases
docker run --rm -v sales_forecast_databases:/data -v $(pwd):/backup alpine \
    tar xvf /backup/databases_backup.tar -C /

# Restore models
docker run --rm -v sales_forecast_models:/data -v $(pwd):/backup alpine \
    tar xvf /backup/models_backup.tar -C /
```

### Reset all data
```bash
# This will delete ALL persistent data!
docker-compose down -v
```

---

## Publishing to Docker Hub

### 1. Login to Docker Hub
```bash
docker login
```

### 2. Build the image
```bash
docker build -t sales-forecast:latest .
```

### 3. Tag the image
```bash
# Replace YOUR_DOCKERHUB_USERNAME with your Docker Hub username
docker tag sales-forecast:latest YOUR_DOCKERHUB_USERNAME/sales-forecast:latest
docker tag sales-forecast:latest YOUR_DOCKERHUB_USERNAME/sales-forecast:v2.0.0
```

### 4. Push to Docker Hub
```bash
docker push YOUR_DOCKERHUB_USERNAME/sales-forecast:latest
docker push YOUR_DOCKERHUB_USERNAME/sales-forecast:v2.0.0
```

### 5. Verify on Docker Hub
Visit: https://hub.docker.com/r/YOUR_DOCKERHUB_USERNAME/sales-forecast

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `APP_HOST` | `0.0.0.0` | Host to bind the server |
| `APP_PORT` | `8000` | Port for the FastAPI server |
| `MLFLOW_TRACKING_URI` | `file:///app/mlruns` | MLflow tracking location |
| `OPTUNA_DB_PATH` | `/app/models/optuna/optuna_studies.db` | Optuna database path |
| `TZ` | `UTC` | Timezone |

### Custom environment variables
```yaml
# In docker-compose.yml
services:
  sales-forecast:
    environment:
      - APP_PORT=9000
      - TZ=America/New_York
```

---

## Troubleshooting

### Container won't start
```bash
# Check logs for errors
docker-compose logs sales-forecast

# Check container status
docker-compose ps
```

### Port already in use
```bash
# Change the port mapping in docker-compose.yml
ports:
  - "9000:8000"  # Map to port 9000 instead
```

### Out of memory
```bash
# Increase memory limit in docker-compose.yml
deploy:
  resources:
    limits:
      memory: 8G
```

### Permission denied errors
```bash
# Fix permissions on volumes
docker-compose down
docker volume rm sales_forecast_databases
docker-compose up
```

### Health check failing
```bash
# Check if the application started
docker-compose logs sales-forecast | grep -i error

# Test manually
curl http://localhost:8000/health
```

### Clear Docker cache and rebuild
```bash
# Full rebuild without cache
docker-compose build --no-cache
docker-compose up
```

---

## Resource Requirements

| Resource | Minimum | Recommended |
|----------|---------|-------------|
| **CPU** | 1 core | 2-4 cores |
| **Memory** | 2 GB | 4-8 GB |
| **Disk** | 5 GB | 10+ GB |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Docker Container                         │
│  ┌────────────────────────────────────────────────────────┐ │
│  │                    FastAPI (Port 8000)                  │ │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐ │ │
│  │  │  REST API   │  │  WebSocket  │  │ Static Files    │ │ │
│  │  │  Endpoints  │  │  Streaming  │  │ (Frontend)      │ │ │
│  │  └─────────────┘  └─────────────┘  └─────────────────┘ │ │
│  └────────────────────────────────────────────────────────┘ │
│                              │                               │
│  ┌────────────────────────────────────────────────────────┐ │
│  │                ML Pipeline & Services                   │ │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌────────┐ │ │
│  │  │ Training │  │Forecaster│  │  MLflow  │  │ Optuna │ │ │
│  │  │ Pipeline │  │          │  │ (5000)   │  │ (8080) │ │ │
│  │  └──────────┘  └──────────┘  └──────────┘  └────────┘ │ │
│  └────────────────────────────────────────────────────────┘ │
│                              │                               │
│  ┌────────────────────────────────────────────────────────┐ │
│  │              Persistent Storage (Volumes)               │ │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌────────┐ │ │
│  │  │ Databases│  │  Models  │  │  MLruns  │  │  Logs  │ │ │
│  │  │ (SQLite) │  │  (.pkl)  │  │          │  │        │ │ │
│  │  └──────────┘  └──────────┘  └──────────┘  └────────┘ │ │
│  └────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

---

## Support

If you encounter issues:

1. Check the [Troubleshooting](#troubleshooting) section
2. View container logs: `docker-compose logs -f`
3. Open an issue on GitHub

---

<p align="center">
  <sub>Built with ❤️ using Docker, FastAPI, and Python</sub>
</p>

