# =============================================================================
# Sales Forecasting System - Docker Image
# =============================================================================
# 
# This Dockerfile creates a production-ready image for the Sales Forecasting
# application with FastAPI backend and interactive web frontend.
#
# Build:
#   docker build -t sales-forecast:latest .
#
# Run:
#   docker run -p 8000:8000 sales-forecast:latest
#
# Push to Docker Hub:
#   docker tag sales-forecast:latest YOUR_USERNAME/sales-forecast:latest
#   docker push YOUR_USERNAME/sales-forecast:latest
#
# =============================================================================

# Stage 1: Builder - Install dependencies
FROM python:3.12.9-slim as builder

# Set build-time arguments
ARG DEBIAN_FRONTEND=noninteractive

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements and install dependencies
# Use the optimized docker requirements for smaller image
COPY requirements-docker.txt /tmp/requirements.txt

# Install Python dependencies using uv for faster installation
# Step 1: Upgrade pip
RUN pip install --no-cache-dir --upgrade pip

# Step 2: Install uv package manager
RUN pip install --no-cache-dir uv

# Step 3: Install requirements using uv into the virtual environment
# Note: Remove --system flag to install into /opt/venv (the active venv)
RUN uv pip install --no-cache-dir -r /tmp/requirements.txt

# =============================================================================
# Stage 2: Production Image
# =============================================================================
FROM python:3.12.9-slim as production

# Labels for Docker Hub
LABEL maintainer="AI Engineer Assessment Project"
LABEL description="Sales Forecasting System with ML Pipeline, FastAPI, and Web Dashboard"
LABEL version="2.0.0"

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    APP_HOST=0.0.0.0 \
    APP_PORT=8000 \
    MLFLOW_TRACKING_URI=file:///app/mlruns \
    OPTUNA_DB_PATH=/app/models/optuna/optuna_studies.db

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Create non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Create application directory
WORKDIR /app

# Create necessary directories
RUN mkdir -p \
    /app/data/raw \
    /app/data/processed \
    /app/data/uploads \
    /app/database \
    /app/models/saved \
    /app/models/optuna \
    /app/models/feature_importance \
    /app/models/learning_curves \
    /app/reports/eda \
    /app/logs \
    /app/mlruns \
    /app/forecasts \
    && chown -R appuser:appuser /app

# Copy application code
COPY --chown=appuser:appuser . /app

# Copy entrypoint script
COPY --chown=appuser:appuser docker-entrypoint.sh /app/docker-entrypoint.sh
RUN chmod +x /app/docker-entrypoint.sh

# Switch to non-root user
USER appuser

# Expose ports
# 8000 - FastAPI application
# 5000 - MLflow UI (optional, started via API)
# 8080 - Optuna Dashboard (optional, started via API)
EXPOSE 8000 5000 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command
ENTRYPOINT ["/app/docker-entrypoint.sh"]
CMD ["--host", "0.0.0.0", "--port", "8000", "--no-browser"]

