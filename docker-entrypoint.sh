#!/bin/bash
# =============================================================================
# Docker Entrypoint Script for Sales Forecasting System
# =============================================================================
#
# This script initializes the application and starts the FastAPI server.
# It checks for required data and provides helpful messages.
#
# =============================================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Banner
echo ""
echo "=============================================================="
echo "  ____        _             _____                          _   "
echo " / ___|  __ _| | ___  ___  |  ___|__  _ __ ___  ___ __ _ ___| |_ "
echo " \___ \ / _\` | |/ _ \/ __| | |_ / _ \| '__/ _ \/ __/ _\` / __| __|"
echo "  ___) | (_| | |  __/\__ \ |  _| (_) | | |  __/ (_| (_| \__ \ |_ "
echo " |____/ \__,_|_|\___||___/ |_|  \___/|_|  \___|\___\__,_|___/\__|"
echo ""
echo "=============================================================="
echo ""

# Print environment info
echo -e "${BLUE}[INFO]${NC} Container Environment:"
echo "  - Python: $(python --version)"
echo "  - Working Directory: $(pwd)"
echo "  - Host: ${APP_HOST:-0.0.0.0}"
echo "  - Port: ${APP_PORT:-8000}"
echo ""

# Check for databases
echo -e "${BLUE}[INFO]${NC} Checking databases..."

if [ -f "/app/database/sales_data.db" ]; then
    echo -e "${GREEN}[OK]${NC} Found sales_data.db"
else
    echo -e "${YELLOW}[WARN]${NC} sales_data.db not found - will be created on first data upload"
fi

if [ -f "/app/database/results.db" ]; then
    echo -e "${GREEN}[OK]${NC} Found results.db"
else
    echo -e "${YELLOW}[WARN]${NC} results.db not found - will be created on first training"
fi

# Check for models
echo -e "${BLUE}[INFO]${NC} Checking trained models..."

MODEL_COUNT=$(find /app/models/saved -name "*.pkl" 2>/dev/null | wc -l)
if [ "$MODEL_COUNT" -gt 0 ]; then
    echo -e "${GREEN}[OK]${NC} Found $MODEL_COUNT trained model(s):"
    find /app/models/saved -name "*.pkl" 2>/dev/null | while read -r file; do
        echo "     - $(basename "$file")"
    done
else
    echo -e "${YELLOW}[WARN]${NC} No trained models found"
    echo "     Run the training pipeline from the web interface to train models"
fi

# Check for sample data
echo -e "${BLUE}[INFO]${NC} Checking sample data..."

if [ -f "/app/ecommerce_sales_data (1).csv" ]; then
    echo -e "${GREEN}[OK]${NC} Sample data available"
else
    echo -e "${YELLOW}[INFO]${NC} No sample data found"
fi

echo ""
echo "=============================================================="
echo -e "${BLUE}[INFO]${NC} Starting Sales Forecasting Application..."
echo "=============================================================="
echo ""
echo "  WEB INTERFACE:"
echo "    Dashboard:    http://localhost:${APP_PORT:-8000}/app/"
echo "    Forecast:     http://localhost:${APP_PORT:-8000}/app/forecast.html"
echo "    Data View:    http://localhost:${APP_PORT:-8000}/app/data.html"
echo "    Retrain:      http://localhost:${APP_PORT:-8000}/app/retraining.html"
echo ""
echo "  API DOCUMENTATION:"
echo "    Swagger UI:   http://localhost:${APP_PORT:-8000}/docs"
echo "    ReDoc:        http://localhost:${APP_PORT:-8000}/redoc"
echo ""
echo "  MONITORING (start from dashboard):"
echo "    MLflow UI:    http://localhost:5000"
echo "    Optuna:       http://localhost:8080"
echo ""
echo "=============================================================="
echo ""

# Execute the main application
exec python app.py "$@"

