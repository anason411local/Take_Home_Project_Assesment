/**
 * Configuration for Sales Forecasting Dashboard
 * Light Theme Version
 */

const CONFIG = {
    // API Configuration
    API_BASE_URL: 'http://127.0.0.1:8000',
    WS_BASE_URL: 'ws://127.0.0.1:8000',
    
    // Session Configuration
    SESSION_KEY: 'sales_forecast_session_id',
    
    // Chart Configuration - Light Theme Colors
    CHART_COLORS: {
        primary: '#4f46e5',
        primaryLight: '#818cf8',
        success: '#10b981',
        warning: '#f59e0b',
        danger: '#ef4444',
        info: '#0ea5e9',
        secondary: '#64748b',
        background: '#ffffff',
        gridColor: '#e2e8f0',
        textColor: '#1e293b',
        textSecondary: '#64748b',
        // Chart specific
        historical: '#4f46e5',
        forecast: '#f59e0b'
    },
    
    // Plotly Layout Defaults - Light Theme
    PLOTLY_LAYOUT: {
        paper_bgcolor: 'rgba(255,255,255,0)',
        plot_bgcolor: 'rgba(255,255,255,0)',
        font: {
            family: 'Outfit, sans-serif',
            color: '#1e293b'
        },
        margin: { l: 60, r: 30, t: 30, b: 50 },
        xaxis: {
            gridcolor: '#e2e8f0',
            linecolor: '#e2e8f0',
            tickfont: { color: '#64748b' },
            zerolinecolor: '#e2e8f0'
        },
        yaxis: {
            gridcolor: '#e2e8f0',
            linecolor: '#e2e8f0',
            tickfont: { color: '#64748b' },
            zerolinecolor: '#e2e8f0'
        },
        legend: {
            font: { color: '#1e293b' },
            bgcolor: 'rgba(255,255,255,0.8)'
        },
        hoverlabel: {
            bgcolor: '#ffffff',
            bordercolor: '#4f46e5',
            font: { color: '#1e293b' }
        }
    },
    
    // Plotly Config
    PLOTLY_CONFIG: {
        responsive: true,
        displayModeBar: true,
        modeBarButtonsToRemove: ['lasso2d', 'select2d'],
        displaylogo: false
    },
    
    // Model Display Names
    MODEL_NAMES: {
        'linear_trend': 'Linear Trend',
        'xgboost': 'XGBoost',
        'random_forest': 'Random Forest',
        'prophet': 'Prophet',
        'sarima': 'SARIMA',
        'ensemble': 'Ensemble',
        'best': 'Best Model'
    },
    
    // Refresh Intervals (ms)
    REFRESH_INTERVALS: {
        health: 30000,
        metrics: 60000
    }
};

// Utility Functions
const Utils = {
    /**
     * Format currency value
     */
    formatCurrency(value, decimals = 0) {
        return new Intl.NumberFormat('en-US', {
            style: 'currency',
            currency: 'USD',
            minimumFractionDigits: decimals,
            maximumFractionDigits: decimals
        }).format(value);
    },
    
    /**
     * Format number with commas
     */
    formatNumber(value, decimals = 0) {
        return new Intl.NumberFormat('en-US', {
            minimumFractionDigits: decimals,
            maximumFractionDigits: decimals
        }).format(value);
    },
    
    /**
     * Format percentage
     */
    formatPercent(value, decimals = 2) {
        return `${value.toFixed(decimals)}%`;
    },
    
    /**
     * Format date
     */
    formatDate(dateStr) {
        const date = new Date(dateStr);
        return date.toLocaleDateString('en-US', {
            year: 'numeric',
            month: 'short',
            day: 'numeric'
        });
    },
    
    /**
     * Get day name from date
     */
    getDayName(dateStr) {
        const date = new Date(dateStr);
        return date.toLocaleDateString('en-US', { weekday: 'short' });
    },
    
    /**
     * Show error toast
     */
    showError(message) {
        const toast = document.getElementById('errorToast');
        const toastMessage = document.getElementById('errorToastMessage');
        if (toast && toastMessage) {
            toastMessage.textContent = message;
            const bsToast = new bootstrap.Toast(toast);
            bsToast.show();
        }
        console.error(message);
    },
    
    /**
     * Show success toast
     */
    showSuccess(message) {
        const toast = document.getElementById('successToast');
        const toastMessage = document.getElementById('successToastMessage');
        if (toast && toastMessage) {
            toastMessage.textContent = message;
            const bsToast = new bootstrap.Toast(toast);
            bsToast.show();
        }
    },
    
    /**
     * Update last updated timestamp
     */
    updateTimestamp() {
        const element = document.getElementById('lastUpdated');
        if (element) {
            const now = new Date();
            element.textContent = `Last updated: ${now.toLocaleTimeString()}`;
        }
    },
    
    /**
     * Get model display name
     */
    getModelName(key) {
        return CONFIG.MODEL_NAMES[key] || key;
    },
    
    /**
     * Debounce function
     */
    debounce(func, wait) {
        let timeout;
        return function executedFunction(...args) {
            const later = () => {
                clearTimeout(timeout);
                func(...args);
            };
            clearTimeout(timeout);
            timeout = setTimeout(later, wait);
        };
    },
    
    /**
     * Get next day after a date
     */
    getNextDay(dateStr) {
        const date = new Date(dateStr);
        date.setDate(date.getDate() + 1);
        return date.toISOString().split('T')[0];
    },
    
    /**
     * Format date to YYYY-MM-DD
     */
    formatDateISO(dateStr) {
        const date = new Date(dateStr);
        return date.toISOString().split('T')[0];
    }
};

// Export for use in other files
window.CONFIG = CONFIG;
window.Utils = Utils;
