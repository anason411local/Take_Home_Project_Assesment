/**
 * Overview Page JavaScript
 * Handles dashboard initialization and data loading
 */

// Global state
let historicalData = [];
let metricsData = null;

/**
 * Initialize the overview page
 */
async function initOverview() {
    try {
        // Initialize session
        await api.initSession();
        
        // Check API health
        await checkAPIHealth();
        
        // Load all data in parallel
        await Promise.all([
            loadHistoricalData(),
            loadMetricsData(),
            loadQuickForecast()
        ]);
        
        // Setup event listeners
        setupEventListeners();
        
        // Update timestamp
        Utils.updateTimestamp();
        
    } catch (error) {
        console.error('Failed to initialize overview:', error);
        Utils.showError('Failed to load dashboard data. Please refresh the page.');
    }
}

/**
 * Check API health and update status
 */
async function checkAPIHealth() {
    try {
        const health = await api.checkHealth();
        
        const indicator = document.getElementById('apiStatusIndicator');
        const text = document.getElementById('apiStatusText');
        
        if (indicator && text) {
            indicator.className = 'status-indicator online';
            text.textContent = `API Online (${health.records} records)`;
        }
        
        return health;
    } catch (error) {
        const indicator = document.getElementById('apiStatusIndicator');
        const text = document.getElementById('apiStatusText');
        
        if (indicator && text) {
            indicator.className = 'status-indicator offline';
            text.textContent = 'API Offline';
        }
        
        throw error;
    }
}

/**
 * Load historical data and update KPIs
 */
async function loadHistoricalData() {
    try {
        const response = await api.getHistoricalData();
        
        if (response.success) {
            historicalData = response.data;
            
            // Update KPIs
            updateKPIs(response);
            
            // Create historical chart
            createHistoricalChart();
        }
    } catch (error) {
        console.error('Failed to load historical data:', error);
        Utils.showError('Failed to load historical data');
    }
}

/**
 * Update KPI cards
 */
function updateKPIs(response) {
    // Total Records
    document.getElementById('kpiTotalRecords').textContent = 
        Utils.formatNumber(response.total_records);
    document.getElementById('kpiDateRange').textContent = 
        `${response.date_range.start} to ${response.date_range.end}`;
    
    // Average Sales
    document.getElementById('kpiAvgSales').textContent = 
        Utils.formatCurrency(response.summary.mean);
    document.getElementById('kpiSalesRange').textContent = 
        `Min: ${Utils.formatCurrency(response.summary.min)} | Max: ${Utils.formatCurrency(response.summary.max)}`;
}

/**
 * Load metrics data
 */
async function loadMetricsData() {
    try {
        metricsData = await api.getMetrics();
        
        if (metricsData.success) {
            // Update best model KPI
            document.getElementById('kpiBestModel').textContent = 
                Utils.getModelName(metricsData.best_model);
            
            // Find best model metrics
            const bestModelMetrics = metricsData.models.find(
                m => m.model_name === metricsData.best_model
            );
            
            if (bestModelMetrics) {
                document.getElementById('kpiBestMape').textContent = 
                    `MAPE: ${Utils.formatPercent(bestModelMetrics.test_mape)}`;
                document.getElementById('kpiBestMape').className = 
                    `kpi-change ${bestModelMetrics.test_mape < 20 ? 'positive' : 'negative'}`;
            }
            
            // Update target status
            document.getElementById('kpiTargetStatus').textContent = 
                metricsData.target_met ? 'Achieved!' : 'Not Met';
            document.getElementById('kpiTargetDetail').textContent = 
                `Best: ${Utils.formatPercent(bestModelMetrics?.test_mape || 0)}`;
            document.getElementById('kpiTargetDetail').className = 
                `kpi-change ${metricsData.target_met ? 'positive' : 'negative'}`;
            
            // Create model performance chart
            createModelPerformanceChart();
            
            // Update available models
            updateAvailableModels();
        }
    } catch (error) {
        console.error('Failed to load metrics:', error);
    }
}

/**
 * Create historical chart
 */
function createHistoricalChart() {
    const container = document.getElementById('historicalChart');
    if (!container || historicalData.length === 0) return;
    
    // Clear loading state
    container.innerHTML = '';
    
    Charts.createHistoricalChart('historicalChart', historicalData);
}

/**
 * Create model performance chart
 */
function createModelPerformanceChart() {
    const container = document.getElementById('modelPerformanceChart');
    if (!container || !metricsData) return;
    
    container.innerHTML = '';
    
    Charts.createModelComparisonChart('modelPerformanceChart', metricsData.models);
}

/**
 * Load quick forecast (7 days)
 */
async function loadQuickForecast() {
    const container = document.getElementById('quickForecastChart');
    if (!container) return;
    
    try {
        // Show loading
        Charts.showLoading('quickForecastChart');
        
        // Generate 7-day forecast
        const response = await api.generateForecast(7, 'best');
        
        if (response.success) {
            // Clear loading
            container.innerHTML = '';
            
            // Create chart with last 30 days of historical + forecast
            const recentHistorical = historicalData.slice(-30);
            Charts.createForecastChart('quickForecastChart', recentHistorical, response.predictions, {
                historicalDays: 30
            });
            
            // Update summary
            updateForecastSummary(response);
        }
    } catch (error) {
        console.error('Failed to load quick forecast:', error);
        Charts.showEmpty('quickForecastChart', 'Failed to generate forecast');
    }
}

/**
 * Update forecast summary
 */
function updateForecastSummary(response) {
    const summaryDiv = document.getElementById('forecastSummary');
    if (!summaryDiv) return;
    
    summaryDiv.innerHTML = `
        <div class="row text-center">
            <div class="col-3">
                <small class="text-muted d-block">Model</small>
                <strong>${Utils.getModelName(response.model_used)}</strong>
            </div>
            <div class="col-3">
                <small class="text-muted d-block">Total</small>
                <strong class="text-success">${Utils.formatCurrency(response.summary.total)}</strong>
            </div>
            <div class="col-3">
                <small class="text-muted d-block">Avg/Day</small>
                <strong>${Utils.formatCurrency(response.summary.mean)}</strong>
            </div>
            <div class="col-3">
                <small class="text-muted d-block">Range</small>
                <strong>${Utils.formatCurrency(response.summary.min)} - ${Utils.formatCurrency(response.summary.max)}</strong>
            </div>
        </div>
    `;
}

/**
 * Update available models display
 */
function updateAvailableModels() {
    const container = document.getElementById('availableModels');
    if (!container || !metricsData) return;
    
    const badges = metricsData.models.map(m => {
        const isBest = m.model_name === metricsData.best_model;
        const badgeClass = isBest ? 'bg-success' : 'bg-secondary';
        const icon = isBest ? '<i class="bi bi-trophy-fill me-1"></i>' : '';
        return `<span class="badge ${badgeClass} me-1">${icon}${Utils.getModelName(m.model_name)}</span>`;
    }).join('');
    
    container.innerHTML = badges;
}

/**
 * Setup event listeners
 */
function setupEventListeners() {
    // Chart period selector
    const chartPeriod = document.getElementById('chartPeriod');
    if (chartPeriod) {
        chartPeriod.addEventListener('change', (e) => {
            filterHistoricalChart(e.target.value);
        });
    }
    
    // Refresh forecast button
    const refreshBtn = document.getElementById('refreshForecast');
    if (refreshBtn) {
        refreshBtn.addEventListener('click', () => {
            loadQuickForecast();
        });
    }
}

/**
 * Filter historical chart by period
 */
function filterHistoricalChart(period) {
    if (historicalData.length === 0) return;
    
    let filteredData = historicalData;
    
    if (period !== 'all') {
        const days = parseInt(period);
        filteredData = historicalData.slice(-days);
    }
    
    Charts.createHistoricalChart('historicalChart', filteredData);
}

// Initialize when DOM is ready
$(document).ready(function() {
    initOverview();
    
    // Set up periodic health check
    setInterval(checkAPIHealth, CONFIG.REFRESH_INTERVALS.health);
});

