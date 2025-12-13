/**
 * Forecast Page JavaScript
 * Handles forecast generation with WebSocket streaming
 */

// Global state
let historicalData = [];
let currentForecast = null;
let wsConnected = false;

/**
 * Initialize the forecast page
 */
async function initForecast() {
    try {
        // Initialize session
        await api.initSession();
        
        // Load historical data for context
        await loadHistoricalData();
        
        // Setup event listeners
        setupEventListeners();
        
        // Setup WebSocket callbacks
        setupWebSocketCallbacks();
        
        // Update timestamp
        Utils.updateTimestamp();
        
    } catch (error) {
        console.error('Failed to initialize forecast page:', error);
        Utils.showError('Failed to initialize. Please refresh the page.');
    }
}

/**
 * Load historical data
 */
async function loadHistoricalData() {
    try {
        const response = await api.getHistoricalData();
        if (response.success) {
            historicalData = response.data;
        }
    } catch (error) {
        console.error('Failed to load historical data:', error);
    }
}

/**
 * Setup event listeners
 */
function setupEventListeners() {
    // Forecast form submission
    const form = document.getElementById('forecastForm');
    if (form) {
        form.addEventListener('submit', handleForecastSubmit);
    }
    
    // Download buttons
    const downloadCSV = document.getElementById('downloadCSV');
    if (downloadCSV) {
        downloadCSV.addEventListener('click', handleDownloadCSV);
    }
    
    const downloadPNG = document.getElementById('downloadPNG');
    if (downloadPNG) {
        downloadPNG.addEventListener('click', handleDownloadPNG);
    }
    
    // Search in predictions table
    const searchInput = document.getElementById('predictionsSearch');
    if (searchInput) {
        searchInput.addEventListener('input', Utils.debounce(handlePredictionsSearch, 300));
    }
}

/**
 * Setup WebSocket callbacks
 */
function setupWebSocketCallbacks() {
    // Progress updates
    api.onWebSocketMessage('forecast_progress', (data) => {
        updateProgress(data.data.progress, data.data.current_step);
    });
    
    // Forecast start
    api.onWebSocketMessage('forecast_start', (data) => {
        showProgressSection();
        updateProgress(0, 'Starting forecast generation...');
    });
    
    // Forecast complete
    api.onWebSocketMessage('forecast_complete', (data) => {
        hideProgressSection();
        displayForecastResults(data.data);
        Utils.showSuccess('Forecast generated successfully!');
    });
    
    // Forecast error
    api.onWebSocketMessage('forecast_error', (data) => {
        hideProgressSection();
        enableForm();
        Utils.showError(`Forecast failed: ${data.data.message}`);
    });
    
    // Connection status
    api.onWebSocketMessage('connected', () => {
        wsConnected = true;
        api.updateConnectionStatus(true);
    });
}

/**
 * Handle forecast form submission
 */
async function handleForecastSubmit(e) {
    e.preventDefault();
    
    const horizon = parseInt(document.getElementById('horizonInput').value);
    const model = document.getElementById('modelSelect').value;
    const useWebSocket = document.getElementById('useWebSocket').checked;
    
    // Validate
    if (horizon < 1 || horizon > 365) {
        Utils.showError('Horizon must be between 1 and 365 days');
        return;
    }
    
    // Disable form
    disableForm();
    
    try {
        if (useWebSocket) {
            // Use WebSocket for streaming
            await generateForecastWS(horizon, model);
        } else {
            // Use REST API
            await generateForecastREST(horizon, model);
        }
    } catch (error) {
        console.error('Forecast error:', error);
        Utils.showError(`Failed to generate forecast: ${error.message}`);
        enableForm();
        hideProgressSection();
    }
}

/**
 * Generate forecast via WebSocket
 */
async function generateForecastWS(horizon, model) {
    // Connect WebSocket if not connected
    if (!api.isWebSocketConnected()) {
        showProgressSection();
        updateProgress(0, 'Connecting to server...');
        await api.connectWebSocket();
    }
    
    // Request forecast
    showProgressSection();
    updateProgress(5, 'Sending forecast request...');
    api.requestForecastWS(horizon, model);
}

/**
 * Generate forecast via REST API
 */
async function generateForecastREST(horizon, model) {
    showProgressSection();
    updateProgress(10, 'Sending request...');
    
    // Simulate progress for REST call
    const progressInterval = setInterval(() => {
        const currentProgress = parseInt(document.getElementById('progressBar').style.width);
        if (currentProgress < 80) {
            updateProgress(currentProgress + 10, 'Generating predictions...');
        }
    }, 500);
    
    try {
        const response = await api.generateForecast(horizon, model);
        
        clearInterval(progressInterval);
        updateProgress(100, 'Complete!');
        
        setTimeout(() => {
            hideProgressSection();
            displayForecastResults({
                predictions: response.predictions,
                model_used: response.model_used,
                summary: response.summary
            });
            Utils.showSuccess('Forecast generated successfully!');
        }, 500);
        
    } catch (error) {
        clearInterval(progressInterval);
        throw error;
    }
}

/**
 * Display forecast results
 */
function displayForecastResults(data) {
    currentForecast = data;
    
    // Show results section
    document.getElementById('resultsSection').style.display = 'flex';
    document.getElementById('chartActions').style.display = 'flex';
    
    // Create forecast chart
    createForecastChart(data.predictions);
    
    // Update summary
    updateSummary(data);
    
    // Populate predictions table
    populatePredictionsTable(data.predictions, data.summary.mean);
    
    // Enable form
    enableForm();
    
    // Update timestamp
    Utils.updateTimestamp();
}

/**
 * Create forecast chart
 */
function createForecastChart(predictions) {
    const container = document.getElementById('forecastChart');
    if (!container) return;
    
    container.innerHTML = '';
    
    Charts.createForecastChart('forecastChart', historicalData, predictions, {
        historicalDays: Math.min(60, historicalData.length)
    });
}

/**
 * Update summary section with CI information
 */
function updateSummary(data) {
    document.getElementById('summaryModel').textContent = Utils.getModelName(data.model_used);
    
    if (data.predictions.length > 0) {
        const startDate = data.predictions[0].date;
        const endDate = data.predictions[data.predictions.length - 1].date;
        document.getElementById('summaryPeriod').textContent = 
            `${Utils.formatDate(startDate)} - ${Utils.formatDate(endDate)}`;
    }
    
    document.getElementById('summaryTotal').textContent = Utils.formatCurrency(data.summary.total);
    document.getElementById('summaryMean').textContent = Utils.formatCurrency(data.summary.mean);
    document.getElementById('summaryMin').textContent = Utils.formatCurrency(data.summary.min);
    document.getElementById('summaryMax').textContent = Utils.formatCurrency(data.summary.max);
    
    // Update CI info if available
    const ciInfoEl = document.getElementById('summaryCIInfo');
    if (ciInfoEl) {
        if (data.ci_method && data.ci_method !== 'none') {
            const ciMethodName = data.ci_method === 'native' ? 'Native (SD-based)' : 
                                 data.ci_method === 'mad' ? 'MAD-based (robust)' :
                                 data.ci_method === 'ensemble' ? 'Ensemble weighted' : data.ci_method;
            const ciWidth = data.summary.ci_mean_width 
                ? Utils.formatCurrency(data.summary.ci_mean_width) 
                : 'N/A';
            ciInfoEl.innerHTML = `
                <span class="badge bg-info me-2">95% CI</span>
                <small class="text-muted">Method: ${ciMethodName} | Avg Width: ${ciWidth}</small>
            `;
            ciInfoEl.style.display = 'block';
        } else {
            ciInfoEl.style.display = 'none';
        }
    }
}

/**
 * Populate predictions table with confidence intervals
 */
function populatePredictionsTable(predictions, avgValue) {
    const tbody = document.getElementById('predictionsBody');
    if (!tbody) return;
    
    if (predictions.length === 0) {
        tbody.innerHTML = `
            <tr>
                <td colspan="6" class="text-center text-muted py-4">
                    No predictions available
                </td>
            </tr>
        `;
        return;
    }
    
    // Check if CI data is available
    const hasCI = predictions[0].lower_bound !== undefined && 
                  predictions[0].upper_bound !== undefined;
    
    const rows = predictions.map((p, i) => {
        const percentOfAvg = ((p.value / avgValue) * 100).toFixed(1);
        const percentClass = p.value >= avgValue ? 'text-success' : 'text-danger';
        
        // Format CI range if available
        let ciRange = '-';
        if (hasCI && p.lower_bound !== null && p.upper_bound !== null) {
            ciRange = `${Utils.formatCurrency(p.lower_bound)} - ${Utils.formatCurrency(p.upper_bound)}`;
        }
        
        return `
            <tr>
                <td>${i + 1}</td>
                <td>${Utils.formatDate(p.date)}</td>
                <td>${Utils.getDayName(p.date)}</td>
                <td><strong>${Utils.formatCurrency(p.value)}</strong></td>
                <td class="text-muted small">${ciRange}</td>
                <td class="${percentClass}">${percentOfAvg}%</td>
            </tr>
        `;
    }).join('');
    
    tbody.innerHTML = rows;
    
    // Update table header if CI is available
    updateTableHeader(hasCI);
}

/**
 * Update table header to show CI column
 */
function updateTableHeader(hasCI) {
    const thead = document.querySelector('#predictionsTable thead tr');
    if (!thead) return;
    
    // Check if we need to add/update CI header
    const existingHeaders = thead.querySelectorAll('th');
    const expectedHeaders = hasCI ? 6 : 5;
    
    if (existingHeaders.length !== expectedHeaders) {
        thead.innerHTML = `
            <th>#</th>
            <th>Date</th>
            <th>Day</th>
            <th>Predicted Sales</th>
            ${hasCI ? '<th>95% CI Range</th>' : ''}
            <th>vs Avg</th>
        `;
    }
}

/**
 * Handle predictions search
 */
function handlePredictionsSearch(e) {
    const searchTerm = e.target.value.toLowerCase();
    const rows = document.querySelectorAll('#predictionsBody tr');
    
    rows.forEach(row => {
        const text = row.textContent.toLowerCase();
        row.style.display = text.includes(searchTerm) ? '' : 'none';
    });
}

/**
 * Handle CSV download
 */
function handleDownloadCSV() {
    if (!currentForecast || !currentForecast.predictions) {
        Utils.showError('No forecast data to download');
        return;
    }
    
    // Create CSV content
    const headers = ['Date', 'Day', 'Predicted Sales'];
    const rows = currentForecast.predictions.map(p => [
        p.date,
        Utils.getDayName(p.date),
        p.value.toFixed(2)
    ]);
    
    const csvContent = [
        headers.join(','),
        ...rows.map(r => r.join(','))
    ].join('\n');
    
    // Download
    const blob = new Blob([csvContent], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `forecast_${currentForecast.model_used}_${new Date().toISOString().split('T')[0]}.csv`;
    a.click();
    URL.revokeObjectURL(url);
    
    Utils.showSuccess('CSV downloaded successfully!');
}

/**
 * Handle PNG download
 */
function handleDownloadPNG() {
    Charts.downloadAsPNG('forecastChart', `forecast_${currentForecast?.model_used || 'chart'}`);
    Utils.showSuccess('Chart image downloaded!');
}

/**
 * Show progress section
 */
function showProgressSection() {
    const section = document.getElementById('progressSection');
    if (section) {
        section.classList.remove('d-none');
    }
}

/**
 * Hide progress section
 */
function hideProgressSection() {
    const section = document.getElementById('progressSection');
    if (section) {
        section.classList.add('d-none');
    }
}

/**
 * Update progress bar
 */
function updateProgress(percent, text) {
    const bar = document.getElementById('progressBar');
    const progressText = document.getElementById('progressText');
    
    if (bar) {
        bar.style.width = `${percent}%`;
        bar.textContent = `${percent}%`;
    }
    
    if (progressText) {
        progressText.textContent = text;
    }
}

/**
 * Disable form during processing
 */
function disableForm() {
    const btn = document.getElementById('generateBtn');
    const horizon = document.getElementById('horizonInput');
    const model = document.getElementById('modelSelect');
    
    if (btn) {
        btn.disabled = true;
        btn.innerHTML = '<span class="spinner-border spinner-border-sm me-2"></span>Generating...';
    }
    if (horizon) horizon.disabled = true;
    if (model) model.disabled = true;
}

/**
 * Enable form after processing
 */
function enableForm() {
    const btn = document.getElementById('generateBtn');
    const horizon = document.getElementById('horizonInput');
    const model = document.getElementById('modelSelect');
    
    if (btn) {
        btn.disabled = false;
        btn.innerHTML = '<i class="bi bi-lightning-charge me-2"></i>Generate Forecast';
    }
    if (horizon) horizon.disabled = false;
    if (model) model.disabled = false;
}

// Initialize when DOM is ready
$(document).ready(function() {
    initForecast();
});

