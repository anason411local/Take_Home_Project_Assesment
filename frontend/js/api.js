/**
 * API Client for Sales Forecasting Dashboard
 * Handles REST API calls and WebSocket connections
 */

class APIClient {
    constructor() {
        this.baseUrl = CONFIG.API_BASE_URL;
        this.wsUrl = CONFIG.WS_BASE_URL;
        this.sessionId = null;
        this.websocket = null;
        this.wsCallbacks = {};
    }
    
    /**
     * Initialize session
     */
    async initSession() {
        // Check for existing session
        const storedSession = localStorage.getItem(CONFIG.SESSION_KEY);
        if (storedSession) {
            try {
                // Verify session is still valid
                const response = await this.get(`/api/session/${storedSession}`);
                if (response.success) {
                    this.sessionId = storedSession;
                    this.updateSessionDisplay();
                    return this.sessionId;
                }
            } catch (e) {
                // Session invalid, create new one
                localStorage.removeItem(CONFIG.SESSION_KEY);
            }
        }
        
        // Create new session
        try {
            const response = await this.post('/api/session');
            if (response.success) {
                this.sessionId = response.session_id;
                localStorage.setItem(CONFIG.SESSION_KEY, this.sessionId);
                this.updateSessionDisplay();
                return this.sessionId;
            }
        } catch (e) {
            console.error('Failed to create session:', e);
        }
        
        return null;
    }
    
    /**
     * Update session display in navbar
     */
    updateSessionDisplay() {
        const sessionInfo = document.getElementById('sessionInfo');
        if (sessionInfo && this.sessionId) {
            sessionInfo.textContent = `Session: ${this.sessionId.substring(0, 8)}...`;
        }
    }
    
    /**
     * Make GET request
     */
    async get(endpoint, params = {}) {
        const url = new URL(this.baseUrl + endpoint);
        Object.keys(params).forEach(key => url.searchParams.append(key, params[key]));
        
        const response = await fetch(url, {
            method: 'GET',
            headers: {
                'Content-Type': 'application/json'
            }
        });
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.message || `HTTP ${response.status}`);
        }
        
        return response.json();
    }
    
    /**
     * Make POST request
     */
    async post(endpoint, data = {}) {
        const response = await fetch(this.baseUrl + endpoint, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(data)
        });
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.message || `HTTP ${response.status}`);
        }
        
        return response.json();
    }
    
    /**
     * Upload file
     */
    async uploadFile(endpoint, file, onProgress = null) {
        return new Promise((resolve, reject) => {
            const formData = new FormData();
            formData.append('file', file);
            
            const xhr = new XMLHttpRequest();
            
            if (onProgress) {
                xhr.upload.addEventListener('progress', (e) => {
                    if (e.lengthComputable) {
                        const percent = Math.round((e.loaded / e.total) * 100);
                        onProgress(percent);
                    }
                });
            }
            
            xhr.addEventListener('load', () => {
                if (xhr.status >= 200 && xhr.status < 300) {
                    resolve(JSON.parse(xhr.responseText));
                } else {
                    try {
                        const error = JSON.parse(xhr.responseText);
                        reject(new Error(error.message || `HTTP ${xhr.status}`));
                    } catch {
                        reject(new Error(`HTTP ${xhr.status}`));
                    }
                }
            });
            
            xhr.addEventListener('error', () => {
                reject(new Error('Network error'));
            });
            
            xhr.open('POST', this.baseUrl + endpoint);
            xhr.send(formData);
        });
    }
    
    // ==================== API ENDPOINTS ====================
    
    /**
     * Check API health
     */
    async checkHealth() {
        return this.get('/health');
    }
    
    /**
     * Get historical data
     */
    async getHistoricalData(limit = null) {
        const params = {};
        if (limit && limit !== 'all') {
            params.limit = limit;
        }
        const response = await this.get('/api/historical', params);
        
        // Sort data by date to ensure chronological order
        if (response.success && response.data) {
            response.data.sort((a, b) => new Date(a.date) - new Date(b.date));
        }
        
        return response;
    }
    
    /**
     * Get model metrics
     */
    async getMetrics() {
        return this.get('/api/metrics');
    }
    
    /**
     * Get feature importance
     */
    async getFeatureImportance(model = 'xgboost') {
        return this.get('/api/feature-importance', { model });
    }
    
    /**
     * Get available models
     */
    async getModels() {
        return this.get('/api/models');
    }
    
    /**
     * Generate forecast (REST) with confidence intervals
     */
    async generateForecast(horizon, model = 'best', includeCI = true) {
        return this.post('/api/forecast', { 
            horizon, 
            model,
            include_ci: includeCI
        });
    }
    
    /**
     * Upload CSV data
     */
    async uploadData(file, onProgress = null) {
        return this.uploadFile('/api/upload', file, onProgress);
    }
    
    // ==================== WEBSOCKET ====================
    
    /**
     * Connect to WebSocket
     */
    async connectWebSocket() {
        if (!this.sessionId) {
            await this.initSession();
        }
        
        if (!this.sessionId) {
            throw new Error('No session available');
        }
        
        return new Promise((resolve, reject) => {
            const wsUrl = `${this.wsUrl}/ws/${this.sessionId}`;
            this.websocket = new WebSocket(wsUrl);
            
            this.websocket.onopen = () => {
                console.log('WebSocket connected');
                this.updateConnectionStatus(true);
                resolve(this.websocket);
            };
            
            this.websocket.onclose = () => {
                console.log('WebSocket disconnected');
                this.updateConnectionStatus(false);
                this.websocket = null;
            };
            
            this.websocket.onerror = (error) => {
                console.error('WebSocket error:', error);
                this.updateConnectionStatus(false);
                reject(error);
            };
            
            this.websocket.onmessage = (event) => {
                this.handleWebSocketMessage(event);
            };
        });
    }
    
    /**
     * Handle WebSocket message
     */
    handleWebSocketMessage(event) {
        try {
            const data = JSON.parse(event.data);
            const type = data.type;
            
            // Call registered callbacks
            if (this.wsCallbacks[type]) {
                this.wsCallbacks[type].forEach(callback => callback(data));
            }
            
            // Call 'all' callbacks
            if (this.wsCallbacks['all']) {
                this.wsCallbacks['all'].forEach(callback => callback(data));
            }
        } catch (e) {
            console.error('Failed to parse WebSocket message:', e);
        }
    }
    
    /**
     * Register WebSocket callback
     */
    onWebSocketMessage(type, callback) {
        if (!this.wsCallbacks[type]) {
            this.wsCallbacks[type] = [];
        }
        this.wsCallbacks[type].push(callback);
    }
    
    /**
     * Send WebSocket message
     */
    sendWebSocketMessage(action, payload = {}) {
        if (this.websocket && this.websocket.readyState === WebSocket.OPEN) {
            this.websocket.send(JSON.stringify({ action, payload }));
        } else {
            throw new Error('WebSocket not connected');
        }
    }
    
    /**
     * Request forecast via WebSocket
     */
    requestForecastWS(horizon, model = 'best') {
        this.sendWebSocketMessage('forecast', { horizon, model });
    }
    
    /**
     * Send heartbeat
     */
    sendHeartbeat() {
        this.sendWebSocketMessage('heartbeat');
    }
    
    /**
     * Update connection status display
     */
    updateConnectionStatus(connected) {
        const statusBadge = document.getElementById('connectionStatus');
        if (statusBadge) {
            if (connected) {
                statusBadge.className = 'badge bg-success me-2';
                statusBadge.innerHTML = '<i class="bi bi-circle-fill me-1"></i> Connected';
            } else {
                statusBadge.className = 'badge bg-danger me-2';
                statusBadge.innerHTML = '<i class="bi bi-circle-fill me-1"></i> Disconnected';
            }
        }
    }
    
    /**
     * Disconnect WebSocket
     */
    disconnectWebSocket() {
        if (this.websocket) {
            this.websocket.close();
            this.websocket = null;
        }
    }
    
    /**
     * Check if WebSocket is connected
     */
    isWebSocketConnected() {
        return this.websocket && this.websocket.readyState === WebSocket.OPEN;
    }
}

// Create global API client instance
window.api = new APIClient();


// ==================== DASHBOARD FUNCTIONS ====================

/**
 * Open MLflow Dashboard in a new tab
 */
async function openMLflowDashboard() {
    const btn = document.getElementById('btnMLflow');
    const originalText = btn.innerHTML;
    
    try {
        // Show loading state
        btn.disabled = true;
        btn.innerHTML = '<span class="spinner-border spinner-border-sm me-1"></span>Starting...';
        
        // Start MLflow dashboard
        const response = await api.post('/api/dashboards/mlflow/start');
        
        if (response.success && response.url) {
            // Open in new tab
            const newWindow = window.open(response.url, '_blank');
            
            // Check if popup was blocked
            if (!newWindow || newWindow.closed || typeof newWindow.closed === 'undefined') {
                Utils.showError('Popup blocked! Please allow popups for this site to open MLflow UI. URL: ' + response.url);
            } else {
                Utils.showSuccess('MLflow UI opened in new tab');
            }
        } else {
            Utils.showError(response.message || 'Failed to start MLflow UI');
        }
    } catch (error) {
        console.error('Failed to open MLflow dashboard:', error);
        Utils.showError('Failed to start MLflow UI: ' + error.message);
    } finally {
        // Restore button
        btn.disabled = false;
        btn.innerHTML = originalText;
    }
}

/**
 * Open Optuna Dashboard in a new tab
 */
async function openOptunaDashboard() {
    const btn = document.getElementById('btnOptuna');
    const originalText = btn.innerHTML;
    
    try {
        // Show loading state
        btn.disabled = true;
        btn.innerHTML = '<span class="spinner-border spinner-border-sm me-1"></span>Starting...';
        
        // Start Optuna dashboard
        const response = await api.post('/api/dashboards/optuna/start');
        
        if (response.success && response.url) {
            // Open in new tab
            const newWindow = window.open(response.url, '_blank');
            
            // Check if popup was blocked
            if (!newWindow || newWindow.closed || typeof newWindow.closed === 'undefined') {
                Utils.showError('Popup blocked! Please allow popups for this site to open Optuna Dashboard. URL: ' + response.url);
            } else {
                Utils.showSuccess('Optuna Dashboard opened in new tab');
            }
        } else {
            Utils.showError(response.message || 'Failed to start Optuna Dashboard');
        }
    } catch (error) {
        console.error('Failed to open Optuna dashboard:', error);
        Utils.showError('Failed to start Optuna Dashboard: ' + error.message);
    } finally {
        // Restore button
        btn.disabled = false;
        btn.innerHTML = originalText;
    }
}

/**
 * Check dashboard status and update button states
 */
async function checkDashboardStatus() {
    try {
        const response = await api.get('/api/dashboards/status');
        
        if (response.success) {
            // Update MLflow button
            const mlflowBtn = document.getElementById('btnMLflow');
            if (mlflowBtn) {
                if (response.mlflow.running) {
                    mlflowBtn.classList.remove('btn-outline-primary');
                    mlflowBtn.classList.add('btn-primary');
                    mlflowBtn.title = 'MLflow UI is running - Click to open';
                } else {
                    mlflowBtn.classList.remove('btn-primary');
                    mlflowBtn.classList.add('btn-outline-primary');
                    mlflowBtn.title = 'Click to start MLflow UI';
                }
            }
            
            // Update Optuna button
            const optunaBtn = document.getElementById('btnOptuna');
            if (optunaBtn) {
                if (response.optuna.running) {
                    optunaBtn.classList.remove('btn-outline-warning');
                    optunaBtn.classList.add('btn-warning');
                    optunaBtn.title = 'Optuna Dashboard is running - Click to open';
                } else {
                    optunaBtn.classList.remove('btn-warning');
                    optunaBtn.classList.add('btn-outline-warning');
                    optunaBtn.title = 'Click to start Optuna Dashboard';
                }
            }
        }
    } catch (error) {
        console.log('Could not check dashboard status:', error.message);
    }
}

// Export dashboard functions globally
window.openMLflowDashboard = openMLflowDashboard;
window.openOptunaDashboard = openOptunaDashboard;
window.checkDashboardStatus = checkDashboardStatus;

// Check dashboard status on page load
$(document).ready(function() {
    checkDashboardStatus();
});

