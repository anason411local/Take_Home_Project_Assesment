/**
 * Data View Page JavaScript
 * Handles historical data display, metrics, feature importance, and upload
 */

// Global state
let historicalData = [];
let metricsData = null;
let featureData = null;
let currentPage = 1;
let pageSize = 100;

/**
 * Initialize the data view page
 */
async function initDataView() {
    try {
        // Initialize session
        await api.initSession();
        
        // Load initial data
        await loadHistoricalData();
        
        // Setup event listeners
        setupEventListeners();
        
        // Update timestamp
        Utils.updateTimestamp();
        
    } catch (error) {
        console.error('Failed to initialize data view:', error);
        Utils.showError('Failed to initialize. Please refresh the page.');
    }
}

/**
 * Setup event listeners
 */
function setupEventListeners() {
    // Tab change events
    const tabEls = document.querySelectorAll('button[data-bs-toggle="tab"]');
    tabEls.forEach(tab => {
        tab.addEventListener('shown.bs.tab', handleTabChange);
    });
    
    // Data search
    const dataSearch = document.getElementById('dataSearch');
    if (dataSearch) {
        dataSearch.addEventListener('input', Utils.debounce(handleDataSearch, 300));
    }
    
    // Data limit
    const dataLimit = document.getElementById('dataLimit');
    if (dataLimit) {
        dataLimit.addEventListener('change', handleDataLimitChange);
    }
    
    // Export button
    const exportBtn = document.getElementById('exportDataBtn');
    if (exportBtn) {
        exportBtn.addEventListener('click', handleExportData);
    }
    
    // Feature model selector
    const featureModelInputs = document.querySelectorAll('input[name="featureModel"]');
    featureModelInputs.forEach(input => {
        input.addEventListener('change', handleFeatureModelChange);
    });
    
    // Upload form
    setupUploadHandlers();
}

/**
 * Handle tab change
 */
async function handleTabChange(e) {
    const targetId = e.target.getAttribute('data-bs-target').replace('#', '');
    
    switch (targetId) {
        case 'historical':
            // Already loaded
            break;
        case 'metrics':
            if (!metricsData) {
                await loadMetricsData();
            }
            break;
        case 'features':
            if (!featureData) {
                await loadFeatureImportance('xgboost');
            }
            break;
        case 'upload':
            // Nothing to load
            break;
    }
}

// ==================== HISTORICAL DATA ====================

/**
 * Load historical data
 */
async function loadHistoricalData() {
    try {
        const limit = document.getElementById('dataLimit')?.value || '100';
        const response = await api.getHistoricalData(limit);
        
        if (response.success) {
            historicalData = response.data;
            
            // Populate table
            populateHistoricalTable(historicalData);
            
            // Update info
            updateDataInfo(response);
            
            // Create charts
            createDistributionChart();
            createMonthlyChart();
        }
    } catch (error) {
        console.error('Failed to load historical data:', error);
        Utils.showError('Failed to load historical data');
    }
}

/**
 * Populate historical data table
 */
function populateHistoricalTable(data) {
    const tbody = document.getElementById('historicalBody');
    if (!tbody) return;
    
    if (data.length === 0) {
        tbody.innerHTML = `
            <tr>
                <td colspan="5" class="text-center text-muted py-4">
                    No data available
                </td>
            </tr>
        `;
        return;
    }
    
    const rows = data.map((d, i) => `
        <tr>
            <td>${i + 1}</td>
            <td>${Utils.formatDate(d.date)}</td>
            <td>${Utils.formatCurrency(d.daily_sales)}</td>
            <td>${d.marketing_spend ? Utils.formatCurrency(d.marketing_spend) : '-'}</td>
            <td>
                ${d.is_holiday ? 
                    '<span class="badge bg-success">Yes</span>' : 
                    '<span class="badge bg-secondary">No</span>'}
            </td>
        </tr>
    `).join('');
    
    tbody.innerHTML = rows;
}

/**
 * Update data info text
 */
function updateDataInfo(response) {
    const info = document.getElementById('dataInfo');
    if (info) {
        info.textContent = `Showing ${response.data.length} of ${response.total_records} records`;
    }
}

/**
 * Create distribution chart
 */
function createDistributionChart() {
    if (historicalData.length === 0) return;
    
    const container = document.getElementById('distributionChart');
    if (!container) return;
    
    container.innerHTML = '';
    Charts.createDistributionChart('distributionChart', historicalData);
}

/**
 * Create monthly chart
 */
function createMonthlyChart() {
    if (historicalData.length === 0) return;
    
    const container = document.getElementById('monthlyChart');
    if (!container) return;
    
    container.innerHTML = '';
    Charts.createMonthlyChart('monthlyChart', historicalData);
}

/**
 * Handle data search
 */
function handleDataSearch(e) {
    const searchTerm = e.target.value.toLowerCase();
    const rows = document.querySelectorAll('#historicalBody tr');
    
    rows.forEach(row => {
        const text = row.textContent.toLowerCase();
        row.style.display = text.includes(searchTerm) ? '' : 'none';
    });
}

/**
 * Handle data limit change
 */
function handleDataLimitChange() {
    loadHistoricalData();
}

/**
 * Handle export data
 */
function handleExportData() {
    if (historicalData.length === 0) {
        Utils.showError('No data to export');
        return;
    }
    
    // Create CSV content
    const headers = ['Date', 'Daily Sales', 'Marketing Spend', 'Is Holiday'];
    const rows = historicalData.map(d => [
        d.date,
        d.daily_sales,
        d.marketing_spend || '',
        d.is_holiday || 0
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
    a.download = `historical_data_${new Date().toISOString().split('T')[0]}.csv`;
    a.click();
    URL.revokeObjectURL(url);
    
    Utils.showSuccess('Data exported successfully!');
}

// ==================== METRICS ====================

/**
 * Load metrics data
 */
async function loadMetricsData() {
    try {
        metricsData = await api.getMetrics();
        
        if (metricsData.success) {
            // Create comparison chart
            createMetricsComparisonChart();
            
            // Update best model card
            updateBestModelCard();
            
            // Populate metrics table
            populateMetricsTable();
        }
    } catch (error) {
        console.error('Failed to load metrics:', error);
        Utils.showError('Failed to load model metrics');
    }
}

/**
 * Create metrics comparison chart
 */
function createMetricsComparisonChart() {
    const container = document.getElementById('metricsComparisonChart');
    if (!container || !metricsData) return;
    
    container.innerHTML = '';
    Charts.createModelComparisonChart('metricsComparisonChart', metricsData.models);
}

/**
 * Update best model card
 */
function updateBestModelCard() {
    if (!metricsData) return;
    
    const bestModel = metricsData.models.find(m => m.model_name === metricsData.best_model);
    if (!bestModel) return;
    
    document.getElementById('bestModelName').textContent = Utils.getModelName(bestModel.model_name);
    document.getElementById('bestModelMape').textContent = Utils.formatPercent(bestModel.test_mape);
    document.getElementById('bestModelMae').textContent = Utils.formatCurrency(bestModel.test_mae);
    document.getElementById('bestModelRmse').textContent = Utils.formatCurrency(bestModel.test_rmse);
    
    const badge = document.getElementById('targetBadge');
    if (badge) {
        if (metricsData.target_met) {
            badge.className = 'badge bg-success';
            badge.textContent = 'Target Met! (< 20% MAPE)';
        } else {
            badge.className = 'badge bg-danger';
            badge.textContent = 'Target Not Met';
        }
    }
}

/**
 * Populate metrics table
 */
function populateMetricsTable() {
    const tbody = document.getElementById('metricsBody');
    if (!tbody || !metricsData) return;
    
    // Sort by test_mape
    const sortedModels = [...metricsData.models].sort((a, b) => a.test_mape - b.test_mape);
    
    const rows = sortedModels.map(m => {
        const isBest = m.model_name === metricsData.best_model;
        const rowClass = isBest ? 'table-success' : '';
        const statusBadge = m.test_mape < 20 ? 
            '<span class="badge bg-success">Pass</span>' : 
            '<span class="badge bg-danger">Fail</span>';
        
        return `
            <tr class="${rowClass}">
                <td>
                    ${isBest ? '<i class="bi bi-trophy-fill text-warning me-1"></i>' : ''}
                    ${Utils.getModelName(m.model_name)}
                </td>
                <td><strong>${Utils.formatPercent(m.test_mape)}</strong></td>
                <td>${Utils.formatCurrency(m.test_mae)}</td>
                <td>${Utils.formatCurrency(m.test_rmse)}</td>
                <td>${m.train_mape ? Utils.formatPercent(m.train_mape) : '-'}</td>
                <td>${m.training_time ? m.training_time.toFixed(2) + 's' : '-'}</td>
                <td>${statusBadge}</td>
            </tr>
        `;
    }).join('');
    
    tbody.innerHTML = rows;
}

// ==================== FEATURE IMPORTANCE ====================

/**
 * Load feature importance
 */
async function loadFeatureImportance(model) {
    try {
        const container = document.getElementById('featureImportanceChart');
        if (container) {
            Charts.showLoading('featureImportanceChart');
        }
        
        featureData = await api.getFeatureImportance(model);
        
        if (featureData.success) {
            // Create chart
            createFeatureImportanceChart();
            
            // Populate table
            populateFeatureTable();
        }
    } catch (error) {
        console.error('Failed to load feature importance:', error);
        Charts.showEmpty('featureImportanceChart', 'Feature importance not available for this model');
        
        const tbody = document.getElementById('featureBody');
        if (tbody) {
            tbody.innerHTML = `
                <tr>
                    <td colspan="3" class="text-center text-muted py-4">
                        Not available
                    </td>
                </tr>
            `;
        }
    }
}

/**
 * Create feature importance chart
 */
function createFeatureImportanceChart() {
    const container = document.getElementById('featureImportanceChart');
    if (!container || !featureData) return;
    
    container.innerHTML = '';
    Charts.createFeatureImportanceChart('featureImportanceChart', featureData.features);
}

/**
 * Populate feature table
 */
function populateFeatureTable() {
    const tbody = document.getElementById('featureBody');
    if (!tbody || !featureData) return;
    
    const rows = featureData.features.map(f => `
        <tr>
            <td><span class="badge bg-primary">${f.rank}</span></td>
            <td><code>${f.feature}</code></td>
            <td>${f.importance.toFixed(4)}</td>
        </tr>
    `).join('');
    
    tbody.innerHTML = rows;
}

/**
 * Handle feature model change
 */
function handleFeatureModelChange(e) {
    const model = e.target.value;
    featureData = null;
    loadFeatureImportance(model);
}

// ==================== UPLOAD ====================

/**
 * Setup upload handlers
 */
function setupUploadHandlers() {
    const uploadArea = document.getElementById('uploadArea');
    const fileInput = document.getElementById('fileInput');
    const browseBtn = document.getElementById('browseBtn');
    const removeBtn = document.getElementById('removeFile');
    const uploadForm = document.getElementById('uploadForm');
    
    if (!uploadArea || !fileInput) return;
    
    // Click to browse
    browseBtn?.addEventListener('click', () => fileInput.click());
    uploadArea.addEventListener('click', (e) => {
        if (e.target === uploadArea || e.target.closest('.upload-area')) {
            fileInput.click();
        }
    });
    
    // File input change
    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFileSelected(e.target.files[0]);
        }
    });
    
    // Drag and drop
    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.classList.add('dragover');
    });
    
    uploadArea.addEventListener('dragleave', () => {
        uploadArea.classList.remove('dragover');
    });
    
    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.classList.remove('dragover');
        
        if (e.dataTransfer.files.length > 0) {
            handleFileSelected(e.dataTransfer.files[0]);
        }
    });
    
    // Remove file
    removeBtn?.addEventListener('click', handleRemoveFile);
    
    // Form submit
    uploadForm?.addEventListener('submit', handleUploadSubmit);
}

/**
 * Handle file selected
 */
function handleFileSelected(file) {
    // Validate file type
    if (!file.name.endsWith('.csv')) {
        Utils.showError('Please select a CSV file');
        return;
    }
    
    // Validate file size (10MB max)
    if (file.size > 10 * 1024 * 1024) {
        Utils.showError('File size must be less than 10MB');
        return;
    }
    
    // Show file info
    const uploadArea = document.getElementById('uploadArea');
    const selectedFile = document.getElementById('selectedFile');
    const fileName = document.getElementById('fileName');
    const fileSize = document.getElementById('fileSize');
    const uploadBtn = document.getElementById('uploadBtn');
    
    uploadArea?.classList.add('d-none');
    selectedFile?.classList.remove('d-none');
    
    if (fileName) fileName.textContent = file.name;
    if (fileSize) fileSize.textContent = formatFileSize(file.size);
    if (uploadBtn) uploadBtn.disabled = false;
    
    // Store file reference
    document.getElementById('fileInput').file = file;
}

/**
 * Handle remove file
 */
function handleRemoveFile() {
    const uploadArea = document.getElementById('uploadArea');
    const selectedFile = document.getElementById('selectedFile');
    const fileInput = document.getElementById('fileInput');
    const uploadBtn = document.getElementById('uploadBtn');
    
    uploadArea?.classList.remove('d-none');
    selectedFile?.classList.add('d-none');
    
    if (fileInput) {
        fileInput.value = '';
        fileInput.file = null;
    }
    if (uploadBtn) uploadBtn.disabled = true;
    
    // Hide result
    document.getElementById('uploadResult')?.classList.add('d-none');
}

/**
 * Handle upload submit
 */
async function handleUploadSubmit(e) {
    e.preventDefault();
    
    const fileInput = document.getElementById('fileInput');
    const file = fileInput.file || fileInput.files[0];
    
    if (!file) {
        Utils.showError('Please select a file');
        return;
    }
    
    // Show progress
    const uploadProgress = document.getElementById('uploadProgress');
    const uploadBtn = document.getElementById('uploadBtn');
    
    uploadProgress?.classList.remove('d-none');
    if (uploadBtn) {
        uploadBtn.disabled = true;
        uploadBtn.innerHTML = '<span class="spinner-border spinner-border-sm me-2"></span>Uploading...';
    }
    
    try {
        const response = await api.uploadFile('/api/upload', file, (percent) => {
            updateUploadProgress(percent);
        });
        
        // Show success
        showUploadResult(true, response);
        
        // Reload historical data
        await loadHistoricalData();
        
    } catch (error) {
        console.error('Upload failed:', error);
        showUploadResult(false, { message: error.message });
    } finally {
        uploadProgress?.classList.add('d-none');
        if (uploadBtn) {
            uploadBtn.disabled = false;
            uploadBtn.innerHTML = '<i class="bi bi-upload me-2"></i>Upload Data';
        }
    }
}

/**
 * Update upload progress
 */
function updateUploadProgress(percent) {
    const bar = document.getElementById('uploadProgressBar');
    const status = document.getElementById('uploadStatus');
    
    if (bar) {
        bar.style.width = `${percent}%`;
        bar.textContent = `${percent}%`;
    }
    
    if (status) {
        if (percent < 100) {
            status.textContent = 'Uploading...';
        } else {
            status.textContent = 'Processing...';
        }
    }
}

/**
 * Show upload result
 */
function showUploadResult(success, response) {
    const resultDiv = document.getElementById('uploadResult');
    const alert = document.getElementById('uploadAlert');
    const icon = document.getElementById('uploadAlertIcon');
    const message = document.getElementById('uploadAlertMessage');
    
    resultDiv?.classList.remove('d-none');
    
    if (success) {
        alert.className = 'alert alert-success';
        icon.className = 'bi bi-check-circle-fill';
        message.textContent = `Successfully imported ${response.records_imported} records (${response.date_range.start} to ${response.date_range.end})`;
        Utils.showSuccess('Data uploaded successfully!');
    } else {
        alert.className = 'alert alert-danger';
        icon.className = 'bi bi-exclamation-triangle-fill';
        message.textContent = response.message || 'Upload failed';
    }
}

/**
 * Format file size
 */
function formatFileSize(bytes) {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
    return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
}

// Initialize when DOM is ready
$(document).ready(function() {
    initDataView();
});

