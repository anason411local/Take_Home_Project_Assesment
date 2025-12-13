/**
 * Retraining Page JavaScript
 * 
 * Handles:
 * - CSV file upload
 * - Pipeline configuration
 * - WebSocket connection for real-time logs
 * - Progress tracking
 */

// API URLs from config
const API_BASE_URL = CONFIG.API_BASE_URL + '/api';
const WS_BASE_URL = CONFIG.WS_BASE_URL + '/ws';

// State
let selectedFile = null;
let isFileUploaded = false;
let isPipelineRunning = false;
let trainingWebSocket = null;
let logLines = [];
let autoScroll = true;
let isConfirmed = false;

// DOM Elements
const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('fileInput');
const browseBtn = document.getElementById('browseBtn');
const selectedFileDiv = document.getElementById('selectedFile');
const fileName = document.getElementById('fileName');
const fileSize = document.getElementById('fileSize');
const removeFileBtn = document.getElementById('removeFile');
const uploadProgressSection = document.getElementById('uploadProgressSection');
const uploadProgressBar = document.getElementById('uploadProgressBar');
const uploadStatus = document.getElementById('uploadStatus');
const uploadResult = document.getElementById('uploadResult');
const uploadResultText = document.getElementById('uploadResultText');
const startPipelineBtn = document.getElementById('startPipelineBtn');
const cancelPipelineBtn = document.getElementById('cancelPipelineBtn');
const pipelineProgressSection = document.getElementById('pipelineProgressSection');
const pipelineProgressBar = document.getElementById('pipelineProgressBar');
const pipelineStepText = document.getElementById('pipelineStepText');
const pipelineStepNumber = document.getElementById('pipelineStepNumber');
const pipelineStatusBadge = document.getElementById('pipelineStatusBadge');
const terminalOutput = document.getElementById('terminalOutput');
const terminalContainer = document.getElementById('terminalContainer');
const clearLogsBtn = document.getElementById('clearLogsBtn');
const downloadLogsBtn = document.getElementById('downloadLogsBtn');
const autoScrollToggle = document.getElementById('autoScrollToggle');
const wsStatus = document.getElementById('wsStatus');
const wsStatusText = document.getElementById('wsStatusText');
const logCount = document.getElementById('logCount');

// Confirmation elements
const confirmationSection = document.getElementById('confirmationSection');
const mainContentSection = document.getElementById('mainContentSection');
const confirmationInput = document.getElementById('confirmationInput');
const confirmBtn = document.getElementById('confirmBtn');
const confirmationHint = document.getElementById('confirmationHint');

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    initializeConfirmation();
    initializeUpload();
    initializeWebSocket();
    initializeControls();
    checkPipelineStatus();
    
    // Update last updated time
    document.getElementById('lastUpdated').textContent = 
        'Last updated: ' + new Date().toLocaleString();
});

/**
 * Initialize confirmation section
 */
function initializeConfirmation() {
    // Check if already confirmed in session
    if (sessionStorage.getItem('training_confirmed') === 'true') {
        enableUploadSection();
        return;
    }
    
    // Input validation on keyup
    confirmationInput.addEventListener('keyup', (e) => {
        const value = e.target.value.toLowerCase().trim();
        
        if (value === 'confirmed') {
            confirmationInput.classList.remove('invalid');
            confirmationInput.classList.add('valid');
            confirmationHint.innerHTML = '<i class="bi bi-check-circle text-success me-1"></i>Correct! Click Confirm to proceed.';
            confirmationHint.classList.add('text-success');
        } else if (value.length > 0) {
            confirmationInput.classList.remove('valid');
            confirmationInput.classList.add('invalid');
            confirmationHint.innerHTML = '<i class="bi bi-x-circle text-danger me-1"></i>Please type exactly "confirmed"';
            confirmationHint.classList.remove('text-success');
        } else {
            confirmationInput.classList.remove('valid', 'invalid');
            confirmationHint.innerHTML = '<i class="bi bi-info-circle me-1"></i>Type exactly "confirmed" (without quotes) to enable upload';
            confirmationHint.classList.remove('text-success');
        }
        
        // Allow Enter key to confirm
        if (e.key === 'Enter' && value === 'confirmed') {
            handleConfirmation();
        }
    });
    
    // Confirm button click
    confirmBtn.addEventListener('click', handleConfirmation);
}

/**
 * Handle confirmation
 */
function handleConfirmation() {
    const value = confirmationInput.value.toLowerCase().trim();
    
    if (value === 'confirmed') {
        isConfirmed = true;
        sessionStorage.setItem('training_confirmed', 'true');
        enableUploadSection();
        showSuccess('Confirmation accepted! You can now upload your dataset.');
        addLogLine('User confirmed requirements - Upload enabled', 'success');
    } else {
        showError('Please type "confirmed" exactly to proceed');
        confirmationInput.focus();
        confirmationInput.classList.add('invalid');
    }
}

/**
 * Enable upload section after confirmation
 */
function enableUploadSection() {
    isConfirmed = true;
    
    // Animate the transition
    confirmationSection.style.transition = 'all 0.5s ease';
    confirmationSection.style.opacity = '0';
    confirmationSection.style.transform = 'translateY(-20px)';
    
    setTimeout(() => {
        confirmationSection.style.display = 'none';
        
        // Unlock main content
        mainContentSection.classList.remove('upload-locked');
        mainContentSection.style.transition = 'all 0.5s ease';
        mainContentSection.style.opacity = '1';
    }, 500);
}

/**
 * Initialize file upload functionality
 */
function initializeUpload() {
    // Browse button click
    browseBtn.addEventListener('click', () => fileInput.click());
    
    // File input change
    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFileSelect(e.target.files[0]);
        }
    });
    
    // Drag and drop
    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.classList.add('drag-over');
    });
    
    uploadArea.addEventListener('dragleave', () => {
        uploadArea.classList.remove('drag-over');
    });
    
    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.classList.remove('drag-over');
        if (e.dataTransfer.files.length > 0) {
            handleFileSelect(e.dataTransfer.files[0]);
        }
    });
    
    // Remove file button
    removeFileBtn.addEventListener('click', () => {
        resetUpload();
    });
}

/**
 * Handle file selection
 */
function handleFileSelect(file) {
    if (!file.name.endsWith('.csv')) {
        showError('Please select a CSV file');
        return;
    }
    
    selectedFile = file;
    fileName.textContent = file.name;
    fileSize.textContent = formatFileSize(file.size);
    
    uploadArea.classList.add('d-none');
    selectedFileDiv.classList.remove('d-none');
    
    // Auto-upload
    uploadFile();
}

/**
 * Upload selected file
 */
async function uploadFile() {
    if (!selectedFile) return;
    
    uploadProgressSection.classList.remove('d-none');
    uploadProgressBar.style.width = '0%';
    uploadStatus.textContent = 'Uploading...';
    
    try {
        const formData = new FormData();
        formData.append('file', selectedFile);
        
        // Simulate progress
        let progress = 0;
        const progressInterval = setInterval(() => {
            progress = Math.min(progress + 10, 90);
            uploadProgressBar.style.width = progress + '%';
        }, 100);
        
        const response = await fetch(`${API_BASE_URL}/training/upload`, {
            method: 'POST',
            body: formData
        });
        
        clearInterval(progressInterval);
        uploadProgressBar.style.width = '100%';
        
        const data = await response.json();
        
        if (data.success) {
            isFileUploaded = true;
            uploadStatus.textContent = 'Upload complete!';
            uploadResult.classList.remove('d-none');
            uploadResultText.textContent = 
                `Uploaded ${data.rows} rows (${data.date_range.start} to ${data.date_range.end})`;
            
            startPipelineBtn.disabled = false;
            
            addLogLine(`File uploaded: ${selectedFile.name}`, 'success');
            addLogLine(`Rows: ${data.rows}, Columns: ${data.columns.length}`, 'info');
            addLogLine(`Date range: ${data.date_range.start} to ${data.date_range.end}`, 'info');
            addLogLine('Ready to start training pipeline...', 'info');
            
        } else {
            throw new Error(data.error || 'Upload failed');
        }
        
    } catch (error) {
        uploadProgressSection.classList.add('d-none');
        showError('Upload failed: ' + error.message);
        resetUpload();
    }
}

/**
 * Reset upload state
 */
function resetUpload() {
    selectedFile = null;
    isFileUploaded = false;
    fileInput.value = '';
    
    uploadArea.classList.remove('d-none');
    selectedFileDiv.classList.add('d-none');
    uploadProgressSection.classList.add('d-none');
    uploadResult.classList.add('d-none');
    
    startPipelineBtn.disabled = true;
}

/**
 * Initialize WebSocket connection
 */
function initializeWebSocket() {
    try {
        const sessionId = 'training_' + Date.now();
        const wsUrl = `${WS_BASE_URL}/training/${sessionId}`;
        
        console.log('Connecting to WebSocket:', wsUrl);
        
        trainingWebSocket = new WebSocket(wsUrl);
        
        trainingWebSocket.onopen = () => {
            console.log('WebSocket connected');
            wsStatus.classList.remove('text-danger', 'text-warning');
            wsStatus.classList.add('text-success');
            wsStatusText.textContent = 'Connected';
        };
        
        trainingWebSocket.onclose = (event) => {
            console.log('WebSocket closed:', event.code, event.reason);
            wsStatus.classList.remove('text-success', 'text-warning');
            wsStatus.classList.add('text-danger');
            wsStatusText.textContent = 'Disconnected';
            
            // Try to reconnect after 3 seconds
            setTimeout(initializeWebSocket, 3000);
        };
        
        trainingWebSocket.onerror = (error) => {
            console.error('WebSocket error:', error);
            wsStatus.classList.remove('text-success');
            wsStatus.classList.add('text-warning');
            wsStatusText.textContent = 'Connection error';
        };
        
        trainingWebSocket.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                handleWebSocketMessage(data);
            } catch (e) {
                console.error('WebSocket message error:', e);
            }
        };
        
    } catch (error) {
        console.error('Failed to initialize WebSocket:', error);
        wsStatus.classList.remove('text-success');
        wsStatus.classList.add('text-danger');
        wsStatusText.textContent = 'Failed to connect';
    }
}

// Start ping interval separately
setInterval(() => {
    if (trainingWebSocket && trainingWebSocket.readyState === WebSocket.OPEN) {
        trainingWebSocket.send('ping');
    }
}, 25000);

/**
 * Handle WebSocket messages
 */
function handleWebSocketMessage(data) {
    switch (data.type) {
        case 'connected':
            addLogLine('WebSocket connected', 'success');
            if (data.status && data.status.is_running) {
                isPipelineRunning = true;
                updateUIForRunning();
            }
            break;
            
        case 'log':
            addLogLine(data.message, data.log_type || 'info');
            break;
            
        case 'status':
            updatePipelineStatus(data.status);
            break;
            
        case 'complete':
            isPipelineRunning = false;
            updateUIForComplete(data.success);
            if (data.success) {
                addLogLine('', 'info');
                addLogLine('=' .repeat(50), 'success');
                addLogLine('TRAINING PIPELINE COMPLETED SUCCESSFULLY!', 'success');
                addLogLine('=' .repeat(50), 'success');
                showSuccess('Training completed! Models are ready for forecasting.');
            } else {
                addLogLine('Pipeline failed: ' + (data.error || 'Unknown error'), 'error');
                showError('Training failed: ' + (data.error || 'Unknown error'));
            }
            break;
            
        case 'heartbeat':
            // Ignore heartbeat
            break;
    }
}

/**
 * Initialize controls
 */
function initializeControls() {
    // Start pipeline button
    startPipelineBtn.addEventListener('click', startPipeline);
    
    // Cancel button
    cancelPipelineBtn.addEventListener('click', cancelPipeline);
    
    // Clear logs button
    clearLogsBtn.addEventListener('click', () => {
        logLines = [];
        terminalOutput.innerHTML = '';
        logCount.textContent = '0 lines';
    });
    
    // Download logs button
    downloadLogsBtn.addEventListener('click', downloadLogs);
    
    // Auto-scroll toggle
    autoScrollToggle.addEventListener('change', (e) => {
        autoScroll = e.target.checked;
        if (autoScroll) {
            scrollToBottom();
        }
    });
}

/**
 * Start training pipeline
 */
async function startPipeline() {
    if (!isFileUploaded) {
        showError('Please upload a CSV file first');
        return;
    }
    
    // Get configuration
    const models = [];
    if (document.getElementById('modelLinearTrend').checked) models.push('linear_trend');
    if (document.getElementById('modelXgboost').checked) models.push('xgboost');
    if (document.getElementById('modelRandomForest').checked) models.push('random_forest');
    if (document.getElementById('modelProphet').checked) models.push('prophet');
    if (document.getElementById('modelSarima').checked) models.push('sarima');
    
    if (models.length === 0) {
        showError('Please select at least one model');
        return;
    }
    
    const config = {
        optuna_trials: parseInt(document.getElementById('optunaTrials').value),
        models: models.join(','),
        test_size: parseFloat(document.getElementById('testSize').value),
        use_holdout: document.getElementById('modeHoldout').checked,
        skip_eda: document.getElementById('skipEda').checked,
        skip_training: false
    };
    
    try {
        const params = new URLSearchParams(config);
        const response = await fetch(`${API_BASE_URL}/training/start?${params}`, {
            method: 'POST'
        });
        
        const data = await response.json();
        
        if (data.success) {
            isPipelineRunning = true;
            updateUIForRunning();
            addLogLine('Pipeline started...', 'info');
        } else {
            throw new Error(data.detail || 'Failed to start pipeline');
        }
        
    } catch (error) {
        showError('Failed to start pipeline: ' + error.message);
    }
}

/**
 * Cancel training pipeline
 */
async function cancelPipeline() {
    try {
        const response = await fetch(`${API_BASE_URL}/training/cancel`, {
            method: 'POST'
        });
        
        const data = await response.json();
        
        if (data.success) {
            addLogLine('Cancellation requested...', 'warning');
        }
        
    } catch (error) {
        showError('Failed to cancel: ' + error.message);
    }
}

/**
 * Check current pipeline status
 */
async function checkPipelineStatus() {
    try {
        const response = await fetch(`${API_BASE_URL}/training/status`);
        const data = await response.json();
        
        if (data.is_running) {
            isPipelineRunning = true;
            updateUIForRunning();
            updatePipelineStatus(data);
        }
        
        // Load recent logs
        if (data.recent_logs && data.recent_logs.length > 0) {
            data.recent_logs.forEach(log => {
                addLogLine(log, 'info');
            });
        }
        
    } catch (error) {
        console.error('Failed to check status:', error);
    }
}

/**
 * Update pipeline status display
 */
function updatePipelineStatus(status) {
    const progress = status.progress || 0;
    const step = status.current_step_name || 'Initializing...';
    const stepNum = status.current_step_number || 0;
    const totalSteps = status.total_steps || 3;
    
    pipelineProgressBar.style.width = progress + '%';
    pipelineProgressBar.textContent = progress + '%';
    pipelineStepText.textContent = step;
    pipelineStepNumber.textContent = `Step ${stepNum}/${totalSteps}`;
    
    // Update status badge
    let badgeClass = 'bg-secondary';
    let badgeIcon = 'bi-circle';
    let badgeText = 'Ready';
    
    switch (status.step) {
        case 'preprocessing':
            badgeClass = 'bg-primary';
            badgeIcon = 'bi-arrow-clockwise';
            badgeText = 'Preprocessing';
            break;
        case 'eda':
            badgeClass = 'bg-success';
            badgeIcon = 'bi-bar-chart-line';
            badgeText = 'EDA Analysis';
            break;
        case 'training':
            badgeClass = 'bg-warning text-dark';
            badgeIcon = 'bi-cpu';
            badgeText = 'Training Models';
            break;
        case 'completed':
            badgeClass = 'bg-success';
            badgeIcon = 'bi-check-circle';
            badgeText = 'Completed';
            break;
        case 'failed':
            badgeClass = 'bg-danger';
            badgeIcon = 'bi-x-circle';
            badgeText = 'Failed';
            break;
        case 'cancelled':
            badgeClass = 'bg-secondary';
            badgeIcon = 'bi-stop-circle';
            badgeText = 'Cancelled';
            break;
    }
    
    pipelineStatusBadge.innerHTML = `
        <span class="badge ${badgeClass} fs-6">
            <i class="bi ${badgeIcon} me-1"></i>${badgeText}
        </span>
    `;
}

/**
 * Update UI for running state
 */
function updateUIForRunning() {
    startPipelineBtn.disabled = true;
    cancelPipelineBtn.disabled = false;
    pipelineProgressSection.style.display = 'block';
    
    // Disable config form
    document.querySelectorAll('#configForm input, #configForm select').forEach(el => {
        el.disabled = true;
    });
}

/**
 * Update UI for complete state
 */
function updateUIForComplete(success) {
    isPipelineRunning = false;
    startPipelineBtn.disabled = !isFileUploaded;
    cancelPipelineBtn.disabled = true;
    
    // Re-enable config form
    document.querySelectorAll('#configForm input, #configForm select').forEach(el => {
        el.disabled = false;
    });
}

/**
 * Add log line to terminal
 */
function addLogLine(message, logType = 'info') {
    const line = document.createElement('div');
    line.className = `terminal-line terminal-${logType}`;
    
    // Parse timestamp if present
    let timestamp = '';
    let text = message;
    
    const match = message.match(/^\[(\d{2}:\d{2}:\d{2})\]\s*/);
    if (match) {
        timestamp = match[1];
        text = message.substring(match[0].length);
    } else {
        timestamp = new Date().toLocaleTimeString('en-US', { hour12: false });
    }
    
    line.innerHTML = `
        <span class="terminal-timestamp">[${timestamp}]</span>
        <span class="terminal-text">${escapeHtml(text)}</span>
    `;
    
    terminalOutput.appendChild(line);
    logLines.push(message);
    logCount.textContent = logLines.length + ' lines';
    
    if (autoScroll) {
        scrollToBottom();
    }
}

/**
 * Scroll terminal to bottom
 */
function scrollToBottom() {
    terminalContainer.scrollTop = terminalContainer.scrollHeight;
}

/**
 * Download logs as text file
 */
function downloadLogs() {
    const content = logLines.join('\n');
    const blob = new Blob([content], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    
    const a = document.createElement('a');
    a.href = url;
    a.download = `training_logs_${new Date().toISOString().replace(/[:.]/g, '-')}.txt`;
    a.click();
    
    URL.revokeObjectURL(url);
}

/**
 * Format file size
 */
function formatFileSize(bytes) {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
    return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
}

/**
 * Escape HTML
 */
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

/**
 * Show error toast
 */
function showError(message) {
    const toast = document.getElementById('errorToast');
    document.getElementById('errorToastMessage').textContent = message;
    new bootstrap.Toast(toast).show();
}

/**
 * Show success toast
 */
function showSuccess(message) {
    const toast = document.getElementById('successToast');
    document.getElementById('successToastMessage').textContent = message;
    new bootstrap.Toast(toast).show();
}

