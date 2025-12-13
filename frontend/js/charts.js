/**
 * Chart utilities for Sales Forecasting Dashboard
 * Uses Plotly.js for interactive charts
 * Light Theme Version with proper date continuity
 */

const Charts = {
    /**
     * Create time series chart
     */
    createTimeSeriesChart(containerId, data, options = {}) {
        const {
            title = '',
            showLegend = true,
            height = 300
        } = options;
        
        const layout = {
            ...CONFIG.PLOTLY_LAYOUT,
            title: title ? { text: title, font: { size: 14 } } : null,
            showlegend: showLegend,
            height: height,
            xaxis: {
                ...CONFIG.PLOTLY_LAYOUT.xaxis,
                title: 'Date',
                type: 'date'
            },
            yaxis: {
                ...CONFIG.PLOTLY_LAYOUT.yaxis,
                title: 'Sales ($)',
                tickformat: ',.0f',
                tickprefix: '$'
            }
        };
        
        Plotly.newPlot(containerId, data, layout, CONFIG.PLOTLY_CONFIG);
    },
    
    /**
     * Create historical sales chart with optional forecast
     * Ensures proper date continuity
     */
    createHistoricalChart(containerId, historicalData, forecastData = null) {
        const traces = [];
        
        // Sort historical data by date
        const sortedHistorical = [...historicalData].sort((a, b) => 
            new Date(a.date) - new Date(b.date)
        );
        
        // Historical trace
        if (sortedHistorical && sortedHistorical.length > 0) {
            traces.push({
                x: sortedHistorical.map(d => d.date),
                y: sortedHistorical.map(d => d.daily_sales),
                type: 'scatter',
                mode: 'lines',
                name: 'Historical Sales',
                line: {
                    color: CONFIG.CHART_COLORS.historical,
                    width: 2
                },
                hovertemplate: '<b>%{x|%b %d, %Y}</b><br>Sales: $%{y:,.0f}<extra></extra>'
            });
        }
        
        // Forecast trace - connect from last historical point
        if (forecastData && forecastData.length > 0 && sortedHistorical.length > 0) {
            const lastHistorical = sortedHistorical[sortedHistorical.length - 1];
            
            // Start forecast from last historical point for visual continuity
            const forecastX = [lastHistorical.date, ...forecastData.map(d => d.date)];
            const forecastY = [lastHistorical.daily_sales, ...forecastData.map(d => d.value)];
            
            traces.push({
                x: forecastX,
                y: forecastY,
                type: 'scatter',
                mode: 'lines',
                name: 'Forecast',
                line: {
                    color: CONFIG.CHART_COLORS.forecast,
                    width: 2,
                    dash: 'dash'
                },
                hovertemplate: '<b>%{x|%b %d, %Y}</b><br>Forecast: $%{y:,.0f}<extra></extra>'
            });
        }
        
        this.createTimeSeriesChart(containerId, traces, {
            showLegend: true,
            height: 350
        });
    },
    
    /**
     * Create forecast chart with historical context
     * Shows last N days of historical data + forecast with proper continuity
     */
    createForecastChart(containerId, historicalData, forecastData, options = {}) {
        const {
            historicalDays = 30
        } = options;
        
        const traces = [];
        
        // Sort and get last N days of historical data
        const sortedHistorical = [...historicalData].sort((a, b) => 
            new Date(a.date) - new Date(b.date)
        );
        const recentHistorical = sortedHistorical.slice(-historicalDays);
        
        if (recentHistorical.length === 0) {
            this.showEmpty(containerId, 'No historical data available');
            return;
        }
        
        // Get the last historical point
        const lastHistoricalPoint = recentHistorical[recentHistorical.length - 1];
        const lastHistoricalDate = new Date(lastHistoricalPoint.date);
        
        // Historical trace
        traces.push({
            x: recentHistorical.map(d => d.date),
            y: recentHistorical.map(d => d.daily_sales),
            type: 'scatter',
            mode: 'lines+markers',
            name: 'Historical',
            line: {
                color: CONFIG.CHART_COLORS.historical,
                width: 2.5
            },
            marker: {
                size: 5,
                color: CONFIG.CHART_COLORS.historical
            },
            hovertemplate: '<b>%{x|%b %d, %Y}</b><br>Actual: $%{y:,.0f}<extra></extra>'
        });
        
        // Forecast trace - ensure continuity from last historical point
        if (forecastData && forecastData.length > 0) {
            // Build forecast line starting from last historical point
            const forecastX = [lastHistoricalPoint.date];
            const forecastY = [lastHistoricalPoint.daily_sales];
            
            // Add all forecast points
            forecastData.forEach(d => {
                forecastX.push(d.date);
                forecastY.push(d.value);
            });
            
            traces.push({
                x: forecastX,
                y: forecastY,
                type: 'scatter',
                mode: 'lines+markers',
                name: 'Forecast',
                line: {
                    color: CONFIG.CHART_COLORS.forecast,
                    width: 3,
                    dash: 'dash'
                },
                marker: {
                    size: 7,
                    color: CONFIG.CHART_COLORS.forecast,
                    symbol: 'diamond'
                },
                hovertemplate: '<b>%{x|%b %d, %Y}</b><br>Forecast: $%{y:,.0f}<extra></extra>'
            });
            
            // Add vertical line at forecast start
            var shapes = [{
                type: 'line',
                x0: lastHistoricalPoint.date,
                x1: lastHistoricalPoint.date,
                y0: 0,
                y1: 1,
                yref: 'paper',
                line: {
                    color: CONFIG.CHART_COLORS.secondary,
                    width: 1.5,
                    dash: 'dot'
                }
            }];
            
            // Add annotation
            var annotations = [{
                x: lastHistoricalPoint.date,
                y: 1.02,
                yref: 'paper',
                text: 'Forecast Start',
                showarrow: false,
                font: { 
                    color: CONFIG.CHART_COLORS.secondary, 
                    size: 10 
                }
            }];
        } else {
            var shapes = [];
            var annotations = [];
        }
        
        const layout = {
            ...CONFIG.PLOTLY_LAYOUT,
            height: 400,
            showlegend: true,
            legend: {
                ...CONFIG.PLOTLY_LAYOUT.legend,
                orientation: 'h',
                y: -0.15,
                x: 0.5,
                xanchor: 'center'
            },
            xaxis: {
                ...CONFIG.PLOTLY_LAYOUT.xaxis,
                title: 'Date',
                type: 'date',
                tickformat: '%b %d<br>%Y'
            },
            yaxis: {
                ...CONFIG.PLOTLY_LAYOUT.yaxis,
                title: 'Sales ($)',
                tickformat: ',.0f',
                tickprefix: '$'
            },
            shapes: shapes,
            annotations: annotations
        };
        
        Plotly.newPlot(containerId, traces, layout, CONFIG.PLOTLY_CONFIG);
    },
    
    /**
     * Create bar chart for model comparison
     */
    createModelComparisonChart(containerId, metricsData, metric = 'test_mape') {
        const models = metricsData.map(m => Utils.getModelName(m.model_name));
        const values = metricsData.map(m => m[metric] || 0);
        
        // Sort by value
        const sorted = models.map((model, i) => ({ model, value: values[i] }))
            .sort((a, b) => a.value - b.value);
        
        const colors = sorted.map((_, i) => 
            i === 0 ? CONFIG.CHART_COLORS.success : CONFIG.CHART_COLORS.primary
        );
        
        const trace = {
            x: sorted.map(d => d.model),
            y: sorted.map(d => d.value),
            type: 'bar',
            marker: {
                color: colors,
                line: {
                    color: colors,
                    width: 1
                }
            },
            hovertemplate: '<b>%{x}</b><br>MAPE: %{y:.2f}%<extra></extra>'
        };
        
        const layout = {
            ...CONFIG.PLOTLY_LAYOUT,
            height: 300,
            showlegend: false,
            xaxis: {
                ...CONFIG.PLOTLY_LAYOUT.xaxis,
                title: ''
            },
            yaxis: {
                ...CONFIG.PLOTLY_LAYOUT.yaxis,
                title: 'MAPE (%)',
                ticksuffix: '%'
            },
            shapes: [{
                type: 'line',
                x0: -0.5,
                x1: sorted.length - 0.5,
                y0: 20,
                y1: 20,
                line: {
                    color: CONFIG.CHART_COLORS.danger,
                    width: 2,
                    dash: 'dash'
                }
            }],
            annotations: [{
                x: sorted.length - 1,
                y: 20,
                text: 'Target: 20%',
                showarrow: false,
                font: { color: CONFIG.CHART_COLORS.danger, size: 10 },
                yshift: 12
            }]
        };
        
        Plotly.newPlot(containerId, [trace], layout, CONFIG.PLOTLY_CONFIG);
    },
    
    /**
     * Create horizontal bar chart for feature importance
     */
    createFeatureImportanceChart(containerId, features) {
        // Sort by importance and take top 15
        const sorted = [...features]
            .sort((a, b) => b.importance - a.importance)
            .slice(0, 15)
            .reverse();
        
        const trace = {
            y: sorted.map(f => f.feature),
            x: sorted.map(f => f.importance),
            type: 'bar',
            orientation: 'h',
            marker: {
                color: sorted.map((_, i) => {
                    const ratio = i / sorted.length;
                    return `rgba(79, 70, 229, ${0.4 + ratio * 0.6})`;
                })
            },
            hovertemplate: '<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>'
        };
        
        const layout = {
            ...CONFIG.PLOTLY_LAYOUT,
            height: 400,
            showlegend: false,
            xaxis: {
                ...CONFIG.PLOTLY_LAYOUT.xaxis,
                title: 'Importance Score'
            },
            yaxis: {
                ...CONFIG.PLOTLY_LAYOUT.yaxis,
                title: '',
                tickfont: { size: 11 }
            },
            margin: { l: 120, r: 30, t: 20, b: 50 }
        };
        
        Plotly.newPlot(containerId, [trace], layout, CONFIG.PLOTLY_CONFIG);
    },
    
    /**
     * Create distribution histogram
     */
    createDistributionChart(containerId, data, field = 'daily_sales') {
        const values = data.map(d => d[field]);
        
        const trace = {
            x: values,
            type: 'histogram',
            nbinsx: 30,
            marker: {
                color: CONFIG.CHART_COLORS.primary,
                line: {
                    color: CONFIG.CHART_COLORS.primaryLight,
                    width: 1
                }
            },
            hovertemplate: 'Range: $%{x:,.0f}<br>Count: %{y}<extra></extra>'
        };
        
        const layout = {
            ...CONFIG.PLOTLY_LAYOUT,
            height: 300,
            showlegend: false,
            xaxis: {
                ...CONFIG.PLOTLY_LAYOUT.xaxis,
                title: 'Sales ($)',
                tickformat: ',.0f',
                tickprefix: '$'
            },
            yaxis: {
                ...CONFIG.PLOTLY_LAYOUT.yaxis,
                title: 'Frequency'
            }
        };
        
        Plotly.newPlot(containerId, [trace], layout, CONFIG.PLOTLY_CONFIG);
    },
    
    /**
     * Create monthly aggregation chart
     */
    createMonthlyChart(containerId, data) {
        // Aggregate by month
        const monthlyData = {};
        data.forEach(d => {
            const date = new Date(d.date);
            const monthKey = `${date.getFullYear()}-${String(date.getMonth() + 1).padStart(2, '0')}`;
            if (!monthlyData[monthKey]) {
                monthlyData[monthKey] = { total: 0, count: 0 };
            }
            monthlyData[monthKey].total += d.daily_sales;
            monthlyData[monthKey].count += 1;
        });
        
        const months = Object.keys(monthlyData).sort();
        const averages = months.map(m => monthlyData[m].total / monthlyData[m].count);
        
        const trace = {
            x: months,
            y: averages,
            type: 'bar',
            marker: {
                color: CONFIG.CHART_COLORS.info
            },
            hovertemplate: '<b>%{x}</b><br>Avg: $%{y:,.0f}<extra></extra>'
        };
        
        const layout = {
            ...CONFIG.PLOTLY_LAYOUT,
            height: 300,
            showlegend: false,
            xaxis: {
                ...CONFIG.PLOTLY_LAYOUT.xaxis,
                title: 'Month',
                tickangle: -45
            },
            yaxis: {
                ...CONFIG.PLOTLY_LAYOUT.yaxis,
                title: 'Avg Daily Sales ($)',
                tickformat: ',.0f',
                tickprefix: '$'
            }
        };
        
        Plotly.newPlot(containerId, [trace], layout, CONFIG.PLOTLY_CONFIG);
    },
    
    /**
     * Download chart as PNG
     */
    downloadAsPNG(containerId, filename = 'chart.png') {
        Plotly.downloadImage(containerId, {
            format: 'png',
            width: 1200,
            height: 600,
            filename: filename.replace('.png', '')
        });
    },
    
    /**
     * Clear chart
     */
    clearChart(containerId) {
        const container = document.getElementById(containerId);
        if (container) {
            Plotly.purge(containerId);
            container.innerHTML = '';
        }
    },
    
    /**
     * Show loading state
     */
    showLoading(containerId) {
        const container = document.getElementById(containerId);
        if (container) {
            container.innerHTML = `
                <div class="chart-loading">
                    <div class="spinner-border text-primary" role="status">
                        <span class="visually-hidden">Loading...</span>
                    </div>
                    <p class="mt-2">Loading chart data...</p>
                </div>
            `;
        }
    },
    
    /**
     * Show empty state
     */
    showEmpty(containerId, message = 'No data available') {
        const container = document.getElementById(containerId);
        if (container) {
            container.innerHTML = `
                <div class="empty-state">
                    <i class="bi bi-bar-chart-line"></i>
                    <h5>No Data</h5>
                    <p>${message}</p>
                </div>
            `;
        }
    }
};

// Export for use in other files
window.Charts = Charts;
