// ==============================================================================
// MAIN JAVASCRIPT FOR STOCK ANALYSIS DASHBOARD
// ==============================================================================

// Global variables
let currentTicker = '';
let currentAnalysisType = 'short'; // 'short' | 'long' | 'day'
let priceChart = null;
let rsiChart = null;
let macdChart = null;
let lastAnalysisData = null;

// API base URL
const API_BASE_URL = 'http://localhost:5000/api';

// ==============================================================================
// UTILITY FUNCTIONS
// ==============================================================================

function formatCurrency(value) {
    return new Intl.NumberFormat('en-US', {
        style: 'currency',
        currency: 'USD',
        minimumFractionDigits: 2
    }).format(value);
}

function formatNumber(value) {
    return new Intl.NumberFormat('en-US').format(value);
}

function formatPercent(value) {
    return `${value.toFixed(1)}%`;
}

function getSentimentEmoji(sentiment) {
    const emojiMap = {
        '🚀 Strong Bullish': '🚀',
        '📈 Bullish': '📈',
        '➡️ Neutral': '➡️',
        '📉 Bearish': '📉',
        '⚠️ Strong Bearish': '⚠️'
    };
    return emojiMap[sentiment] || '📊';
}

function getSentimentColor(sentiment) {
    if (sentiment.includes('Strong Bullish') || sentiment.includes('Bullish')) {
        return 'text-green-600';
    } else if (sentiment.includes('Bearish')) {
        return 'text-red-600';
    } else {
        return 'text-gray-600';
    }
}

// ==============================================================================
// API FUNCTIONS
// ==============================================================================

async function fetchWithErrorHandling(url, options = {}) {
    try {
        const response = await fetch(url, {
            headers: {
                'Content-Type': 'application/json',
                ...options.headers
            },
            ...options
        });
        
        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.error || `HTTP ${response.status}`);
        }
        
        return await response.json();
    } catch (error) {
        console.error('API Error:', error);
        throw error;
    }
}

// ==============================================================================
// UI STATE MANAGEMENT
// ==============================================================================

function showLoading(step = 'Initializing analysis...') {
    document.getElementById('loadingSection').classList.remove('hidden');
    document.getElementById('resultsSection').classList.add('hidden');
    document.getElementById('errorSection').classList.add('hidden');
    document.getElementById('loadingStep').textContent = step;
}

function hideLoading() {
    document.getElementById('loadingSection').classList.add('hidden');
}

function showResults() {
    document.getElementById('loadingSection').classList.add('hidden');
    document.getElementById('resultsSection').classList.remove('hidden');
    document.getElementById('errorSection').classList.add('hidden');
}

function showError(message) {
    document.getElementById('loadingSection').classList.add('hidden');
    document.getElementById('resultsSection').classList.add('hidden');
    document.getElementById('errorSection').classList.remove('hidden');
    document.getElementById('errorMessage').textContent = message;
}

function hideError() {
    document.getElementById('errorSection').classList.add('hidden');
}

function updateLoadingStep(step) {
    document.getElementById('loadingStep').textContent = step;
}

// ==============================================================================
// ANALYSIS TYPE SELECTION
// ==============================================================================

function selectAnalysisType(type) {
    currentAnalysisType = type;
    
    // Update UI
    const shortOption = document.getElementById('shortTermOption');
    const longOption = document.getElementById('longTermOption');
    const dayOption = document.getElementById('dayTradeOption');
    const descriptionText = document.getElementById('descriptionText');
    
    // Reset all options
    shortOption.querySelector('div').className = 'border-2 border-blue-200 rounded-xl p-4 cursor-pointer transition-all hover:border-blue-400 hover:bg-blue-50';
    longOption.querySelector('div').className = 'border-2 border-green-200 rounded-xl p-4 cursor-pointer transition-all hover:border-green-400 hover:bg-green-50';
    if (dayOption) {
        dayOption.querySelector('div').className = 'border-2 border-yellow-200 rounded-xl p-4 cursor-pointer transition-all hover:border-yellow-400 hover:bg-yellow-50';
    }
    
    if (type === 'short') {
        shortOption.querySelector('div').className = 'border-2 border-blue-500 bg-blue-50 rounded-xl p-4 cursor-pointer transition-all';
        descriptionText.innerHTML = `
            <strong>Short-Term Analysis Selected:</strong><br>
            📰 News sentiment analysis • 📈 Technical momentum • ⚡ Quick price movements<br>
            <span class="text-blue-600">Perfect for day trading and swing trading strategies</span>
        `;
    } else if (type === 'long') {
        longOption.querySelector('div').className = 'border-2 border-green-500 bg-green-50 rounded-xl p-4 cursor-pointer transition-all';
        descriptionText.innerHTML = `
            <strong>Long-Term Analysis Selected:</strong><br>
            🏢 Company fundamentals • 📊 Financial health • 🎯 Growth potential<br>
            <span class="text-green-600">Perfect for investment portfolios and long-term holdings</span>
        `;
    } else if (type === 'day') {
        if (dayOption) {
            dayOption.querySelector('div').className = 'border-2 border-yellow-500 bg-yellow-50 rounded-xl p-4 cursor-pointer transition-all';
        }
        descriptionText.innerHTML = `
            <strong>Day Trade Analysis Selected:</strong><br>
            ⚡ Intraday/next-session move • 📉 Volatility-aware • 🧮 Options confidence<br>
            <span class="text-yellow-600">Great for same-day or next-morning strategies</span>
        `;
    }

    // Re-render reasoning list if switching tabs after results
    if (lastAnalysisData) {
        updateAIReasoning(
            lastAnalysisData.reasoning,
            lastAnalysisData.reasoning_tabs,
            lastAnalysisData.reasoning_charts
        );
    }
}

// ==============================================================================
// MAIN ANALYSIS FUNCTION
// ==============================================================================

async function analyzeStock() {
    const tickerInput = document.getElementById('stockInput');
    const ticker = tickerInput.value.trim().toUpperCase();
    
    if (!ticker) {
        showError('Please enter a stock ticker symbol');
        return;
    }
    
    if (!currentAnalysisType) {
        showError('Please select an analysis type (Short-term or Long-term)');
        return;
    }
    
    currentTicker = ticker;
    
    try {
        showLoading('Clearing previous data...');
        
        // Simulate loading steps based on analysis type
        if (currentAnalysisType === 'short') {
            setTimeout(() => updateLoadingStep('Analyzing news sentiment...'), 1000);
            setTimeout(() => updateLoadingStep('Calculating momentum indicators...'), 2000);
            setTimeout(() => updateLoadingStep('Generating short-term predictions...'), 3000);
            setTimeout(() => updateLoadingStep('Assessing market volatility...'), 4000);
        } else if (currentAnalysisType === 'long') {
            setTimeout(() => updateLoadingStep('Analyzing company fundamentals...'), 1000);
            setTimeout(() => updateLoadingStep('Evaluating financial health...'), 2000);
            setTimeout(() => updateLoadingStep('Generating long-term predictions...'), 3000);
            setTimeout(() => updateLoadingStep('Assessing growth potential...'), 4000);
        } else {
            setTimeout(() => updateLoadingStep('Scanning intraday momentum...'), 1000);
            setTimeout(() => updateLoadingStep('Estimating next-session direction...'), 2000);
            setTimeout(() => updateLoadingStep('Calculating risk and volatility...'), 3000);
            setTimeout(() => updateLoadingStep('Computing options confidence...'), 4000);
        }
        
        const response = await fetchWithErrorHandling(`${API_BASE_URL}/analyze/${ticker}?type=${currentAnalysisType}`, {
            method: 'POST'
        });
        
        if (response.success) {
            updateLoadingStep('AI analysis complete!');
            setTimeout(() => {
                hideLoading();
                displayAIResults(response.data);
                showResults();
            }, 1000);
        } else {
            throw new Error(response.error || 'AI analysis failed');
        }
        
    } catch (error) {
        console.error('Analysis error:', error);
        showError(`Failed to analyze ${ticker}: ${error.message}`);
    }
}

// ==============================================================================
// RESULTS DISPLAY FUNCTIONS
// ==============================================================================

function displayAIResults(data) {
    lastAnalysisData = data;
    // Update stock overview
    const analysisTypeLabel = currentAnalysisType === 'short' ? 'Short-Term' : (currentAnalysisType === 'long' ? 'Long-Term' : 'Day Trade');
    document.getElementById('stockTitle').textContent = `${data.ticker} - ${analysisTypeLabel} AI Analysis`;
    document.getElementById('analysisDate').textContent = `Analysis Date: ${data.analysis_date}`;
    
    // Update key metrics
    document.getElementById('currentPrice').textContent = formatCurrency(data.current_price);
    document.getElementById('sentimentScore').textContent = `${data.predicted_return.toFixed(2)}%`;
    document.getElementById('rsiValue').textContent = `${data.confidence.toFixed(1)}%`;
    document.getElementById('volumeValue').textContent = `${data.risk_score.toFixed(1)}%`;
    
    // Update AI prediction results
    const recommendation = data.recommendation;
    const emoji = getRecommendationEmoji(recommendation.action);
    const colorClass = getRecommendationColor(recommendation.action);
    
    document.getElementById('sentimentEmoji').textContent = emoji;
    document.getElementById('sentimentLabel').textContent = recommendation.action;
    document.getElementById('sentimentLabel').className = `text-lg font-semibold ${colorClass}`;
    
    // Update scores and labels based on analysis type
    if (currentAnalysisType === 'short') {
        document.getElementById('shortTermScore').textContent = `${data.predicted_return.toFixed(2)}%`;
        document.getElementById('longTermScore').textContent = formatCurrency(data.predicted_price);
        document.getElementById('metric1Label').textContent = 'Predicted Return (30 days)';
        document.getElementById('metric2Label').textContent = 'Target Price';
    } else if (currentAnalysisType === 'long') {
        document.getElementById('shortTermScore').textContent = formatCurrency(data.predicted_price);
        document.getElementById('longTermScore').textContent = `${data.predicted_return.toFixed(2)}%`;
        document.getElementById('metric1Label').textContent = 'Target Price (12 months)';
        document.getElementById('metric2Label').textContent = 'Expected Return';
    } else {
        document.getElementById('shortTermScore').textContent = `${data.predicted_return.toFixed(2)}%`;
        document.getElementById('longTermScore').textContent = formatCurrency(data.predicted_price);
        document.getElementById('metric1Label').textContent = 'Predicted Return (next session)';
        document.getElementById('metric2Label').textContent = 'Expected Open/Close';
    }
    
    // Update confidence levels
    document.getElementById('shortTermConfidence').textContent = `Confidence: ${data.confidence.toFixed(1)}%`;
    document.getElementById('longTermConfidence').textContent = `Risk Score: ${data.risk_score.toFixed(1)}%`;
    
    // Update prediction details
    document.getElementById('articleCount').textContent = data.recommendation.score;
    document.getElementById('marketClassification').textContent = recommendation.action;
    document.getElementById('overallConfidence').textContent = `${data.confidence.toFixed(1)}%`;
    const optionsConfidenceEl = document.getElementById('optionsConfidence');
    if (optionsConfidenceEl) {
        optionsConfidenceEl.textContent = data.option_confidence !== undefined ? `${data.option_confidence.toFixed(1)}%` : '-';
    }
    
    // Update AI reasoning
    updateAIReasoning(data.reasoning, data.reasoning_tabs, data.reasoning_charts);
    
    // Update recommendation
    updateAIRecommendation(data);
    
    // Create simple prediction chart
    createPredictionChart(data);
    if (data.reasoning_charts) {
        renderReasoningCharts(data.reasoning_charts);
    }
    
    // Update prediction table
    updatePredictionTable(data);
}

// Reasoning tabs state
let currentReasoningTab = 'technical';

function selectReasoningTab(tab) {
    currentReasoningTab = tab;
    const tabTechnical = document.getElementById('tabTechnical');
    const tabNonTechnical = document.getElementById('tabNonTechnical');
    if (tabTechnical && tabNonTechnical) {
        if (tab === 'technical') {
            tabTechnical.className = 'px-3 py-1 rounded bg-blue-100 text-blue-700';
            tabNonTechnical.className = 'px-3 py-1 rounded bg-gray-100 text-gray-700';
        } else {
            tabTechnical.className = 'px-3 py-1 rounded bg-gray-100 text-gray-700';
            tabNonTechnical.className = 'px-3 py-1 rounded bg-blue-100 text-blue-700';
        }
    }
    if (lastAnalysisData) {
        updateAIReasoning(
            lastAnalysisData.reasoning,
            lastAnalysisData.reasoning_tabs,
            lastAnalysisData.reasoning_charts
        );
    }
}

function updateAIReasoning(reasoning, reasoningTabs, charts) {
    const factorsList = document.getElementById('keyFactors');
    factorsList.innerHTML = '';
    
    const listToRender = reasoningTabs && reasoningTabs[currentReasoningTab]
        ? reasoningTabs[currentReasoningTab]
        : (reasoning || []);

    if (listToRender && listToRender.length > 0) {
        listToRender.forEach(reason => {
            const li = document.createElement('li');
            li.textContent = `• ${reason}`;
            li.className = 'text-sm text-gray-600';
            factorsList.appendChild(li);
        });
    } else {
        const li = document.createElement('li');
        li.textContent = '• AI analysis completed with machine learning models';
        li.className = 'text-sm text-gray-600';
        factorsList.appendChild(li);
    }

    // Charts in reasoning section (if provided)
    if (charts) {
        renderReasoningCharts(charts);
    }
}

let directionChartInstance = null;
let confidenceRiskChartInstance = null;

function renderReasoningCharts(charts) {
    const dirCtx = document.getElementById('directionChart');
    const crCtx = document.getElementById('confidenceRiskChart');
    if (!dirCtx || !crCtx) return;
    
    if (directionChartInstance) directionChartInstance.destroy();
    if (confidenceRiskChartInstance) confidenceRiskChartInstance.destroy();
    
    directionChartInstance = new Chart(dirCtx.getContext('2d'), {
        type: 'doughnut',
        data: {
            labels: ['Up', 'Down'],
            datasets: [{
                data: [charts.direction.up, charts.direction.down],
                backgroundColor: ['rgba(34,197,94,0.8)', 'rgba(239,68,68,0.8)'],
                borderColor: ['rgb(34,197,94)', 'rgb(239,68,68)'],
                borderWidth: 1
            }]
        },
        options: { responsive: true, plugins: { legend: { position: 'bottom' } } }
    });
    
    confidenceRiskChartInstance = new Chart(crCtx.getContext('2d'), {
        type: 'bar',
        data: {
            labels: ['Confidence', 'Risk'],
            datasets: [{
                data: [charts.confidence_vs_risk.confidence, charts.confidence_vs_risk.risk],
                backgroundColor: ['rgba(59,130,246,0.8)', 'rgba(245,158,11,0.8)'],
                borderColor: ['rgb(59,130,246)', 'rgb(245,158,11)'],
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            plugins: { legend: { display: false } },
            scales: { y: { beginAtZero: true, max: 100 } }
        }
    });
}

function getRecommendationEmoji(action) {
    const emojiMap = {
        'STRONG BUY': '🚀',
        'BUY': '📈',
        'HOLD': '➡️',
        'SELL': '📉',
        'STRONG SELL': '⚠️'
    };
    return emojiMap[action] || '📊';
}

function getRecommendationColor(action) {
    if (action.includes('BUY')) {
        return 'text-green-600';
    } else if (action.includes('SELL')) {
        return 'text-red-600';
    } else {
        return 'text-gray-600';
    }
}

function updateAIRecommendation(data) {
    const recommendationDiv = document.getElementById('recommendation');
    const recommendation = data.recommendation;
    const emoji = getRecommendationEmoji(recommendation.action);
    
    let bgColor, textColor;
    if (recommendation.action.includes('BUY')) {
        bgColor = 'bg-green-50 border-green-200';
        textColor = 'text-green-800';
    } else if (recommendation.action.includes('SELL')) {
        bgColor = 'bg-red-50 border-red-200';
        textColor = 'text-red-800';
    } else {
        bgColor = 'bg-yellow-50 border-yellow-200';
        textColor = 'text-yellow-800';
    }
    
    recommendationDiv.className = `text-center p-6 ${bgColor} border rounded-lg`;
    recommendationDiv.innerHTML = `
        <div class="text-4xl mb-2">${emoji}</div>
        <div class="text-lg font-semibold ${textColor}">${recommendation.action}</div>
        <div class="text-sm text-gray-600 mt-2">${recommendation.strength}</div>
        <div class="text-xs text-gray-500 mt-1">Score: ${recommendation.score}/100 | Confidence: ${data.confidence.toFixed(1)}%</div>
    `;
}

function createPredictionChart(data) {
    const ctx = document.getElementById('priceChart').getContext('2d');
    
    if (priceChart) {
        priceChart.destroy();
    }
    
    // Create a simple prediction visualization
    const currentPrice = data.current_price;
    const predictedPrice = data.predicted_price;
    const predictedReturn = data.predicted_return;
    
    priceChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: ['Current Price', 'Predicted Price'],
            datasets: [{
                label: 'Price',
                data: [currentPrice, predictedPrice],
                backgroundColor: ['rgba(59, 130, 246, 0.8)', predictedReturn > 0 ? 'rgba(34, 197, 94, 0.8)' : 'rgba(239, 68, 68, 0.8)'],
                borderColor: ['rgb(59, 130, 246)', predictedReturn > 0 ? 'rgb(34, 197, 94)' : 'rgb(239, 68, 68)'],
                borderWidth: 2
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: false
                },
                title: {
                    display: true,
                    text: `AI Prediction: ${predictedReturn > 0 ? '+' : ''}${predictedReturn.toFixed(2)}%`
                }
            },
            scales: {
                y: {
                    beginAtZero: false,
                    ticks: {
                        callback: function(value) {
                            return formatCurrency(value);
                        }
                    }
                }
            }
        }
    });
}

function updatePredictionTable(data) {
    const tableBody = document.getElementById('indicatorsTable');
    tableBody.innerHTML = '';
    
    const predictionData = [
        {
            name: 'Predicted Return',
            value: `${data.predicted_return > 0 ? '+' : ''}${data.predicted_return.toFixed(2)}%`,
            interpretation: data.predicted_return > 0.02 ? 'Strong bullish prediction' : data.predicted_return < -0.02 ? 'Strong bearish prediction' : 'Minimal movement expected',
            signal: data.predicted_return > 0 ? 'Bullish' : 'Bearish'
        },
        {
            name: 'Confidence Score',
            value: `${data.confidence.toFixed(1)}%`,
            interpretation: data.confidence > 80 ? 'High confidence prediction' : data.confidence > 60 ? 'Moderate confidence' : 'Low confidence',
            signal: data.confidence > 70 ? 'Reliable' : 'Uncertain'
        },
        {
            name: 'Risk Score',
            value: `${data.risk_score.toFixed(1)}%`,
            interpretation: data.risk_score < 30 ? 'Low risk investment' : data.risk_score < 60 ? 'Moderate risk' : 'High risk investment',
            signal: data.risk_score < 40 ? 'Low Risk' : 'High Risk'
        },
        {
            name: 'Recommendation Score',
            value: `${data.recommendation.score}/100`,
            interpretation: data.recommendation.score > 75 ? 'Strong recommendation' : data.recommendation.score > 50 ? 'Moderate recommendation' : 'Weak recommendation',
            signal: data.recommendation.action
        },
        ...(data.option_confidence !== undefined ? [{
            name: 'Options Confidence',
            value: `${data.option_confidence.toFixed(1)}%`,
            interpretation: data.option_confidence > 75 ? 'High options confidence' : data.option_confidence > 50 ? 'Moderate options confidence' : 'Low options confidence',
            signal: data.option_confidence > 60 ? 'Favorable' : 'Caution'
        }] : [])
    ];
    
    predictionData.forEach(item => {
        const row = document.createElement('tr');
        row.innerHTML = `
            <td class="px-4 py-3 font-medium">${item.name}</td>
            <td class="px-4 py-3">${item.value}</td>
            <td class="px-4 py-3">${item.interpretation}</td>
            <td class="px-4 py-3">
                <span class="px-2 py-1 text-xs rounded-full ${getSignalColor(item.signal)}">
                    ${item.signal}
                </span>
            </td>
        `;
        tableBody.appendChild(row);
    });
}

function updateIndicatorsTable(indicators) {
    const tableBody = document.getElementById('indicatorsTable');
    tableBody.innerHTML = '';
    
    const indicatorData = [
        {
            name: 'RSI',
            value: indicators.rsi ? indicators.rsi.toFixed(2) : 'N/A',
            interpretation: getRSIInterpretation(indicators.rsi),
            signal: getRSISignal(indicators.rsi)
        },
        {
            name: 'MACD',
            value: indicators.macd ? indicators.macd.toFixed(4) : 'N/A',
            interpretation: getMACDInterpretation(indicators.macd),
            signal: getMACDSignal(indicators.macd)
        },
        {
            name: 'ATR',
            value: indicators.atr ? indicators.atr.toFixed(2) : 'N/A',
            interpretation: getATRInterpretation(indicators.atr),
            signal: 'Volatility'
        },
        {
            name: 'ADX',
            value: indicators.adx ? indicators.adx.toFixed(2) : 'N/A',
            interpretation: getADXInterpretation(indicators.adx),
            signal: 'Trend Strength'
        }
    ];
    
    indicatorData.forEach(indicator => {
        const row = document.createElement('tr');
        row.innerHTML = `
            <td class="px-4 py-3 font-medium">${indicator.name}</td>
            <td class="px-4 py-3">${indicator.value}</td>
            <td class="px-4 py-3">${indicator.interpretation}</td>
            <td class="px-4 py-3">
                <span class="px-2 py-1 text-xs rounded-full ${getSignalColor(indicator.signal)}">
                    ${indicator.signal}
                </span>
            </td>
        `;
        tableBody.appendChild(row);
    });
}

// ==============================================================================
// INDICATOR INTERPRETATION FUNCTIONS
// ==============================================================================

function getRSIInterpretation(rsi) {
    if (!rsi) return 'No data';
    if (rsi > 70) return 'Overbought';
    if (rsi < 30) return 'Oversold';
    if (rsi > 50) return 'Bullish momentum';
    return 'Bearish momentum';
}

function getRSISignal(rsi) {
    if (!rsi) return 'Neutral';
    if (rsi > 70) return 'Sell';
    if (rsi < 30) return 'Buy';
    return 'Hold';
}

function getMACDInterpretation(macd) {
    if (!macd) return 'No data';
    if (macd > 0) return 'Bullish divergence';
    return 'Bearish divergence';
}

function getMACDSignal(macd) {
    if (!macd) return 'Neutral';
    return macd > 0 ? 'Bullish' : 'Bearish';
}

function getATRInterpretation(atr) {
    if (!atr) return 'No data';
    if (atr > 5) return 'High volatility';
    if (atr < 2) return 'Low volatility';
    return 'Moderate volatility';
}

function getADXInterpretation(adx) {
    if (!adx) return 'No data';
    if (adx > 25) return 'Strong trend';
    if (adx > 15) return 'Moderate trend';
    return 'Weak trend';
}

function getSignalColor(signal) {
    switch (signal.toLowerCase()) {
        case 'buy': return 'bg-green-100 text-green-800';
        case 'sell': return 'bg-red-100 text-red-800';
        case 'bullish': return 'bg-green-100 text-green-800';
        case 'bearish': return 'bg-red-100 text-red-800';
        default: return 'bg-gray-100 text-gray-800';
    }
}

// ==============================================================================
// CHART CREATION FUNCTIONS
// ==============================================================================

function createPriceChart(priceHistory) {
    const ctx = document.getElementById('priceChart').getContext('2d');
    
    if (priceChart) {
        priceChart.destroy();
    }
    
    const labels = priceHistory.map(item => item.date);
    const prices = priceHistory.map(item => item.close);
    
    priceChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [{
                label: 'Closing Price',
                data: prices,
                borderColor: 'rgb(59, 130, 246)',
                backgroundColor: 'rgba(59, 130, 246, 0.1)',
                borderWidth: 2,
                fill: true,
                tension: 0.4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: false
                }
            },
            scales: {
                y: {
                    beginAtZero: false,
                    ticks: {
                        callback: function(value) {
                            return formatCurrency(value);
                        }
                    }
                },
                x: {
                    ticks: {
                        maxTicksLimit: 10
                    }
                }
            },
            interaction: {
                intersect: false,
                mode: 'index'
            }
        }
    });
}

function createRSIChart(indicatorsHistory) {
    const ctx = document.getElementById('rsiChart').getContext('2d');
    
    if (rsiChart) {
        rsiChart.destroy();
    }
    
    const labels = indicatorsHistory.map(item => item.date);
    const rsiValues = indicatorsHistory.map(item => item.rsi).filter(val => val !== null);
    
    rsiChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels.slice(-rsiValues.length),
            datasets: [{
                label: 'RSI',
                data: rsiValues,
                borderColor: 'rgb(147, 51, 234)',
                backgroundColor: 'rgba(147, 51, 234, 0.1)',
                borderWidth: 2,
                tension: 0.4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: false
                }
            },
            scales: {
                y: {
                    min: 0,
                    max: 100,
                    ticks: {
                        stepSize: 20
                    }
                },
                x: {
                    ticks: {
                        maxTicksLimit: 8
                    }
                }
            }
        }
    });
    
    // Add RSI reference lines
    rsiChart.options.plugins.annotation = {
        annotations: {
            line1: {
                type: 'line',
                yMin: 70,
                yMax: 70,
                borderColor: 'red',
                borderWidth: 1,
                borderDash: [5, 5],
                label: {
                    content: 'Overbought (70)',
                    enabled: true
                }
            },
            line2: {
                type: 'line',
                yMin: 30,
                yMax: 30,
                borderColor: 'green',
                borderWidth: 1,
                borderDash: [5, 5],
                label: {
                    content: 'Oversold (30)',
                    enabled: true
                }
            }
        }
    };
}

function createMACDChart(indicatorsHistory) {
    const ctx = document.getElementById('macdChart').getContext('2d');
    
    if (macdChart) {
        macdChart.destroy();
    }
    
    const labels = indicatorsHistory.map(item => item.date);
    const macdValues = indicatorsHistory.map(item => item.macd).filter(val => val !== null);
    
    macdChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels.slice(-macdValues.length),
            datasets: [{
                label: 'MACD',
                data: macdValues,
                borderColor: 'rgb(16, 185, 129)',
                backgroundColor: 'rgba(16, 185, 129, 0.1)',
                borderWidth: 2,
                tension: 0.4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: false
                }
            },
            scales: {
                y: {
                    ticks: {
                        callback: function(value) {
                            return value.toFixed(4);
                        }
                    }
                },
                x: {
                    ticks: {
                        maxTicksLimit: 8
                    }
                }
            }
        }
    });
}

// ==============================================================================
// QUICK ANALYSIS FUNCTIONS
// ==============================================================================

function quickAnalyze(ticker) {
    document.getElementById('stockInput').value = ticker;
    analyzeStock();
}

// ==============================================================================
// EVENT LISTENERS
// ==============================================================================

document.addEventListener('DOMContentLoaded', function() {
    // Enter key support for stock input
    document.getElementById('stockInput').addEventListener('keypress', function(e) {
        if (e.key === 'Enter') {
            analyzeStock();
        }
    });
    
    // Focus on input when page loads
    document.getElementById('stockInput').focus();
    
    // Add some example data for demonstration
    console.log('Stock Analysis Dashboard loaded successfully!');
    console.log('Enter a stock ticker to begin analysis...');
});

// ==============================================================================
// ERROR HANDLING
// ==============================================================================

window.addEventListener('unhandledrejection', function(event) {
    console.error('Unhandled promise rejection:', event.reason);
    showError('An unexpected error occurred. Please try again.');
});

window.addEventListener('error', function(event) {
    console.error('JavaScript error:', event.error);
    showError('A technical error occurred. Please refresh the page and try again.');
});
