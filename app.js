const tabs = document.querySelectorAll('.tab-btn');
const sections = document.querySelectorAll('.tab-content');
const form = document.getElementById('prediction-form');
const resultDiv = document.getElementById('result');
const predictBtn = document.getElementById('predict-btn');

function escapeHtml(value) {
    return String(value)
        .replaceAll('&', '&amp;')
        .replaceAll('<', '&lt;')
        .replaceAll('>', '&gt;')
        .replaceAll('"', '&quot;')
        .replaceAll("'", '&#039;');
}

function formatPercent(value) {
    return `${Number(value).toFixed(2)}%`;
}

function showTab(tabName) {
    tabs.forEach((btn) => {
        btn.classList.toggle('active', btn.dataset.tab === tabName);
    });

    sections.forEach((section) => {
        section.classList.toggle('active', section.id === `tab-${tabName}`);
    });
}

tabs.forEach((btn) => {
    btn.addEventListener('click', () => showTab(btn.dataset.tab));
});

async function loadMetadata() {
    try {
        const response = await fetch('/metadata');
        const result = await response.json();
        if (result.status !== 'ok') {
            return;
        }

        const metadata = result.metadata;
        document.getElementById('rf-accuracy').textContent = formatPercent(metadata.rf_accuracy * 100);
        document.getElementById('rf-rocauc').textContent = Number(metadata.rf_roc_auc).toFixed(4);
        document.getElementById('training-date').textContent = metadata.training_date;
        document.getElementById('train-size').textContent = metadata.train_size;
        document.getElementById('test-size').textContent = metadata.test_size;

        document.getElementById('info-rf-accuracy').textContent = formatPercent(metadata.rf_accuracy * 100);
        document.getElementById('info-rf-rocauc').textContent = Number(metadata.rf_roc_auc).toFixed(4);
        document.getElementById('info-log-accuracy').textContent = formatPercent(metadata.logistic_accuracy * 100);
        document.getElementById('info-log-rocauc').textContent = Number(metadata.logistic_roc_auc).toFixed(4);

        const paramsList = document.getElementById('best-params');
        paramsList.innerHTML = '';
        Object.entries(metadata.best_params).forEach(([key, value]) => {
            const item = document.createElement('li');
            item.textContent = `${key}: ${value === null ? 'None' : value}`;
            paramsList.appendChild(item);
        });
    } catch (_err) {
        // Metadata is optional for page usage; UI can still run with fallback text.
    }
}

form.addEventListener('submit', async (e) => {
    e.preventDefault();
    const formData = new FormData(form);
    const payload = {};
    formData.forEach((value, key) => {
        payload[key] = value;
    });

    predictBtn.disabled = true;
    predictBtn.textContent = 'Predicting...';
    resultDiv.innerHTML = '<div class="card" style="padding: 12px;">Running prediction...</div>';

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
        const result = await response.json();

        if (result.status !== 'ok') {
            resultDiv.innerHTML = `<div class="error">${escapeHtml(result.error || 'Prediction failed.')}</div>`;
            return;
        }

        const isPositive = result.prediction === 'Positive';
        const pos = Number(result.probabilities?.Positive || 0);
        const neg = Number(result.probabilities?.Negative || 0);

        resultDiv.innerHTML = `
            <div class="result-card ${isPositive ? 'positive' : 'negative'}">
                <p class="result-title">${isPositive ? 'High Risk: Diabetes Positive' : 'Low Risk: Diabetes Negative'}</p>
                <p class="confidence">Confidence: ${formatPercent(result.confidence)}</p>
            </div>
            <div class="probabilities">
                <div class="prob-row">
                    <span>Negative</span>
                    <div class="bar"><span style="width:${Math.max(0, Math.min(100, neg))}%"></span></div>
                    <strong>${formatPercent(neg)}</strong>
                </div>
                <div class="prob-row">
                    <span>Positive</span>
                    <div class="bar"><span style="width:${Math.max(0, Math.min(100, pos))}%"></span></div>
                    <strong>${formatPercent(pos)}</strong>
                </div>
            </div>
            <div class="recommendation">
                ${isPositive
                    ? 'Please consult with a healthcare professional immediately for proper diagnosis and treatment.'
                    : 'Maintain a healthy lifestyle and continue regular health check-ups.'}
            </div>
        `;

        resultDiv.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    } catch (err) {
        resultDiv.innerHTML = `<div class="error">Error: ${escapeHtml(err.message)}</div>`;
    } finally {
        predictBtn.disabled = false;
        predictBtn.textContent = 'Predict Diabetes Risk';
    }
});

loadMetadata();
