/* Production Drift Detection Dashboard — Frontend Logic */

Chart.register(ChartAnnotation);

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------
const COLORS = { kl: '#ef4444', psi: '#22c55e', mmd: '#3b82f6', adwin: '#f59e0b' };
const SEV_COLORS = { critical: '#ef4444', warning: '#f97316', watch: '#eab308', healthy: '#22c55e' };
let charts = {};
let activeDetectors = ['kl', 'psi', 'mmd', 'adwin'];

// ---------------------------------------------------------------------------
// API helper
// ---------------------------------------------------------------------------
let lastError = '';
function showError(msg) {
  if (msg === lastError) return;
  lastError = msg;
  const el = document.createElement('div');
  el.style.cssText = 'position:fixed;top:16px;right:16px;background:#ef4444;color:#fff;padding:12px 20px;border-radius:8px;font-size:.85rem;z-index:999;animation:fadeIn .3s ease;box-shadow:0 4px 20px rgba(239,68,68,.4);max-width:400px';
  el.textContent = msg;
  document.body.appendChild(el);
  setTimeout(() => { el.remove(); lastError = ''; }, 5000);
}

async function api(path) {
  const r = await fetch(path);
  if (!r.ok) throw new Error(`${r.status} ${r.statusText}`);
  return r.json();
}

function safeRender(fn) {
  return (...args) => {
    fn(...args).catch(err => showError(err.message));
  };
}

// ---------------------------------------------------------------------------
// Theme toggle
// ---------------------------------------------------------------------------
document.getElementById('themeToggle').addEventListener('click', () => {
  const html = document.documentElement;
  const next = html.getAttribute('data-theme') === 'dark' ? 'light' : 'dark';
  html.setAttribute('data-theme', next);
  localStorage.setItem('theme', next);
  // Rebuild charts with new theme colors
  Object.values(charts).forEach(c => { if (c) c.destroy(); });
  charts = {};
  loadPage(document.querySelector('.nav-item.active').dataset.page);
});
const savedTheme = localStorage.getItem('theme');
if (savedTheme) document.documentElement.setAttribute('data-theme', savedTheme);

// ---------------------------------------------------------------------------
// Navigation
// ---------------------------------------------------------------------------
document.querySelectorAll('.nav-item').forEach(item => {
  item.addEventListener('click', () => {
    document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));
    item.classList.add('active');
    loadPage(item.dataset.page);
  });
});

function loadPage(page) {
  document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
  const el = document.getElementById(`page-${page}`);
  if (el) el.classList.add('active');
  const dispatch = {
    overview: safeRender(renderOverview),
    drift: safeRender(renderDrift),
    features: safeRender(renderFeatures),
    confidence: safeRender(renderConfidence),
    correlation: safeRender(renderCorrelation),
    alerts: safeRender(renderAlerts),
  };
  if (dispatch[page]) dispatch[page]();
}

// ---------------------------------------------------------------------------
// Chart helpers
// ---------------------------------------------------------------------------
function chartOpts(extra = {}) {
  const isDark = document.documentElement.getAttribute('data-theme') === 'dark';
  return {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: { labels: { color: isDark ? '#9ca0b8' : '#6b7089', usePointStyle: true, padding: 16 } },
    },
    scales: {
      x: { grid: { color: isDark ? '#2a2d3e' : '#e5e7eb' }, ticks: { color: isDark ? '#9ca0b8' : '#6b7089' } },
      y: { grid: { color: isDark ? '#2a2d3e' : '#e5e7eb' }, ticks: { color: isDark ? '#9ca0b8' : '#6b7089' } },
    },
    ...extra,
  };
}

function createChart(id, type, data, opts = {}) {
  if (charts[id]) charts[id].destroy();
  const ctx = document.getElementById(id);
  if (!ctx) return null;
  charts[id] = new Chart(ctx, { type, data, options: chartOpts(opts) });
  return charts[id];
}

// ---------------------------------------------------------------------------
// Metric card
// ---------------------------------------------------------------------------
function metric(label, value, sub = '') {
  return `<div class="metric"><div class="metric-label">${label}</div><div class="metric-value">${value}</div>${sub ? `<div class="metric-sub">${sub}</div>` : ''}</div>`;
}

// ---------------------------------------------------------------------------
// Sidebar status
// ---------------------------------------------------------------------------
function updateSidebar(status, alerts, batches) {
  const badge = document.getElementById('statusBadge');
  const colors = { healthy: '#22c55e', watch: '#eab308', warning: '#f97316', critical: '#ef4444' };
  badge.style.background = colors[status] || '#22c55e';
  badge.textContent = status.charAt(0).toUpperCase() + status.slice(1);
  document.getElementById('alertCount').textContent = alerts;
  document.getElementById('batchCount').textContent = batches;
}

// ---------------------------------------------------------------------------
// Pages
// ---------------------------------------------------------------------------
async function renderOverview() {
  const [overview, scores, alerts] = await Promise.all([
    api('/api/overview'), api('/api/drift-scores'), api('/api/alerts?limit=5'),
  ]);
  updateSidebar(overview.status, overview.total_alerts, overview.batches_processed);

  document.getElementById('overviewMetrics').innerHTML =
    metric('Status', overview.status.toUpperCase(), `${overview.active_detectors} detectors`) +
    metric('Batches', overview.batches_processed, 'monitored') +
    metric('Alerts', overview.total_alerts, `${overview.alerts_by_severity.critical || 0} critical`) +
    metric('Early Warning', `${overview.early_warning_score.toFixed(1)}/100`, `Confidence: ${overview.confidence_drop.toFixed(1)}% drop`);

  const datasets = Object.entries(scores.detectors).map(([k, v]) => ({
    label: k.toUpperCase(), data: v, borderColor: COLORS[k] || '#6366f1',
    backgroundColor: COLORS[k] || '#6366f1', fill: false, tension: .3, pointRadius: 3,
  }));
  createChart('overviewChart', 'line', {
    labels: scores.batches, datasets,
  }, { plugins: { legend: { display: datasets.length > 1 } } });

  const list = document.getElementById('overviewAlerts');
  list.innerHTML = alerts.length
    ? alerts.map(a => `<div class="alert-item ${a.severity}"><span class="alert-severity ${a.severity}">${a.severity}</span><span class="alert-detector">${a.detector}</span><span>${a.explanation}</span><span class="alert-score">${a.score.toFixed(4)}</span></div>`).join('')
    : '<p style="color:var(--text2)">No recent alerts.</p>';
}

async function renderDrift() {
  const scores = await api('/api/drift-scores');
  // Toggles
  const toggleHtml = Object.entries(scores.detectors).map(([k, v]) =>
    `<button class="detector-tog active" data-det="${k}">${k.toUpperCase()}</button>`
  ).join('');
  document.getElementById('detectorToggles').innerHTML = toggleHtml;
  document.querySelectorAll('.detector-tog').forEach(btn => {
    btn.addEventListener('click', () => {
      btn.classList.toggle('active');
      updateDriftChart();
    });
  });
  async function updateDriftChart() {
    const data = await api('/api/drift-scores');
    const active = [...document.querySelectorAll('.detector-tog.active')].map(b => b.dataset.det);
    const datasets = active.map(k => ({
      label: k.toUpperCase(), data: data.detectors[k] || [],
      borderColor: COLORS[k] || '#6366f1', backgroundColor: COLORS[k] || '#6366f1',
      fill: false, tension: .3, pointRadius: 3,
    }));
    createChart('driftChart', 'line', { labels: data.batches, datasets });

    // Detector details
    const details = await Promise.all(active.map(k => api(`/api/detector-details/${k}`)));
    document.getElementById('detectorDetails').innerHTML = `
      <h3>Detector Details</h3>
      <div class="detector-grid">${details.map(d => `
        <div class="detector-card">
          <h4>${d.name.toUpperCase()}</h4>
          <div class="stat-row"><span>Current</span><span>${d.current_score?.toFixed(4) || 'N/A'}</span></div>
          <div class="stat-row"><span>Mean</span><span>${d.mean_score.toFixed(4)}</span></div>
          <div class="stat-row"><span>Max</span><span>${d.max_score.toFixed(4)}</span></div>
          <div class="stat-row"><span>Alerts</span><span>${d.num_alerts}</span></div>
        </div>`).join('')}</div>`;
  }
  updateDriftChart();
}

async function renderFeatures() {
  const data = await api('/api/feature-scores');
  if (data.error) {
    document.getElementById('featureChart').parentElement.innerHTML = '<p style="color:var(--text2)">PSI detector not configured for per-feature analysis.</p>';
    return;
  }
  const names = data.feature_names;
  const scores = data.scores;
  const batches = scores.map((_, i) => i + 1);

  // Heatmap as a matrix chart
  const heatData = names.map((f, fi) => ({
    label: f, data: scores.map(b => b[f] || 0), backgroundColor: ctx => {
      const v = ctx.raw;
      return v > 0.25 ? '#ef4444' : v > 0.1 ? '#f97316' : v > 0.05 ? '#eab308' : '#22c55e';
    }, borderColor: 'rgba(0,0,0,0)',
  }));
  createChart('featureChart', 'bar', {
    labels: batches, datasets: heatData,
  }, { plugins: { legend: { display: true, labels: { boxWidth: 12 } } }, scales: { x: { stacked: false }, y: { stacked: false } } });

  // Rankings
  const latest = scores[scores.length - 1];
  const sorted = Object.entries(latest).sort((a, b) => b[1] - a[1]);
  const maxScore = Math.max(...sorted.map(s => s[1]), 0.01);
  document.getElementById('featureRankings').innerHTML = sorted.map(([name, score]) => {
    const pct = (score / maxScore * 100).toFixed(0);
    const color = score > 0.25 ? '#ef4444' : score > 0.1 ? '#f97316' : '#22c55e';
    const status = score > 0.25 ? 'Critical' : score > 0.1 ? 'Watch' : 'Stable';
    return `<div class="feature-row"><span class="feature-name">${name}</span><div class="feature-bar"><div class="feature-bar-fill" style="width:${pct}%;background:${color}"></div></div><span class="feature-score">${score.toFixed(4)}</span><span style="font-size:.8rem;color:var(--text2)">${status}</span></div>`;
  }).join('');
}

async function renderConfidence() {
  const data = await api('/api/confidence');
  document.getElementById('confidenceMetrics').innerHTML =
    metric('Confidence', data.trends.current_confidence?.toFixed(3) || 'N/A', data.trends.confidence_trend) +
    metric('Entropy', data.trends.current_entropy?.toFixed(3) || 'N/A', data.trends.entropy_trend) +
    metric('Margin', data.trends.current_margin?.toFixed(3) || 'N/A', data.trends.margin_trend) +
    metric('Status', data.degraded ? 'Degraded' : 'Stable', data.degradation_reason);

  const batches = data.confidence_history.map((_, i) => i + 1);
  createChart('confidenceChart', 'line', {
    labels: batches,
    datasets: [
      { label: 'Confidence', data: data.confidence_history, borderColor: '#22c55e', backgroundColor: 'rgba(34,197,94,0.1)', fill: true, tension: .3, pointRadius: 2 },
      { label: 'Entropy', data: data.entropy_history, borderColor: '#ef4444', backgroundColor: 'rgba(239,68,68,0.1)', fill: true, tension: .3, pointRadius: 2, yAxisID: 'y1' },
      { label: 'Margin', data: data.margin_history, borderColor: '#3b82f6', backgroundColor: 'rgba(59,130,246,0.1)', fill: true, tension: .3, pointRadius: 2, yAxisID: 'y1' },
    ],
  }, {
    scales: { y: { min: 0, max: 1 }, y1: { position: 'right', grid: { drawOnChartArea: false }, min: 0 } },
  });

  const cal = data.calibration;
  document.getElementById('calibrationData').innerHTML = cal && Object.keys(cal).length
    ? `<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px">
        <div class="detector-card"><div class="stat-row"><span>Low Conf Ratio</span><span>${(cal.underconfident_ratio * 100).toFixed(1)}%</span></div></div>
        <div class="detector-card"><div class="stat-row"><span>High Conf Ratio</span><span>${(cal.overconfident_ratio * 100).toFixed(1)}%</span></div></div>
        <div class="detector-card"><div class="stat-row"><span>ECE Approx</span><span>${cal.ece_approximation.toFixed(4)}</span></div></div>
        <div class="detector-card"><div class="stat-row"><span>Mean Conf</span><span>${cal.mean_confidence_overall?.toFixed(4) || 'N/A'}</span></div></div>
      </div>`
    : '<p style="color:var(--text2)">No calibration data available.</p>';
}

async function renderCorrelation() {
  const data = await api('/api/correlation');
  const viz = data.visualization;
  const score = data.early_warning_score;
  document.getElementById('correlationMetrics').innerHTML =
    metric('Early Warning', `${(score.early_warning_score || 0).toFixed(1)}/100', score.confidence_is_leading ? 'Leading' : 'Not leading') +
    metric('Degradation', `${((score.degradation_rate || 0) * 100).toFixed(1)}%`, 'confidence change') +
    metric('Drift Accel', `${((score.drift_acceleration || 0) * 100).toFixed(1)}%`, 'drift rate change') +
    metric('Warnings', viz.early_warnings.length, 'detected');

  // Time series
  const batches = viz.confidence_history.map((_, i) => i + 1);
  const ds = [{ label: 'Confidence', data: viz.confidence_history, borderColor: '#22c55e', yAxisID: 'y', tension: .3, pointRadius: 2, fill: false }];
  const detName = Object.keys(viz.drift_history)[0];
  if (detName) {
    const driftVals = viz.drift_history[detName];
    ds.push({ label: `${detName.toUpperCase()} Drift`, data: driftVals, borderColor: '#ef4444', yAxisID: 'y1', tension: .3, pointRadius: 2, fill: false });
  }
  createChart('corrTimeChart', 'line', { labels: batches, datasets: ds }, {
    scales: { y: { min: 0, max: 1, position: 'left' }, y1: { position: 'right', grid: { drawOnChartArea: false }, min: 0 } },
  });

  // Lead-lag
  const cc = Object.values(data.cross_correlations)[0];
  if (cc && cc.lead_lag_analysis) {
    const lags = cc.lead_lag_analysis.map(c => c.lag);
    const lead = cc.lead_lag_analysis.map(c => c.confidence_leads_drift !== undefined ? c.confidence_leads_drift : c.correlation || 0);
    const lag = cc.lead_lag_analysis.map(c => c.drift_leads_confidence !== undefined ? c.drift_leads_confidence : 0);
    createChart('corrLagChart', 'bar', {
      labels: lags,
      datasets: [
        { label: 'Confidence Leads', data: lead, backgroundColor: '#22c55e' },
        { label: 'Drift Leads', data: lag, backgroundColor: '#ef4444' },
      ],
    }, { scales: { y: { min: -1, max: 1 } } });
  }

  // Details
  document.getElementById('correlationDetails').innerHTML = `
    <h3>Cross-Correlation Details</h3>
    <div class="detector-summary-grid">${Object.entries(data.cross_correlations).map(([name, cc]) => `
      <div class="detector-summary-card">
        <h4>${name.toUpperCase()} vs Confidence</h4>
        <div class="row"><span>Simultaneous Corr</span><span>${(cc.simultaneous_correlation?.correlation || 0).toFixed(4)}</span></div>
        <div class="row"><span>Confidence Leads</span><span>${cc.confidence_is_leading_indicator ? 'Yes' : 'No'}</span></div>
        <div class="row"><span>Optimal Lag</span><span>${cc.optimal_lag || 0}</span></div>
      </div>`).join('')}</div>`;
}

async function renderAlerts() {
  const sev = document.getElementById('alertSeverityFilter').value;
  const alerts = await api(`/api/alerts?severity=${sev}&limit=500`);
  document.getElementById('alertTable').innerHTML = alerts.length
    ? `<table class="data-table"><thead><tr><th>Time</th><th>Severity</th><th>Detector</th><th>Score</th><th>Explanation</th></tr></thead>
        <tbody>${alerts.map(a => `<tr>
          <td style="font-size:.8rem;color:var(--text2)">${new Date(a.timestamp).toLocaleTimeString()}</td>
          <td><span style="color:${SEV_COLORS[a.severity] || '#9ca0b8'};font-weight:600">${a.severity.toUpperCase()}</span></td>
          <td style="font-weight:600">${a.detector}</td>
          <td style="font-family:monospace">${a.score.toFixed(4)}</td>
          <td style="color:var(--text2);font-size:.85rem">${a.explanation}</td>
        </tr>`).join('')}</tbody></table>`
    : '<p style="color:var(--text2)">No alerts to display.</p>';

  // Stats chart
  const counts = { critical: 0, warning: 0, watch: 0, healthy: 0 };
  alerts.forEach(a => { if (counts[a.severity] !== undefined) counts[a.severity]++; });
  createChart('alertChart', 'bar', {
    labels: Object.keys(counts),
    datasets: [{ label: 'Alerts', data: Object.values(counts), backgroundColor: ['#ef4444', '#f97316', '#eab308', '#22c55e'] }],
  }, { plugins: { legend: { display: false } }, scales: { y: { beginAtZero: true, ticks: { precision: 0 } } } });
}

document.getElementById('alertSeverityFilter').addEventListener('change', safeRender(renderAlerts));

// ---------------------------------------------------------------------------
// Auto-refresh
// ---------------------------------------------------------------------------
setInterval(() => {
  const active = document.querySelector('.nav-item.active');
  if (active) loadPage(active.dataset.page);
}, 15000);

// Initial load
loadPage('overview');
