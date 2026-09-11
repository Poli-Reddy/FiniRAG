/**
 * FiniRAG — Frontend Application Logic
 *
 * Handles: file upload, API calls, result rendering, tab switching,
 * history loading, health check, and drag & drop.
 */

// ─── State ───
let selectedFile = null;
let historyDocuments = [];
let historyPage = 1;
const HISTORY_PAGE_SIZE = 15;
const historySearch = document.getElementById('historySearch');
const historyTypeFilter = document.getElementById('historyTypeFilter');
const historyValidationFilter = document.getElementById('historyValidationFilter');
const clearHistoryFilters = document.getElementById('clearHistoryFilters');

// ─── DOM Elements ───
const dropZone = document.getElementById('dropZone');
const fileInput = document.getElementById('fileInput');
const filePreview = document.getElementById('filePreview');
const fileName = document.getElementById('fileName');
const clearFile = document.getElementById('clearFile');
const documentType = document.getElementById('documentType');
const submitBtn = document.getElementById('submitBtn');
const uploadForm = document.getElementById('uploadForm');
const uploadSection = document.getElementById('uploadSection');
const processingSection = document.getElementById('processingSection');
const resultSection = document.getElementById('resultSection');
const newUploadBtn = document.getElementById('newUploadBtn');

// ─── File Drop Zone ───
dropZone.addEventListener('click', () => fileInput.click());

dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('dragover');
});

dropZone.addEventListener('dragleave', () => {
    dropZone.classList.remove('dragover');
});

dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('dragover');
    if (e.dataTransfer.files.length > 0) {
        handleFileSelect(e.dataTransfer.files[0]);
    }
});

fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
        handleFileSelect(e.target.files[0]);
    }
});

clearFile.addEventListener('click', () => {
    selectedFile = null;
    filePreview.style.display = 'none';
    dropZone.style.display = 'block';
    fileInput.value = '';
    updateSubmitButton();
});

function handleFileSelect(file) {
    selectedFile = file;
    fileName.textContent = `${file.name} (${formatBytes(file.size)})`;
    filePreview.style.display = 'flex';
    dropZone.style.display = 'none';
    updateSubmitButton();
}

function updateSubmitButton() {
    submitBtn.disabled = !(selectedFile && documentType.value);
}

documentType.addEventListener('change', updateSubmitButton);

// ─── Form Submit ───
uploadForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    if (!selectedFile || !documentType.value) return;

    // Show processing state
    uploadSection.style.display = 'none';
    resultSection.style.display = 'none';
    processingSection.style.display = 'block';

    // Animate steps
    animateStep('stepValidation');

    const formData = new FormData();
    formData.append('file', selectedFile);
    formData.append('document_type', documentType.value);

    const timers = [
        setTimeout(() => animateStep('stepExtraction'), 800),
        setTimeout(() => animateStep('stepLLM'), 1600),
        setTimeout(() => animateStep('stepValidate'), 2400),
    ];
    try {
        const response = await fetch('/api/v1/documents/process', {
            method: 'POST',
            body: formData,
        });

        const result = await response.json();
        if (!response.ok) {
            const code = result?.detail?.code || result?.error?.code || `HTTP ${response.status}`;
            const msg = result?.detail?.message || result?.error?.message || '';
            showError(msg ? `${code}: ${msg}` : String(code));
            return;
        }
        showResult(result);
    } catch (err) {
        showError(err.message || 'Processing failed');
    } finally {
        timers.forEach(clearTimeout);
    }

    // Refresh history
    loadHistory();
});

function animateStep(stepId) {
    // Mark all previous steps as done
    const steps = document.querySelectorAll('.step');
    let found = false;
    steps.forEach(step => {
        if (step.id === stepId) {
            step.classList.add('active');
            step.classList.remove('done');
            found = true;
        } else if (!found) {
            step.classList.remove('active');
            step.classList.add('done');
        }
    });
}

// ─── Result Display ───
function showResult(result) {
    processingSection.style.display = 'none';
    resultSection.style.display = 'block';

    // Status banner
    const banner = document.getElementById('statusBanner');
    const isPass = result.processing_status === 'PASS';
    banner.className = `status-banner ${isPass ? 'pass' : 'failed'}`;

    document.getElementById('statusIcon').textContent = isPass ? '✅' : '❌';
    document.getElementById('statusLabel').textContent = isPass
        ? 'Processing Successful'
        : 'Processing Failed';

    const validationStatus = result.validation?.overall_status || 'N/A';
    document.getElementById('statusDetail').textContent = isPass
        ? `Validation: ${validationStatus}`
        : `Error: ${result.error?.code || 'Unknown'}`;

    const meta = result.processing_metadata || {};
    document.getElementById('statusMeta').innerHTML = `
        <div>Method: ${escapeHtml(meta.text_extraction_method || 'N/A')}</div>
        <div>Time: ${escapeHtml(meta.processing_time_ms || 0)}ms</div>
    `;

    // Render tabs
    renderExtractedData(result.extracted_data, result.document_type);
    renderValidation(result.validation);
    renderMetadata(result);
    document.getElementById('rawJson').textContent = JSON.stringify(result, null, 2);

    // Reset tab to first
    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
    document.querySelector('.tab[data-tab="extracted"]').classList.add('active');
    document.getElementById('tab-extracted').classList.add('active');
}

function showError(message) {
    processingSection.style.display = 'none';
    resultSection.style.display = 'block';

    const banner = document.getElementById('statusBanner');
    banner.className = 'status-banner failed';
    document.getElementById('statusIcon').textContent = '❌';
    document.getElementById('statusLabel').textContent = 'Request Failed';
    document.getElementById('statusDetail').textContent = message;
    document.getElementById('statusMeta').innerHTML = '';
}

// ─── Render Extracted Data ───
function extractedDataMarkup(data, docType) {
    if (!data) {
        return '<p class="empty-state">No data extracted</p>';
    }

    let html = '';

    if (docType === 'invoice') {
        html += renderFieldGroup('Document Info', {
            'Invoice Number': data.invoice_number,
            'Invoice Date': data.invoice_date,
            'Due Date': data.due_date,
            'Currency': data.currency,
            'Payment Method': data.payment_method,
        });
        html += renderFieldGroup('Vendor', {
            'Name': data.vendor_name,
            'Address': data.vendor_address,
            'Tax ID': data.vendor_tax_id,
        });
        html += renderFieldGroup('Customer', {
            'Name': data.customer_name,
            'Address': data.customer_address,
            'Tax ID': data.customer_tax_id,
        });
        html += renderFieldGroup('Amounts', {
            'Subtotal': data.subtotal,
            'Tax Amount': data.tax_amount,
            'Tax Rate': data.tax_rate,
            'Discount': data.discount,
            'Total Amount': data.total_amount,
            'Cash Paid': data.cash_paid,
            'Change Given': data.change_given,
            'Rounding': data.rounding_adjustment,
        });

        if (data.line_items?.length > 0) {
            html += '<div class="data-group"><div class="data-group-title">Line Items</div>';
            data.line_items.forEach((item, i) => {
                html += `<div class="data-row"><span class="data-label">#${i+1}: ${escapeHtml(getVal(item.description))}</span>`;
                html += `<span class="data-value">${escapeHtml(getVal(item.quantity))} × ${escapeHtml(getVal(item.unit_price))} = ${escapeHtml(getVal(item.amount))}</span></div>`;
            });
            html += '</div>';
        }

        if (data.additional_fields?.length > 0) {
            const fields = {};
            data.additional_fields.forEach(f => { fields[f.label] = f.value; });
            html += renderFieldGroup('Additional Fields', fields);
        }
    } else if (docType === 'balance_sheet' || docType === 'profit_and_loss' || docType === 'cash_flow_statement') {
        html += renderFieldGroup('Statement Info', {
            'Company': data.company_name,
            'Type': data.statement_type,
            'Unit': data.unit_descriptor,
        });

        if (data.periods) {
            for (const [year, period] of Object.entries(data.periods)) {
                html += renderPeriodData(year, period, docType);
            }
        }
    }

    return html;
}

function renderExtractedData(data, docType) {
    document.getElementById('extractedGrid').innerHTML = extractedDataMarkup(data, docType);
}

function renderPeriodData(year, period, docType) {
    let html = '';

    if (docType === 'balance_sheet') {
        if (period.assets) {
            html += renderDictGroup(`${year} — Assets`, period.assets);
        }
        const assetTotals = {};
        if (period.total_assets) assetTotals['Total Assets'] = period.total_assets;
        if (Object.keys(assetTotals).length) html += renderFieldGroup(`${year} — Asset Totals`, assetTotals);

        if (period.capital_and_liabilities) {
            html += renderDictGroup(`${year} — Capital & Liabilities`, period.capital_and_liabilities);
        }
        const liabTotals = {};
        if (period.total_capital_and_liabilities) liabTotals['Total C&L'] = period.total_capital_and_liabilities;
        if (Object.keys(liabTotals).length) html += renderFieldGroup(`${year} — Liability Totals`, liabTotals);
    } else if (docType === 'profit_and_loss') {
        html += renderFieldGroup(`${year} — Income`, {
            'Interest Earned': period.interest_earned,
            'Other Income': period.other_income,
            'Total Income': period.total_income,
        });
        html += renderFieldGroup(`${year} — Expenditure`, {
            'Interest Expended': period.interest_expended,
            'Operating Expenses': period.operating_expenses,
            'Provisions': period.provisions_and_contingencies,
            'Total Expenditure': period.total_expenditure,
        });
        html += renderFieldGroup(`${year} — Profit`, {
            'Consolidated Net Profit': period.consolidated_net_profit,
            'Basic EPS': period.basic_eps,
            'Diluted EPS': period.diluted_eps,
        });
    } else if (docType === 'cash_flow_statement') {
        html += renderFieldGroup(`${year} — Cash Flow`, {
            'Net Cash - Operating': period.net_cash_from_operating,
            'Net Cash - Investing': period.net_cash_from_investing,
            'Net Cash - Financing': period.net_cash_from_financing,
            'FX Adjustment': period.fx_translation_adjustment,
            'Net Increase': period.net_increase_in_cash,
            'Amalgamation': period.cash_acquired_on_amalgamation,
            'Opening Cash': period.opening_cash,
            'Closing Cash': period.closing_cash,
        });
    }

    return html;
}

function renderFieldGroup(title, fields) {
    let html = `<div class="data-group"><div class="data-group-title">${escapeHtml(title)}</div>`;
    for (const [label, field] of Object.entries(fields)) {
        const val = getVal(field);
        const cls = (val === 'Missing' || val === '—') ? ' missing' : '';
        html += `<div class="data-row"><span class="data-label">${escapeHtml(label)}</span><span class="data-value${cls}">${escapeHtml(val)}</span></div>`;
    }
    html += '</div>';
    return html;
}

function renderDictGroup(title, dict) {
    let html = `<div class="data-group"><div class="data-group-title">${escapeHtml(title)}</div>`;
    for (const [label, field] of Object.entries(dict)) {
        const val = getVal(field);
        const cls = (val === 'Missing' || val === '—') ? ' missing' : '';
        html += `<div class="data-row"><span class="data-label">${escapeHtml(label)}</span><span class="data-value${cls}">${escapeHtml(val)}</span></div>`;
    }
    html += '</div>';
    return html;
}

function getVal(field) {
    if (!field) return '—';
    if (typeof field === 'string') return field || '—';
    if (typeof field === 'number') return field.toString();
    if (field.value !== null && field.value !== undefined) {
        return field.value.toString();
    }
    if (field.status === 'MISSING') return 'Missing';
    if (field.status === 'UNREADABLE') return 'Unreadable';
    if (field.status === 'EXPLICIT_ZERO') return '0';
    return field.original_value || '—';
}

// ─── Render Validation ───
function validationMarkup(validation) {
    if (!validation || !validation.checks) {
        return { summary: '', checks: '<p class="empty-state">No validation data</p>' };
    }

    const s = validation.summary || { pass: 0, fail: 0, not_applicable: 0 };

    const summary = `
        <div class="summary-card pass">
            <div class="summary-count">${s.pass}</div>
            <div class="summary-label">Passed</div>
        </div>
        <div class="summary-card fail">
            <div class="summary-count">${s.fail}</div>
            <div class="summary-label">Failed</div>
        </div>
        <div class="summary-card na">
            <div class="summary-count">${s.not_applicable}</div>
            <div class="summary-label">N/A</div>
        </div>
    `;

    let checksHtml = '';
    validation.checks.forEach(check => {
        const statusClass = check.status === 'PASS' ? 'pass' :
                           check.status === 'FAIL' ? 'fail' : 'na';

        let detail = '';
        if (check.formula) detail += `Formula: ${check.formula}`;
        if (check.calculated_value !== null && check.calculated_value !== undefined) {
            detail += ` | Calculated: ${check.calculated_value}`;
        }
        if (check.reported_value !== null && check.reported_value !== undefined) {
            detail += ` | Reported: ${check.reported_value}`;
        }
        if (check.variance !== null && check.variance !== undefined) {
            detail += ` | Variance: ${check.variance}`;
        }
        if (check.reason) detail = check.reason;

        checksHtml += `
            <div class="check-item">
                <span class="check-badge ${statusClass}">${escapeHtml(check.status)}</span>
                <div>
                    <div class="check-name">${escapeHtml(check.name)}</div>
                    <div class="check-detail">${escapeHtml(detail)}</div>
                </div>
            </div>
        `;
    });

    return { summary, checks: checksHtml };
}

function renderValidation(validation) {
    const rendered = validationMarkup(validation);
    document.getElementById('validationSummary').innerHTML = rendered.summary;
    document.getElementById('validationChecks').innerHTML = rendered.checks;
}

// ─── Render Metadata ───
function metadataMarkup(result) {
    const meta = result.processing_metadata || {};
    const fv = result.file_validation || {};

    return `
        <div class="metadata-card">
            <div class="metadata-label">Document Name</div>
            <div class="metadata-value">${escapeHtml(result.document_name || '—')}</div>
        </div>
        <div class="metadata-card">
            <div class="metadata-label">Document Type</div>
            <div class="metadata-value">${escapeHtml(result.document_type || '—')}</div>
        </div>
        <div class="metadata-card">
            <div class="metadata-label">File Type</div>
            <div class="metadata-value">${escapeHtml(fv.file_type || '—')}</div>
        </div>
        <div class="metadata-card">
            <div class="metadata-label">Page Count</div>
            <div class="metadata-value">${escapeHtml(fv.page_count ?? '—')}</div>
        </div>
        <div class="metadata-card">
            <div class="metadata-label">Text Extraction</div>
            <div class="metadata-value">${escapeHtml(meta.text_extraction_method || '—')}</div>
        </div>
        <div class="metadata-card">
            <div class="metadata-label">OCR Provider</div>
            <div class="metadata-value">${escapeHtml(meta.ocr_provider || 'N/A (native text)')}</div>
        </div>
        <div class="metadata-card">
            <div class="metadata-label">LLM Provider</div>
            <div class="metadata-value">${escapeHtml(meta.llm_provider || '—')}</div>
        </div>
        <div class="metadata-card">
            <div class="metadata-label">Processing Time</div>
            <div class="metadata-value">${escapeHtml(meta.processing_time_ms || 0)}ms</div>
        </div>
    `;
}

function renderMetadata(result) {
    document.getElementById('metadataGrid').innerHTML = metadataMarkup(result);
}

// ─── Tab Switching ───
document.querySelectorAll('.tab').forEach(tab => {
    tab.addEventListener('click', () => {
        document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
        document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
        tab.classList.add('active');
        document.getElementById(`tab-${tab.dataset.tab}`).classList.add('active');
    });
});

// ─── New Upload ───
newUploadBtn.addEventListener('click', () => {
    resultSection.style.display = 'none';
    uploadSection.style.display = 'block';
    selectedFile = null;
    filePreview.style.display = 'none';
    dropZone.style.display = 'block';
    fileInput.value = '';
    documentType.value = '';
    updateSubmitButton();

    // Reset processing steps
    document.querySelectorAll('.step').forEach(s => {
        s.classList.remove('active', 'done');
    });
});

// ─── History ───
async function loadHistory() {
    try {
        // Fetch the complete persisted dashboard set once. Pagination is
        // client-side so changing pages never loses the stored results.
        const response = await fetch('/api/v1/documents?limit=1000&offset=0');
        const data = await response.json();
        historyDocuments = (data.documents || []).sort((a, b) => {
            return new Date(b.created_at || 0) - new Date(a.created_at || 0);
        });
        historyPage = 1;
        renderHistoryPage();
    } catch (err) {
        console.error('Failed to load history:', err);
    }
}

function renderHistoryPage() {
    const query = (historySearch?.value || '').trim().toLowerCase();
    const type = historyTypeFilter?.value || '';
    const validation = historyValidationFilter?.value || '';
    let filtered = historyDocuments.filter(doc => {
        const matchesSearch = !query || (doc.document_name || '').toLowerCase().includes(query);
        const matchesType = !type || doc.document_type === type;
        const matchesValidation = !validation || (doc.validation_status || 'NOT_RUN') === validation;
        return matchesSearch && matchesType && matchesValidation;
    });

    // Default view is newest-first, so a new upload is immediately first.
    // Once a filter is active, group the matching results by type while
    // preserving newest-first order within each type.
    if (type || validation || query) {
        const typeOrder = { balance_sheet: 1, profit_and_loss: 2, cash_flow_statement: 3, invoice: 4 };
        filtered = filtered.sort((a, b) => {
            const typeDiff = (typeOrder[a.document_type] || 99) - (typeOrder[b.document_type] || 99);
            return typeDiff || (new Date(b.created_at || 0) - new Date(a.created_at || 0));
        });
    }

    const pages = Math.max(1, Math.ceil(filtered.length / HISTORY_PAGE_SIZE));
    if (historyPage > pages) historyPage = pages;
    const start = (historyPage - 1) * HISTORY_PAGE_SIZE;
    renderHistory(filtered.slice(start, start + HISTORY_PAGE_SIZE));
    renderHistoryPagination(pages);
}

function renderHistoryPagination(pageCount = Math.max(1, Math.ceil(historyDocuments.length / HISTORY_PAGE_SIZE))) {
    const pagination = document.getElementById('historyPagination');
    const pages = pageCount;
    if (pages <= 1) {
        pagination.innerHTML = '';
        return;
    }

    let html = `<button class="history-page-btn" ${historyPage === 1 ? 'disabled' : ''} data-page="${historyPage - 1}">Previous</button>`;
    for (let page = 1; page <= pages; page += 1) {
        html += `<button class="history-page-btn ${page === historyPage ? 'active' : ''}" data-page="${page}">${page}</button>`;
    }
    html += `<button class="history-page-btn" ${historyPage === pages ? 'disabled' : ''} data-page="${historyPage + 1}">Next</button>`;
    pagination.innerHTML = html;
    pagination.querySelectorAll('.history-page-btn').forEach(button => {
        button.addEventListener('click', () => {
            const page = Number(button.dataset.page);
            if (page >= 1 && page <= pages) {
                historyPage = page;
                renderHistoryPage();
                document.getElementById('historySection').scrollIntoView({ behavior: 'smooth' });
            }
        });
    });
}

function refreshHistoryFromFilters() {
    historyPage = 1;
    renderHistoryPage();
}

historySearch?.addEventListener('input', refreshHistoryFromFilters);
historyTypeFilter?.addEventListener('change', refreshHistoryFromFilters);
historyValidationFilter?.addEventListener('change', refreshHistoryFromFilters);
clearHistoryFilters?.addEventListener('click', () => {
    historySearch.value = '';
    historyTypeFilter.value = '';
    historyValidationFilter.value = '';
    refreshHistoryFromFilters();
});

function renderHistory(documents) {
    const tbody = document.getElementById('historyBody');
    if (documents.length === 0) {
        tbody.innerHTML = '<tr><td colspan="7" class="empty-state">No documents processed yet</td></tr>';
        return;
    }

    tbody.innerHTML = documents.map(doc => {
        const meta = doc.processing_metadata || {};
        const statusClass = doc.processing_status === 'PASS' ? 'status-pass' : 'status-fail';
        // Processing and validation are different questions: a readable
        // document whose arithmetic does not reconcile is PASS / FAIL.
        const vs = doc.validation_status || 'NOT_RUN';
        const vsClass = vs === 'PASS' ? 'status-pass'
            : vs === 'FAIL' ? 'status-fail' : 'status-na';
        const date = doc.created_at ? new Date(doc.created_at).toLocaleString() : '—';

        return `
            <tr class="history-row" data-name="${escapeHtml(doc.document_name)}" title="Click to view full result">
                <td>${escapeHtml(doc.document_name)}</td>
                <td>${escapeHtml(doc.document_type)}</td>
                <td class="${statusClass}">${escapeHtml(doc.processing_status)}</td>
                <td class="${vsClass}">${escapeHtml(vs)}</td>
                <td>${escapeHtml(meta.text_extraction_method || '—')}</td>
                <td>${escapeHtml(meta.processing_time_ms || 0)}ms</td>
                <td>${escapeHtml(date)}</td>
            </tr>
        `;
    }).join('');

    tbody.querySelectorAll('.history-row').forEach(row => {
        row.addEventListener('click', () => {
            const name = row.getAttribute('data-name');
            if (name) toggleHistoryResult(row, name);
        });
    });
}

async function toggleHistoryResult(row, documentName) {
    const existing = row.nextElementSibling;
    if (existing?.classList.contains('history-detail-row')) {
        existing.remove();
        return;
    }

    const detailRow = document.createElement('tr');
    detailRow.className = 'history-detail-row';
    detailRow.innerHTML = '<td colspan="7"><div class="history-detail-loading">Loading stored result...</div></td>';
    row.after(detailRow);

    try {
        const response = await fetch(`/api/v1/documents/${encodeURIComponent(documentName)}`);
        const result = await response.json();
        if (!response.ok) {
            const code = result?.detail?.code || `HTTP ${response.status}`;
            detailRow.querySelector('td').innerHTML = `<div class="history-detail-error">${escapeHtml(String(code))}</div>`;
            return;
        }
        detailRow.querySelector('td').innerHTML = renderInlineHistoryResult(result);
    } catch (err) {
        detailRow.querySelector('td').innerHTML = `<div class="history-detail-error">${escapeHtml(err.message || 'Failed to load stored result')}</div>`;
    }
}

function renderInlineHistoryResult(result) {
    const validation = result.validation || {};
    const renderedValidation = validationMarkup(validation);
    const isPass = result.processing_status === 'PASS';
    const statusClass = isPass ? 'pass' : 'failed';

    return `
        <div class="history-detail">
            <div class="history-detail-header">
                <div>
                    <strong>${isPass ? '✅' : '❌'} ${escapeHtml(result.document_name || 'Stored result')}</strong>
                    <span class="history-detail-status ${statusClass}">${isPass ? 'Processing Successful' : 'Processing Failed'} · Validation: ${escapeHtml(validation.overall_status || 'NOT_RUN')}</span>
                </div>
                <span>${escapeHtml(result.processing_metadata?.processing_time_ms || 0)}ms</span>
            </div>
            <div class="history-detail-tabs">
                <section><h4>Extracted Data</h4><div class="data-grid">${extractedDataMarkup(result.extracted_data, result.document_type)}</div></section>
                <section><h4>Validation</h4><div class="validation-summary">${renderedValidation.summary}</div><div class="validation-checks">${renderedValidation.checks}</div></section>
                <section><h4>Metadata</h4><div class="metadata-grid">${metadataMarkup(result)}</div></section>
                <section><h4>Raw JSON</h4><pre class="json-view">${escapeHtml(JSON.stringify(result, null, 2))}</pre></section>
            </div>
        </div>
    `;
}

// ─── Health Check ───
async function checkHealth() {
    const statusEl = document.getElementById('healthStatus');
    const dot = statusEl.querySelector('.status-dot');
    const text = statusEl.querySelector('.status-text');

    try {
        const response = await fetch('/api/v1/health');
        const data = await response.json();

        if (data.database === 'connected') {
            dot.className = 'status-dot connected';
            text.textContent = 'System Online';
        } else {
            dot.className = 'status-dot disconnected';
            text.textContent = 'DB Disconnected';
        }
    } catch (err) {
        dot.className = 'status-dot disconnected';
        text.textContent = 'Offline';
    }
}

// ─── Utilities ───
function escapeHtml(s) {
    return String(s ?? '')
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#39;');
}

function formatBytes(bytes) {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
}

// ─── Init ───
checkHealth();
loadHistory();
