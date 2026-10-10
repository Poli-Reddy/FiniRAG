/**
 * FiniRAG — Frontend Application Logic
 *
 * Handles: file upload, API calls, result rendering, tab switching,
 * history loading, health check, Q&A with retrieval provenance,
 * per-document indexing status + reindex, operations metrics, and drag & drop.
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
    loadResultIndex(result.document_name);

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
        if (check.tolerance !== null && check.tolerance !== undefined) {
            detail += ` | Tolerance: ${check.tolerance}`;
        }
        if (check.reason) detail += ` | Reason: ${check.reason}`;

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

// ─── Indexing (per-document status + reindex) ───
function indexStatusBadge(status) {
    const cls = status === 'COMPLETED' ? 'status-pass'
        : (status === 'FAILED' ? 'status-fail' : 'status-na');
    return `<span class="${cls}">${escapeHtml(status || 'UNKNOWN')}</span>`;
}

function indexPanelMarkup(st) {
    if (!st || st.error) {
        return `<p class="empty-state">${escapeHtml(st?.error || 'Index status unavailable')}</p>`;
    }
    let html = `<div class="data-group"><div class="data-group-title">Index status</div>`;
    html += `<div class="data-row"><span class="data-label">Status</span><span class="data-value">${indexStatusBadge(st.status)}</span></div>`;
    if (st.chunk_count !== undefined) html += `<div class="data-row"><span class="data-label">Active chunks</span><span class="data-value">${escapeHtml(st.chunk_count)}</span></div>`;
    if (st.index_version) html += `<div class="data-row"><span class="data-label">Index version</span><span class="data-value">${escapeHtml(st.index_version)}</span></div>`;
    if (st.attempt_count !== undefined) html += `<div class="data-row"><span class="data-label">Attempts</span><span class="data-value">${escapeHtml(st.attempt_count)}</span></div>`;
    if (st.job_id) html += `<div class="data-row"><span class="data-label">Job ID</span><span class="data-value"><code>${escapeHtml(st.job_id)}</code></span></div>`;
    if (st.error_code) html += `<div class="data-row"><span class="data-label">Error</span><span class="data-value">${escapeHtml(st.error_code)}${st.error_message ? ` — ${escapeHtml(st.error_message)}` : ''}</span></div>`;
    html += `</div>`;
    if (st.document_id) {
        html += `<button type="button" class="btn-secondary" data-reindex="${escapeHtml(st.document_id)}">Reindex document</button>
                 <div class="index-job-note"></div>`;
    }
    if (st.status === 'PENDING' || st.status === 'PROCESSING') {
        html += `<p class="empty-state">Jobs are processed by the indexing worker (<code>python scripts/run_indexing_worker.py</code>); status updates when it runs.</p>`;
    }
    return html;
}

async function fetchIndexStatus(documentId) {
    const response = await fetch(`/api/v1/documents/${encodeURIComponent(documentId)}/index-status`);
    const data = await response.json();
    if (!response.ok) {
        const code = data?.detail?.code || `HTTP ${response.status}`;
        return { error: String(code) };
    }
    return data;
}

async function loadResultIndex(documentName) {
    const panel = document.getElementById('indexPanel');
    if (!panel) return;
    panel.innerHTML = '<p class="empty-state">Loading index status…</p>';
    try {
        // The process response carries no attempt id; resolve it by name.
        const rec = await fetch(`/api/v1/documents/${encodeURIComponent(documentName)}`);
        const recData = await rec.json();
        if (!rec.ok || !recData.id) {
            panel.innerHTML = '<p class="empty-state">Index status unavailable for this result.</p>';
            return;
        }
        const st = await fetchIndexStatus(recData.id);
        panel.innerHTML = indexPanelMarkup(st);
        wireReindexButtons(panel, () => loadResultIndex(documentName));
    } catch (err) {
        panel.innerHTML = `<p class="empty-state">${escapeHtml(err.message || 'Index status unavailable')}</p>`;
    }
}

function wireReindexButtons(root, rerender) {
    root.querySelectorAll('[data-reindex]').forEach(btn => {
        btn.addEventListener('click', async () => {
            const note = root.querySelector('.index-job-note');
            const docId = btn.getAttribute('data-reindex');
            btn.disabled = true;
            try {
                const response = await fetch(`/api/v1/documents/${encodeURIComponent(docId)}/reindex`, { method: 'POST' });
                const data = await response.json();
                if (!response.ok) {
                    const code = data?.detail?.code || `HTTP ${response.status}`;
                    throw new Error(String(code));
                }
                if (note) note.textContent = `Reindex job ${data.job_id} (${data.status}). Refreshing status…`;
                // Poll the job a few times; indexing is bounded server-side.
                for (let i = 0; i < 4; i++) {
                    await new Promise(r => setTimeout(r, 2000));
                    try {
                        const js = await fetch(`/api/v1/indexing/jobs/${encodeURIComponent(data.job_id)}`);
                        const jd = await js.json();
                        if (js.ok && (jd.status === 'COMPLETED' || jd.status === 'FAILED')) break;
                    } catch (_) { /* keep polling */ }
                }
                await rerender();
            } catch (err) {
                if (note) note.textContent = `Reindex failed: ${err.message || err}`;
            } finally {
                btn.disabled = false;
            }
        });
    });
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
        loadHistoryIndex(detailRow, result.id);
    } catch (err) {
        detailRow.querySelector('td').innerHTML = `<div class="history-detail-error">${escapeHtml(err.message || 'Failed to load stored result')}</div>`;
    }
}

async function loadHistoryIndex(detailRow, documentId) {
    const slot = detailRow.querySelector('[data-index-slot]');
    if (!slot || !documentId) return;
    try {
        const st = await fetchIndexStatus(documentId);
        slot.innerHTML = indexPanelMarkup(st);
        wireReindexButtons(slot, () => loadHistoryIndex(detailRow, documentId));
    } catch (err) {
        slot.innerHTML = `<p class="empty-state">${escapeHtml(err.message || 'Index status unavailable')}</p>`;
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
                <section><h4>Indexing</h4><div data-index-slot><p class="empty-state">Loading index status…</p></div></section>
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
        statusEl.title = healthTooltip(data);
    } catch (err) {
        dot.className = 'status-dot disconnected';
        text.textContent = 'Offline';
        statusEl.title = 'Health endpoint unreachable';
    }
}

function healthTooltip(data) {
    if (!data || typeof data !== 'object') return 'Health unavailable';
    const parts = [
        `status: ${data.status || '?'}`,
        `database: ${data.database || '?'}`,
        `retrieval: ${data.retrieval_mode || '?'}`,
        `embedding: ${data.embedding_model || '?'}${data.embedding_configured ? '' : ' (not configured)'}`,
        `indexed: ${data.indexed_documents ?? '?'} documents / ${data.indexed_chunks ?? '?'} chunks`,
    ];
    const open = (data.provider_breakers || []).filter(b => b.state && b.state !== 'closed');
    if (open.length) parts.push(`breakers open: ${open.map(b => b.provider || '?').join(', ')}`);
    return parts.join(' • ');
}

// ─── Operations ───
function metricCard(label, value) {
    return `<div class="metadata-card"><div class="metadata-label">${escapeHtml(label)}</div>` +
        `<div class="metadata-value">${escapeHtml(value)}</div></div>`;
}

async function loadOps() {
    const healthGrid = document.getElementById('opsHealth');
    const metricsGrid = document.getElementById('opsMetrics');
    try {
        const [healthRes, metricsRes] = await Promise.all([
            fetch('/api/v1/health'),
            fetch('/metrics?format=json'),
        ]);
        const health = await healthRes.json();
        const snap = await metricsRes.json();
        if (!healthRes.ok) throw new Error(`health HTTP ${healthRes.status}`);
        if (!metricsRes.ok) throw new Error(`metrics HTTP ${metricsRes.status}`);

        const breakers = (health.provider_breakers || []).map(b =>
            `${b.provider || '?'}: ${b.state || '?'}`).join(', ') || 'none tripped';
        healthGrid.innerHTML =
            metricCard('Status', health.status || '?') +
            metricCard('Database', health.database || '?') +
            metricCard('Retrieval mode', health.retrieval_mode || '?') +
            metricCard('Embedding model', `${health.embedding_model || '?'}${health.embedding_configured ? '' : ' (not configured)'}`) +
            metricCard('Indexed documents', health.indexed_documents ?? '?') +
            metricCard('Active chunks', health.indexed_chunks ?? '?') +
            metricCard('Index version', health.index_version || '?') +
            metricCard('Provider breakers', breakers);

        metricsGrid.innerHTML = opsMetricsMarkup(snap);
    } catch (err) {
        const msg = `<p class="empty-state">${escapeHtml(err.message || 'Operations data unavailable')}</p>`;
        if (healthGrid) healthGrid.innerHTML = msg;
        if (metricsGrid) metricsGrid.innerHTML = msg;
    }
}

function opsMetricsMarkup(snap) {
    if (!snap || typeof snap !== 'object') return '<p class="empty-state">No metrics</p>';
    const counters = snap.counters || {};
    const gauges = snap.gauges || {};
    const sumCounter = (name) => Object.values(counters[name] || {}).reduce((a, b) => a + Number(b || 0), 0);
    const sumWhere = (name, substr) => Object.entries(counters[name] || {})
        .filter(([k]) => k.includes(substr)).reduce((a, [, b]) => a + Number(b || 0), 0);
    const gaugeVal = (name) => {
        const vals = Object.values(gauges[name] || {});
        return vals.length ? vals.reduce((a, b) => a + Number(b || 0), 0) : '—';
    };
    const answersBy = (outcome) => sumWhere('finirag_answer_requests_total', `'outcome', '${outcome}'`);
    const httpErr = Object.entries(counters.finirag_http_requests_total || {})
        .filter(([k]) => /'status', '([45]\d\d)'/.test(k))
        .reduce((a, [, b]) => a + Number(b || 0), 0);
    return (
        metricCard('HTTP requests', sumCounter('finirag_http_requests_total')) +
        metricCard('HTTP errors (4xx/5xx)', httpErr) +
        metricCard('Answers: answered', answersBy('answered')) +
        metricCard('Answers: abstained', answersBy('abstained')) +
        metricCard('Answer fallbacks', sumCounter('finirag_answer_fallbacks_total')) +
        metricCard('Claims dropped', sumCounter('finirag_claims_dropped_total')) +
        metricCard('Retrieval degraded', sumWhere('finirag_retrieval_requests_total', "'degraded', 'true'")) +
        metricCard('Retrieval empty', sumCounter('finirag_retrieval_empty_total')) +
        metricCard('Index jobs completed', sumWhere('finirag_indexing_jobs_total', "'status', 'completed'")) +
        metricCard('Index jobs failed', sumWhere('finirag_indexing_jobs_total', "'status', 'failed'")) +
        metricCard('Pending index jobs', gaugeVal('finirag_indexing_pending_jobs')) +
        metricCard('Documents processed', sumCounter('finirag_documents_processed_total'))
    );
}

document.getElementById('opsRefreshBtn')?.addEventListener('click', loadOps);

// ─── Job lookup ───
document.getElementById('jobLookupBtn')?.addEventListener('click', async () => {
    const input = document.getElementById('jobIdInput');
    const errorEl = document.getElementById('jobError');
    const resultEl = document.getElementById('jobResult');
    errorEl.style.display = 'none';
    resultEl.innerHTML = '';
    const jobId = input?.value?.trim() || '';
    if (!jobId) {
        errorEl.textContent = 'Enter a job ID.';
        errorEl.style.display = 'block';
        return;
    }
    try {
        const response = await fetch(`/api/v1/indexing/jobs/${encodeURIComponent(jobId)}`);
        const data = await response.json();
        if (!response.ok) {
            const code = data?.detail?.code || `HTTP ${response.status}`;
            throw new Error(String(code));
        }
        let html = '<div class="data-group"><div class="data-group-title">Indexing job</div>';
        const rows = [
            ['Job ID', data.job_id], ['Document ID', data.document_id],
            ['Status', data.status], ['Attempts', data.attempt_count],
            ['Index version', data.index_version],
        ];
        for (const [label, value] of rows) {
            html += `<div class="data-row"><span class="data-label">${escapeHtml(label)}</span><span class="data-value">${escapeHtml(value ?? '—')}</span></div>`;
        }
        if (data.error_code) {
            html += `<div class="data-row"><span class="data-label">Error</span><span class="data-value">${escapeHtml(data.error_code)}${data.error_message ? ` — ${escapeHtml(data.error_message)}` : ''}</span></div>`;
        }
        for (const [label, key] of [['Created', 'created_at'], ['Started', 'started_at'], ['Completed', 'completed_at']]) {
            if (data[key]) html += `<div class="data-row"><span class="data-label">${escapeHtml(label)}</span><span class="data-value">${escapeHtml(new Date(data[key]).toLocaleString())}</span></div>`;
        }
        resultEl.innerHTML = html + '</div>';
    } catch (err) {
        errorEl.textContent = err.message || 'Job lookup failed';
        errorEl.style.display = 'block';
    }
});

// ─── Q&A ───
const qaAskBtn = document.getElementById('qaAskBtn');
qaAskBtn?.addEventListener('click', async () => {
    const question = document.getElementById('qaQuestion')?.value?.trim() || '';
    const document_id = document.getElementById('qaDocId')?.value?.trim() || undefined;
    const document_name = document.getElementById('qaDocName')?.value?.trim() || undefined;
    const document_type = document.getElementById('qaDocType')?.value || undefined;
    const reporting_period = document.getElementById('qaPeriod')?.value?.trim() || undefined;
    const company = document.getElementById('qaCompany')?.value?.trim() || undefined;
    const loading = document.getElementById('qaLoading');
    const errorEl = document.getElementById('qaError');
    const answerEl = document.getElementById('qaAnswer');
    errorEl.style.display = 'none';
    answerEl.style.display = 'none';
    if (!question) {
        errorEl.textContent = 'Enter a question.';
        errorEl.style.display = 'block';
        return;
    }
    loading.style.display = 'block';
    qaAskBtn.disabled = true;
    try {
        const controller = new AbortController();
        const timeout = setTimeout(() => controller.abort(), 60000);
        const response = await fetch('/api/v1/qa/ask', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ question, document_id, document_name, document_type, reporting_period, company }),
            signal: controller.signal,
        });
        clearTimeout(timeout);
        const data = await response.json();
        if (!response.ok) {
            throw new Error(qaErrorMessage(response, data));
        }
        renderQaAnswer(data);
        answerEl.style.display = 'grid';
    } catch (err) {
        errorEl.textContent = err?.name === 'AbortError' ? 'Request timed out.' : (err.message || 'Q&A failed');
        errorEl.style.display = 'block';
    } finally {
        loading.style.display = 'none';
        qaAskBtn.disabled = false;
    }
});

function qaErrorMessage(response, data) {
    const code = data?.detail?.code || data?.error?.code || `HTTP ${response.status}`;
    const msg = data?.detail?.message || data?.error?.message || '';
    // Surface server guidance for throttling and pipeline timeouts.
    if (response.status === 429) {
        const retryAfter = response.headers?.get?.('Retry-After');
        const hint = retryAfter ? ` Retry after ${retryAfter}s.` : ' Please slow down and retry.';
        return `${code}: ${(msg || 'Too many requests.') + hint}`;
    }
    if (code === 'QA_TIMEOUT') {
        return `${code}: ${msg || 'The answer took too long to produce. Please retry.'}`;
    }
    return msg ? `${code}: ${msg}` : String(code);
}

function renderQaAnswer(data) {
    const errorEl = document.getElementById('qaError');
    const meta = data.retrieval_metadata || {};
    // The API contract is a list, but normalize defensively so one shaped
    // response can never blank the whole answer panel.
    const asArray = (v) => Array.isArray(v) ? v : (v ? [v] : []);
    const provider = data.answer_provider || 'unknown';
    document.getElementById('qaProvider').innerHTML =
        `Answered by <strong>${escapeHtml(provider)}</strong>` +
        (asArray(data.intents).length ? ` · intents: ${escapeHtml(asArray(data.intents).join(', '))}` : '') +
        (data.request_id ? ` · request <code>${escapeHtml(data.request_id)}</code>` : '');

    // Retrieval health: never bury a degraded path in raw JSON.
    const degradedBox = document.getElementById('qaDegraded');
    const problems = [];
    if (meta.degraded) problems.push('retrieval was degraded');
    if (meta.semantic_available === false) problems.push('semantic search unavailable — keyword retrieval only');
    const nativeErrs = meta.native_errors && typeof meta.native_errors === 'object'
        ? Object.entries(meta.native_errors).map(([k, v]) => `${k}: ${v}`).join(', ') : '';
    if (nativeErrs) problems.push(`native retrieval errors (${nativeErrs})`);
    if (problems.length) {
        degradedBox.textContent = `Warning: ${problems.join('; ')}. Treat the answer with extra caution.`;
        degradedBox.style.display = 'block';
    } else {
        degradedBox.style.display = 'none';
        const paths = [meta.vector_path, meta.keyword_path].filter(Boolean).join(' + ');
        if (paths) {
            degradedBox.textContent = `Retrieval: ${paths}.`;
            degradedBox.className = 'qa-degraded qa-degraded-ok';
            degradedBox.style.display = 'block';
        } else {
            degradedBox.className = 'qa-degraded';
        }
    }

    document.getElementById('qaAnswerText').textContent = data.answer || '';
    document.getElementById('qaCitations').innerHTML = (asArray(data.citations).map(c => {
        const kind = c.evidence_kind === 'verbatim' ? 'verbatim' : 'derived';
        const badge = `<span class="prov-badge ${kind}">${kind === 'verbatim' ? 'verbatim quote' : 'derived summary'}</span>`;
        return `<div class="cite-row"><span>${escapeHtml(c.citation_id)} — ${escapeHtml(c.document_name || '')} p.${escapeHtml(c.page_number ?? '')}</span>${badge}</div>`;
    }).join('')) || '<div>None</div>';
    document.getElementById('qaFacts').innerHTML = (asArray(data.financial_facts).map(f => {
        if (typeof f === 'string') return `<div>${escapeHtml(f)}</div>`;
        return `<div class="data-row"><span class="data-label">${escapeHtml(f.field || 'fact')}</span>` +
            `<span class="data-value">${escapeHtml(f.value ?? '—')}${f.unit ? ` (${escapeHtml(f.unit)})` : ''}${f.status ? ` [${escapeHtml(f.status)}]` : ''}</span></div>`;
    }).join('')) || '<div>None</div>';
    document.getElementById('qaCalcs').innerHTML = (asArray(data.calculations).map(c => {
        const periods = Array.isArray(c.periods) ? c.periods.join(' → ') : '';
        return `<div class="data-row"><span class="data-label">${escapeHtml(c.field || 'calculation')}${periods ? ` (${escapeHtml(periods)})` : ''}</span>` +
            `<span class="data-value">Δ ${escapeHtml(c.difference ?? '—')}${c.unit ? ` (${escapeHtml(c.unit)})` : ''}</span></div>`;
    }).join('')) || '<div>None</div>';
    document.getElementById('qaValidation').innerHTML = (asArray(data.validation_checks).map(v =>
        `<div class="check-item"><span class="check-badge ${(v?.status === 'PASS' ? 'pass' : v?.status === 'FAIL' ? 'fail' : 'na')}">${escapeHtml(v?.status || 'N/A')}</span>` +
        `<div><div class="check-name">${escapeHtml(v?.name || '')}</div>` +
        (v?.formula ? `<div class="check-detail">${escapeHtml(v.formula)}</div>` : '') + `</div></div>`
    ).join('')) || '<div>None</div>';
    document.getElementById('qaLimits').innerHTML = (asArray(data.limitations).map(l =>
        `<div>• ${escapeHtml(typeof l === 'string' ? l : JSON.stringify(l))}</div>`).join('')) || '<div>None</div>';
    document.getElementById('qaMeta').textContent = JSON.stringify({
        status: data.status, intents: data.intents, retrieval_metadata: data.retrieval_metadata,
    }, null, 2);
    if (data.status === 'INSUFFICIENT_EVIDENCE') {
        errorEl.textContent = 'Insufficient evidence — the answer was withheld.';
        errorEl.style.display = 'block';
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
loadOps();
