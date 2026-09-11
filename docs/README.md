# FiniRAG Installation and Submission Guide

## 1. Solution Overview

FiniRAG is an AI-powered financial document extraction and validation platform.
It accepts invoices, balance sheets, profit and loss statements, and cash flow
statements as PDF, JPG, or PNG files. The system validates the file, extracts
text with OCR or native PDF processing, uses structured LLM extraction,
normalizes values, performs deterministic financial checks, stores the result
in PostgreSQL, and displays it through the dashboard and REST API.

The processing flow is:

```text
Upload
  -> File validation
  -> Native text or OCR
  -> Structured AI extraction
  -> Normalization and evidence
  -> Deterministic financial validation
  -> PostgreSQL persistence
  -> Dashboard and REST response
```

## 2. Architecture

```text
User
 |
 v
FastAPI-served HTML/CSS/JavaScript frontend
 |
 v
POST /api/v1/documents/process
 |
 v
Document validation
 |
 +--> Native PDF text extraction
 |
 +--> Mistral OCR for scanned/image documents
 |
 v
Gemini structured extraction
 |
 v
Groq fallback extraction
 |
 v
Normalization + evidence
 |
 v
Decimal financial validation
 |
 v
PostgreSQL JSONB persistence
 |
 +--> Dashboard history
 +--> GET document result API
 +--> Swagger/OpenAPI
```

The frontend and backend are hosted by the same Render web service. A separate
frontend service is not required.

Place the final visual architecture diagram in this directory as:

- `architecture.png`, or
- `architecture.pdf`

## 3. Technology Stack

- Python 3.12.10
- FastAPI 0.115.0
- Uvicorn 0.30.6
- SQLAlchemy 2.0.35 with async PostgreSQL support
- PostgreSQL with JSONB result storage
- PyMuPDF 1.24.10
- Pillow 10.4.0
- Mistral OCR SDK 2.10.0
- Google Generative AI SDK 0.8.2
- Groq SDK 1.7.0
- Pydantic 2.11.2
- HTML, CSS, and browser JavaScript frontend
- Render web service and Render PostgreSQL

## 4. Why These Technologies Were Selected

- **FastAPI** provides typed request handling, multipart upload support, async
  service integration, and automatic Swagger/OpenAPI documentation.
- **HTML/CSS/JavaScript** keeps the frontend simple, deployable with the API,
  and aligned with the assignment, which does not require React or Node.js.
- **PostgreSQL** provides durable hosted persistence and JSONB for flexible
  document-specific fields, tables, periods, and evidence.
- **Mistral OCR** provides OCR text plus layout blocks and page coordinates for
  scanned documents and evidence support.
- **Gemini** is the primary structured extraction model.
- **Groq** is the fallback provider for provider quota or transport failures.
- **PyMuPDF** handles PDF validation, page counts, rendering, and native text
  probing.
- **Pydantic** provides consistent machine-readable schemas and API contracts.
- **Decimal arithmetic** prevents floating-point errors in financial checks.

## 5. Local Setup

Requirements:

- Python 3.12+
- PostgreSQL
- Gemini, Groq, and Mistral provider credentials

From the project root:

```powershell
python -m venv venv
venv/Scripts/python.exe -m pip install -r backend/requirements.txt
Copy-Item .env.example .env
```

Set the values in `.env`, then run:

```powershell
venv/Scripts/python.exe -m uvicorn backend.app.main:app --reload
```

Open:

```text
http://127.0.0.1:8000/
http://127.0.0.1:8000/docs
http://127.0.0.1:8000/api/v1/health
```

Run the automated tests:

```powershell
venv/Scripts/python.exe -m pytest backend/tests -q
```

The normal test run excludes live PostgreSQL integration tests. Those tests
can be run with the integration marker when the database is available.

## 6. Environment Variables

Use `.env.example` as the template. Never commit `.env` or real credentials.

Required production variables:

```env
DATABASE_URL=postgresql+asyncpg://user:password@host:5432/database
GEMINI_API_KEY=your_gemini_key
GROQ_API_KEY=your_groq_key
MISTRAL_API_KEY=your_mistral_key
OCR_PROVIDER=mistral
LLM_PROVIDER=gemini
ENVIRONMENT=production
LOG_LEVEL=INFO
MAX_FILE_SIZE_MB=10
MAX_PAGES=3
```

Render supplies `DATABASE_URL` from the Render PostgreSQL database. Provider
keys are entered through the Render environment settings and are not stored
in GitHub.

## 7. Frontend URL

After Render deployment:

```text
https://YOUR-APP.onrender.com/
```

The frontend is served directly by FastAPI from `frontend/templates` and
`frontend/static`.

## 8. Backend URL

The backend base URL is the same Render service:

```text
https://YOUR-APP.onrender.com
```

## 9. Swagger URL

FastAPI automatically serves Swagger UI at:

```text
https://YOUR-APP.onrender.com/docs
```

## 10. API Examples

### Process a document

```bash
curl -X POST https://YOUR-APP.onrender.com/api/v1/documents/process \
  -F "file=@Dataset/Invoices/batch3-1354.jpg" \
  -F "document_type=invoice"
```

Supported `document_type` values:

```text
invoice
balance_sheet
profit_and_loss
cash_flow_statement
```

### List persisted documents

```text
GET /api/v1/documents?limit=100&offset=0
```

### Retrieve the latest result by document name

```text
GET /api/v1/documents/Consolidated%20Balance%20Sheet%202026.pdf
```

If the same document name is processed multiple times, the newest attempt is
returned. Older attempts remain available in the database-backed history.

### Health check

```text
GET /api/v1/health
```

Expected healthy response:

```json
{
  "status": "healthy",
  "database": "connected",
  "timestamp": "..."
}
```

### Controlled error response

```json
{
  "error": {
    "code": "UNSUPPORTED_FILE_TYPE",
    "message": "Only PDF / JPG / PNG documents are supported."
  }
}
```

## 11. OCR Approach

The service first validates the upload and probes PDFs for usable native text.
If native text is unavailable, PDF pages are rendered and image documents are
preprocessed with EXIF orientation correction. Mistral OCR then extracts page
text and layout blocks.

The OCR path supports:

- Native PDFs
- Scanned PDFs
- JPEG invoices and receipts
- PNG documents
- Multi-page documents up to three pages
- Rotated images with EXIF orientation metadata
- Parenthesized financial values and OCR-noisy layouts

OCR output is treated as source material, not as unquestioned truth. The LLM
extracts structured fields, while deterministic validation checks arithmetic.

## 12. LLM / Model Used

- **Primary:** Gemini using `GEMINI_API_KEY`
- **Fallback:** Groq using `GROQ_API_KEY`
- **OCR:** Mistral OCR using `MISTRAL_API_KEY`

The LLM returns structured document-type-specific JSON. It is instructed to:

- Extract all visible meaningful fields
- Preserve original values
- Preserve comparative periods
- Preserve line items and tables
- Return null for genuinely missing data
- Avoid inventing or silently correcting unreadable values
- Include evidence source text and page numbers where available

The LLM does not perform trusted financial arithmetic. Python Decimal
validation performs the financial checks after extraction.

## 13. Financial Validation Logic

### Invoice

- Quantity × unit price, adjusted for an explicit line discount, compared with
  line amount
- Sum of line amounts compared with subtotal or total where applicable
- Tax-inclusive and tax-exclusive interpretations supported
- Subtotal + tax compared with total when appropriate
- Cash paid − total compared with change
- Signed discounts handled by magnitude so a printed `-5.59` discount reduces
  the line amount by `5.59`

### Balance Sheet

- Total assets compared with total capital and liabilities
- Asset component sums compared with total assets
- Capital/liability component sums compared with total capital and liabilities
- Checks performed independently per extracted period

### Profit and Loss

- Interest earned + other income compared with total income
- Interest expended + operating expenses + provisions compared with total
  expenditure
- Total income − total expenditure compared with profit before minority
- Minority interest and associates handled where present
- Appropriation transfers reconciled with total available profit
- Carried-forward residual rows excluded from the transfer sum and disclosed

### Cash Flow

- Operating + investing + financing + FX compared with net increase
- Opening cash + net increase + applicable amalgamation adjustment compared with
  closing cash
- Parenthesized values normalized as negative
- Multi-page statement rows preserved
- Intermediate subtotal rows excluded from component sums and disclosed

Missing operands produce `NOT_APPLICABLE`, not an invented zero.

## 14. Numerical Tolerance

All financial checks use Python `Decimal` and:

```text
max(0.01, abs(reported_value) * 0.001)
```

This provides a one-cent absolute floor and a 0.1% relative tolerance for
rounding differences at larger scales. Every check includes its formula,
operands, calculated value, reported value, variance, tolerance, and status.

## 15. Database / Persistence

Render PostgreSQL stores one row per processing attempt in the `documents`
table. JSONB columns hold:

- File validation
- Complete extracted data
- Validation checks
- Processing metadata
- Error information

The database also stores document name, type, processing status, UUID, and
timestamp. The dashboard loads persisted records through
`GET /api/v1/documents`, and clicking a history row loads the latest complete
result by name.

The evaluation harness intentionally bypasses persistence to measure the
pipeline without polluting dashboard history. The measured results can be
seeded with:

```powershell
venv/Scripts/python.exe scripts/seed_evaluation_results.py
```

## 16. Known Limitations

- Evidence guarantees source text and page number. The schema supports
  bounding boxes, but field-to-OCR-block bounding-box matching is not reliable
  for every extracted field yet.
- Provider free-tier quotas can affect latency or availability.
- Render free services can sleep and have cold-start latency.
- History filters operate in the browser over the loaded dashboard result set.
- Authentication, authorization, rate limiting, migrations, background queues,
  and distributed quota state are not included in this case-study prototype.
- Confidence is intentionally omitted because arbitrary LLM confidence values
  are not a defensible measurement.

## 17. Production Improvements

- Add authentication, authorization, upload rate limiting, and audit logging.
- Add Alembic migrations instead of startup `create_all`.
- Store OCR block references and implement verified field-to-block matching.
- Add a background job queue for long-running multi-page processing.
- Add provider circuit breakers and distributed key/quota state.
- Add request timeouts and per-provider observability dashboards.
- Add staging deployment and end-to-end tests against deployed services.
- Use managed secret rotation and separate production/staging databases.

## 18. AI Tools Used

Generative AI assistants were used for codebase exploration, implementation
assistance, debugging, test generation, evaluation analysis, and documentation
drafting. The final pipeline architecture, prompts, financial formulas,
security behavior, deployment configuration, and evaluation results were
reviewed against the assignment and verified with automated tests and the
50-document evaluation dataset.

## Submission Files

Place the final submission artifacts in this directory:

- `architecture.png` or `architecture.pdf`
- `solution_presentation.pdf`
- Optional editable source: `solution_presentation.pptx`

The application runtime does not read these files. They are repository and
submission documentation only.
