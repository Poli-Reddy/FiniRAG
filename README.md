# FiniRAG

FiniRAG is an AI-powered financial document extraction, validation, and
persistence platform for invoices, balance sheets, profit and loss statements,
and cash flow statements.

## Quick Start

```powershell
python -m venv venv
venv/Scripts/python.exe -m pip install -r backend/requirements.txt
Copy-Item .env.example .env
venv/Scripts/python.exe -m uvicorn backend.app.main:app --reload
```

Open the dashboard at `http://127.0.0.1:8000/`, Swagger at

## Project Guide

The complete installation, architecture, API, validation, persistence,
deployment, testing, limitations, production-improvement, and AI usage guide
is in [`docs/README.md`](docs/README.md).

## Main Components

- `backend/`: FastAPI API, OCR, extraction, normalization, validation, and PostgreSQL persistence.
- `frontend/`: HTML/CSS/JavaScript dashboard served by FastAPI.
- `Dataset/`: supplied development and evaluation documents.
- `evaluation/`: measured dataset results and reproducible evaluation scripts.
- `sample_outputs/`: representative structured JSON outputs.
- `scripts/`: deployment smoke check and PostgreSQL evaluation-result seeder.
- `docs/`: architecture diagram and solution presentation placement.

## Test

```powershell
python -m pytest backend/tests -q
```

The project has been evaluated against all 50 supplied documents. The current
measured state is 50/50 processing success, with intentional validation FAIL
results retained for the three inconsistent invoice trap documents.

## Deployment

Render configuration is provided in `render.yaml`. The same Render web service
serves the frontend, REST API, Swagger, and health endpoint. Render PostgreSQL
provides persistence. Provider credentials must be configured as Render
environment variables and must never be committed.
