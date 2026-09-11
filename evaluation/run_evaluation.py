"""
FiniRAG dataset evaluation harness (PHASE 5 — "Run the entire real dataset").

Runs every file in `Dataset/` through the *real* pipeline — the same
`DocumentService.process_document` the API calls — and writes one JSON file
per document under `evaluation/results/`.

Two deliberate differences from a live API request:

  1. **Nothing is persisted.** The repository is replaced with a recorder that
     returns the row unsaved. The evaluation is a measurement of extraction and
     validation quality, and writing 54 synthetic rows into the dashboard the
     user actually reads would corrupt it. Persistence has its own tests
     against the live database (`backend/tests/test_repository.py -m integration`).

  2. **Four synthetic edge cases are appended.** The dataset contains only
     well-formed documents, but the file gate is half the requirement, so an
     unsupported type, an empty file, a corrupted PDF and a 4-page PDF are
     generated here and pushed through the same entry point.

Documents are processed one at a time on purpose: `generate_content` is a
blocking call, so `asyncio.gather` would not overlap anything, and a serial
run keeps well inside the Gemini rate limit.

The run is resumable — a document whose result JSON already exists is skipped
unless `--force` is passed. Re-running after a crash costs nothing.

If the LLM provider starts refusing calls on quota, the run stops rather than
recording the refusals: a throttled document says nothing about the document,
and writing it would let the next run skip a file that was never read. A
document the fallback truncates is likewise not recorded, but the run keeps
going — truncation depends on the document's size, not on time passing.

Usage:
    ./venv/Scripts/python.exe evaluation/run_evaluation.py
    ./venv/Scripts/python.exe evaluation/run_evaluation.py --force
    ./venv/Scripts/python.exe evaluation/run_evaluation.py --only "Cash Flow"

Exit codes: 0 = finished, 2 = stopped early on provider quota.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import fitz  # PyMuPDF

from backend.app.core.logging import setup_logging  # noqa: E402  (configures "finirag")
from backend.app.services.document_service import DocumentService  # noqa: E402

RESULTS_DIR = Path(__file__).resolve().parent / "results"
DATASET_DIR = ROOT / "Dataset"

# Folder name → the document type a user would pick in the UI. The pipeline
# never classifies (Trap P10); the type is always supplied by the caller.
FOLDER_TO_TYPE = {
    "Balance Sheet": "balance_sheet",
    "Cash Flows": "cash_flow_statement",
    "Profit & Loss": "profit_and_loss",
    "Invoices": "invoice",
}

logger = logging.getLogger("finirag")


# ─────────────────────────────────────────────────────────────────────
# No-persist repository
# ─────────────────────────────────────────────────────────────────────

class NoPersistRepository:
    """
    Stands in for DocumentRepository so the evaluation writes no rows.

    `create` returns the Document unchanged, which is all the pipeline does
    with it — the response is built from the in-memory objects, not from a
    round trip through the database.
    """

    def __init__(self, db=None):
        self.created = []

    async def create(self, document):
        self.created.append(document)
        return document


# ─────────────────────────────────────────────────────────────────────
# Synthetic edge cases (file-gate coverage)
# ─────────────────────────────────────────────────────────────────────

def _pdf_with_pages(count: int) -> bytes:
    """A minimal but genuinely valid PDF of `count` pages."""
    doc = fitz.open()
    for i in range(count):
        page = doc.new_page()
        page.insert_text((72, 72), f"Edge case page {i + 1}")
    data = doc.tobytes()
    doc.close()
    return data


def edge_cases() -> list[tuple[str, str, bytes]]:
    """(filename, document_type, bytes) for each file-gate scenario."""
    valid_pdf = _pdf_with_pages(1)

    return [
        # Unsupported extension and unsupported magic bytes.
        ("edge_unsupported_type.txt", "invoice", b"Total: 100.00\nTax: 6.00\n"),
        # Zero bytes — a browser will happily upload one.
        ("edge_empty_file.pdf", "invoice", b""),
        # A real PDF header followed by a truncated body.
        ("edge_corrupted.pdf", "balance_sheet", valid_pdf[: len(valid_pdf) // 3]),
        # One page over the documented MAX_PAGES=3 limit.
        ("edge_four_pages.pdf", "cash_flow_statement", _pdf_with_pages(4)),
    ]


# ─────────────────────────────────────────────────────────────────────
# Run
# ─────────────────────────────────────────────────────────────────────

def slug(name: str) -> str:
    """A filesystem-safe result name that still reads like the document."""
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in name)


def dataset_files() -> list[tuple[str, str, Path]]:
    """(filename, document_type, path) for every dataset file, sorted."""
    found = []
    for folder, doc_type in FOLDER_TO_TYPE.items():
        directory = DATASET_DIR / folder
        if not directory.is_dir():
            logger.warning("Dataset folder missing", extra={"folder": folder})
            continue
        for path in sorted(directory.iterdir()):
            if path.is_file():
                found.append((path.name, doc_type, path))
    return found


async def process_one(
    service: DocumentService,
    filename: str,
    doc_type: str,
    file_bytes: bytes,
    source: str,
) -> dict:
    """Run one document and return the JSON-ready result envelope."""
    started = time.time()
    try:
        response = await service.process_document(file_bytes, filename, doc_type)
        payload = json.loads(response.model_dump_json())
        payload["_harness"] = {
            "source": source,
            "file_size_bytes": len(file_bytes),
            "wall_time_ms": int((time.time() - started) * 1000),
            "crashed": False,
        }
        return payload
    except Exception as exc:
        # A crash is a result too — it belongs in the report rather than
        # stopping the run 12 documents in.
        logger.error("Harness caught an unhandled error", extra={
            "document_name": filename,
            "error_type": type(exc).__name__,
        })
        return {
            "document_name": filename,
            "document_type": doc_type,
            "processing_status": "CRASHED",
            "file_validation": None,
            "extracted_data": None,
            "validation": None,
            "processing_metadata": None,
            "error": {"code": "HARNESS_UNHANDLED", "message": type(exc).__name__},
            "_harness": {
                "source": source,
                "file_size_bytes": len(file_bytes),
                "wall_time_ms": int((time.time() - started) * 1000),
                "crashed": True,
                "traceback": traceback.format_exc(),
            },
        }


async def main() -> int:
    parser = argparse.ArgumentParser(description="Run the dataset through FiniRAG.")
    parser.add_argument("--force", action="store_true",
                        help="Re-run documents that already have a result file.")
    parser.add_argument("--only", default="",
                        help="Substring filter on the filename.")
    parser.add_argument("--skip-edge-cases", action="store_true",
                        help="Dataset documents only.")
    args = parser.parse_args()

    # The Windows console defaults to cp1252, which cannot encode a document
    # name containing a currency sign or a dash. Without this the harness dies
    # on the print, not on the document.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    setup_logging()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    service = DocumentService(db=None)
    service._repo = NoPersistRepository()

    work: list[tuple[str, str, bytes, str]] = []

    for filename, doc_type, path in dataset_files():
        if args.only and args.only.lower() not in filename.lower():
            continue
        work.append((filename, doc_type, path.read_bytes(), "dataset"))

    if not args.skip_edge_cases:
        for filename, doc_type, data in edge_cases():
            if args.only and args.only.lower() not in filename.lower():
                continue
            work.append((filename, doc_type, data, "edge_case"))

    total = len(work)
    print(f"[harness] {total} documents queued", flush=True)

    done = failures = skipped = 0
    throttled = 0
    truncated: list[str] = []
    run_started = time.time()

    for index, (filename, doc_type, data, source) in enumerate(work, start=1):
        out_path = RESULTS_DIR / f"{slug(filename)}.json"

        if out_path.exists() and not args.force:
            skipped += 1
            print(f"[{index}/{total}] SKIP (already done) {filename}", flush=True)
            continue

        print(f"[{index}/{total}] {filename} ({doc_type}) ...", flush=True)
        payload = await process_one(service, filename, doc_type, data, source)

        status = payload["processing_status"]
        error_code = (payload.get("error") or {}).get("code")

        # Neither a throttled nor a truncated document is a measurement.
        # Writing one would make the run resumable *past* it — the next run
        # would skip a document that was never actually read — and the report
        # would count a provider limit as an extraction failure. This is
        # exactly what happened on 2026-09-10: 43 readable documents recorded
        # as failures because the daily Gemini quota ran out 6 documents in.
        #
        # They are handled differently, though. Throttling clears with time,
        # so the run stops and waits. Truncation does not: the document is
        # simply larger than the fallback's output ceiling, and the next
        # document may well fit, so the run continues.
        if error_code == "LLM_RATE_LIMITED":
            throttled += 1
            print(f"    -> THROTTLED, not recorded. Stopping.", flush=True)
            print(
                "\n[harness] The LLM provider is refusing calls on quota.\n"
                "[harness] Nothing was written for this document, so re-running\n"
                "[harness] later picks up exactly here. Documents already\n"
                "[harness] completed are kept.",
                flush=True,
            )
            break

        if error_code == "LLM_OUTPUT_TRUNCATED":
            truncated.append(filename)
            print("    -> TRUNCATED at the provider's output ceiling, "
                  "not recorded. Continuing.", flush=True)
            continue

        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        validation = (payload.get("validation") or {}).get("overall_status", "n/a")
        elapsed = payload["_harness"]["wall_time_ms"]
        print(f"    -> {status} | validation={validation} | {elapsed} ms", flush=True)

        done += 1
        if status != "PASS":
            failures += 1

    minutes = (time.time() - run_started) / 60
    print(
        f"\n[harness] processed={done} skipped={skipped} "
        f"non-PASS={failures} in {minutes:.1f} min",
        flush=True,
    )
    print(f"[harness] results in {RESULTS_DIR}", flush=True)
    if truncated:
        print(f"[harness] {len(truncated)} document(s) exceeded the fallback's "
              f"output ceiling and were not recorded:", flush=True)
        for name in truncated:
            print(f"[harness]   - {name}", flush=True)
        print("[harness] Re-run these once the primary provider's quota "
              "resets.", flush=True)
    if throttled:
        remaining = total - done - skipped
        print(f"[harness] STOPPED EARLY on provider quota — {remaining} document(s) "
              f"still to run.", flush=True)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
