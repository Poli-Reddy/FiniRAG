"""
Turn `evaluation/results/*.json` into the dataset evaluation report.

Reads the raw pipeline responses written by run_evaluation.py and produces:

  evaluation/dataset_report.md  — the per-document table required by
                                  master_prompt.md (filename, document type,
                                  extraction status, validation status, fields
                                  extracted, tables, line items, missing
                                  fields, evidence coverage)
  evaluation/summary.json       — the same numbers as data, for the tolerance
                                  analysis and the README to reuse

Nothing here calls the pipeline or the LLM. It only counts what is already on
disk, so it is cheap to re-run after editing the presentation.

Usage:
    ./venv/Scripts/python.exe evaluation/build_report.py
"""

from __future__ import annotations

import json
import statistics
import sys
from collections import Counter
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
RESULTS_DIR = EVAL_DIR / "results"

# Which dynamic dicts hold document rows rather than fixed schema fields.
# A "line item" is a row the document itself labelled — an invoice line, a
# balance sheet asset row, a cash flow activity line — as opposed to a field
# the schema always has a slot for. Counting fixed slots as line items would
# make every document look equally detailed.
LINE_ITEM_CONTAINERS = {
    "line_items",             # invoice
    "assets",                 # balance sheet
    "capital_and_liabilities",
    "appropriations",         # P&L
    "operating_items",        # cash flow
    "investing_items",
    "financing_items",
}

EXTRACTED_STATUSES = {"EXPLICIT", "EXPLICIT_ZERO"}


# ─────────────────────────────────────────────────────────────────────
# Walking an extraction
# ─────────────────────────────────────────────────────────────────────

def is_extracted_value(node) -> bool:
    """True for a serialized ExtractedValue, whatever schema it came from."""
    return (
        isinstance(node, dict)
        and "status" in node
        and "evidence" in node
        and "original_value" in node
    )


def walk_fields(node, path: str = ""):
    """Yield (dotted_path, extracted_value_dict) for every field in a tree."""
    if is_extracted_value(node):
        yield path, node
        return

    if isinstance(node, dict):
        for key, child in node.items():
            yield from walk_fields(child, f"{path}.{key}" if path else str(key))
    elif isinstance(node, list):
        for i, child in enumerate(node):
            yield from walk_fields(child, f"{path}[{i}]")


def count_line_items(node) -> int:
    """Total document-labelled rows across every period and section."""
    total = 0
    if isinstance(node, dict):
        for key, child in node.items():
            if key in LINE_ITEM_CONTAINERS and isinstance(child, (dict, list)):
                total += len(child)
            else:
                total += count_line_items(child)
    elif isinstance(node, list):
        for child in node:
            total += count_line_items(child)
    return total


def count_tables(node) -> int:
    """Supplementary tables (GST/VAT summaries) captured alongside the fields."""
    total = 0
    if isinstance(node, dict):
        for key, child in node.items():
            if key == "additional_tables" and isinstance(child, list):
                total += len(child)
            else:
                total += count_tables(child)
    elif isinstance(node, list):
        for child in node:
            total += count_tables(child)
    return total


def leaf(path: str) -> str:
    """The readable tail of a dotted path, for the missing-field list."""
    return path.split(".")[-1]


# ─────────────────────────────────────────────────────────────────────
# Per-document metrics
# ─────────────────────────────────────────────────────────────────────

def measure(payload: dict) -> dict:
    """Every number the report needs about one document."""
    extracted = payload.get("extracted_data")
    validation = payload.get("validation") or {}
    metadata = payload.get("processing_metadata") or {}
    file_validation = payload.get("file_validation") or {}
    harness = payload.get("_harness", {})

    fields = list(walk_fields(extracted)) if extracted else []
    by_status = Counter(f["status"] for _, f in fields)

    usable = [(p, f) for p, f in fields if f["status"] in EXTRACTED_STATUSES]
    with_source = [
        (p, f) for p, f in usable
        if (f.get("evidence") or {}).get("source_text", "").strip()
    ]

    summary = validation.get("summary") or {}

    return {
        "filename": payload.get("document_name", ""),
        "source": harness.get("source", "dataset"),
        "document_type": payload.get("document_type", ""),
        "extraction_status": payload.get("processing_status", ""),
        "validation_status": validation.get("overall_status") or "NOT_RUN",
        "error_code": (payload.get("error") or {}).get("code"),

        "file_type": file_validation.get("file_type", ""),
        "page_count": file_validation.get("page_count", 0),
        "extraction_method": metadata.get("text_extraction_method", ""),
        "ocr_provider": metadata.get("ocr_provider"),
        "processing_time_ms": metadata.get("processing_time_ms", 0),

        "fields_total": len(fields),
        "fields_extracted": len(usable),
        "fields_missing": by_status.get("MISSING", 0),
        "fields_unreadable": by_status.get("UNREADABLE", 0),
        "fields_explicit_zero": by_status.get("EXPLICIT_ZERO", 0),
        "missing_fields": sorted({leaf(p) for p, f in fields if f["status"] == "MISSING"}),
        "unreadable_fields": sorted({leaf(p) for p, f in fields if f["status"] == "UNREADABLE"}),

        "tables": count_tables(extracted) if extracted else 0,
        "line_items": count_line_items(extracted) if extracted else 0,

        "evidence_with_source_text": len(with_source),
        "evidence_coverage_pct": (
            round(100 * len(with_source) / len(usable), 1) if usable else None
        ),

        "checks_pass": summary.get("pass", 0),
        "checks_fail": summary.get("fail", 0),
        "checks_not_applicable": summary.get("not_applicable", 0),
    }


# ─────────────────────────────────────────────────────────────────────
# Report rendering
# ─────────────────────────────────────────────────────────────────────

def pct(part: int, whole: int) -> str:
    return f"{100 * part / whole:.1f}%" if whole else "n/a"


def fmt_missing(names: list[str], limit: int = 4) -> str:
    if not names:
        return "none"
    shown = ", ".join(names[:limit])
    return shown if len(names) <= limit else f"{shown} (+{len(names) - limit} more)"


def render(rows: list[dict]) -> str:
    dataset = [r for r in rows if r["source"] == "dataset"]
    edge = [r for r in rows if r["source"] == "edge_case"]

    lines: list[str] = []
    add = lines.append

    add("# FiniRAG — Dataset Evaluation Report")
    add("")
    add("Generated by `evaluation/build_report.py` from the raw pipeline")
    add("responses in `evaluation/results/`. Every row is a real run of")
    add("`DocumentService.process_document` — the same entry point the API uses.")
    add("")
    add("Persistence is the one thing stubbed out: the evaluation writes no rows")
    add("to the database, so the dashboard keeps showing the user's own uploads.")
    add("Persistence is covered separately by 14 integration tests that write and")
    add("read real rows against the live PostgreSQL instance")
    add("(`pytest backend/tests/test_repository.py -m integration`).")
    add("")

    # ── Headline ──
    passed = [r for r in dataset if r["extraction_status"] == "PASS"]
    val_pass = [r for r in dataset if r["validation_status"] == "PASS"]
    val_fail = [r for r in dataset if r["validation_status"] == "FAIL"]

    add("## Headline")
    add("")
    add("| Metric | Value |")
    add("|---|---|")
    add(f"| Dataset documents processed | {len(dataset)} |")
    add(f"| Processing status PASS | {len(passed)} ({pct(len(passed), len(dataset))}) |")
    add(f"| Validation overall PASS | {len(val_pass)} ({pct(len(val_pass), len(dataset))}) |")
    add(f"| Validation overall FAIL | {len(val_fail)} ({pct(len(val_fail), len(dataset))}) |")
    add(f"| Fields extracted (total) | {sum(r['fields_extracted'] for r in dataset):,} |")
    add(f"| Line items extracted (total) | {sum(r['line_items'] for r in dataset):,} |")
    add(f"| Arithmetic checks run | {sum(r['checks_pass'] + r['checks_fail'] for r in dataset):,} |")
    add(f"| Arithmetic checks passed | {sum(r['checks_pass'] for r in dataset):,} |")
    add(f"| Arithmetic checks failed | {sum(r['checks_fail'] for r in dataset):,} |")
    add(f"| Checks not applicable | {sum(r['checks_not_applicable'] for r in dataset):,} |")
    add("")

    times = [r["processing_time_ms"] for r in dataset if r["processing_time_ms"]]
    if times:
        add("### Latency")
        add("")
        add("Every document currently reaches the LLM vision fallback because")
        add("the Vision API is disabled on the GCP project, so these are")
        add("vision-path timings, not OCR-path timings. See *Known limitations*")
        add("in the README.")
        add("")
        add(f"- median **{statistics.median(times) / 1000:.1f} s**")
        add(f"- fastest **{min(times) / 1000:.1f} s**, slowest **{max(times) / 1000:.1f} s**")
        add(f"- total wall time **{sum(times) / 60000:.0f} min** across {len(times)} documents")
        add("")

    # ── By document type ──
    add("## By document type")
    add("")
    add("| Type | Docs | Proc PASS | Valid PASS | Valid FAIL | Fields | Line items | Checks P/F/NA |")
    add("|---|---|---|---|---|---|---|---|")
    for doc_type in ("balance_sheet", "profit_and_loss", "cash_flow_statement", "invoice"):
        group = [r for r in dataset if r["document_type"] == doc_type]
        if not group:
            continue
        add(
            f"| `{doc_type}` | {len(group)} "
            f"| {sum(1 for r in group if r['extraction_status'] == 'PASS')} "
            f"| {sum(1 for r in group if r['validation_status'] == 'PASS')} "
            f"| {sum(1 for r in group if r['validation_status'] == 'FAIL')} "
            f"| {sum(r['fields_extracted'] for r in group):,} "
            f"| {sum(r['line_items'] for r in group):,} "
            f"| {sum(r['checks_pass'] for r in group)}/"
            f"{sum(r['checks_fail'] for r in group)}/"
            f"{sum(r['checks_not_applicable'] for r in group)} |"
        )
    add("")

    # ── Per document ──
    add("## Per document")
    add("")
    add("`Fields` counts fields whose status is EXPLICIT or EXPLICIT_ZERO, out of")
    add("every field slot in that document's schema. `Line items` counts rows the")
    add("document itself labelled (invoice lines, balance sheet rows, cash flow")
    add("activity lines), not fixed schema slots. `Evidence` is the share of")
    add("extracted fields carrying non-empty `evidence.source_text`.")
    add("")
    add("| Document | Type | Pages | Method | Extraction | Validation | Fields | Tables | Line items | Checks P/F/NA | Evidence | Missing fields |")
    add("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in sorted(dataset, key=lambda x: (x["document_type"], x["filename"])):
        coverage = "n/a" if r["evidence_coverage_pct"] is None else f"{r['evidence_coverage_pct']:.0f}%"
        add(
            f"| {r['filename']} | `{r['document_type']}` | {r['page_count']} "
            f"| {r['extraction_method'] or '—'} | {r['extraction_status']} "
            f"| {r['validation_status']} "
            f"| {r['fields_extracted']}/{r['fields_total']} "
            f"| {r['tables']} | {r['line_items']} "
            f"| {r['checks_pass']}/{r['checks_fail']}/{r['checks_not_applicable']} "
            f"| {coverage} | {fmt_missing(r['missing_fields'])} |"
        )
    add("")

    # ── Failures worth reading ──
    problems = [r for r in dataset
                if r["extraction_status"] != "PASS" or r["validation_status"] == "FAIL"]
    add("## Documents needing attention")
    add("")
    if not problems:
        add("None — every dataset document processed and validated cleanly.")
    else:
        add("| Document | Processing | Validation | Failed checks | Error |")
        add("|---|---|---|---|---|")
        for r in sorted(problems, key=lambda x: x["filename"]):
            add(
                f"| {r['filename']} | {r['extraction_status']} "
                f"| {r['validation_status']} | {r['checks_fail']} "
                f"| {r['error_code'] or '—'} |"
            )
    add("")

    # ── Edge cases ──
    add("## File-gate edge cases")
    add("")
    add("The dataset contains only well-formed documents, so these four are")
    add("generated by the harness and pushed through the same entry point. Each")
    add("must fail *gracefully* with a specific error code, not raise.")
    add("")
    add("| Input | Scenario | Detected type | Pages | Status | Error code |")
    add("|---|---|---|---|---|---|")
    scenarios = {
        "edge_unsupported_type.txt": "Plain text uploaded as an invoice",
        "edge_empty_file.pdf": "Zero-byte upload",
        "edge_corrupted.pdf": "Valid PDF header, truncated body",
        "edge_four_pages.pdf": "One page over the 3-page limit",
    }
    for r in sorted(edge, key=lambda x: x["filename"]):
        add(
            f"| `{r['filename']}` | {scenarios.get(r['filename'], '')} "
            f"| {r['file_type'] or '—'} | {r['page_count']} "
            f"| {r['extraction_status']} | `{r['error_code']}` |"
        )
    add("")

    # ── Method notes ──
    add("## How to reproduce")
    add("")
    add("```bash")
    add("./venv/Scripts/python.exe evaluation/run_evaluation.py   # ~30 min, calls Gemini")
    add("./venv/Scripts/python.exe evaluation/build_report.py     # instant, reads results/")
    add("./venv/Scripts/python.exe evaluation/tolerance_sensitivity.py")
    add("```")
    add("")
    add("`run_evaluation.py` is resumable: a document that already has a result")
    add("file is skipped, so an interrupted run costs nothing to restart.")
    add("")

    return "\n".join(lines) + "\n"


def main() -> int:
    if not RESULTS_DIR.is_dir():
        print(f"No results directory at {RESULTS_DIR}. Run run_evaluation.py first.")
        return 1

    files = sorted(RESULTS_DIR.glob("*.json"))
    if not files:
        print("No result files yet.")
        return 1

    rows = [measure(json.loads(p.read_text(encoding="utf-8"))) for p in files]

    (EVAL_DIR / "summary.json").write_text(
        json.dumps({"documents": rows}, indent=2), encoding="utf-8"
    )
    (EVAL_DIR / "dataset_report.md").write_text(render(rows), encoding="utf-8")

    dataset = [r for r in rows if r["source"] == "dataset"]
    print(f"[report] {len(rows)} results ({len(dataset)} dataset documents)")
    print(f"[report] wrote {EVAL_DIR / 'dataset_report.md'}")
    print(f"[report] wrote {EVAL_DIR / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
