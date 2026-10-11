"""
Unit tests for two-stage hybrid text extraction (mixed-content PDF pages).

Stage 1 (page classifier) is pure geometry + text layer — no rendering,
no OCR, no network. Stage 2 merges native blocks with selective OCR
crops, preferring native text on overlap. The pipeline tests below pin
the three method values ("native_pdf" | "ocr" | "hybrid") and prove OCR
is never called for native pages and only on crops (never full pages)
for mixed ones.
"""

from __future__ import annotations

import io
import time
import uuid
from datetime import datetime
from types import SimpleNamespace

import fitz
import pytest
from PIL import Image, ImageDraw

from backend.app.services import document_service as ds_module
from backend.app.services.document_service import DocumentService
from backend.app.services.text_extraction_router import (
    PAGE_MIXED, PAGE_NATIVE, PAGE_SCANNED, classify_page,
)
from backend.tests.test_pipeline import (
    FakeExtraction,
    FakeFinancialValidator,
    FakeOCR,
    FakeRepo,
    _blank_pdf,
)


# ─────────────────────────────────────────────────────────────────────
# Builders: synthetic PDFs with known ground truth
# ─────────────────────────────────────────────────────────────────────

NATIVE_PARAGRAPH = (
    "Consolidated Balance Sheet for the financial year. "
    "Total assets and total liabilities are reported below. "
    "Cash and cash equivalents at end of year follow."
)


def _native_pdf() -> bytes:
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    page.insert_text((40, 60), NATIVE_PARAGRAPH)
    page.insert_text((40, 100), NATIVE_PARAGRAPH)
    data = doc.tobytes()
    doc.close()
    return data


def _photo_png(width: int = 400, height: int = 300) -> bytes:
    """A sizable 'photograph': deterministic pixels, no text layer."""
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    for y in range(0, height, 10):
        draw.line([(0, y), (width, y)], fill="gray")
    draw.rectangle([20, 20, width - 20, height - 20], outline="black", width=3)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _mixed_pdf() -> bytes:
    """Native paragraph on top, large embedded photo below."""
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    page.insert_text((40, 60), NATIVE_PARAGRAPH)
    page.insert_text((40, 100), NATIVE_PARAGRAPH)
    page.insert_image(fitz.Rect(40, 400, 440, 700), stream=_photo_png())
    data = doc.tobytes()
    doc.close()
    return data


def _open_page(data: bytes, number: int = 0):
    doc = fitz.open(stream=data, filetype="pdf")
    return doc, doc[number]


@pytest.fixture()
def svc(monkeypatch) -> DocumentService:
    monkeypatch.setattr(ds_module, "DocumentRepository", FakeRepo)
    monkeypatch.setattr(ds_module, "ExtractionService", FakeExtraction)
    monkeypatch.setattr(ds_module, "OCRService", FakeOCR)

    s = DocumentService(db=None)
    s._financial_validator = FakeFinancialValidator()
    return s


# ─────────────────────────────────────────────────────────────────────
# Stage 1: page classifier
# ─────────────────────────────────────────────────────────────────────

def test_native_page_classified_without_rendering():
    doc, page = _open_page(_native_pdf())
    try:
        cls = classify_page(page, 1)
    finally:
        doc.close()
    assert cls.kind == PAGE_NATIVE
    assert cls.text_chars >= 50
    assert cls.image_fraction == 0.0


def test_scanned_page_classified_without_rendering():
    doc, page = _open_page(_blank_pdf(1))
    try:
        cls = classify_page(page, 1)
    finally:
        doc.close()
    assert cls.kind == PAGE_SCANNED
    assert cls.text_chars == 0


def test_mixed_page_classified():
    doc, page = _open_page(_mixed_pdf())
    try:
        cls = classify_page(page, 1)
    finally:
        doc.close()
    assert cls.kind == PAGE_MIXED
    assert cls.text_chars >= 50
    assert cls.image_fraction > 0.15
    assert len(cls.image_boxes) >= 1


def test_classification_is_fast_and_deterministic():
    """Stage-1 budget: must not render or call OCR. Generous wall bound
    (typical is single-digit ms); determinism asserted exactly."""
    doc, page = _open_page(_mixed_pdf())
    try:
        t0 = time.perf_counter()
        first = classify_page(page, 1)
        elapsed = time.perf_counter() - t0
        second = classify_page(page, 1)
    finally:
        doc.close()
    assert elapsed < 0.5, f"classifier took {elapsed:.3f}s — it must not render"
    assert first.kind == second.kind == PAGE_MIXED
    assert first.text_chars == second.text_chars
    assert first.image_boxes == second.image_boxes


# ─────────────────────────────────────────────────────────────────────
# Stage 2: merge logic
# ─────────────────────────────────────────────────────────────────────

def test_merge_prefers_native_on_overlap():
    from backend.app.services.hybrid_text_extractor import (
        TextBlock, merge_blocks,
    )

    native = [TextBlock(text="Total assets 100", bbox=(40.0, 60.0, 300.0, 80.0),
                        block_type="text", source_type="native")]
    # OCR crop almost fully covering the same region: duplicate, dropped.
    ocr_dup = [TextBlock(text="Total assets 100", bbox=(45.0, 62.0, 295.0, 78.0),
                         block_type="image", source_type="ocr")]
    # OCR crop elsewhere: genuine image content, kept.
    ocr_new = [TextBlock(text="Stamped PAID", bbox=(40.0, 400.0, 200.0, 450.0),
                         block_type="image", source_type="ocr")]
    merged = merge_blocks(native, ocr_dup + ocr_new)
    texts = [b.text for b in merged]
    assert texts.count("Total assets 100") == 1
    assert "Stamped PAID" in texts
    assert all(b.source_type == "native" for b in merged if b.text == "Total assets 100")


def test_merge_orders_top_to_bottom():
    from backend.app.services.hybrid_text_extractor import (
        TextBlock, merge_blocks,
    )

    bottom = TextBlock(text="bottom", bbox=(40.0, 700.0, 200.0, 720.0),
                       block_type="image", source_type="ocr")
    top = TextBlock(text="top", bbox=(40.0, 60.0, 200.0, 80.0),
                    block_type="text", source_type="native")
    merged = merge_blocks([bottom], [top])
    assert [b.text for b in merged] == ["top", "bottom"]


def test_table_suppresses_flat_echo():
    from backend.app.services.hybrid_text_extractor import extract_native_blocks

    table = SimpleNamespace(
        bbox=(40.0, 100.0, 400.0, 200.0),
        extract=lambda: [["Capital", "557.97"], ["Total", "557.97"]],
    )
    dict_blocks = {"blocks": [{
        "type": 0,
        "bbox": [40.0, 100.0, 400.0, 200.0],
        "lines": [{"spans": [{"text": "Capital 557.97"}]},
                  {"spans": [{"text": "Total 557.97"}]}],
    }]}
    page = SimpleNamespace(
        get_text=lambda kind: dict_blocks,
        find_tables=lambda: [table],
    )
    text_blocks, table_blocks = extract_native_blocks(page)
    assert len(table_blocks) == 1
    assert "Capital | 557.97" in table_blocks[0].text
    assert table_blocks[0].block_type == "table"
    # The flat echo of the same region is suppressed, not duplicated.
    assert all("Capital 557.97" not in b.text for b in text_blocks)


# ─────────────────────────────────────────────────────────────────────
# Pipeline wiring: methods, OCR call discipline, provenance
# ─────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_native_pdf_never_touches_ocr(svc):
    r = await svc.process_document(_native_pdf(), "native.pdf", "balance_sheet")

    assert r.processing_metadata.text_extraction_method == "native_pdf"
    assert r.processing_metadata.ocr_provider is None
    assert svc._ocr_service.images_seen == []
    assert r.processing_metadata.text_pages[0]["classification"] == "NATIVE"
    assert r.processing_metadata.text_pages[0]["image_block_count"] == 0
    assert r.processing_metadata.text_pages[0]["ocr_blocks_processed"] == 0
    assert r.processing_metadata.text_pages[0]["source_types"] == ["native"]


@pytest.mark.asyncio
async def test_mixed_pdf_ocrs_only_crops(svc):
    """The photo region is OCR'd; the full page is never rendered for OCR."""
    svc._ocr_service = FakeOCR(texts=["STAMPED TOTAL 5,000"])
    r = await svc.process_document(_mixed_pdf(), "mixed.pdf", "invoice")

    assert r.processing_metadata.text_extraction_method == "hybrid"
    # Exactly one OCR call (the crop) — not a full-page render. The full
    # page at 300 DPI would be 2479x3508 px; the crop must be smaller.
    assert len(svc._ocr_service.images_seen) == 1
    crop = svc._ocr_service.images_seen[0]
    assert crop.width < 2479 and crop.height < 3508
    sent = svc._extraction_service.text_calls[0]
    assert "Total assets" in sent  # native paragraph preserved
    assert "STAMPED TOTAL 5,000" in sent  # OCR crop merged in
    assert "--- PAGE BREAK ---" not in sent  # single page: no separators
    pages = r.processing_metadata.text_pages
    assert len(pages) == 1 and pages[0]["classification"] == "MIXED"
    assert pages[0]["ocr_blocks_processed"] == 1
    assert set(pages[0]["source_types"]) == {"native", "ocr"}


@pytest.mark.asyncio
async def test_failed_crop_marks_unreadable_and_continues(svc):
    """A dead OCR block must not fail the page or the document."""
    svc._ocr_service = FakeOCR(texts=[None])
    r = await svc.process_document(_mixed_pdf(), "mixed.pdf", "invoice")

    assert r.processing_status == "PASS"
    assert r.processing_metadata.text_extraction_method == "hybrid"
    sent = svc._extraction_service.text_calls[0]
    assert "Total assets" in sent  # native text still flows
    assert "[unreadable image region]" in sent


@pytest.mark.asyncio
async def test_scanned_pdf_keeps_full_page_ocr_path(svc):
    svc._ocr_service = FakeOCR(texts=["p1", "p2"])
    r = await svc.process_document(_blank_pdf(2), "bs.pdf", "balance_sheet")

    assert r.processing_metadata.text_extraction_method == "ocr"
    assert len(svc._ocr_service.images_seen) == 2
    pages = r.processing_metadata.text_pages
    assert [p["classification"] for p in pages] == ["SCANNED", "SCANNED"]


@pytest.mark.asyncio
async def test_hybrid_extraction_is_deterministic(svc):
    from backend.app.services.hybrid_text_extractor import extract_pdf_hybrid

    data = _mixed_pdf()
    svc._ocr_service = FakeOCR(texts=["STAMPED TOTAL 5,000"])
    first = await extract_pdf_hybrid(data, svc._preprocessor, svc._ocr_service)
    svc._ocr_service = FakeOCR(texts=["STAMPED TOTAL 5,000"])
    second = await extract_pdf_hybrid(data, svc._preprocessor, svc._ocr_service)
    assert first["page_texts"] == second["page_texts"]
    assert first["method"] == second["method"] == "hybrid"


def test_extraction_metrics_count_pages_and_ocr():
    """ocr_pages_total vs pages_total, by classification (prompt §7)."""
    from backend.app.core import metrics as m

    before_pages = dict(m.metrics.snapshot()["counters"].get("finirag_pages_total", {}))
    before_ocr = dict(m.metrics.snapshot()["counters"].get("finirag_ocr_pages_total", {}))

    from backend.app.services.document_service import DocumentService

    DocumentService._record_extraction_metrics(
        [{"classification": "NATIVE", "ocr_blocks_processed": 0},
         {"classification": "MIXED", "ocr_blocks_processed": 2},
         {"classification": "SCANNED", "ocr_blocks_processed": 1}])

    snap = m.metrics.snapshot()["counters"]
    pages = snap["finirag_pages_total"]
    ocr = snap["finirag_ocr_pages_total"]

    def key(classification):
        return str((("classification", classification),))

    assert pages.get(key("NATIVE"), 0) == before_pages.get(key("NATIVE"), 0) + 1
    assert pages.get(key("MIXED"), 0) == before_pages.get(key("MIXED"), 0) + 1
    assert pages.get(key("SCANNED"), 0) == before_pages.get(key("SCANNED"), 0) + 1
    assert ocr.get(key("MIXED"), 0) == before_ocr.get(key("MIXED"), 0) + 1
    assert ocr.get(key("SCANNED"), 0) == before_ocr.get(key("SCANNED"), 0) + 1
    assert key("NATIVE") not in ocr or ocr.get(key("NATIVE"), 0) == before_ocr.get(key("NATIVE"), 0)
