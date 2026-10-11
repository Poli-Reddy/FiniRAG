"""
Unit tests for DocumentService — the pipeline orchestrator.

The file gate, the router and the preprocessor run for real here; OCR,
the LLM and the database are doubles. What is under test is the wiring:
which stage runs, which is skipped, what error code a failure carries,
and which failures are allowed to degrade rather than abort.
"""

from __future__ import annotations

import io
import uuid
from datetime import datetime

import fitz
import pytest
from PIL import Image

from backend.app.providers.errors import LLMOutputTruncatedError, LLMRateLimitError
from backend.app.providers.ocr_provider import OCRResult
from backend.app.services import document_service as ds_module
from backend.app.services.document_service import DocumentService


# ─────────────────────────────────────────────────────────────────────
# Doubles
# ─────────────────────────────────────────────────────────────────────

class FakeRepo:
    """Collects rows instead of writing them."""

    def __init__(self, db=None):
        self.created: list = []

    async def create(self, document):
        document.id = uuid.uuid4()
        document.created_at = datetime.now()
        self.created.append(document)
        return document


class FakeOCR:
    """
    Returns one scripted OCRResult per image.

    `texts` is indexed by page; a None entry means both providers failed
    on that page, which is what the live service reports as provider
    "none" with empty text.
    """

    def __init__(self, texts=("page text",)):
        self.texts = list(texts)
        self.images_seen: list = []

    async def extract_text_from_images(self, images):
        self.images_seen = list(images)
        results = []
        for i, _ in enumerate(images):
            entry = self.texts[i] if i < len(self.texts) else None
            if entry is None:
                results.append(OCRResult(text="", provider="none", page_number=i + 1))
            else:
                text, provider = entry if isinstance(entry, tuple) else (entry, "google_vision")
                results.append(OCRResult(text=text, provider=provider, page_number=i + 1))
        return results


class FakeExtraction:
    """Stands in for the Gemini round-trip."""

    def __init__(self, result=None, text_error=None, image_error=None):
        self.result = result if result is not None else {"company_name": "ACME"}
        self.text_error = text_error
        self.image_error = image_error
        self.text_calls: list[str] = []
        self.image_calls: list[list] = []

    async def extract(self, text, document_type):
        self.text_calls.append(text)
        if self.text_error:
            raise self.text_error
        return self.result

    async def extract_from_images(self, images, document_type):
        self.image_calls.append(list(images))
        if self.image_error:
            raise self.image_error
        return self.result


class FakeFinancialValidator:
    def __init__(self, error=None):
        self.error = error
        self.calls: list[tuple] = []

    def validate(self, extracted_data, document_type):
        self.calls.append((extracted_data, document_type))
        if self.error:
            raise self.error
        return _StubValidationResult()


class _StubValidationResult:
    def model_dump(self, mode=None):
        return {
            "checks": [{"name": "sum_assets", "status": "PASS"}],
            "overall_status": "PASS",
            "summary": {"pass": 1, "fail": 0, "not_applicable": 0},
        }


# ─────────────────────────────────────────────────────────────────────
# Fixtures and builders
# ─────────────────────────────────────────────────────────────────────

@pytest.fixture()
def svc(monkeypatch) -> DocumentService:
    """
    A DocumentService whose slow edges are doubles.

    ExtractionService is patched at the module symbol so the constructor
    never builds a Gemini client — the service must be constructible
    without credentials.
    """
    monkeypatch.setattr(ds_module, "DocumentRepository", FakeRepo)
    monkeypatch.setattr(ds_module, "ExtractionService", FakeExtraction)
    monkeypatch.setattr(ds_module, "OCRService", FakeOCR)

    s = DocumentService(db=None)
    s._financial_validator = FakeFinancialValidator()
    return s


def _blank_pdf(pages: int = 1) -> bytes:
    """A PDF with no text layer — the vector-outlined case."""
    doc = fitz.open()
    for _ in range(pages):
        doc.new_page(width=200, height=200)
    data = doc.tobytes()
    doc.close()
    return data


def _text_pdf(text: str) -> bytes:
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((20, 40), text)
    data = doc.tobytes()
    doc.close()
    return data


def _png() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (40, 40), "white").save(buf, format="PNG")
    return buf.getvalue()


NATIVE_TEXT = (
    "Consolidated Cash Flow Statement\n"
    "Net cash from operating activities for the period\n"
    "Total cash and cash equivalents at end of year\n"
)


# ─────────────────────────────────────────────────────────────────────
# Happy paths
# ─────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_native_text_pdf_skips_ocr_entirely(svc):
    """No OCR provider should be touched when the PDF carries its text."""
    r = await svc.process_document(_text_pdf(NATIVE_TEXT), "cf.pdf", "cash_flow_statement")

    assert r.processing_status == "PASS"
    assert r.processing_metadata.text_extraction_method == "native_pdf"
    assert r.processing_metadata.ocr_provider is None
    assert svc._ocr_service.images_seen == []
    assert len(svc._extraction_service.text_calls) == 1


@pytest.mark.asyncio
async def test_image_upload_goes_through_ocr(svc):
    r = await svc.process_document(_png(), "receipt.png", "invoice")

    assert r.processing_status == "PASS"
    assert r.processing_metadata.text_extraction_method == "ocr"
    assert r.processing_metadata.ocr_provider == "google_vision"
    assert len(svc._ocr_service.images_seen) == 1


@pytest.mark.asyncio
async def test_textless_pdf_is_rendered_then_ocred(svc):
    svc._ocr_service = FakeOCR(texts=["p1", "p2"])
    r = await svc.process_document(_blank_pdf(2), "bs.pdf", "balance_sheet")

    assert r.processing_metadata.text_extraction_method == "ocr"
    assert len(svc._ocr_service.images_seen) == 2


@pytest.mark.asyncio
async def test_multi_page_text_is_aggregated_with_page_breaks(svc):
    svc._ocr_service = FakeOCR(texts=["ASSETS 100", "LIABILITIES 100"])
    await svc.process_document(_blank_pdf(2), "bs.pdf", "balance_sheet")

    sent = svc._extraction_service.text_calls[0]
    assert "ASSETS 100" in sent and "LIABILITIES 100" in sent
    assert "--- PAGE BREAK ---" in sent


@pytest.mark.asyncio
async def test_every_provider_that_contributed_is_recorded(svc):
    """
    A page no provider could read contributes nothing and must not be
    counted as OCR coverage. Only backends that actually returned text
    are named, each once, in the order they first did.
    """
    svc._ocr_service = FakeOCR(texts=[
        ("p1", "google_vision"), None, ("p3", "google_vision"),
    ])
    r = await svc.process_document(_blank_pdf(3), "bs.pdf", "balance_sheet")

    assert r.processing_metadata.ocr_provider == "google_vision"


@pytest.mark.asyncio
async def test_extracted_data_reaches_validation_unchanged(svc):
    svc._extraction_service = FakeExtraction(result={"total_assets": "17995066442"})
    r = await svc.process_document(_png(), "bs.png", "balance_sheet")

    payload, doc_type = svc._financial_validator.calls[0]
    assert payload == {"total_assets": "17995066442"}
    assert doc_type == "balance_sheet"
    assert r.extracted_data == {"total_assets": "17995066442"}


@pytest.mark.asyncio
async def test_metadata_records_when_and_how_long(svc):
    r = await svc.process_document(_png(), "receipt.png", "invoice")
    m = r.processing_metadata

    assert m.llm_provider == "gemini"
    assert m.processing_time_ms >= 0
    assert datetime.fromisoformat(m.processed_at).tzinfo is not None


# ─────────────────────────────────────────────────────────────────────
# The file gate short-circuits everything downstream
# ─────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_rejected_file_never_reaches_ocr_or_the_llm(svc):
    """The gate exists to stop spend, so nothing after it may run."""
    r = await svc.process_document(b"plain text file" * 20, "notes.txt", "invoice")

    assert r.processing_status == "FAILED"
    assert r.error["code"] == "UNSUPPORTED_FILE_TYPE"
    assert svc._ocr_service.images_seen == []
    assert svc._extraction_service.text_calls == []
    assert svc._extraction_service.image_calls == []


@pytest.mark.asyncio
async def test_page_limit_failure_carries_its_own_code(svc):
    r = await svc.process_document(_blank_pdf(4), "long.pdf", "balance_sheet")

    assert r.processing_status == "FAILED"
    assert r.error["code"] == "PAGE_LIMIT_EXCEEDED"
    assert r.file_validation.page_count == 4
    assert r.extracted_data is None
    assert r.validation is None


@pytest.mark.asyncio
async def test_empty_upload_fails_at_the_gate(svc):
    r = await svc.process_document(b"", "nothing.pdf", "invoice")
    assert r.error["code"] == "EMPTY_FILE"


# ─────────────────────────────────────────────────────────────────────
# Vision fallback (the production failure mode)
# ─────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_total_ocr_failure_on_one_page_triggers_the_vision_fallback(svc):
    """Vision 403 on the only OCR backend — verbatim from the logs."""
    svc._ocr_service = FakeOCR(texts=[None])
    r = await svc.process_document(_png(), "receipt.png", "invoice")

    assert r.processing_status == "PASS"
    assert r.processing_metadata.text_extraction_method == "gemini_vision"
    assert r.processing_metadata.ocr_provider == "gemini_vision"
    assert len(svc._extraction_service.image_calls) == 1
    assert svc._extraction_service.text_calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("pages", [1, 2, 3])
async def test_total_ocr_failure_triggers_the_fallback_at_any_page_count(svc, pages):
    """
    Joining N empty pages leaves the "--- PAGE BREAK ---" separators
    behind. Treating that residue as text would send a document with no
    content at all to the LLM instead of to the vision fallback — and it
    would only ever happen on multi-page files, which is exactly where
    the cash flow statements live.
    """
    svc._ocr_service = FakeOCR(texts=[None] * pages)
    r = await svc.process_document(_blank_pdf(pages), "cf.pdf", "cash_flow_statement")

    assert r.processing_metadata.text_extraction_method == "gemini_vision"
    assert len(svc._extraction_service.image_calls) == 1
    assert len(svc._extraction_service.image_calls[0]) == pages
    assert svc._extraction_service.text_calls == []


@pytest.mark.asyncio
async def test_a_single_readable_page_is_enough_to_stay_on_the_text_path(svc):
    """Partial OCR success is still success; do not pay for vision twice."""
    svc._ocr_service = FakeOCR(texts=[None, "LIABILITIES 100", None])
    r = await svc.process_document(_blank_pdf(3), "bs.pdf", "balance_sheet")

    assert r.processing_metadata.text_extraction_method == "ocr"
    assert len(svc._extraction_service.text_calls) == 1
    assert svc._extraction_service.image_calls == []


@pytest.mark.asyncio
async def test_page_positions_survive_a_blank_page(svc):
    """
    Dropping an unreadable page from the aggregate would renumber every
    page after it, and evidence page numbers are cited back to the user.
    """
    svc._ocr_service = FakeOCR(texts=[None, "SECOND PAGE"])
    await svc.process_document(_blank_pdf(2), "bs.pdf", "balance_sheet")

    sent = svc._extraction_service.text_calls[0]
    assert sent.startswith("\n\n--- PAGE BREAK ---")
    assert sent.endswith("SECOND PAGE")


@pytest.mark.asyncio
async def test_vision_fallback_llm_failure_is_not_reported_as_empty_text(svc):
    """
    The document had pages; the LLM is what failed.

    This branch reported EMPTY_TEXT until the 2026-09-10 dataset run, where a
    quota rejection labelled 43 perfectly readable documents as having no
    text. The code a user sees has to point at the stage that actually broke.
    """
    svc._ocr_service = FakeOCR(texts=[None])
    svc._extraction_service = FakeExtraction(image_error=RuntimeError("model overloaded"))

    r = await svc.process_document(_png(), "receipt.png", "invoice")

    assert r.processing_status == "FAILED"
    assert r.error["code"] == "LLM_EXTRACTION_FAILED"


@pytest.mark.asyncio
async def test_vision_fallback_throttling_has_its_own_code(svc):
    """
    A throttle says nothing about the document — the same file will succeed
    later untouched. LLM_EXTRACTION_FAILED would send the user to rescan it.
    """
    svc._ocr_service = FakeOCR(texts=[None])
    svc._extraction_service = FakeExtraction(image_error=LLMRateLimitError(31.0))

    r = await svc.process_document(_png(), "receipt.png", "invoice")

    assert r.processing_status == "FAILED"
    assert r.error["code"] == "LLM_RATE_LIMITED"


@pytest.mark.asyncio
async def test_text_path_throttling_has_its_own_code(svc):
    """The same distinction has to hold on the path that never renders."""
    svc._extraction_service = FakeExtraction(text_error=LLMRateLimitError())

    r = await svc.process_document(_png(), "receipt.png", "invoice")

    assert r.processing_status == "FAILED"
    assert r.error["code"] == "LLM_RATE_LIMITED"


@pytest.mark.asyncio
async def test_vision_fallback_truncation_has_its_own_code(svc):
    """
    A reply cut off at the provider's output ceiling is a limit of that
    backend, not a property of the document. Folding it into
    LLM_EXTRACTION_FAILED would let a resumable run skip the file for good.
    """
    svc._ocr_service = FakeOCR(texts=[None])
    svc._extraction_service = FakeExtraction(
        image_error=LLMOutputTruncatedError(999)
    )

    r = await svc.process_document(_png(), "receipt.png", "invoice")

    assert r.processing_status == "FAILED"
    assert r.error["code"] == "LLM_OUTPUT_TRUNCATED"


@pytest.mark.asyncio
async def test_text_path_truncation_has_its_own_code(svc):
    svc._extraction_service = FakeExtraction(
        text_error=LLMOutputTruncatedError(999)
    )

    r = await svc.process_document(_png(), "receipt.png", "invoice")

    assert r.processing_status == "FAILED"
    assert r.error["code"] == "LLM_OUTPUT_TRUNCATED"


@pytest.mark.asyncio
async def test_a_truncated_document_is_never_recorded_as_a_pass(svc):
    """
    The failure this guards against is the quiet one: a truncated reply still
    parses, so nothing downstream distinguishes a missing total from a total
    the document never had. No extracted data may survive it.
    """
    svc._ocr_service = FakeOCR(texts=[None])
    svc._extraction_service = FakeExtraction(
        image_error=LLMOutputTruncatedError(999)
    )

    r = await svc.process_document(_png(), "receipt.png", "invoice")

    assert r.extracted_data is None
    assert r.validation is None


@pytest.mark.asyncio
async def test_vision_fallback_render_failure_is_a_render_failure(svc, monkeypatch):
    """Rendering and extracting fail for unrelated reasons; keep them apart."""
    svc._ocr_service = FakeOCR(texts=[None])

    def _boom(*_args, **_kwargs):
        raise RuntimeError("cannot decode image")

    monkeypatch.setattr(svc._preprocessor, "preprocess_images", _boom)
    r = await svc.process_document(_png(), "receipt.png", "invoice")

    assert r.processing_status == "FAILED"
    assert r.error["code"] == "TEXT_EXTRACTION_FAILED"
    assert svc._extraction_service.image_calls == []


@pytest.mark.asyncio
async def test_empty_text_survives_for_the_case_it_actually_describes(svc, monkeypatch):
    """
    No OCR text *and* nothing to look at. This is the only branch where
    EMPTY_TEXT is the honest answer, and the LLM must not be billed for it.
    """
    svc._ocr_service = FakeOCR(texts=[None])
    monkeypatch.setattr(svc._preprocessor, "preprocess_images", lambda *a, **k: [])

    r = await svc.process_document(_png(), "receipt.png", "invoice")

    assert r.processing_status == "FAILED"
    assert r.error["code"] == "EMPTY_TEXT"
    assert svc._extraction_service.image_calls == []


@pytest.mark.asyncio
async def test_throttling_never_leaks_the_provider_message(svc):
    """Correction #10 holds for the new code too."""
    svc._extraction_service = FakeExtraction(
        text_error=LLMRateLimitError(retry_after_seconds=30.878)
    )
    r = await svc.process_document(_png(), "receipt.png", "invoice")

    assert r.error == {"code": "LLM_RATE_LIMITED", "message": "LLM_RATE_LIMITED"}


# ─────────────────────────────────────────────────────────────────────
# Stage failures
# ─────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_render_failure_is_reported_as_text_extraction_failed(svc, monkeypatch):
    def _boom(*_args, **_kwargs):
        raise RuntimeError("mupdf: cannot render page")

    # Hybrid extraction renders per page (SCANNED pages and MIXED crops),
    # so the render seam is render_page_image; a render failure must still
    # abort the document, never degrade into partial text.
    monkeypatch.setattr(svc._preprocessor, "render_page_image", _boom)
    r = await svc.process_document(_blank_pdf(1), "bs.pdf", "balance_sheet")

    assert r.processing_status == "FAILED"
    assert r.error["code"] == "TEXT_EXTRACTION_FAILED"


@pytest.mark.asyncio
async def test_llm_failure_on_the_text_path_has_its_own_code(svc):
    svc._extraction_service = FakeExtraction(text_error=RuntimeError("429 rate limited"))
    r = await svc.process_document(_png(), "receipt.png", "invoice")

    assert r.processing_status == "FAILED"
    assert r.error["code"] == "LLM_EXTRACTION_FAILED"


@pytest.mark.asyncio
async def test_error_codes_never_carry_the_underlying_message(svc):
    """Correction #10 — the client sees a code, not a vendor stack trace."""
    svc._extraction_service = FakeExtraction(
        text_error=RuntimeError("api_key=AIzaSyBADSECRET rejected")
    )
    r = await svc.process_document(_png(), "receipt.png", "invoice")

    assert "AIzaSyBADSECRET" not in str(r.error)
    assert r.error == {"code": "LLM_EXTRACTION_FAILED", "message": "LLM_EXTRACTION_FAILED"}


@pytest.mark.asyncio
async def test_validation_crash_degrades_instead_of_failing_the_document(svc):
    """
    Extraction succeeded; the numbers are worth keeping. A validator bug
    must not throw away a document the user already paid to process.
    """
    svc._financial_validator = FakeFinancialValidator(error=KeyError("total_assets"))
    r = await svc.process_document(_png(), "bs.png", "balance_sheet")

    assert r.processing_status == "PASS"
    assert r.extracted_data is not None
    assert r.validation["overall_status"] == "NOT_APPLICABLE"
    assert r.validation["checks"] == []


# ─────────────────────────────────────────────────────────────────────
# Persistence (every attempt is a new row)
# ─────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_successful_processing_is_persisted_in_full(svc):
    await svc.process_document(_png(), "receipt.png", "invoice")

    [row] = svc._repo.created
    assert row.document_name == "receipt.png"
    assert row.document_type == "invoice"
    assert row.processing_status == "PASS"
    assert row.extracted_data is not None
    assert row.validation["overall_status"] == "PASS"
    assert row.error is None


@pytest.mark.asyncio
async def test_failures_are_persisted_too(svc):
    """A rejected upload is part of the audit trail, not a non-event."""
    await svc.process_document(_blank_pdf(4), "long.pdf", "balance_sheet")

    [row] = svc._repo.created
    assert row.processing_status == "FAILED"
    assert row.error["code"] == "PAGE_LIMIT_EXCEEDED"
    assert row.extracted_data is None
    assert row.validation is None


@pytest.mark.asyncio
async def test_reprocessing_appends_a_row_rather_than_overwriting(svc):
    await svc.process_document(_png(), "receipt.png", "invoice")
    await svc.process_document(_png(), "receipt.png", "invoice")

    assert len(svc._repo.created) == 2
    assert {r.document_name for r in svc._repo.created} == {"receipt.png"}


@pytest.mark.asyncio
async def test_the_stored_row_and_the_response_agree(svc):
    r = await svc.process_document(_png(), "receipt.png", "invoice")
    [row] = svc._repo.created

    assert row.processing_status == r.processing_status
    assert row.extracted_data == r.extracted_data
    assert row.validation == r.validation
    assert row.file_validation["page_count"] == r.file_validation.page_count


@pytest.mark.asyncio
async def test_document_type_is_carried_through_untouched(svc):
    """
    The user picked the type; nothing in the pipeline may second-guess it
    (Trap P10). An invoice filename on a balance sheet stays a balance
    sheet all the way to the validator.
    """
    await svc.process_document(_png(), "invoice.png", "balance_sheet")

    _, doc_type = svc._financial_validator.calls[0]
    assert doc_type == "balance_sheet"
    assert svc._repo.created[0].document_type == "balance_sheet"
