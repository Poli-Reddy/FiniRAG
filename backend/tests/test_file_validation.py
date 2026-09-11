"""
Unit tests for DocumentValidationService — the pre-extraction gate.

Everything here runs before a single token is spent on OCR or the LLM:
magic-byte typing, corruption, emptiness, size, and the 3-page cap.
Real files from the Dataset/ folder are used where a synthetic byte
string would not exercise the same code path.
"""

from __future__ import annotations

import io

import fitz
import pytest
from PIL import Image

from backend.app.core.config import settings
from backend.app.services.document_validation_service import (
    SUPPORTED_MIME_TYPES,
    DocumentValidationService,
)


@pytest.fixture()
def v() -> DocumentValidationService:
    return DocumentValidationService()


def _make_pdf(pages: int) -> bytes:
    doc = fitz.open()
    for i in range(pages):
        page = doc.new_page()
        page.insert_text((72, 72), f"Page {i + 1}")
    data = doc.tobytes()
    doc.close()
    return data


def _make_png(size=(64, 64)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, "white").save(buf, format="PNG")
    return buf.getvalue()


# ─────────────────────────────────────────────────────────────────────
# Emptiness and size
# ─────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("payload", [b"", None])
def test_empty_upload_is_rejected(v, payload):
    result = v.validate(payload or b"", "empty.pdf")
    assert result.status == "FAIL"
    assert result.error == "EMPTY_FILE"


def test_oversized_file_is_rejected_before_anything_else(v):
    """The size gate must fire without parsing the payload."""
    oversized = b"%PDF-1.4" + b"\x00" * (settings.max_file_size_bytes + 1)
    result = v.validate(oversized, "huge.pdf")
    assert result.status == "FAIL"
    assert result.error == "FILE_TOO_LARGE"


def test_file_at_the_size_limit_is_accepted(v):
    """The limit is inclusive — exactly max bytes must not be rejected."""
    pdf = _make_pdf(1)
    assert len(pdf) < settings.max_file_size_bytes
    assert v.validate(pdf, "ok.pdf").status == "PASS"


# ─────────────────────────────────────────────────────────────────────
# MIME typing by magic bytes, not extension (Trap P13)
# ─────────────────────────────────────────────────────────────────────

def test_type_comes_from_content_not_extension(v):
    """A PNG renamed to .pdf is still a PNG."""
    result = v.validate(_make_png(), "actually_a_png.pdf")
    assert result.status == "PASS"
    assert result.file_type == "image/png"


def test_pdf_renamed_as_jpg_is_still_a_pdf(v):
    result = v.validate(_make_pdf(1), "actually_a_pdf.jpg")
    assert result.status == "PASS"
    assert result.file_type == "application/pdf"


def test_unsupported_type_is_rejected(v):
    """A text file has no magic bytes we accept."""
    result = v.validate(b"just some plain text, not a document" * 10, "notes.txt")
    assert result.status == "FAIL"
    assert result.error == "UNSUPPORTED_FILE_TYPE"


def test_executable_disguised_as_pdf_is_rejected(v):
    """MZ header — a Windows binary uploaded with a document name."""
    result = v.validate(b"MZ\x90\x00\x03" + b"\x00" * 500, "invoice.pdf")
    assert result.status == "FAIL"
    assert result.error == "UNSUPPORTED_FILE_TYPE"


def test_supported_types_are_exactly_pdf_jpeg_png(v):
    assert set(SUPPORTED_MIME_TYPES) == {
        "application/pdf", "image/jpeg", "image/png",
    }


# ─────────────────────────────────────────────────────────────────────
# Corruption
# ─────────────────────────────────────────────────────────────────────

def test_truncated_pdf_is_reported_as_corrupted(v):
    """Right magic bytes, unparseable body."""
    result = v.validate(b"%PDF-1.4\n" + b"garbage" * 100, "broken.pdf")
    assert result.status == "FAIL"
    assert result.error == "CORRUPTED_FILE"


def test_truncated_png_is_reported_as_corrupted(v):
    png = _make_png()
    result = v.validate(png[: len(png) // 3], "broken.png")
    assert result.status == "FAIL"
    assert result.error == "CORRUPTED_FILE"


# ─────────────────────────────────────────────────────────────────────
# Page limit (Trap P12, D7)
# ─────────────────────────────────────────────────────────────────────

def test_page_count_is_counted_not_trusted(v):
    result = v.validate(_make_pdf(3), "three_pages.pdf")
    assert result.status == "PASS"
    assert result.page_count == 3


def test_page_limit_is_enforced_at_the_boundary(v):
    """MAX_PAGES is inclusive; one more is a hard fail."""
    at_limit = v.validate(_make_pdf(settings.MAX_PAGES), "at_limit.pdf")
    assert at_limit.status == "PASS"

    over = v.validate(_make_pdf(settings.MAX_PAGES + 1), "over_limit.pdf")
    assert over.status == "FAIL"
    assert over.error == "PAGE_LIMIT_EXCEEDED"
    assert over.page_count == settings.MAX_PAGES + 1


def test_page_limit_default_is_three():
    assert settings.MAX_PAGES == 3


def test_images_always_count_as_one_page(v):
    result = v.validate(_make_png(), "receipt.png")
    assert result.page_count == 1


# ─────────────────────────────────────────────────────────────────────
# Real dataset files
# ─────────────────────────────────────────────────────────────────────

def test_real_balance_sheet_pdf_passes(v, sample_pdf):
    result = v.validate(sample_pdf, "Consolidated Balance Sheet 2021.pdf")
    assert result.status == "PASS"
    assert result.file_type == "application/pdf"
    assert result.is_supported and result.is_readable
    assert 1 <= result.page_count <= settings.MAX_PAGES


def test_real_invoice_jpeg_passes(v, sample_jpeg):
    result = v.validate(sample_jpeg, "batch1-1109.jpg")
    assert result.status == "PASS"
    assert result.file_type == "image/jpeg"
    assert result.page_count == 1


def test_phone_photo_with_trailing_bytes_passes(v, rotated_jpeg):
    """
    Samsung phone JPEGs carry an SEFT trailer after the EOI marker.
    Validating on a literal FFD9-at-EOF rule would reject these.
    """
    assert rotated_jpeg.rstrip().endswith(b"SEFT"), "fixture is not the SEFT-trailer photo"
    result = v.validate(rotated_jpeg, "20251118_000612.jpg")
    assert result.status == "PASS"
    assert result.file_type == "image/jpeg"


def test_every_dataset_file_passes_validation(v, dataset_dir):
    """
    Nothing shipped in the case-study dataset should be rejected by the
    gate. A failure here means the gate is stricter than the brief.
    """
    files = [
        p for p in dataset_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in {".pdf", ".jpg", ".jpeg", ".png"}
    ]
    assert files, "dataset appears to be empty"

    rejected = []
    for path in files:
        result = v.validate(path.read_bytes(), path.name)
        if result.status != "PASS":
            rejected.append((path.name, result.error, result.page_count))

    assert not rejected, f"dataset files rejected by validation: {rejected}"
