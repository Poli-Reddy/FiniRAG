"""
Unit tests for the text-acquisition layer: routing, preprocessing, OCR.

This is the stage that actually failed in production logs — Vision
returned 403 on every page and every document fell through to the LLM
vision fallback. None of these tests touch the network: the providers
are replaced with doubles so the *chain* is what is under test, not the
vendors.
"""

from __future__ import annotations

import io
import logging

import fitz
import pytest
from PIL import Image

from backend.app.providers.ocr_provider import OCRProvider, OCRResult, OCRWord
from backend.app.providers.google_vision_provider import GoogleVisionProvider
from backend.app.services import ocr_service as ocr_module
from backend.app.services.image_preprocessor import ImagePreprocessor
from backend.app.services.ocr_service import OCRService
from backend.app.services.text_extraction_router import (
    FINANCIAL_KEYWORDS,
    TextExtractionRouter,
)


# ─────────────────────────────────────────────────────────────────────
# Doubles and builders
# ─────────────────────────────────────────────────────────────────────

class FakeProvider(OCRProvider):
    """
    A scriptable OCR provider.

    `fail_on` and `blank_on` are sets of page numbers, so a test can make
    a provider fail for one page and succeed for the next — which is the
    behaviour the per-page fallback rule exists to protect.
    """

    def __init__(self, name: str, fail_on=(), blank_on=()):
        self.name = name
        self.fail_on = set(fail_on)
        self.blank_on = set(blank_on)
        self.calls: list[int] = []

    async def extract_text(self, image: Image.Image, page_number: int) -> OCRResult:
        self.calls.append(page_number)
        if page_number in self.fail_on:
            raise RuntimeError(f"{self.name} failed on page {page_number}")
        text = "" if page_number in self.blank_on else f"{self.name} text p{page_number}"
        return OCRResult(
            text=text,
            words=[OCRWord(text=self.name, confidence=0.9)],
            provider=self.name,
            page_number=page_number,
        )


def _img(size=(20, 20)) -> Image.Image:
    return Image.new("RGB", size, "white")


def _pages(n: int) -> list[Image.Image]:
    return [_img() for _ in range(n)]


def _text_pdf(text: str, pages: int = 1) -> bytes:
    doc = fitz.open()
    for _ in range(pages):
        page = doc.new_page()
        page.insert_text((72, 100), text)
    data = doc.tobytes()
    doc.close()
    return data


def _blank_pdf(pages: int = 1, width: float = 612, height: float = 792) -> bytes:
    doc = fitz.open()
    for _ in range(pages):
        doc.new_page(width=width, height=height)
    data = doc.tobytes()
    doc.close()
    return data


@pytest.fixture()
def svc() -> OCRService:
    """An OCRService with both slots replaced by doubles."""
    s = OCRService()
    s._primary = FakeProvider("primary")
    s._fallback = FakeProvider("fallback")
    return s


# ─────────────────────────────────────────────────────────────────────
# Provider wiring
# ─────────────────────────────────────────────────────────────────────

def test_google_vision_is_the_only_ocr_backend(monkeypatch):
    """
    Vision has no OCR provider behind it. The document's second chance is
    the LLM vision path in DocumentService, one layer up — not another
    OCR engine here.
    """
    monkeypatch.setattr(ocr_module.settings, "OCR_PROVIDER", "google_vision")
    s = OCRService()
    assert isinstance(s._primary, GoogleVisionProvider)
    assert s._fallback is None


def test_an_unknown_provider_disables_ocr_rather_than_guessing(monkeypatch, caplog):
    """
    A typo in .env must not be papered over with a default. OCR goes dark,
    every page reports "none", the document still reaches the LLM vision
    fallback, and the log names the offending value.
    """
    monkeypatch.setattr(ocr_module.settings, "OCR_PROVIDER", "goggle_vision")
    with caplog.at_level(logging.ERROR, logger="finirag"):
        s = OCRService()

    assert s._primary is None
    assert s._fallback is None
    assert any("Unknown OCR_PROVIDER" in r.message for r in caplog.records)


def test_constructing_the_service_makes_no_network_call(monkeypatch):
    """
    Provider clients are lazy. Building OCRService at import time must not
    reach out to Google, or the app cannot boot without credentials.
    """
    monkeypatch.setattr(ocr_module.settings, "OCR_PROVIDER", "google_vision")
    s = OCRService()
    assert s._primary._client is None


# ─────────────────────────────────────────────────────────────────────
# The fallback chain
# ─────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_primary_success_leaves_the_fallback_untouched(svc):
    results = await svc.extract_text_from_images(_pages(2))

    assert [r.provider for r in results] == ["primary", "primary"]
    assert svc._fallback.calls == []


@pytest.mark.asyncio
async def test_primary_exception_routes_the_page_to_the_fallback(svc):
    """This is the 403-from-Vision case from the production logs."""
    svc._primary = FakeProvider("primary", fail_on={1})
    results = await svc.extract_text_from_images(_pages(1))

    assert results[0].provider == "fallback"
    assert results[0].text == "fallback text p1"


@pytest.mark.asyncio
async def test_primary_returning_empty_text_counts_as_failure(svc):
    """
    A 200 response with no text is not a successful read. Accepting it
    would hand the LLM an empty page and call the document processed.
    """
    svc._primary = FakeProvider("primary", blank_on={1})
    results = await svc.extract_text_from_images(_pages(1))

    assert results[0].provider == "fallback"
    assert svc._fallback.calls == [1]


@pytest.mark.asyncio
async def test_whitespace_only_text_counts_as_empty(svc):
    class Whitespace(FakeProvider):
        async def extract_text(self, image, page_number):
            self.calls.append(page_number)
            return OCRResult(text="   \n\t  ", provider=self.name, page_number=page_number)

    svc._primary = Whitespace("primary")
    results = await svc.extract_text_from_images(_pages(1))

    assert results[0].provider == "fallback"


@pytest.mark.asyncio
async def test_both_providers_failing_yields_an_empty_result_not_an_exception(svc):
    """
    The exact production scenario, with the chain filled: nothing readable
    comes back. The pipeline must survive it so the LLM vision fallback
    upstream gets its turn.
    """
    svc._primary = FakeProvider("primary", fail_on={1})
    svc._fallback = FakeProvider("fallback", fail_on={1})

    results = await svc.extract_text_from_images(_pages(1))

    assert len(results) == 1
    assert results[0].provider == "none"
    assert results[0].text == ""
    assert results[0].words == []
    assert results[0].page_number == 1


@pytest.mark.asyncio
async def test_fallback_is_the_last_word_even_when_it_returns_nothing(svc):
    """
    Once the fallback answers without raising, its result stands. There
    is no third provider to consult, and a blank page is a real outcome.
    """
    svc._primary = FakeProvider("primary", fail_on={1})
    svc._fallback = FakeProvider("fallback", blank_on={1})

    results = await svc.extract_text_from_images(_pages(1))

    assert results[0].provider == "fallback"
    assert results[0].text == ""


@pytest.mark.asyncio
async def test_fallback_is_per_page_not_per_document(svc):
    """
    A transient failure on page 1 must not demote pages 2 and 3 to the
    weaker provider for the rest of the document.
    """
    svc._primary = FakeProvider("primary", fail_on={1})
    results = await svc.extract_text_from_images(_pages(3))

    assert [r.provider for r in results] == ["fallback", "primary", "primary"]
    assert svc._primary.calls == [1, 2, 3]
    assert svc._fallback.calls == [1]


@pytest.mark.asyncio
async def test_a_mid_document_failure_does_not_disturb_its_neighbours(svc):
    svc._primary = FakeProvider("primary", fail_on={2})
    results = await svc.extract_text_from_images(_pages(3))

    assert [r.provider for r in results] == ["primary", "fallback", "primary"]


@pytest.mark.asyncio
async def test_page_numbers_are_one_indexed_and_ordered(svc):
    results = await svc.extract_text_from_images(_pages(3))

    assert [r.page_number for r in results] == [1, 2, 3]
    assert svc._primary.calls == [1, 2, 3]


@pytest.mark.asyncio
async def test_no_images_means_no_results(svc):
    assert await svc.extract_text_from_images([]) == []


@pytest.mark.asyncio
async def test_without_a_fallback_a_primary_failure_is_terminal():
    """
    The shipped wiring: Vision alone, nothing behind it. A primary failure
    has nowhere to go here, and must still not raise — the document's real
    second chance is the LLM vision path in DocumentService.
    """
    s = OCRService()
    s._primary = FakeProvider("google_vision", fail_on={1})
    s._fallback = None

    results = await s.extract_text_from_images(_pages(1))
    assert results[0].provider == "none"
    assert results[0].text == ""


@pytest.mark.asyncio
async def test_failures_are_logged_without_the_exception_text(svc, caplog):
    """
    Correction #10: log the error *type*, never the message. Vendor
    exception strings carry request IDs, URLs and sometimes credentials.
    """
    svc._primary = FakeProvider("primary", fail_on={1})
    svc._fallback = FakeProvider("fallback", fail_on={1})

    with caplog.at_level(logging.WARNING, logger="finirag"):
        await svc.extract_text_from_images(_pages(1))

    types = [getattr(r, "error_type", None) for r in caplog.records]
    assert types == ["RuntimeError", "RuntimeError"]
    assert "failed on page 1" not in caplog.text


# ─────────────────────────────────────────────────────────────────────
# Image preprocessing (Trap D19 — EXIF rotation)
# ─────────────────────────────────────────────────────────────────────

@pytest.fixture()
def pre() -> ImagePreprocessor:
    return ImagePreprocessor()


def test_exif_rotation_is_applied_before_ocr(pre, rotated_jpeg):
    """
    The phone photo is stored landscape with an orientation tag of 6.
    Handing that to OCR sideways is the whole of Trap D19.
    """
    raw = Image.open(io.BytesIO(rotated_jpeg))
    assert raw.getexif().get(274) == 6, "fixture no longer carries the orientation tag"
    assert raw.width > raw.height, "fixture is not stored landscape"

    [out] = pre.preprocess_images(rotated_jpeg, "image/jpeg")

    assert out.height > out.width, "EXIF rotation was not applied"
    assert (out.width, out.height) == (raw.height, raw.width)


def test_rotation_reported_in_logs_matches_what_happened(pre, rotated_jpeg, caplog):
    """An audit line that always says `exif_rotated: false` is worse than none."""
    with caplog.at_level(logging.INFO, logger="finirag"):
        pre.preprocess_images(rotated_jpeg, "image/jpeg")

    record = next(r for r in caplog.records if r.message == "Image preprocessing")
    assert record.exif_rotated is True


def test_unrotated_image_is_not_reported_as_rotated(pre, sample_jpeg, caplog):
    with caplog.at_level(logging.INFO, logger="finirag"):
        pre.preprocess_images(sample_jpeg, "image/jpeg")

    record = next(r for r in caplog.records if r.message == "Image preprocessing")
    assert record.exif_rotated is False


def test_an_image_is_always_exactly_one_page(pre, sample_jpeg):
    assert len(pre.preprocess_images(sample_jpeg, "image/jpeg")) == 1


@pytest.mark.parametrize("mode", ["RGBA", "P", "CMYK", "1"])
def test_exotic_colour_modes_are_normalised_to_rgb(pre, mode):
    """OCR providers want RGB or greyscale; a palette PNG is neither."""
    buf = io.BytesIO()
    Image.new(mode, (32, 32)).save(buf, format="PNG" if mode != "CMYK" else "TIFF")

    [out] = pre.preprocess_images(buf.getvalue(), "image/png")
    assert out.mode == "RGB"


def test_greyscale_is_left_alone(pre):
    """Converting L to RGB triples the bytes sent to OCR for no gain."""
    buf = io.BytesIO()
    Image.new("L", (32, 32), 128).save(buf, format="PNG")

    [out] = pre.preprocess_images(buf.getvalue(), "image/png")
    assert out.mode == "L"


def test_truncated_jpeg_still_preprocesses(pre, sample_jpeg):
    """
    Phone cameras and mail gateways both produce JPEGs that end early.
    The gate already accepted this file; the preprocessor must not be
    the thing that drops it.
    """
    truncated = sample_jpeg[: int(len(sample_jpeg) * 0.6)]

    [out] = pre.preprocess_images(truncated, "image/jpeg")
    assert out.width > 0 and out.height > 0


def test_truncated_loading_is_not_left_enabled_globally(pre, sample_jpeg):
    """
    Flipping Pillow's LOAD_TRUNCATED_IMAGES is process-wide. Leaving it
    on would silently weaken the corruption gate for every later upload.
    """
    from PIL import ImageFile

    before = ImageFile.LOAD_TRUNCATED_IMAGES
    pre.preprocess_images(sample_jpeg[: int(len(sample_jpeg) * 0.6)], "image/jpeg")
    assert ImageFile.LOAD_TRUNCATED_IMAGES == before


def test_unreadable_bytes_still_raise(pre):
    """Resilience is not the same as swallowing garbage."""
    with pytest.raises(Exception):
        pre.preprocess_images(b"not an image at all", "image/png")


# ─────────────────────────────────────────────────────────────────────
# PDF rendering
# ─────────────────────────────────────────────────────────────────────

def test_pdf_renders_one_image_per_page(pre):
    images = pre.render_pdf_pages(_blank_pdf(pages=3))
    assert len(images) == 3


def test_pdf_is_rendered_at_300_dpi(pre):
    """
    Vector-outlined statements are unreadable below ~300 DPI. A US-Letter
    page is 612x792 points, so 300 DPI means 2550x3300 pixels.
    """
    [img] = pre.render_pdf_pages(_blank_pdf(pages=1, width=612, height=792))

    assert ImagePreprocessor.RENDER_DPI == 300
    assert abs(img.width - 2550) <= 2
    assert abs(img.height - 3300) <= 2


def test_rendered_pages_are_rgb(pre):
    """alpha=False — an alpha channel would confuse binarisation downstream."""
    [img] = pre.render_pdf_pages(_blank_pdf(pages=1))
    assert img.mode == "RGB"


def test_real_vector_pdf_renders_every_page(pre, sample_pdf):
    expected = fitz.open(stream=sample_pdf, filetype="pdf").page_count
    images = pre.render_pdf_pages(sample_pdf)

    assert len(images) == expected
    assert all(i.width > 1000 and i.height > 1000 for i in images)


# ─────────────────────────────────────────────────────────────────────
# Native-text vs OCR routing (Correction #1)
# ─────────────────────────────────────────────────────────────────────

@pytest.fixture()
def router() -> TextExtractionRouter:
    return TextExtractionRouter()


FINANCIAL_TEXT = (
    "Consolidated Balance Sheet\n"
    "Total assets and liabilities for the period\n"
    "Cash and balances with Reserve Bank of India\n"
)

# Long enough to clear the length gate, and deliberately free of every
# entry in FINANCIAL_KEYWORDS — substring matches included.
FILLER_TEXT = "Lorem ipsum dolor sit amet consectetur adipiscing elit sed do eiusmod."


def test_filler_text_really_contains_no_financial_keywords():
    """Guards the test below from a keyword sneaking in via substring match."""
    lower = FILLER_TEXT.lower()
    assert not [kw for kw in FINANCIAL_KEYWORDS if kw in lower]


def test_pdf_with_real_text_layer_skips_ocr(router):
    result = router.should_use_native_text(_text_pdf(FINANCIAL_TEXT), "application/pdf")

    assert result is not None
    assert result.method == "native_pdf"
    assert "Total assets" in " ".join(result.text_by_page)


def test_images_are_never_routed_to_native_text(router, sample_jpeg):
    """A JPEG has no text layer to probe, whatever the extension claimed."""
    assert router.should_use_native_text(sample_jpeg, "image/jpeg") is None
    assert router.should_use_native_text(sample_jpeg, "image/png") is None


def test_vector_outlined_statement_falls_through_to_ocr(router, sample_pdf):
    """
    The dataset's balance sheets are drawn as paths — no text layer at
    all. Trusting a native probe here would yield an empty extraction.
    """
    assert router.should_use_native_text(sample_pdf, "application/pdf") is None


def test_only_the_2022_cash_flow_has_a_usable_text_layer(router, dataset_dir):
    """
    A standing property of the shipped dataset. If this ever changes,
    the routing thresholds need re-examining, not silently adapting.
    """
    native = [
        p.name
        for p in sorted(dataset_dir.rglob("*.pdf"))
        if router.should_use_native_text(p.read_bytes(), "application/pdf")
    ]
    assert native == ["Consolidated Cash Flow Statement 2022.pdf"]


def test_short_text_layer_is_not_trusted(router):
    """A stray page number is not an extraction."""
    assert router.should_use_native_text(_text_pdf("Total 7"), "application/pdf") is None


def test_length_threshold_is_fifty_characters(router):
    assert TextExtractionRouter.MIN_TEXT_LENGTH == 50

    just_under = "Total " + "x" * (TextExtractionRouter.MIN_TEXT_LENGTH - 7)
    assert router.should_use_native_text(_text_pdf(just_under), "application/pdf") is None


def test_long_text_without_financial_keywords_is_rejected(router):
    """
    Boilerplate — a cover page, a disclaimer — clears the length bar but
    carries none of the numbers we are here for.
    """
    assert len(FILLER_TEXT) > TextExtractionRouter.MIN_TEXT_LENGTH
    assert router.should_use_native_text(_text_pdf(FILLER_TEXT), "application/pdf") is None


def test_corrupt_pdf_routes_to_ocr_instead_of_raising(router):
    assert router.probe_native_text(b"%PDF-1.4\n" + b"garbage" * 50) is None
    assert router.probe_native_text(b"") is None


def test_text_is_returned_per_page_with_a_matching_total(router):
    result = router.should_use_native_text(
        _text_pdf(FINANCIAL_TEXT, pages=3), "application/pdf"
    )

    assert len(result.text_by_page) == 3
    assert result.total_chars == sum(len(t) for t in result.text_by_page)
    assert result.total_chars > 0


def test_real_native_text_pdf_reads_its_line_items(router, dataset_dir):
    path = dataset_dir / "Cash Flows" / "Consolidated Cash Flow Statement 2022.pdf"
    if not path.is_file():
        pytest.skip(f"Missing native-text PDF: {path}")

    result = router.should_use_native_text(path.read_bytes(), "application/pdf")

    assert result is not None and result.method == "native_pdf"
    joined = " ".join(result.text_by_page).lower()
    assert "cash flow" in joined
    assert "operating" in joined
