"""
FiniRAG hybrid text extraction — two-stage routing for mixed-content PDFs.

Stage 1 (fast filter): each page is classified NATIVE / SCANNED / MIXED by
text-layer coverage vs raster-image area. No rendering, no OCR, no network.

Stage 2 (precise extraction):
  NATIVE  → PyMuPDF text (+ structured tables), never OCR'd.
  SCANNED → full-page render + OCR (existing path, unchanged).
  MIXED   → native blocks + tables from the text layer, OCR ONLY on
            cropped image regions, merged in reading order with
            native-preferred deduplication.

Every block carries provenance (source_type / page_number / bbox /
block_type) so citations stay honest about verbatim-native vs
OCR-derived text. Provenance is small JSON in processing metadata;
merged page text flows through the unchanged PAGE_SEPARATOR pipeline
into source_text retention and the LLM — structured extraction,
validation, persistence and indexing are untouched.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger("finirag")

# Reading-order band height (PDF points). Blocks whose tops fall in the
# same 12pt band read left-to-right; bands read top-to-bottom. Documented
# limit: multi-column layouts may interleave — v1 accepts that rather
# than pretending to solve column segmentation.
READING_ORDER_BAND_PT = 12.0
# An OCR crop this covered by native text is a duplicate (caption over a
# figure, text baked beside a stamp): prefer native, discard the OCR copy.
OCR_NATIVE_OVERLAP_DISCARD = 0.5
# A dict text block this covered by an extracted native table is the same
# content twice: keep the structured table, drop the flat echo.
TABLE_TEXT_OVERLAP_SUPPRESS = 0.5
# Crops smaller than this (rendered pixels) are decor noise, not content.
MIN_CROP_PIXELS = 8
# Marker for image regions the OCR chain could not read. Short and
# explicit: downstream must see that content exists here, not silence.
UNREADABLE_MARKER = "[unreadable image region]"


@dataclass
class TextBlock:
    """One ordered unit of page text with its provenance."""
    text: str
    bbox: tuple | None = None          # PDF points (x0, y0, x1, y1)
    block_type: str = "text"           # text | table | image | unreadable
    source_type: str = "native"        # native | ocr | unreadable


@dataclass
class HybridPageResult:
    """Everything one PDF page produced (JSON-serializable provenance)."""
    page_number: int
    classification: str
    text: str = ""
    blocks: list = field(default_factory=list)
    ocr_provider: str | None = None
    provenance: dict | None = None


def _box_area(box) -> float:
    return max(0.0, (box[2] - box[0]) * (box[3] - box[1]))


def _intersect(a, b):
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    if x1 <= x0 or y1 <= y0:
        return None
    return (x0, y0, x1, y1)


def _coverage_fraction(inner, outer) -> float:
    """Fraction of `inner` covered by `outer` (asymmetric — the dedup question
    is always 'is the OCR/table region already represented', never IoU)."""
    inter = _intersect(inner, outer)
    if inter is None:
        return 0.0
    area = _box_area(inner)
    return _box_area(inter) / area if area > 0 else 0.0


def extract_native_blocks(page) -> tuple[list[TextBlock], list[TextBlock]]:
    """Native text blocks + structured tables from the text layer.

    Returns (text_blocks, table_blocks). A dict text block mostly covered
    by a table bbox is the same content twice — the structured table wins
    and the flat echo is dropped. Table extraction itself is best-effort:
    find_tables() can raise on pathological pages, in which case tables
    are skipped and plain text still flows.
    """
    text_blocks: list[TextBlock] = []
    try:
        d = page.get_text("dict")
    except Exception as e:
        raise ValueError(f"unreadable page text layer: {type(e).__name__}") from e
    for block in d.get("blocks", []):
        if block.get("type", 0) != 0:
            continue
        lines = []
        for line in block.get("lines", []):
            spans = [s.get("text") or "" for s in line.get("spans", [])]
            line_text = " ".join(t for t in spans if t).strip()
            if line_text:
                lines.append(line_text)
        if not lines:
            continue
        bbox = block.get("bbox")
        text_blocks.append(TextBlock(
            text="\n".join(lines),
            bbox=tuple(bbox) if bbox else None,
            block_type="text", source_type="native"))

    table_blocks: list[TextBlock] = []
    try:
        tables = page.find_tables()
    except Exception as e:
        logger.warning("Native table detection skipped", extra={
            "error_type": type(e).__name__})
        tables = None
    if tables:
        try:
            for table in tables:
                rows = []
                try:
                    extracted = table.extract()
                except Exception:
                    continue
                for row in extracted or []:
                    cells = [(c or "").strip() for c in row]
                    if any(cells):
                        rows.append(" | ".join(cells))
                if not rows:
                    continue
                bbox = tuple(table.bbox) if getattr(table, "bbox", None) is not None else None
                table_blocks.append(TextBlock(
                    text="\n".join(rows), bbox=bbox,
                    block_type="table", source_type="native"))
        except Exception as e:
            logger.warning("Native table iteration skipped", extra={
                "error_type": type(e).__name__})

    if table_blocks:
        kept = []
        for tb in text_blocks:
            if tb.bbox is None:
                kept.append(tb)
                continue
            if any(_coverage_fraction(tb.bbox, t.bbox) >= TABLE_TEXT_OVERLAP_SUPPRESS
                   for t in table_blocks if t.bbox is not None):
                continue  # flat echo of a structured table
            kept.append(tb)
        text_blocks = kept
    return text_blocks, table_blocks


def _reading_key(block: TextBlock):
    if block.bbox is None:
        return (10 ** 9, 0.0)
    return (int(block.bbox[1] // READING_ORDER_BAND_PT), block.bbox[0])


def merge_blocks(native: list[TextBlock], ocr: list[TextBlock]) -> list[TextBlock]:
    """Merge native + OCR blocks into reading order with native-preferred
    deduplication. Deterministic: same inputs, same output, always."""
    ocr_kept = []
    for ob in ocr:
        if ob.bbox is None:
            ocr_kept.append(ob)
            continue
        covered = 0.0
        for nb in native:
            if nb.bbox is None:
                continue
            inter = _intersect(ob.bbox, nb.bbox)
            if inter is not None:
                covered += _box_area(inter)
        if _box_area(ob.bbox) > 0 and covered / _box_area(ob.bbox) >= OCR_NATIVE_OVERLAP_DISCARD:
            continue  # native text already says this — discard the OCR copy
        ocr_kept.append(ob)
    return sorted(native + ocr_kept, key=_reading_key)


def blocks_to_text(blocks: list[TextBlock]) -> str:
    return "\n".join(b.text for b in blocks if (b.text or "").strip()).strip()


async def extract_pdf_hybrid(file_bytes: bytes, preprocessor, ocr_service,
                             zoom: float | None = None) -> dict:
    """Two-stage hybrid extraction for one PDF.

    Opens the document ONCE, classifies each page, renders only pages
    that need pixels (SCANNED fully, MIXED per image crop), batches ALL
    OCR images into a SINGLE provider call (same call discipline as the
    old full-page path — no extra per-page round trips), then merges.
    Returns {"page_texts", "method", "ocr_provider", "pages",
             "ocr_pages", "ocr_blocks"} — all JSON-serializable except
    page_texts (plain strings, PAGE_SEPARATOR-joined by the caller).

    Pixel-cap violations still raise (security gate, never degraded);
    everything else degrades per page, never per document.
    """
    import fitz

    from backend.app.services.text_extraction_router import (
        PAGE_MIN_OCR_BLOCK_AREA_FRACTION, PAGE_MIXED, PAGE_NATIVE,
        PAGE_SCANNED, classify_page,
    )

    render_zoom = zoom or (preprocessor.RENDER_DPI / 72.0)
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    try:
        # ── Pass 1: classify every page (no rendering, no OCR) ──
        plan: list[dict] = []
        for page_num in range(len(doc)):
            page_number = page_num + 1
            try:
                classification = classify_page(doc[page_num], page_number)
                plan.append({"page": doc[page_num], "classification": classification,
                             "unreadable": None})
            except Exception as e:
                logger.warning("Page classification failed, marking unreadable", extra={
                    "page_number": page_number, "error_type": type(e).__name__})
                plan.append({"page": None, "classification": None,
                             "unreadable": type(e).__name__})

        # ── Pass 2: native blocks now; collect OCR jobs (render on demand) ──
        ocr_jobs: list[dict] = []  # {page_idx, kind, image, box?}
        native_per_page: list[list[TextBlock]] = []
        for page_idx, entry in enumerate(plan):
            if entry["unreadable"] is not None:
                native_per_page.append([])
                continue
            page = entry["page"]
            classification = entry["classification"]
            if classification.kind == PAGE_NATIVE:
                text_blocks, table_blocks = extract_native_blocks(page)
                native_per_page.append(text_blocks + table_blocks)
                continue
            image = preprocessor.render_page_image(page)
            if classification.kind == PAGE_SCANNED:
                ocr_jobs.append({"page_idx": page_idx, "kind": "full",
                                 "image": image, "box": None})
                native_per_page.append([])
                continue
            # MIXED: native layer + crop list for selective OCR.
            text_blocks, table_blocks = extract_native_blocks(page)
            native_per_page.append(text_blocks + table_blocks)
            pw, ph = image.width, image.height
            page_area = max(float(page.rect.width * page.rect.height), 1.0)
            for box in classification.image_boxes:
                if _box_area(box) / page_area < PAGE_MIN_OCR_BLOCK_AREA_FRACTION:
                    continue  # decor, not content
                px = (int(box[0] * render_zoom), int(box[1] * render_zoom),
                      int(box[2] * render_zoom), int(box[3] * render_zoom))
                px = (max(0, px[0]), max(0, px[1]), min(pw, px[2]), min(ph, px[3]))
                if px[2] - px[0] < MIN_CROP_PIXELS or px[3] - px[1] < MIN_CROP_PIXELS:
                    continue
                ocr_jobs.append({"page_idx": page_idx, "kind": "crop",
                                 "image": image.crop(px), "box": box})

        # ── Pass 3: ONE batched OCR call for the whole document ──
        job_results: list = []
        if ocr_jobs:
            try:
                job_results = await ocr_service.extract_text_from_images(
                    [j["image"] for j in ocr_jobs])
            except Exception as e:
                logger.warning("Batched OCR failed, marking blocks unreadable", extra={
                    "error_type": type(e).__name__})
                job_results = []
        while len(job_results) < len(ocr_jobs):
            job_results.append(None)

        # ── Pass 4: assemble pages ──
        page_texts: list[str] = []
        pages: list[dict] = []
        providers: list[str] = []
        ocr_pages = 0
        ocr_blocks = 0
        kinds: list[str] = []
        by_page: dict[int, list] = {}
        for job, result in zip(ocr_jobs, job_results):
            by_page.setdefault(job["page_idx"], []).append((job, result))
        for page_idx, entry in enumerate(plan):
            page_number = page_idx + 1
            if entry["unreadable"] is not None:
                page_texts.append("")
                pages.append({"page_number": page_number, "classification": "UNREADABLE",
                              "native_text_coverage": 0.0, "image_block_count": 0,
                              "ocr_blocks_processed": 0, "source_types": []})
                kinds.append("UNREADABLE")
                continue
            classification = entry["classification"]
            kinds.append(classification.kind)
            ocr_pieces: list[TextBlock] = []
            ran = 0
            for job, result in by_page.get(page_idx, []):
                ran += 1
                text = (result.text or "").strip() if result is not None else ""
                provider = getattr(result, "provider", None) if result is not None else None
                if provider and provider not in providers:
                    providers.append(provider)
                if job["kind"] == "full":
                    if text:
                        ocr_pieces.append(TextBlock(
                            text=text, bbox=None,
                            block_type="image", source_type="ocr"))
                    # An empty full-page result simply contributes no block;
                    # an all-empty document still reaches vision fallback.
                else:
                    if text:
                        ocr_pieces.append(TextBlock(
                            text=text, bbox=job["box"],
                            block_type="image", source_type="ocr"))
                    else:
                        ocr_pieces.append(TextBlock(
                            text=UNREADABLE_MARKER, bbox=job["box"],
                            block_type="unreadable", source_type="unreadable"))
            blocks = merge_blocks(native_per_page[page_idx], ocr_pieces)
            page_texts.append(blocks_to_text(blocks))
            if ran:
                ocr_pages += 1
                ocr_blocks += ran
            source_types = sorted({b.source_type for b in blocks})
            pages.append({"page_number": page_number,
                          "classification": classification.kind,
                          "native_text_coverage": round(classification.native_text_coverage, 4),
                          "image_block_count": len(classification.image_boxes),
                          "ocr_blocks_processed": ran,
                          "source_types": source_types})
            logger.info("Hybrid page extraction", extra={
                "page_number": page_number,
                "classification": classification.kind,
                "native_text_coverage": round(classification.native_text_coverage, 4),
                "image_block_count": len(classification.image_boxes),
                "ocr_blocks_processed": ran})

        if not kinds:
            method = "ocr"
        elif all(k == PAGE_NATIVE for k in kinds):
            method = "native_pdf"
        elif all(k == PAGE_SCANNED for k in kinds):
            method = "ocr"
        else:
            method = "hybrid"
        used = [p for p in providers if p != "none"]
        ocr_provider = "+".join(dict.fromkeys(used)) or ("none" if "none" in providers else None)
        return {"page_texts": page_texts, "method": method,
                "ocr_provider": ocr_provider, "pages": pages,
                "ocr_pages": ocr_pages, "ocr_blocks": ocr_blocks}
    finally:
        doc.close()
