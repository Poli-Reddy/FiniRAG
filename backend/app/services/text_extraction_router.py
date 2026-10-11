"""
FiniRAG Text Extraction Router (Correction #1).

Decides whether to use native PDF text extraction (PyMuPDF) or
render-and-OCR based on the document content.

Flow:
  PDF → probe for native text → usable? → native extraction
                                     └──→ render 300 DPI → OCR
  JPG/PNG → always render → OCR

A PDF has "usable" native text when:
  1. Text length >= 50 characters
  2. Contains at least one financial keyword (Amount, Total, etc.)
  3. Text is NOT just whitespace or noise

Only one PDF in the dataset (CF Statement 2022) has usable native text.
All others are vector-outlined (no text layer, no images — just paths).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import fitz  # PyMuPDF

logger = logging.getLogger("finirag")

# Keywords that indicate genuinely useful text content
FINANCIAL_KEYWORDS = [
    "total", "amount", "balance", "income", "expenditure",
    "profit", "loss", "cash", "net", "assets", "liabilities",
    "interest", "tax", "date", "period", "schedule",
    "operating", "investing", "financing",
    "invoice", "receipt", "qty", "quantity", "price",
    "subtotal", "discount", "payment", "due",
]


@dataclass
class TextExtractionResult:
    """Result of text extraction routing."""
    method: str                    # "native_pdf" or "ocr"
    text_by_page: list[str]        # Text per page
    total_chars: int = 0


class TextExtractionRouter:
    """Routes between native PDF text extraction and OCR."""

    MIN_TEXT_LENGTH = 50  # Minimum chars to consider native text usable
    MIN_KEYWORD_MATCHES = 1  # Minimum financial keywords required
    def probe_native_text(self, file_bytes: bytes) -> TextExtractionResult | None:
        """
        Attempt native text extraction from PDF using PyMuPDF.

        Returns TextExtractionResult if native text is usable, None otherwise.
        """
        try:
            doc = fitz.open(stream=file_bytes, filetype="pdf")
        except Exception:
            return None

        text_by_page = []
        total_chars = 0

        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text("text").strip()
            text_by_page.append(text)
            total_chars += len(text)

        doc.close()

        # ── Usability check ──
        if total_chars < self.MIN_TEXT_LENGTH:
            logger.info("Text extraction routing", extra={
                "method": "ocr",
                "reason": f"native text too short ({total_chars} chars)",
            })
            return None

        # Check for financial keywords
        all_text_lower = " ".join(text_by_page).lower()
        keyword_count = sum(
            1 for kw in FINANCIAL_KEYWORDS if kw in all_text_lower
        )

        if keyword_count < self.MIN_KEYWORD_MATCHES:
            logger.info("Text extraction routing", extra={
                "method": "ocr",
                "reason": f"insufficient keywords ({keyword_count})",
            })
            return None

        logger.info("Text extraction routing", extra={
            "method": "native_pdf",
            "chars": total_chars,
            "keywords_matched": keyword_count,
        })

        return TextExtractionResult(
            method="native_pdf",
            text_by_page=text_by_page,
            total_chars=total_chars,
        )

    def should_use_native_text(self, file_bytes: bytes, file_type: str) -> TextExtractionResult | None:
        """
        Main routing decision.

        Only PDFs can have native text. Images always go through OCR.
        """
        if file_type != "application/pdf":
            return None

        return self.probe_native_text(file_bytes)


# ─────────────────────────────────────────────────────────────────────
# Two-stage hybrid routing: per-page classification (Stage 1)
#
# The document-level probe above is all-or-nothing: a page that mixes
# selectable text with a scanned region (photographed receipt, stamped
# total, scanned table) either loses the image content (native wins) or
# pays full-page OCR (OCR wins). The classifier below runs per page so
# each page takes the cheapest path that loses nothing.
# ─────────────────────────────────────────────────────────────────────

# A page needs at least this much native text to be considered for NATIVE.
PAGE_MIN_NATIVE_CHARS = 50
# Image area fraction below which raster regions are decor (logos, rules)
# rather than content: the page stays NATIVE.
PAGE_MAX_IMAGE_FRACTION_FOR_NATIVE = 0.15
# Image boxes below this fraction of page area are never OCR'd on MIXED
# pages (icons, bullets, ornamental rules — OCR on them is pure noise).
PAGE_MIN_OCR_BLOCK_AREA_FRACTION = 0.005

# Classification labels. MIXED is also the default for anything uncertain:
# block-level analysis degrades to native-only when OCR is unavailable,
# so over-classifying MIXED costs a little work but never loses content.
PAGE_NATIVE = "NATIVE"
PAGE_SCANNED = "SCANNED"
PAGE_MIXED = "MIXED"


@dataclass
class PageClassification:
    """Stage-1 verdict for one PDF page. Pure function of page geometry +
    text layer — no OCR, no network, deterministic for the same bytes."""
    page_number: int                 # 1-indexed
    kind: str                        # NATIVE | SCANNED | MIXED
    text_chars: int = 0
    text_coverage: float = 0.0       # native text area / page area
    image_fraction: float = 0.0      # raster image area / page area
    image_boxes: list = field(default_factory=list)  # PDF-point rects
    native_text_coverage: float = 0.0  # alias kept for logging parity


def _clip_box(box, page_rect):
    """Intersect a rect with the page; None when fully outside."""
    x0 = max(box[0], page_rect[0])
    y0 = max(box[1], page_rect[1])
    x1 = min(box[2], page_rect[2])
    y1 = min(box[3], page_rect[3])
    if x1 <= x0 or y1 <= y0:
        return None
    return (x0, y0, x1, y1)


def classify_page(page, page_number: int) -> PageClassification:
    """Classify one open PyMuPDF page without rendering or OCR.

    - NATIVE: usable text layer (>= PAGE_MIN_NATIVE_CHARS) and only
      decor-level imagery (< PAGE_MAX_IMAGE_FRACTION_FOR_NATIVE).
    - SCANNED: no usable text layer — full-page OCR is the only option.
    - MIXED: text plus significant image regions — block-level selective
      OCR. Also the verdict for anything uncertain (partial/broken text
      layer), since MIXED degrades safely to native-only.

    Raises on a broken page object so the caller can mark the page
    UNREADABLE instead of guessing.
    """
    rect = page.rect
    page_area = max(float(rect.width * rect.height), 1.0)

    # Native text signal from the text layer (no rendering).
    text_chars = 0
    text_area = 0.0
    try:
        d = page.get_text("dict")
    except Exception as e:
        raise ValueError(f"unreadable page text layer: {type(e).__name__}") from e
    for block in d.get("blocks", []):
        if block.get("type", 0) != 0:
            continue
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                text_chars += len((span.get("text") or "").strip())
        b = block.get("bbox")
        if b:
            clipped = _clip_box(b, (rect.x0, rect.y0, rect.x1, rect.y1))
            if clipped:
                text_area += (clipped[2] - clipped[0]) * (clipped[3] - clipped[1])

    # Raster signal: embedded images by bbox area (clipped, deduped).
    image_boxes: list[tuple] = []
    try:
        for img in page.get_images(full=True):
            try:
                bboxes = page.get_image_bbox(img)
            except Exception:
                continue
            if isinstance(bboxes, (list, tuple)) and bboxes and isinstance(bboxes[0], (list, tuple)):
                candidates = list(bboxes)
            else:
                candidates = [bboxes]
            for b in candidates:
                try:
                    clipped = _clip_box((float(b[0]), float(b[1]), float(b[2]), float(b[3])),
                                        (rect.x0, rect.y0, rect.x1, rect.y1))
                except Exception:
                    continue
                if clipped:
                    image_boxes.append(clipped)
    except Exception:
        # Image listing itself failed — treat as no imagery signal rather
        # than failing the page; text evidence below still decides.
        image_boxes = []
    image_boxes = _dedupe_boxes(image_boxes)
    image_area = sum((b[2] - b[0]) * (b[3] - b[1]) for b in image_boxes)

    coverage = text_area / page_area
    img_frac = image_area / page_area
    result = PageClassification(
        page_number=page_number,
        kind=PAGE_MIXED,  # default: uncertain classifies MIXED (safe side)
        text_chars=text_chars,
        text_coverage=coverage,
        image_fraction=img_frac,
        image_boxes=image_boxes,
        native_text_coverage=coverage,
    )
    if text_chars < PAGE_MIN_NATIVE_CHARS:
        result.kind = PAGE_SCANNED
    elif img_frac < PAGE_MAX_IMAGE_FRACTION_FOR_NATIVE:
        result.kind = PAGE_NATIVE
    else:
        result.kind = PAGE_MIXED
    return result


def _dedupe_boxes(boxes: list[tuple]) -> list[tuple]:
    """Drop near-duplicate boxes (same image referenced twice), keep largest."""
    kept: list[tuple] = []
    for b in sorted(boxes, key=lambda r: -((r[2] - r[0]) * (r[3] - r[1]))):
        if any(_iou(b, k) > 0.9 for k in kept):
            continue
        kept.append(b)
    return kept


def _iou(a, b) -> float:
    """Symmetric intersection-over-union of two (x0, y0, x1, y1) rects."""
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    if inter <= 0:
        return 0.0
    area_a = max(0.0, (a[2] - a[0]) * (a[3] - a[1]))
    area_b = max(0.0, (b[2] - b[0]) * (b[3] - b[1]))
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0
