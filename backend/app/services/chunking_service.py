"""
FiniRAG Chunking service — structure-aware, deterministic, rebuildable.

Provenance rule: `content` is a GENERATED structured summary for retrieval;
`source_evidence` carries the VERBATIM snippet only when the extraction
evidence actually contains one. Generated text is NEVER labeled verbatim:
source_evidence includes evidence_kind="verbatim"|"derived" and
verbatim_available. Narrative chunks from source_text are verbatim;
all structured chunks without a snippet are derived.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field

from backend.app.core.config import settings

INDEX_VERSION_DEFAULT = "v1"

# Approx chars per token for chunk sizing without a tokenizer.
CHARS_PER_TOKEN = 4


@dataclass
class ChunkDraft:
    content: str
    chunk_type: str
    chunk_index: int = 0
    page_number: int | None = None
    section_name: str | None = None
    reporting_period: str | None = None
    source_evidence: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)


def content_hash(content: str, index_version: str) -> str:
    return hashlib.sha256(f"{index_version}\x00{content}".encode()).hexdigest()


def split_narrative(text: str, target_tokens: int | None = None,
                    overlap_tokens: int | None = None) -> list[str]:
    """Split long narrative into ~target chunks with small overlap."""
    target = (target_tokens or settings.CHUNK_TARGET_TOKENS) * CHARS_PER_TOKEN
    overlap = (overlap_tokens or settings.CHUNK_OVERLAP_TOKENS) * CHARS_PER_TOKEN
    text = (text or "").strip()
    if not text:
        return []
    if len(text) <= target:
        return [text]
    # Split on paragraph boundaries first.
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[str] = []
    buf = ""
    for para in paras:
        if len(buf) + len(para) + 2 <= target or not buf:
            buf = f"{buf}\n\n{para}".strip() if buf else para
        else:
            chunks.append(buf)
            # Overlap: tail of previous buffer.
            tail = buf[-overlap:] if overlap > 0 else ""
            buf = f"{tail}\n\n{para}".strip() if tail else para
    if buf:
        chunks.append(buf)
    # Hard-split any still-oversized chunk by sentences.
    final: list[str] = []
    for c in chunks:
        if len(c) <= int(target * 1.5):
            final.append(c)
            continue
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", c) if s.strip()]
        sbuf = ""
        for s in sentences:
            if len(sbuf) + len(s) + 1 <= target or not sbuf:
                sbuf = f"{sbuf} {s}".strip()
            else:
                final.append(sbuf)
                tail = sbuf[-overlap:] if overlap > 0 else ""
                sbuf = f"{tail} {s}".strip() if tail else s
        if sbuf:
            final.append(sbuf)
    return final


def _ev_text(ev: dict | None) -> str:
    if not isinstance(ev, dict):
        return ""
    e = ev.get("evidence") or {}
    return str(e.get("source_text") or "")


def _ev_page(ev: dict | None) -> int | None:
    """Claimed 1-indexed page, or None when the evidence states none.

    No silent default: a missing page must not become "page 1" provenance.
    Callers that need a positional hint apply their own fallback; the
    verification pass decides what is presented as fact.
    """
    if not isinstance(ev, dict):
        return None
    e = ev.get("evidence") or {}
    if "page_number" not in e:
        return None
    try:
        return int(e["page_number"])
    except Exception:
        return None


def _val_str(ev: dict | None) -> str:
    if not isinstance(ev, dict):
        return "—"
    v = ev.get("value")
    if v is None:
        st = ev.get("status")
        return {"MISSING": "Missing", "UNREADABLE": "Unreadable",
                "EXPLICIT_ZERO": "0"}.get(st, "—")
    return str(v)


def _normalize_ws(text: str | None) -> str:
    """Lowercase whitespace-collapsed form for snippet containment checks."""
    return re.sub(r"\s+", " ", (text or "").lower()).strip()


# Generic corporate suffixes stripped for canonical company matching. Kept
# deliberately short: "bank", "steel", "motors" and the like are meaningful
# and must never be stripped.
_COMPANY_SUFFIXES = ("ltd", "limited", "inc", "incorporated", "corp",
                     "corporation", "llc", "pvt", "private", "co", "company")


def normalize_company(name: str | None) -> str | None:
    """Canonical company key for exact-match filtering (finding #19).

    Lowercase, punctuation dropped, generic suffixes removed. Returns None
    for empty input so "no company" stays distinguishable from "company X".
    """
    words = re.findall(r"[a-z0-9]+", (name or "").lower())
    words = [w for w in words if w not in _COMPANY_SUFFIXES]
    return " ".join(words) or None


def _split_source_pages(source_text: str) -> list[str]:
    """Retained source split back into per-page payloads."""
    return (source_text or "").split("--- PAGE BREAK ---")


def _src(ev: dict | None, fallback_page: int | None = None) -> dict:
    """Claimed evidence, explicitly NOT verified.

    A non-empty snippet is kept as transcribed context, but the kind stays
    "derived": only the verification pass in chunk_document() (which holds
    the retained source text) may promote a snippet to verbatim. Page numbers
    come only from explicit evidence — never a default, never a fallback.
    """
    text = _ev_text(ev)
    page = _ev_page(ev)
    return {"source_text": text[:2000] if text.strip() else "",
            "page_number": page,
            "evidence_kind": "derived",
            "verbatim_available": False}


def _verify_draft(d: ChunkDraft, pages: list[str] | None,
                  page_count: int | None) -> None:
    """Promote a draft's evidence to verbatim only on proof.

    Proof = the normalized snippet occurs in the normalized retained source
    of the claimed page (or, failing that, of exactly one locatable page, in
    which case the page is corrected to where the text actually is). Anything
    else — no source retained, snippet absent, page out of range — stays
    derived with no presented page number. Narrative chunks built directly
    from retained source text are verbatim by construction and skipped here
    by the caller.
    """
    sev = dict(d.source_evidence or {})
    snippet = (sev.get("source_text") or "").strip()
    if not snippet or not pages:
        sev["source_text"] = snippet[:2000]
        sev["evidence_kind"] = "derived"
        sev["verbatim_available"] = False
        sev["page_number"] = None
        d.source_evidence = sev
        d.page_number = None
        return
    norm_snip = _normalize_ws(snippet)
    norm_pages = [_normalize_ws(p) for p in pages]

    def _page_ok(p) -> bool:
        return (isinstance(p, int) and p >= 1
                and (page_count is None or p <= page_count))

    claimed = sev.get("page_number")
    page = None
    if _page_ok(claimed) and claimed <= len(norm_pages) \
            and norm_snip and norm_snip in norm_pages[claimed - 1]:
        page = claimed
    else:
        for i, p in enumerate(norm_pages, start=1):
            if norm_snip and norm_snip in p:
                page = i
                break
    if page is not None and _page_ok(page):
        sev["evidence_kind"] = "verbatim"
        sev["verbatim_available"] = True
        sev["page_number"] = page
        d.page_number = page
    else:
        sev["evidence_kind"] = "derived"
        sev["verbatim_available"] = False
        sev["page_number"] = None
        d.page_number = None
    d.source_evidence = sev


class ChunkingService:
    def __init__(self, index_version: str | None = None):
        self.index_version = index_version or settings.INDEX_VERSION

    def chunk_document(self, *, document_id: str, document_type: str,
                       extracted_data: dict | None, source_text: str = "",
                       document_name: str = "",
                       page_count: int | None = None) -> list[dict]:
        """Build deterministic chunk dicts (without embeddings).

        `source_text` is the retained OCR/native text and `page_count` the
        validated page total. Every non-narrative draft passes through
        _verify_draft: snippets proven against the retained source become
        verbatim with a proven page; everything else is derived with no
        presented page. Narrative chunks are verbatim by construction.
        """
        drafts: list[ChunkDraft] = []
        extracted_data = extracted_data or {}

        # Canonical company for exact-match filtering: company_name for
        # statements, vendor_name for invoices (a buyer-side filter would
        # need its own key; conflating the two would misattribute).
        if document_type == "invoice":
            _company_src = ((extracted_data.get("vendor_name") or {}).get("value")
                            or (extracted_data.get("vendor_name") or {}).get("original_value"))
        else:
            _company_src = ((extracted_data.get("company_name") or {}).get("value")
                            or (extracted_data.get("company_name") or {}).get("original_value"))
        company_norm = normalize_company(_company_src)

        if document_type == "invoice":
            drafts.extend(self._chunk_invoice(extracted_data, document_name))
        elif document_type in ("balance_sheet", "profit_and_loss", "cash_flow_statement"):
            drafts.extend(self._chunk_statement(document_type, extracted_data, document_name))
        else:
            drafts.extend(self._chunk_generic(extracted_data, document_name, document_type))

        pages = _split_source_pages(source_text) if source_text and source_text.strip() else None
        for d in drafts:
            if d.chunk_type != "narrative":
                _verify_draft(d, pages, page_count)

        # Narrative source text as paragraph chunks (VERBATIM, preserves page breaks).
        if source_text and source_text.strip():
            for page_no, page_text in enumerate(source_text.split("--- PAGE BREAK ---"), start=1):
                for part in split_narrative(page_text):
                    drafts.append(ChunkDraft(
                        content=f"Document: {document_name}\nType: {document_type}\nPage: {page_no}\n{part}",
                        chunk_type="narrative", page_number=page_no,
                        section_name="source_text",
                        source_evidence={"source_text": part[:2000], "page_number": page_no,
                                         "evidence_kind": "verbatim", "verbatim_available": True},
                        metadata={"document_name": document_name},
                    ))

        # Assign stable indexes + hashes, dedupe identical content per version.
        seen: set[str] = set()
        out: list[dict] = []
        idx = 0
        for d in drafts:
            content = (d.content or "").strip()
            if not content:
                continue
            h = content_hash(content, self.index_version)
            if h in seen:
                continue
            seen.add(h)
            out.append({
                "document_id": str(document_id),
                "chunk_index": idx,
                "chunk_type": d.chunk_type,
                "content": content,
                "page_number": d.page_number,
                "section_name": d.section_name,
                "document_type": document_type,
                "reporting_period": d.reporting_period,
                "company_norm": company_norm,
                "source_evidence": d.source_evidence,
                # Attribute name is chunk_metadata (column "metadata").
                # Never use the bare "metadata" key: DeclarativeBase already
                # defines .metadata (the MetaData object), so that kwarg is
                # silently swallowed and document identity is lost.
                "chunk_metadata": {**d.metadata, "document_name": document_name,
                                   "index_version": self.index_version},
                "embedding_model": settings.EMBEDDING_MODEL,
                "index_version": self.index_version,
                "content_hash": h,
            })
            idx += 1
        return out

    # ── Invoice ──
    def _chunk_invoice(self, data: dict, doc_name: str) -> list[ChunkDraft]:
        drafts: list[ChunkDraft] = []
        def txt(key: str) -> str:
            v = data.get(key) or {}
            return str(v.get("value") or v.get("original_value") or "")

        header = (
            f"Invoice: {txt('invoice_number')} | Date: {txt('invoice_date')} | "
            f"Vendor: {txt('vendor_name')} | Customer: {txt('customer_name')} | "
            f"Currency: {txt('currency')} | Subtotal: {txt('subtotal')} | "
            f"Tax: {txt('tax_amount')} ({txt('tax_rate')}) | Total: {txt('total_amount')} | "
            f"Paid: {txt('amount_paid')} | Due: {txt('amount_due')} | "
            f"Payment: {txt('payment_method')}"
        )
        page = _ev_page(data.get("total_amount")) or _ev_page(data.get("invoice_number"))
        # Header content is GENERATED (derived), not a verbatim quotation.
        # The joined snippets are transcribed context; the verification pass
        # promotes them only on proof against retained source text.
        header_snippets = [s for s in (
            _ev_text(data.get("invoice_number")), _ev_text(data.get("vendor_name")),
            _ev_text(data.get("total_amount"))) if s.strip()]
        drafts.append(ChunkDraft(content=f"Document: {doc_name}\n{header}",
                                 chunk_type="section", page_number=page,
                                 section_name="invoice_header",
                                 source_evidence={
                                     "source_text": " | ".join(header_snippets)[:2000],
                                     "page_number": page,
                                     "evidence_kind": "derived",
                                     "verbatim_available": False}))
        for i, item in enumerate(data.get("line_items") or []):
            desc = (item.get("description") or {}).get("value") or (item.get("description") or {}).get("original_value") or ""
            qty = _val_str(item.get("quantity"))
            price = _val_str(item.get("unit_price"))
            amt = _val_str(item.get("amount"))
            content = (f"Invoice: {txt('invoice_number')} | Line {i+1}: {desc} | "
                       f"Qty: {qty} | Unit price: {price} | Amount: {amt} | Currency: {txt('currency')}")
            pg = _ev_page(item.get("amount")) or page
            drafts.append(ChunkDraft(content=f"Document: {doc_name}\n{content}",
                                     chunk_type="table_row", page_number=pg,
                                     section_name="line_items",
                                     source_evidence=_src(item.get("amount"), fallback_page=pg)))
        for f in (data.get("additional_fields") or []):
            label = f.get("label", "")
            val = f.get("value") or {}
            content = f"Invoice: {txt('invoice_number')} | Field: {label} | Value: {_val_str(val)}"
            drafts.append(ChunkDraft(content=f"Document: {doc_name}\n{content}",
                                     chunk_type="financial_field", page_number=_ev_page(val) or page,
                                     section_name="additional_fields",
                                     source_evidence=_src(val, fallback_page=_ev_page(val) or page)))
        # Compact structured-field chunks for key facts.
        for key in ("subtotal", "tax_amount", "total_amount", "amount_due"):
            ev = data.get(key) or {}
            if (ev.get("status") in ("EXPLICIT", "EXPLICIT_ZERO")) and ev.get("value") is not None:
                fact = (f"Invoice: {txt('invoice_number')} | Field: {key} | Reported value: {ev.get('value')} | "
                        f"Unit: {ev.get('unit') or txt('currency')} | Source page: {_ev_page(ev) or page}")
                drafts.append(ChunkDraft(
                    content=f"Document: {doc_name}\n{fact}",
                    chunk_type="financial_field", page_number=_ev_page(ev) or page,
                    section_name=key,
                    source_evidence=_src(ev, fallback_page=_ev_page(ev) or page)))
        return drafts

    # ── Statements ──
    def _chunk_statement(self, doc_type: str, data: dict, doc_name: str) -> list[ChunkDraft]:
        drafts: list[ChunkDraft] = []
        company = str(((data.get("company_name") or {}).get("value") or "")) or doc_name
        unit = str(((data.get("unit_descriptor") or {}).get("value") or "")) or str(data.get("unit_code") or "")
        label = {"balance_sheet": "Balance Sheet", "profit_and_loss": "Profit and Loss",
                 "cash_flow_statement": "Cash Flow Statement"}[doc_type]
        for period_key, period in (data.get("periods") or {}).items():
            # Group dict sections into one chunk per section (keeps header+rows together).
            for section_key in ("capital_and_liabilities", "assets", "appropriations",
                                "operating_items", "investing_items", "financing_items"):
                items = period.get(section_key) or {}
                if not items:
                    continue
                lines = [f"{k}: {_val_str(v)}" for k, v in items.items()]
                # Batch rows to stay near target size.
                batch: list[str] = []
                batch_len = 0
                target = settings.CHUNK_TARGET_TOKENS * CHARS_PER_TOKEN
                pg = 1
                for line, (k, v) in zip(lines, items.items()):
                    pg = _ev_page(v) or pg
                    if batch_len + len(line) > target and batch:
                        drafts.append(self._section_chunk(label, company, period_key, unit, section_key, batch, pg, doc_name))
                        batch, batch_len = [], 0
                    batch.append(line)
                    batch_len += len(line)
                if batch:
                    drafts.append(self._section_chunk(label, company, period_key, unit, section_key, batch, pg, doc_name))
            # Scalar fields as individual financial-field chunks (verbatim when snippet exists).
            for fkey, fval in period.items():
                if isinstance(fval, dict) and "value" in fval:
                    if fval.get("status") not in ("EXPLICIT", "EXPLICIT_ZERO"):
                        continue
                    content = (f"Statement: {label} | Company: {company} | Period: {period_key} | "
                               f"Field: {fkey} | Reported value: {fval.get('value')} | Unit: {unit} | "
                               f"Source page: {_ev_page(fval) or 1}")
                    drafts.append(ChunkDraft(content=f"Document: {doc_name}\n{content}",
                                             chunk_type="financial_field", page_number=_ev_page(fval) or 1,
                                             section_name=fkey, reporting_period=str(period_key),
                                             source_evidence=_src(fval, fallback_page=_ev_page(fval) or 1)))
        return drafts

    def _section_chunk(self, label: str, company: str, period: str, unit: str,
                       section: str, lines: list[str], page: int, doc_name: str) -> ChunkDraft:
        body = "\n".join(f"- {ln}" for ln in lines)
        content = (f"Statement: {label} | Company: {company} | Period: {period} | "
                   f"Section: {section} | Unit: {unit}\n{body}")
        # Section body is a generated aggregation of row labels+values (derived).
        return ChunkDraft(content=f"Document: {doc_name}\n{content}",
                          chunk_type="table_row", page_number=page,
                          section_name=section, reporting_period=str(period),
                          source_evidence={"source_text": "", "page_number": page,
                                           "evidence_kind": "derived",
                                           "verbatim_available": False})

    def _chunk_generic(self, data: dict, doc_name: str, doc_type: str) -> list[ChunkDraft]:
        import json

        text = json.dumps(data)[:6000]
        return [ChunkDraft(content=f"Document: {doc_name}\nType: {doc_type}\n{text}",
                           chunk_type="section", page_number=1,
                           section_name="extracted_data",
                           source_evidence={"source_text": text[:2000], "page_number": 1,
                                            "evidence_kind": "derived",
                                            "verbatim_available": False})]
