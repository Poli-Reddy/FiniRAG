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
    if not isinstance(ev, dict):
        return None
    e = ev.get("evidence") or {}
    try:
        return int(e.get("page_number", 1))
    except Exception:
        return 1


def _val_str(ev: dict | None) -> str:
    if not isinstance(ev, dict):
        return "—"
    v = ev.get("value")
    if v is None:
        st = ev.get("status")
        return {"MISSING": "Missing", "UNREADABLE": "Unreadable",
                "EXPLICIT_ZERO": "0"}.get(st, "—")
    return str(v)


def _src(ev: dict | None, fallback_page: int | None = 1) -> dict:
    """Verbatim snippet when present, else explicit derived marker.

    Never fabricates a quotation: empty snippet + evidence_kind=derived.
    """
    text = _ev_text(ev)
    page = _ev_page(ev)
    if text.strip():
        out: dict = {"source_text": text[:2000], "evidence_kind": "verbatim",
                     "verbatim_available": True}
        if page is not None:
            out["page_number"] = page
        bbox = (ev.get("evidence") or {}).get("bounding_box") if isinstance(ev, dict) else None
        if bbox:
            out["bounding_box"] = bbox
        return out
    out = {"source_text": "", "evidence_kind": "derived",
           "verbatim_available": False}
    if fallback_page is not None:
        out["page_number"] = fallback_page
    return out


class ChunkingService:
    def __init__(self, index_version: str | None = None):
        self.index_version = index_version or settings.INDEX_VERSION

    def chunk_document(self, *, document_id: str, document_type: str,
                       extracted_data: dict | None, source_text: str = "",
                       document_name: str = "") -> list[dict]:
        """Build deterministic chunk dicts (without embeddings)."""
        drafts: list[ChunkDraft] = []
        extracted_data = extracted_data or {}

        if document_type == "invoice":
            drafts.extend(self._chunk_invoice(extracted_data, document_name))
        elif document_type in ("balance_sheet", "profit_and_loss", "cash_flow_statement"):
            drafts.extend(self._chunk_statement(document_type, extracted_data, document_name))
        else:
            drafts.extend(self._chunk_generic(extracted_data, document_name, document_type))

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
        page = _ev_page(data.get("total_amount")) or _ev_page(data.get("invoice_number")) or 1
        # Header content is GENERATED (derived), not a verbatim quotation.
        header_snippets = [s for s in (
            _ev_text(data.get("invoice_number")), _ev_text(data.get("vendor_name")),
            _ev_text(data.get("total_amount"))) if s.strip()]
        drafts.append(ChunkDraft(content=f"Document: {doc_name}\n{header}",
                                 chunk_type="section", page_number=page,
                                 section_name="invoice_header",
                                 source_evidence={
                                     "source_text": " | ".join(header_snippets)[:2000],
                                     "page_number": page,
                                     "evidence_kind": "verbatim" if header_snippets else "derived",
                                     "verbatim_available": bool(header_snippets)}))
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
                           source_evidence={"source_text": text[:2000], "page_number": 1})]
