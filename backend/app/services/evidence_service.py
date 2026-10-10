"""
FiniRAG Evidence service — single evidence package with stable citations.
Priority: structured values+validation > field source evidence > passages.
Contradictions preserved; failed checks stay visible.
"""

from __future__ import annotations


class EvidenceService:
    def build(self, *, sql_results: list[dict] | None = None,
              validation: dict | None = None,
              candidates: list[dict] | None = None) -> dict:
        items: list[dict] = []
        seen: set[str] = set()

        def add(item: dict):
            key = (item.get("document_id", ""), item.get("content", "") or
                   str(item.get("value", "")) + (item.get("field") or ""))
            if key in seen:
                return
            seen.add(key)
            item["citation_id"] = f"CIT-{len(items) + 1}"
            items.append(item)

        for r in (sql_results or []):
            if not isinstance(r, dict):
                continue
            if r.get("ok"):
                ev = r.get("evidence") or {}
                add({"kind": "structured_value", "document_id": r.get("document_id", ""),
                     "document_name": r.get("document_name", ""),
                     "document_type": r.get("document_type", ""),
                     "page_number": ev.get("page_number"),
                     "section": r.get("section") or r.get("field"),
                     "field": r.get("field"), "period": r.get("period"),
                     "content": f"{r.get('field')} = {r.get('value')} "
                                f"(original: {r.get('original_value')})",
                     "value": r.get("value"), "unit": r.get("unit"),
                     "status": r.get("status"),
                     "source_text": ev.get("source_text", ""),
                     "retrieval_method": "sql"})
            else:
                # Keep insufficient markers so the answer can abstain precisely.
                add({"kind": "unavailable", "document_id": r.get("document_id", ""),
                     "document_name": r.get("document_name", ""),
                     "content": f"Unavailable: {r.get('field')} ({r.get('error')})",
                     "retrieval_method": "sql", "status": r.get("status"),
                     "field": r.get("field")})
        if isinstance(validation, dict) and validation.get("ok", True):
            for c in (validation.get("checks") or [])[:20]:
                add({"kind": "validation_check",
                     "document_id": validation.get("document_id", ""),
                     "document_name": validation.get("document_name", ""),
                     "content": f"Check {c.get('name')}: {c.get('status')} "
                                f"(formula: {c.get('formula')})",
                     "check": c, "retrieval_method": "validation"})
        for c in (candidates or []):
            sev = c.get("source_evidence") or {}
            add({"kind": "passage", "document_id": c.get("document_id", ""),
                 "document_name": c.get("document_name", ""),
                 "document_type": c.get("document_type", ""),
                 "page_number": c.get("page_number"),
                 "section": c.get("section_name"),
                 "content": c.get("content", "")[:2000],
                 "source_text": sev.get("source_text", ""),
                 "evidence_kind": sev.get("evidence_kind", "derived"),
                 "verbatim_available": sev.get("verbatim_available", False),
                 "retrieval_method": c.get("retrieval_method", "hybrid"),
                 "index_version": c.get("index_version")})
        return {"items": items, "count": len(items)}

    @staticmethod
    def support_pool(item: dict) -> set[str]:
        """All tokens an item can vouch for: snippet/content plus the
        structured value, period, section and document name."""
        import re

        def toks(s) -> set[str]:
            return set(re.findall(r"[a-z0-9]+", str(s or "").lower()))

        pool: set[str] = set()
        src = str(item.get("source_text") or "")
        content = str(item.get("content") or "")
        if item.get("evidence_kind") == "verbatim" and src.strip():
            pool |= toks(src)
        pool |= toks(content)
        for key in ("value", "period", "section", "document_name"):
            pool |= toks(item.get(key))
        return pool

    @staticmethod
    def supports(item: dict, claim_text: str) -> bool:
        """Claim-level support guard: normalized claim tokens appear in the
        item's support pool (snippet and/or content plus structured
        value/period/section). Existence of a citation ID is NOT support.

        This is a basic deterministic guard, not proof of semantic
        entailment — the answer pipeline additionally cross-checks exact
        financial values and rejects wrong periods/currencies/sources.
        """
        import re

        def toks(s: str) -> set[str]:
            return set(re.findall(r"[a-z0-9]+", (s or "").lower()))

        claim = toks(claim_text) - {"the", "a", "an", "of", "and", "is", "in", "for"}
        if not claim:
            return True
        pool = EvidenceService.support_pool(item)
        # Numbers must match exactly when the claim contains digits.
        nums = {t for t in claim if any(ch.isdigit() for ch in t)}
        if nums and not nums.issubset(pool):
            return False
        # Negation guard: explicit negations in the claim but absent from the
        # evidence means the polarity cannot be established from this source
        # — overlap alone must not pass it (finding #11). Deliberately narrow:
        # "no"/"none"/"nor" are excluded because "No." (invoice numbers) and
        # "None" (rendered nulls) are not negations.
        negations = {"not", "never", "without",
                     "n't", "cannot", "cant", "wont", "dont", "isnt", "arent"}
        if (claim & negations) and not (pool & negations):
            return False
        return len(claim & pool) >= min(2, len(claim))
