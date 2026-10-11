#!/usr/bin/env python
"""Build a grounded question bank from the REAL 50-document corpus.

Reads evaluation/results/*.json (the same source the database was
recovered from) and generates diverse questions whose expected answers
are taken from the actual extracted values — no invented ground truth.

Categories (each pins the expected router behavior):
- exact      — needs_sql=True,  needs_retrieval=False (whitelisted field)
- compare    — needs_sql=True,  multi-period deterministic comparison
- lookup     — needs_retrieval=True (document/line-item lookup)
- conceptual — needs_retrieval=True (company/period prose questions)
- validation — needs_validation=True (reconciliation status)
- unsupported— abstains (no tools should fire)

Output: evaluation/rag_real_questions.json
"""

import hashlib
import json
import sys
from decimal import Decimal, InvalidOperation
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _stable_pick(name: str, n: int) -> int:
    """Deterministic slot from a name. NOTE: never use hash() here —
    CPython salts str hashing per process, so hash(name) would rebuild a
    different bank on every run."""
    return int(hashlib.sha256(name.encode()).hexdigest(), 16) % n

RESULTS_DIR = Path(__file__).resolve().parent / "results"
OUT = Path(__file__).resolve().parent / "rag_real_questions.json"

INVOICE_LABELS = {
    "total_amount": "total amount",
    "subtotal": "subtotal",
    "tax_amount": "tax amount",
    "amount_due": "amount due",
    "amount_paid": "amount paid",
    "vendor_name": "vendor",
    "customer_name": "customer",
    "invoice_number": "invoice number",
    "invoice_date": "invoice date",
    "due_date": "due date",
    "payment_method": "payment method",
    "currency": "currency",
}

# Statement fields worth asking about per period (scalar, EXPLICIT).
STATEMENT_LABELS = {
    "total_assets": "total assets",
    "total_capital_and_liabilities": "total capital and liabilities",
    "total_income": "total income",
    "total_expenditure": "total expenditure",
    "consolidated_net_profit": "consolidated net profit",
    "net_cash_from_operating": "net cash from operating",
    "net_increase_in_cash": "net increase in cash",
    "closing_cash": "closing cash",
    "opening_cash": "opening cash",
    "basic_eps": "basic EPS",
}


def _usable(blob) -> bool:
    return (isinstance(blob, dict)
            and blob.get("status") in ("EXPLICIT", "EXPLICIT_ZERO")
            and blob.get("value") is not None)


def _num(value) -> Decimal | None:
    try:
        return Decimal(str(value).replace(",", ""))
    except (InvalidOperation, ValueError, AttributeError):
        return None


def build() -> list[dict]:
    files = sorted(RESULTS_DIR.glob("*.json"))
    questions: list[dict] = []

    def add(qid, question, category, relevant, expect_sql,
            expect_retrieval, expected_value=None, expected_document=None,
            requires_abstention=False, expect_validation=False):
        questions.append({
            "id": qid,
            "question": question,
            "category": category,
            "relevant_documents": relevant,
            "expected_document": expected_document,
            "expected_value": expected_value,
            "expect_sql": expect_sql,
            "expect_retrieval": expect_retrieval,
            "expect_validation": expect_validation,
            "requires_abstention": requires_abstention,
        })

    for path in files:
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("processing_status") != "PASS":
            continue
        name = data["document_name"]
        dtype = data["document_type"]
        ed = data.get("extracted_data") or {}

        if dtype == "invoice":
            # Two exact-value probes per invoice (total + rotating field).
            # Dual-path by design: SQL answers, retrieval corroborates —
            # so expect_sql AND expect_retrieval are both True.
            total = ed.get("total_amount")
            if _usable(total):
                add(f"inv-total-{name}", f"What is the total amount on invoice {name}?",
                    "exact", [name], True, True,
                    expected_value=str(total["value"]), expected_document=name)
            others = [f for f in ("subtotal", "tax_amount", "amount_due",
                                  "vendor_name", "invoice_number", "amount_paid")
                      if _usable(ed.get(f))]
            if others:
                f = others[_stable_pick(name, len(others))]
                blob = ed[f]
                add(f"inv-{f}-{name}", f"What is the {INVOICE_LABELS[f]} on invoice {name}?",
                    "exact", [name], True, True,
                    expected_value=str(blob["value"]), expected_document=name)
            # Line-item lookup (retrieval path).
            for item in (ed.get("line_items") or [])[:1]:
                desc = (item.get("description") or {}).get("value")
                amt = (item.get("amount") or {}).get("value")
                if desc and amt:
                    add(f"inv-line-{name}",
                        f"What does invoice {name} charge for {desc}?",
                        "lookup", [name], False, True,
                        expected_value=str(amt), expected_document=name)
            # Validation probe on every 3rd invoice (bounded count).
            # Reconciliation runs through the structured validation layer,
            # so needs_sql is True by design alongside needs_validation.
            if _stable_pick(name, 3) == 0:
                add(f"inv-valid-{name}", f"Does invoice {name} balance?",
                    "validation", [name], True, False,
                    expected_document=name, expect_validation=True)

        elif dtype in ("balance_sheet", "profit_and_loss", "cash_flow_statement"):
            periods = ed.get("periods") or {}
            years = sorted(periods)
            company = ((ed.get("company_name") or {}).get("value")) or ""
            label = {"balance_sheet": "balance sheet",
                     "profit_and_loss": "profit and loss statement",
                     "cash_flow_statement": "cash flow statement"}[dtype]
            if company and years:
                add(f"stmt-co-{name}",
                    f"Which company is the {label} {name} for?",
                    "conceptual", [name], False, True,
                    expected_value=company, expected_document=name)
            # Exact per-period value on the latest year.
            if years:
                latest = years[-1]
                for field, flabel in STATEMENT_LABELS.items():
                    blob = periods[latest].get(field)
                    if _usable(blob):
                        add(f"stmt-exact-{name}-{latest}",
                            f"What were {flabel} in {latest} on {name}?",
                            "exact", [name], True, True,
                            expected_value=str(blob["value"]),
                            expected_document=name)
                        break
            # Comparison across the two latest years on a shared field.
            if len(years) >= 2:
                a, b = years[-2], years[-1]
                for field, flabel in STATEMENT_LABELS.items():
                    ba, bb = periods[a].get(field), periods[b].get(field)
                    if _usable(ba) and _usable(bb):
                        na, nb = _num(ba["value"]), _num(bb["value"])
                        if na is not None and nb is not None:
                            add(f"stmt-cmp-{name}-{a}-{b}",
                                f"How did {flabel} change between {a} and {b} on {name}?",
                                "compare", [name], True, True,
                                expected_value=str(nb - na),
                                expected_document=name)
                        break

    # Unsupported probes (must abstain; no tools should fire).
    for i, q in enumerate([
        "Should I invest in HDFC Bank?",
        "What will total income be next year?",
        "Is this invoice a good deal?",
        "Give me tax advice for this company.",
    ]):
        add(f"unsup-{i}", q, "unsupported", [], False, False,
            requires_abstention=True)

    OUT.write_text(json.dumps(questions, indent=2), encoding="utf-8")
    cats: dict[str, int] = {}
    for q in questions:
        cats[q["category"]] = cats.get(q["category"], 0) + 1
    print(f"wrote {OUT} ({len(questions)} questions: {cats})")
    return questions


if __name__ == "__main__":
    build()
