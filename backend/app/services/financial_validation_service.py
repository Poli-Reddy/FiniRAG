"""
FiniRAG Financial Validation Service.

Deterministic Python validation — the LLM extracts, this module validates.
Every check compares CALCULATED values against REPORTED values.

Implements all 4 document-type validators:
  - InvoiceValidator (Corrections #2, #7)
  - BalanceSheetValidator
  - ProfitLossValidator
  - CashFlowValidator (Correction #4 — amalgamation guards)

Tolerance: PROPOSED only, subject to empirical validation (Correction #8).
All arithmetic uses Decimal.
"""

from __future__ import annotations

import logging
from decimal import Decimal

from backend.app.schemas.extraction import (
    ExtractedValue, FieldStatus,
    InvoiceExtraction, BalanceSheetExtraction, BalanceSheetPeriod,
    ProfitLossExtraction, ProfitLossPeriod,
    CashFlowExtraction, CashFlowPeriod,
    is_usable, safe_value,
)
from backend.app.schemas.validation import ValidationCheck, ValidationResult

logger = logging.getLogger("finirag")


# ─────────────────────────────────────────────────────────────────────
# Tolerance Function (PROPOSED — Correction #8)
# ─────────────────────────────────────────────────────────────────────

def is_approximately_equal(
    calculated: Decimal,
    reported: Decimal,
) -> tuple[bool, Decimal, Decimal]:
    """
    PROPOSED tolerance — subject to empirical validation before finalization.

    Returns: (is_equal, variance, tolerance)
    """
    if reported == 0:
        tolerance = Decimal("0.01")
    else:
        tolerance = max(Decimal("0.01"), abs(reported) * Decimal("0.001"))
    variance = abs(calculated - reported)
    return variance <= tolerance, variance, tolerance


def operands_of(**values: object) -> dict[str, str]:
    """
    Record the inputs to a calculation as exact strings.

    A check that only reports calculated-vs-reported is unauditable: a
    reviewer cannot tell which figures went into it. Values are stringified
    so the audit trail survives JSON persistence without float rounding.
    """
    return {k: ("" if v is None else str(v)) for k, v in values.items()}


def sum_operands(items: dict[str, ExtractedValue]) -> dict[str, str]:
    """Operands for a summation check: every component keyed by its document label."""
    return {label: str(safe_value(v)) for label, v in items.items() if is_usable(v)}


# Row labels that are a *result* of the appropriation, not an appropriation.
# The statements print the residual inside the same block as the transfers
# ("Balance carried over to Balance Sheet", and from 2024 "…to consolidated
# balance sheet"), so an unfiltered sum(appropriations) swallows the very
# figure the check is meant to reconcile against and the identity collapses
# to total_available − total_available ≈ 0.
CARRIED_FORWARD_MARKERS = (
    "balance carried",
    "carried over to",
    "carried forward",
)


def is_carried_forward_label(label: str) -> bool:
    """
    True when an appropriation row is the carried-forward residual.

    Matched on the label rather than on the value: two transfers could
    coincidentally equal the residual, and the identity being checked is
    structural — that row closes the block, it is not a use of profit.
    """
    text = " ".join(str(label or "").lower().split())
    return any(marker in text for marker in CARRIED_FORWARD_MARKERS)


def appropriation_components(
    appropriations: dict[str, ExtractedValue],
) -> dict[str, ExtractedValue]:
    """The true appropriation rows — the carried-forward residual removed."""
    return {
        label: value
        for label, value in appropriations.items()
        if not is_carried_forward_label(label)
    }


def split_running_subtotals(
    items: dict[str, ExtractedValue],
) -> tuple[dict[str, ExtractedValue], list[str]]:
    """
    Separate genuine line items from intermediate subtotal rows.

    Cash flow statements print running subtotals *inside* a section — e.g.
    "Consolidated profit before income tax and after minority interest",
    which is the sum of the two rows above it. Both the label and the value
    are real, so the extraction is right to report the row; but adding it to
    its own components double-counts and the section can never reconcile.

    A row is treated as a subtotal only when its value equals the running
    sum of the rows already accepted, within the same tolerance used for
    every other check, and only once at least two rows have been accepted
    (one row "reconciling" with itself is meaningless). Nothing is deleted:
    the caller discloses the dropped labels in the check's operands, so a
    reviewer can see what was excluded and why.
    """
    kept: dict[str, ExtractedValue] = {}
    dropped: list[str] = []
    running = Decimal(0)
    summed = 0

    for label, value in items.items():
        if not is_usable(value):
            kept[label] = value
            continue
        current = safe_value(value)
        if summed >= 2:
            _, variance, tolerance = is_approximately_equal(current, running)
            if variance <= tolerance:
                dropped.append(label)
                continue
        kept[label] = value
        running += current
        summed += 1

    return kept, dropped


def compare_candidates(
    name: str,
    candidates: list[tuple[str, Decimal]],
    reported: Decimal,
    operands: dict | None = None,
) -> ValidationCheck:
    """
    Compare several admissible readings and report the one that reconciles.

    Receipts state the same arithmetic in two ways — tax added to the line
    amounts, or tax already inside them — and which applies is often not
    stated, or is stated wrongly (X51005361895 is labelled inclusive yet
    prints 8.49 + 0.51 = 9.00). Picking one reading and failing the other
    marks self-consistent documents as broken, which is the tax-inclusive
    trap (P6/D11).

    Candidates are given in preference order; the first that reconciles
    wins and its formula is recorded, so the check still says exactly which
    relation held. If none reconciles the first is reported and the check
    FAILs — a document whose numbers work under no standard reading is
    genuinely inconsistent.
    """
    for formula, value in candidates:
        passed, _, _ = is_approximately_equal(value, reported)
        if passed:
            return compare(
                name=name, calculated=value, reported=reported,
                formula=formula, operands=operands,
            )
    formula, value = candidates[0]
    return compare(
        name=name, calculated=value, reported=reported,
        formula=formula, operands=operands,
    )


def compare(
    name: str,
    calculated: Decimal,
    reported: Decimal,
    formula: str = "",
    operands: dict | None = None,
) -> ValidationCheck:
    """Build a PASS/FAIL ValidationCheck by comparing two values."""
    is_equal, variance, tolerance = is_approximately_equal(calculated, reported)

    return ValidationCheck(
        name=name,
        formula=formula,
        operands=operands or {},
        calculated_value=calculated,
        reported_value=reported,
        variance=variance,
        tolerance=tolerance,
        status="PASS" if is_equal else "FAIL",
    )


def NOT_APPLICABLE(name: str, reason: str = "Missing operand(s)") -> ValidationCheck:
    """Build a NOT_APPLICABLE ValidationCheck."""
    return ValidationCheck(
        name=name,
        status="NOT_APPLICABLE",
        reason=reason,
    )


# ─────────────────────────────────────────────────────────────────────
# Invoice Validator (Corrections #2, #7)
# ─────────────────────────────────────────────────────────────────────

class InvoiceValidator:
    """Validates invoice/receipt arithmetic."""

    def validate(self, ext: InvoiceExtraction) -> ValidationResult:
        """Run all invoice validation checks."""
        checks = []

        # ── Check 1: Per-line Quantity × Unit Price [− discount] ≈ Line Total ──
        for i, item in enumerate(ext.line_items):
            if not (is_usable(item.quantity) and
                    is_usable(item.unit_price) and
                    is_usable(item.amount)):
                checks.append(NOT_APPLICABLE(f"line_total_check_{i+1}"))
            else:
                quantity = safe_value(item.quantity)
                unit_price = safe_value(item.unit_price)
                reported = safe_value(item.amount)
                gross = quantity * unit_price
                inputs = {
                    "quantity": quantity,
                    "unit_price": unit_price,
                }
                formula = "quantity × unit_price"
                calculated = gross

                # A line that prints its own discount reports the NET amount.
                # Comparing the gross against it would FAIL every discounted
                # line on an otherwise perfectly consistent invoice. Applied
                # only when the document actually shows the discount, so a
                # missing one is never invented as zero.
                #
                # The magnitude is subtracted, not the signed value: receipts
                # print the deduction both ways (X00016469619 shows -5.59 for
                # a 5.59 discount) and a discount only ever reduces a line.
                if is_usable(item.discount_amount):
                    discount = safe_value(item.discount_amount)
                    calculated = gross - abs(discount)
                    formula = "quantity × unit_price − |discount_amount|"
                    inputs["discount_amount"] = discount
                elif is_usable(item.discount_rate):
                    rate = safe_value(item.discount_rate)
                    rate_multiplier, rate_reading = self._interpret_rate(
                        item.discount_rate, rate
                    )
                    calculated = gross * (Decimal(1) - rate_multiplier)
                    formula = "quantity × unit_price × (1 − discount_rate)"
                    inputs["discount_rate"] = item.discount_rate.original_value or rate
                    inputs["rate_reading"] = rate_reading

                inputs["amount"] = reported
                checks.append(compare(
                    name=f"line_total_check_{i+1}",
                    formula=formula,
                    calculated=calculated,
                    reported=reported,
                    operands=operands_of(**inputs),
                ))

        # ── Check 2: Sum of line totals ≈ subtotal ──
        usable_lines = [item for item in ext.line_items if is_usable(item.amount)]
        if not usable_lines or not is_usable(ext.subtotal):
            checks.append(NOT_APPLICABLE("sum_lines_vs_subtotal"))
        else:
            line_sum = sum(safe_value(item.amount) for item in usable_lines)
            operands = operands_of(
                line_amounts=", ".join(
                    str(safe_value(item.amount)) for item in usable_lines
                ),
                line_count=len(usable_lines),
                subtotal=safe_value(ext.subtotal),
            )
            # On a GST-inclusive receipt the line amounts carry the tax while
            # the printed subtotal is the net figure (X51005663293: lines sum
            # to 30.30, subtotal 28.58, GST 1.72). Removing the tax is then
            # the correct reading; a single gross comparison would FAIL it.
            candidates = [("sum(line_amounts)", line_sum)]
            if is_usable(ext.tax_amount):
                operands["tax_amount"] = str(safe_value(ext.tax_amount))
                candidates.append((
                    "sum(line_amounts) − tax_amount (lines are tax-inclusive)",
                    line_sum - safe_value(ext.tax_amount),
                ))
            checks.append(compare_candidates(
                name="sum_lines_vs_subtotal",
                candidates=candidates,
                reported=safe_value(ext.subtotal),
                operands=operands,
            ))

        # ── Check 2b: Sum of line totals vs TOTAL when no subtotal ──
        # A receipt that prints lines + total but no subtotal (batch3-1495:
        # lines sum to 3480, total says 5257) would otherwise slip through:
        # Check 2 needs a subtotal and Check 3 needs subtotal+tax. Skipped
        # whenever a subtotal exists — there Check 2/3 already own the
        # reconciliation and a naive lines-vs-total would false-FAIL every
        # tax-exclusive invoice.
        if is_usable(ext.subtotal) or not usable_lines or not is_usable(ext.total_amount):
            checks.append(NOT_APPLICABLE("sum_lines_vs_total"))
        else:
            line_sum = sum(safe_value(item.amount) for item in usable_lines)
            base = line_sum
            base_formula = "sum(line_amounts)"
            extra_operands: dict = {}
            if is_usable(ext.discount):
                base -= safe_value(ext.discount)
                base_formula += " − discount"
                extra_operands["discount"] = safe_value(ext.discount)

            # Whether tax is ADDED to the line amounts or ALREADY INSIDE them
            # decides the expectation, and most receipts in this dataset are
            # GST-inclusive (Trap P6/D11): "Total Includes GST 6%" means the
            # line amounts already carry the tax, so adding it again FAILs a
            # perfectly consistent receipt.
            #
            # Both readings are always offered when a tax figure is present,
            # and the one that reconciles is reported. The extracted
            # `tax_inclusive` flag only decides which is tried FIRST, because
            # it is a model interpretation while the document's own
            # arithmetic is direct evidence: X51005361895 is labelled
            # inclusive yet prints 8.49 + 0.51 = 9.00. Trusting the label
            # over the arithmetic would fail a self-consistent receipt. A
            # document where neither reading holds still FAILs.
            inclusive_reading = (f"{base_formula} (tax already included)", base)
            readings: list[tuple[str, Decimal]] = []
            if is_usable(ext.tax_amount):
                exclusive_reading = (
                    f"{base_formula} + tax_amount",
                    base + safe_value(ext.tax_amount),
                )
                readings = (
                    [inclusive_reading, exclusive_reading]
                    if ext.tax_inclusive is True
                    else [exclusive_reading, inclusive_reading]
                )
            else:
                readings = [(base_formula, base)]

            reported = safe_value(ext.total_amount)
            if is_usable(ext.tax_amount):
                extra_operands["tax_amount"] = safe_value(ext.tax_amount)
                extra_operands["tax_inclusive"] = (
                    "not stated" if ext.tax_inclusive is None else ext.tax_inclusive
                )

            # Prefer a reading that reconciles; otherwise report the first.
            chosen_formula, chosen_value = readings[0]
            for formula, value in readings:
                passed, _, _ = is_approximately_equal(value, reported)
                if passed:
                    chosen_formula, chosen_value = formula, value
                    break

            checks.append(compare(
                name="sum_lines_vs_total",
                formula=chosen_formula,
                calculated=chosen_value,
                reported=reported,
                operands=operands_of(
                    line_amounts=", ".join(
                        str(safe_value(item.amount)) for item in usable_lines
                    ),
                    line_count=len(usable_lines),
                    total_amount=reported,
                    **extra_operands,
                ),
            ))

        # ── Check 3: Tax validation (Corrections #2, #7) ──
        checks.extend(self._validate_tax(ext))

        # ── Check 4: Cash − Total ≈ Change ──
        if not (is_usable(ext.cash_paid) and
                is_usable(ext.change_given) and
                is_usable(ext.total_amount)):
            checks.append(NOT_APPLICABLE("cash_change_check"))
        else:
            cash_paid = safe_value(ext.cash_paid)
            total = safe_value(ext.total_amount)
            checks.append(compare(
                name="cash_change_check",
                formula="cash_paid − total",
                calculated=cash_paid - total,
                reported=safe_value(ext.change_given),
                operands=operands_of(
                    cash_paid=cash_paid,
                    total_amount=total,
                    change_given=safe_value(ext.change_given),
                ),
            ))

        result = ValidationResult(checks=checks)
        result.compute_summary()
        return result

    def _validate_tax(self, ext: InvoiceExtraction) -> list[ValidationCheck]:
        """
        Non-tautological tax validation (Corrections #2 + #7).

        Correction #7: Do NOT assume "subtotal" is always pre-tax on
        inclusive documents without document evidence confirming it.
        """
        checks = []

        has_subtotal = is_usable(ext.subtotal)
        has_tax = is_usable(ext.tax_amount)
        has_total = is_usable(ext.total_amount)
        has_rate = is_usable(ext.tax_rate)

        # ── Sub-check 3a: subtotal + tax ≈ total ──
        if has_subtotal and has_tax and has_total:
            subtotal = safe_value(ext.subtotal)
            tax = safe_value(ext.tax_amount)
            operands = operands_of(
                subtotal=subtotal,
                tax_amount=tax,
                total_amount=safe_value(ext.total_amount),
                tax_inclusive=(
                    "not stated" if ext.tax_inclusive is None else ext.tax_inclusive
                ),
            )
            # The canonical tax-inclusive trap: "Subtotal 29.00, GST 1.64,
            # Total 29.00" (X51005806685, X51006328913). The total already
            # contains the tax, so subtotal + tax over-counts it. Both
            # readings are admissible and the reconciling one is reported.
            exclusive = ("subtotal + tax_amount", subtotal + tax)
            inclusive = ("subtotal (tax already included in total)", subtotal)
            candidates = (
                [inclusive, exclusive] if ext.tax_inclusive is True
                else [exclusive, inclusive]
            )
            checks.append(compare_candidates(
                name="subtotal_plus_tax_vs_total",
                candidates=candidates,
                reported=safe_value(ext.total_amount),
                operands=operands,
            ))
        else:
            checks.append(NOT_APPLICABLE("subtotal_plus_tax_vs_total"))

        # ── Sub-check 3b: rate × taxable_base ≈ tax_amount ──
        # Correction #7: taxable base must be independently confirmed
        if has_rate and has_tax:
            rate = safe_value(ext.tax_rate)
            tax_reported = safe_value(ext.tax_amount)

            taxable_base = None
            base_source = ""

            if ext.tax_inclusive is True:
                # On inclusive documents, subtotal is pre-tax ONLY if
                # confirmed by document evidence (Correction #7)
                if has_subtotal and ext.subtotal_confirmed_pretax:
                    taxable_base = safe_value(ext.subtotal)
                    base_source = "subtotal (confirmed pre-tax)"
            else:
                # On exclusive documents, subtotal is typically pre-tax
                if has_subtotal:
                    taxable_base = safe_value(ext.subtotal)
                    base_source = "subtotal"
                else:
                    usable_lines = [
                        item for item in ext.line_items
                        if is_usable(item.amount)
                    ]
                    if usable_lines:
                        taxable_base = sum(
                            safe_value(item.amount) for item in usable_lines
                        )
                        base_source = "sum(line_amounts)"

            if taxable_base is not None and rate > 0:
                rate_decimal, rate_reading = self._interpret_rate(ext.tax_rate, rate)
                calculated_tax = taxable_base * rate_decimal
                checks.append(compare(
                    name="rate_times_base_vs_tax",
                    formula=f"taxable_base × {rate_decimal}",
                    calculated=calculated_tax,
                    reported=tax_reported,
                    operands=operands_of(
                        taxable_base=taxable_base,
                        base_source=base_source,
                        tax_rate=ext.tax_rate.original_value or rate,
                        rate_reading=rate_reading,
                        rate_as_multiplier=rate_decimal,
                        tax_amount=tax_reported,
                    ),
                ))
            else:
                checks.append(NOT_APPLICABLE(
                    "rate_times_base_vs_tax",
                    reason="Cannot establish independent taxable base",
                ))
        else:
            checks.append(NOT_APPLICABLE("rate_times_base_vs_tax"))

        return checks

    @staticmethod
    def _interpret_rate(field: ExtractedValue, rate: Decimal) -> tuple[Decimal, str]:
        """
        Decide whether an extracted tax rate is a percentage or a fraction.

        Documents print rates both ways ("6%" and "0.06"), so the multiplier
        cannot be read off the number alone. Resolution order:

          1. A literal '%' in the source text settles it — that is document
             evidence, not a guess, and it holds at any magnitude (0.5% too).
          2. Otherwise a value of 1 or more can only be a percentage: a
             multiplier of 1.0 would mean 100% tax.
          3. Below 1 with no percent sign, read it as a fraction (0.06 = 6%).

        The previous rule keyed on `rate > 1`, which read a 1% rate as a
        multiplier of 1 and reported 100% tax on every such invoice.
        """
        printed = field.original_value or ""
        if "%" in printed:
            return rate / Decimal("100"), "percent (explicit % in document)"
        if rate >= 1:
            return rate / Decimal("100"), "percent (magnitude >= 1)"
        return rate, "fraction (magnitude < 1, no % sign)"


# ─────────────────────────────────────────────────────────────────────
# Balance Sheet Validator
# ─────────────────────────────────────────────────────────────────────

class BalanceSheetValidator:
    """Validates Balance Sheet arithmetic per period."""

    def validate(self, ext: BalanceSheetExtraction) -> ValidationResult:
        """Run all Balance Sheet validation checks across all periods."""
        checks = []

        for period_key, period in ext.periods.items():
            checks.extend(self._validate_period(period, period_key))

        result = ValidationResult(checks=checks)
        result.compute_summary()
        return result

    def _validate_period(
        self, period: BalanceSheetPeriod, prefix: str
    ) -> list[ValidationCheck]:
        """Validate a single Balance Sheet period."""
        checks = []

        # ── Check 1: Total Assets ≈ Total Capital & Liabilities ──
        if not (is_usable(period.total_assets) and
                is_usable(period.total_capital_and_liabilities)):
            checks.append(NOT_APPLICABLE(f"{prefix}_assets_vs_liabilities"))
        else:
            checks.append(compare(
                name=f"{prefix}_assets_vs_liabilities",
                formula="total_assets ≈ total_capital_and_liabilities",
                calculated=safe_value(period.total_assets),
                reported=safe_value(period.total_capital_and_liabilities),
                operands=operands_of(
                    total_assets=safe_value(period.total_assets),
                    total_capital_and_liabilities=safe_value(
                        period.total_capital_and_liabilities
                    ),
                ),
            ))

        # ── Check 2: Sum of asset components ≈ Total Assets ──
        asset_components = [v for v in period.assets.values() if is_usable(v)]
        if len(asset_components) < 2 or not is_usable(period.total_assets):
            checks.append(NOT_APPLICABLE(f"{prefix}_sum_assets"))
        else:
            asset_sum = sum(safe_value(v) for v in asset_components)
            checks.append(compare(
                name=f"{prefix}_sum_assets",
                formula="sum(asset_components)",
                calculated=asset_sum,
                reported=safe_value(period.total_assets),
                operands=sum_operands(period.assets),
            ))

        # ── Check 3: Sum of liabilities components ≈ Total C&L ──
        liab_components = [
            v for v in period.capital_and_liabilities.values() if is_usable(v)
        ]
        if (len(liab_components) < 2 or
                not is_usable(period.total_capital_and_liabilities)):
            checks.append(NOT_APPLICABLE(f"{prefix}_sum_liabilities"))
        else:
            liab_sum = sum(safe_value(v) for v in liab_components)
            checks.append(compare(
                name=f"{prefix}_sum_liabilities",
                formula="sum(capital_and_liabilities_components)",
                calculated=liab_sum,
                reported=safe_value(period.total_capital_and_liabilities),
                operands=sum_operands(period.capital_and_liabilities),
            ))

        return checks


# ─────────────────────────────────────────────────────────────────────
# P&L Validator
# ─────────────────────────────────────────────────────────────────────

class ProfitLossValidator:
    """Validates P&L arithmetic per period."""

    def validate(self, ext: ProfitLossExtraction) -> ValidationResult:
        """Run all P&L validation checks across all periods."""
        checks = []

        for period_key, period in ext.periods.items():
            checks.extend(self._validate_period(period, period_key))

        result = ValidationResult(checks=checks)
        result.compute_summary()
        return result

    def _validate_period(
        self, period: ProfitLossPeriod, prefix: str
    ) -> list[ValidationCheck]:
        """Validate a single P&L period."""
        checks = []

        # ── Check 1: Income − Expenditure ≈ Net Profit ──
        if not (is_usable(period.total_income) and
                is_usable(period.total_expenditure) and
                is_usable(period.consolidated_net_profit)):
            checks.append(NOT_APPLICABLE(f"{prefix}_income_minus_expenditure"))
        else:
            income = safe_value(period.total_income)
            expenditure = safe_value(period.total_expenditure)
            calculated = income - expenditure
            inputs = {
                "total_income": income,
                "total_expenditure": expenditure,
            }

            # Account for minority interest and associates if present
            if is_usable(period.minority_interest):
                calculated -= safe_value(period.minority_interest)
                inputs["minority_interest"] = -safe_value(period.minority_interest)
            if is_usable(period.share_in_profits_of_associates):
                calculated += safe_value(period.share_in_profits_of_associates)
                inputs["share_in_profits_of_associates"] = safe_value(
                    period.share_in_profits_of_associates
                )

            inputs["consolidated_net_profit"] = safe_value(
                period.consolidated_net_profit
            )
            checks.append(compare(
                name=f"{prefix}_income_minus_expenditure",
                formula="total_income − total_expenditure [± adjustments]",
                calculated=calculated,
                reported=safe_value(period.consolidated_net_profit),
                operands=operands_of(**inputs),
            ))

        # ── Check 1b: Income − Expenditure ≈ Profit BEFORE minority interest ──
        # This is a distinct equation from Check 1. Income minus expenditure
        # yields profit before minority interest and associates; the
        # consolidated figure is that number after those adjustments.
        # Conflating them hides an error in either half.
        if not (is_usable(period.total_income) and
                is_usable(period.total_expenditure) and
                is_usable(period.net_profit_before_minority)):
            checks.append(NOT_APPLICABLE(f"{prefix}_net_profit_before_minority"))
        else:
            income = safe_value(period.total_income)
            expenditure = safe_value(period.total_expenditure)
            reported = safe_value(period.net_profit_before_minority)
            checks.append(compare(
                name=f"{prefix}_net_profit_before_minority",
                formula="total_income − total_expenditure",
                calculated=income - expenditure,
                reported=reported,
                operands=operands_of(
                    total_income=income,
                    total_expenditure=expenditure,
                    net_profit_before_minority=reported,
                ),
            ))

        # ── Check 2: Sum of income components ≈ Total Income ──
        income_components = {
            "interest_earned": period.interest_earned,
            "other_income": period.other_income,
        }
        usable_income = [v for v in income_components.values() if is_usable(v)]
        if len(usable_income) < 2 or not is_usable(period.total_income):
            checks.append(NOT_APPLICABLE(f"{prefix}_sum_income"))
        else:
            income_sum = sum(safe_value(v) for v in usable_income)
            checks.append(compare(
                name=f"{prefix}_sum_income",
                formula="interest_earned + other_income",
                calculated=income_sum,
                reported=safe_value(period.total_income),
                operands=sum_operands(income_components),
            ))

        # ── Check 3: Sum of expenditure components ≈ Total Expenditure ──
        exp_components = {
            "interest_expended": period.interest_expended,
            "operating_expenses": period.operating_expenses,
            "provisions_and_contingencies": period.provisions_and_contingencies,
        }
        usable_exp = [v for v in exp_components.values() if is_usable(v)]
        if len(usable_exp) < 2 or not is_usable(period.total_expenditure):
            checks.append(NOT_APPLICABLE(f"{prefix}_sum_expenditure"))
        else:
            exp_sum = sum(safe_value(v) for v in usable_exp)
            checks.append(compare(
                name=f"{prefix}_sum_expenditure",
                formula="interest_expended + operating_expenses + provisions",
                calculated=exp_sum,
                reported=safe_value(period.total_expenditure),
                operands=sum_operands(exp_components),
            ))

        # ── Check 4: Appropriation reconciliation ──
        # total_available − sum(appropriations) ≈ balance_carried_forward.
        # The carried-forward row is excluded from the sum: the statements
        # print it inside the appropriation block, and leaving it in makes
        # the identity self-cancelling (see appropriation_components).
        components = appropriation_components(period.appropriations)
        approp_values = [v for v in components.values() if is_usable(v)]
        if (not approp_values or
                not is_usable(period.total_available_for_appropriation) or
                not is_usable(period.balance_carried_forward)):
            checks.append(NOT_APPLICABLE(f"{prefix}_appropriation"))
        else:
            approp_sum = sum(safe_value(v) for v in approp_values)
            available = safe_value(period.total_available_for_appropriation)
            operands = sum_operands(components)
            operands["total_available_for_appropriation"] = str(available)
            operands["balance_carried_forward"] = str(
                safe_value(period.balance_carried_forward)
            )
            excluded = [
                label for label in period.appropriations
                if is_carried_forward_label(label)
            ]
            if excluded:
                operands["excluded_from_sum"] = ", ".join(excluded)
            checks.append(compare(
                name=f"{prefix}_appropriation",
                formula="total_available − sum(appropriations)",
                calculated=available - approp_sum,
                reported=safe_value(period.balance_carried_forward),
                operands=operands,
            ))

        return checks


# ─────────────────────────────────────────────────────────────────────
# Cash Flow Validator (Correction #4 — Amalgamation Guards)
# ─────────────────────────────────────────────────────────────────────

class CashFlowValidator:
    """
    Validates Cash Flow Statement arithmetic per period.

    Implements Correction #4: amalgamation guards with is_usable()
    checked before every .value access.
    """

    def validate(self, ext: CashFlowExtraction) -> ValidationResult:
        """Run all Cash Flow validation checks across all periods."""
        checks = []

        for period_key, period in ext.periods.items():
            checks.extend(self._validate_period(period, period_key))

        result = ValidationResult(checks=checks)
        result.compute_summary()
        return result

    def _validate_period(
        self, period: CashFlowPeriod, prefix: str
    ) -> list[ValidationCheck]:
        """Validate a single Cash Flow period."""
        checks = []

        amalg = period.cash_acquired_on_amalgamation
        amalg_position = period.amalgamation_position
        amalg_usable = is_usable(amalg)

        # ── Check 1: Operating + Investing + Financing [+ FX] [+ Amalg?] ≈ Net Increase ──
        if not (is_usable(period.net_cash_from_operating) and
                is_usable(period.net_cash_from_investing) and
                is_usable(period.net_cash_from_financing) and
                is_usable(period.net_increase_in_cash)):
            checks.append(NOT_APPLICABLE(f"{prefix}_cash_flow_components"))
        else:
            operating = safe_value(period.net_cash_from_operating)
            investing = safe_value(period.net_cash_from_investing)
            financing = safe_value(period.net_cash_from_financing)
            total = operating + investing + financing
            inputs = {
                "net_cash_from_operating": operating,
                "net_cash_from_investing": investing,
                "net_cash_from_financing": financing,
            }

            if is_usable(period.fx_translation_adjustment):
                total += safe_value(period.fx_translation_adjustment)
                inputs["fx_translation_adjustment"] = safe_value(
                    period.fx_translation_adjustment
                )

            # Amalgamation: add to equation 1 ONLY if position is "before_net_increase"
            if amalg_position == "before_net_increase" and amalg_usable:
                total += safe_value(amalg)
                inputs["cash_acquired_on_amalgamation"] = safe_value(amalg)
            elif amalg_usable:
                inputs["cash_acquired_on_amalgamation"] = (
                    f"excluded (position={amalg_position or 'unknown'})"
                )

            inputs["net_increase_in_cash"] = safe_value(period.net_increase_in_cash)
            checks.append(compare(
                name=f"{prefix}_cash_flow_components",
                formula="operating + investing + financing [+ FX] [+ amalgamation]",
                calculated=total,
                reported=safe_value(period.net_increase_in_cash),
                operands=operands_of(**inputs),
            ))

        # ── Check 2: Opening + Net Increase [+ Amalg?] ≈ Closing ──
        if not (is_usable(period.opening_cash) and
                is_usable(period.net_increase_in_cash) and
                is_usable(period.closing_cash)):
            checks.append(NOT_APPLICABLE(f"{prefix}_cash_reconciliation"))
        else:
            opening = safe_value(period.opening_cash)
            net_increase = safe_value(period.net_increase_in_cash)
            total = opening + net_increase
            inputs = {
                "opening_cash": opening,
                "net_increase_in_cash": net_increase,
            }

            # Amalgamation: add to equation 2 ONLY if position is "after_net_increase"
            if amalg_position == "after_net_increase" and amalg_usable:
                total += safe_value(amalg)
                inputs["cash_acquired_on_amalgamation"] = safe_value(amalg)
            elif amalg_usable:
                inputs["cash_acquired_on_amalgamation"] = (
                    f"excluded (position={amalg_position or 'unknown'})"
                )

            inputs["closing_cash"] = safe_value(period.closing_cash)
            checks.append(compare(
                name=f"{prefix}_cash_reconciliation",
                formula="opening + net_increase [+ amalgamation]",
                calculated=total,
                reported=safe_value(period.closing_cash),
                operands=operands_of(**inputs),
            ))

        # ── Check 3: Each section's line items ≈ that section's net figure ──
        # The balance sheet reconciles its components against their totals;
        # a cash flow statement's three activity sections deserve the same
        # treatment, otherwise a mis-transcribed line item inside a section
        # is invisible as long as the section subtotal itself was read right.
        sections = (
            ("operating", period.operating_items, period.net_cash_from_operating),
            ("investing", period.investing_items, period.net_cash_from_investing),
            ("financing", period.financing_items, period.net_cash_from_financing),
        )
        for section_name, items, net_field in sections:
            name = f"{prefix}_sum_{section_name}_items"
            # Intermediate subtotals printed inside the section are excluded
            # from the sum and disclosed in the operands — they are genuine
            # rows, but adding them to their own components double-counts.
            components, subtotal_rows = split_running_subtotals(items)
            usable_items = [v for v in components.values() if is_usable(v)]

            if len(usable_items) < 2:
                # One item reconciling with its own section total is tautological.
                checks.append(NOT_APPLICABLE(
                    name,
                    reason="Fewer than two readable line items in this section",
                ))
            elif not is_usable(net_field):
                checks.append(NOT_APPLICABLE(
                    name,
                    reason=f"net_cash_from_{section_name} not readable",
                ))
            else:
                item_sum = sum(safe_value(v) for v in usable_items)
                operands = sum_operands(components)
                if subtotal_rows:
                    operands["excluded_intermediate_subtotals"] = ", ".join(
                        subtotal_rows
                    )
                checks.append(compare(
                    name=name,
                    formula=f"sum({section_name}_items)",
                    calculated=item_sum,
                    reported=safe_value(net_field),
                    operands=operands,
                ))

        return checks


# ─────────────────────────────────────────────────────────────────────
# Validator Factory
# ─────────────────────────────────────────────────────────────────────

VALIDATOR_MAP = {
    "invoice": InvoiceValidator,
    "balance_sheet": BalanceSheetValidator,
    "profit_and_loss": ProfitLossValidator,
    "cash_flow_statement": CashFlowValidator,
}


class FinancialValidationService:
    """Dispatches validation to the appropriate document-type validator."""

    def validate(
        self,
        extracted_data: dict,
        document_type: str,
    ) -> ValidationResult:
        """
        Run deterministic financial validation on extracted data.

        Args:
            extracted_data: The extraction result dict
            document_type: Document type string

        Returns:
            ValidationResult with all checks
        """
        validator_class = VALIDATOR_MAP.get(document_type)
        if not validator_class:
            return ValidationResult()

        validator = validator_class()

        # Parse dict back into typed extraction schema
        if document_type == "invoice":
            extraction = InvoiceExtraction.model_validate(extracted_data)
        elif document_type == "balance_sheet":
            extraction = BalanceSheetExtraction.model_validate(extracted_data)
        elif document_type == "profit_and_loss":
            extraction = ProfitLossExtraction.model_validate(extracted_data)
        elif document_type == "cash_flow_statement":
            extraction = CashFlowExtraction.model_validate(extracted_data)
        else:
            return ValidationResult()

        result = validator.validate(extraction)

        logger.info("Financial validation", extra={
            "document_type": document_type,
            "pass": result.summary["pass"],
            "fail": result.summary["fail"],
            "na": result.summary["not_applicable"],
        })

        return result
