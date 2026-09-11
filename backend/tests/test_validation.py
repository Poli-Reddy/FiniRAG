"""
Unit tests for the deterministic financial validation engine.

Nothing here touches the LLM: these tests feed hand-built extraction
objects to the validators and assert the arithmetic, the tolerance
behaviour, and the PASS / FAIL / NOT_APPLICABLE contract.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from backend.app.schemas.extraction import (
    BalanceSheetExtraction,
    BalanceSheetPeriod,
    CashFlowExtraction,
    CashFlowPeriod,
    FieldStatus,
    InvoiceExtraction,
    InvoiceLineItem,
    ProfitLossExtraction,
    ProfitLossPeriod,
    is_usable,
    safe_value,
)
from backend.app.services.financial_validation_service import (
    BalanceSheetValidator,
    CashFlowValidator,
    FinancialValidationService,
    InvoiceValidator,
    ProfitLossValidator,
    is_approximately_equal,
)

from backend.tests.conftest import check_by_name, ev, ev_missing, ev_unreadable, ev_zero


# ─────────────────────────────────────────────────────────────────────
# Tolerance function (Trap P4)
# ─────────────────────────────────────────────────────────────────────

def test_tolerance_is_one_cent_for_small_values():
    """Below ~₹10 the relative band is under a cent, so the floor applies."""
    ok, variance, tol = is_approximately_equal(Decimal("5.00"), Decimal("5.005"))
    assert tol == Decimal("0.01")
    assert ok


def test_tolerance_scales_with_magnitude():
    """0.1% of the reported value once that beats one cent."""
    ok, variance, tol = is_approximately_equal(Decimal("1000000"), Decimal("1000000"))
    assert tol == Decimal("1000")
    assert variance == 0
    assert ok


def test_tolerance_rejects_variance_just_outside_band():
    ok, variance, tol = is_approximately_equal(Decimal("100.05"), Decimal("100.00"))
    assert tol == Decimal("0.1")
    assert variance == Decimal("0.05")
    assert ok

    ok, variance, tol = is_approximately_equal(Decimal("100.20"), Decimal("100.00"))
    assert not ok


def test_tolerance_when_reported_is_zero():
    ok, _, tol = is_approximately_equal(Decimal("0.005"), Decimal("0"))
    assert tol == Decimal("0.01")
    assert ok
    ok, _, _ = is_approximately_equal(Decimal("5"), Decimal("0"))
    assert not ok


def test_tolerance_handles_negative_reported_values():
    """Cash-flow outflows are negative; tolerance must use magnitude."""
    ok, variance, tol = is_approximately_equal(Decimal("-3988.06"), Decimal("-3988.06"))
    assert tol > 0
    assert ok


# ─────────────────────────────────────────────────────────────────────
# safe_value / is_usable contract (Trap P7)
# ─────────────────────────────────────────────────────────────────────

def test_is_usable_accepts_only_explicit_and_explicit_zero():
    assert is_usable(ev(5))
    assert is_usable(ev_zero())
    assert not is_usable(ev_missing())
    assert not is_usable(ev_unreadable())


def test_safe_value_refuses_missing_values_loudly():
    """
    A MISSING field must never silently become 0.
    This must hold even when Python runs with -O (asserts stripped),
    so the guard cannot be a bare `assert`.
    """
    with pytest.raises((AssertionError, ValueError, TypeError)):
        safe_value(ev_missing())


def test_safe_value_refuses_unreadable_values_loudly():
    with pytest.raises((AssertionError, ValueError, TypeError)):
        safe_value(ev_unreadable())


def test_safe_value_returns_zero_for_explicit_zero():
    assert safe_value(ev_zero()) == Decimal("0")


# ─────────────────────────────────────────────────────────────────────
# Invoice validator
# ─────────────────────────────────────────────────────────────────────

def _line(qty, price, amount) -> InvoiceLineItem:
    return InvoiceLineItem(quantity=ev(qty), unit_price=ev(price), amount=ev(amount))


def test_invoice_clean_document_passes():
    inv = InvoiceExtraction(
        line_items=[_line(2, "10.00", "20.00"), _line(1, "5.50", "5.50")],
        subtotal=ev("25.50"),
        tax_amount=ev("2.55"),
        total_amount=ev("28.05"),
    )
    result = InvoiceValidator().validate(inv)
    assert result.overall_status == "PASS"
    assert result.summary["fail"] == 0


def test_invoice_line_total_mismatch_fails():
    inv = InvoiceExtraction(line_items=[_line(3, "10.00", "25.00")])
    result = InvoiceValidator().validate(inv)
    check = check_by_name(result, "line_total_check_1")
    assert check.status == "FAIL"
    assert check.calculated_value == Decimal("30.00")
    assert check.reported_value == Decimal("25.00")
    assert result.overall_status == "FAIL"


def test_invoice_subtotal_mismatch_fails():
    inv = InvoiceExtraction(
        line_items=[_line(1, "10.00", "10.00")],
        subtotal=ev("99.00"),
    )
    result = InvoiceValidator().validate(inv)
    assert check_by_name(result, "sum_lines_vs_subtotal").status == "FAIL"


def test_invoice_cash_change_reconciliation():
    inv = InvoiceExtraction(
        total_amount=ev("28.05"),
        cash_paid=ev("50.00"),
        change_given=ev("21.95"),
    )
    result = InvoiceValidator().validate(inv)
    assert check_by_name(result, "cash_change_check").status == "PASS"


def test_invoice_missing_operands_are_not_applicable_not_fail():
    """An absent field must produce NOT_APPLICABLE, never a false FAIL."""
    inv = InvoiceExtraction(total_amount=ev("28.05"))
    result = InvoiceValidator().validate(inv)
    assert check_by_name(result, "cash_change_check").status == "NOT_APPLICABLE"
    assert check_by_name(result, "sum_lines_vs_subtotal").status == "NOT_APPLICABLE"


def test_invoice_lines_vs_total_catches_total_mismatch_without_subtotal():
    """batch3-1495 shape: lines sum to 3480, total says 5257, no subtotal."""
    inv = InvoiceExtraction(
        line_items=[_line(9, "367", "3303"), _line(2, "17", "34")],
        total_amount=ev("5257"),
    )
    result = InvoiceValidator().validate(inv)
    check = check_by_name(result, "sum_lines_vs_total")
    assert check.status == "FAIL"
    assert result.overall_status == "FAIL"


def test_invoice_lines_vs_total_passes_when_consistent():
    inv = InvoiceExtraction(
        line_items=[_line(2, "10.00", "20.00")],
        total_amount=ev("20.00"),
    )
    result = InvoiceValidator().validate(inv)
    assert check_by_name(result, "sum_lines_vs_total").status == "PASS"


def test_invoice_lines_vs_total_skipped_when_subtotal_present():
    """With a subtotal, Checks 2/3 own the reconciliation (no double jeopardy)."""
    inv = InvoiceExtraction(
        line_items=[_line(2, "10.00", "20.00")],
        subtotal=ev("20.00"),
        total_amount=ev("22.00"),
        tax_amount=ev("2.00"),
    )
    result = InvoiceValidator().validate(inv)
    assert check_by_name(result, "sum_lines_vs_total").status == "NOT_APPLICABLE"
    assert result.overall_status == "PASS"


def test_invoice_lines_vs_total_accounts_for_tax_and_discount():
    inv = InvoiceExtraction(
        line_items=[_line(2, "10.00", "20.00")],
        discount=ev("2.00"),
        tax_amount=ev("1.80"),
        total_amount=ev("19.80"),
        tax_inclusive=False,
    )
    result = InvoiceValidator().validate(inv)
    assert check_by_name(result, "sum_lines_vs_total").status == "PASS"


def test_invoice_lines_vs_total_respects_tax_inclusive_totals():
    """
    Malaysian receipts print GST inside the line amounts ("Total Includes
    GST 6%"). Adding the tax again FAILs a perfectly consistent receipt —
    the shape of X51005757342 / X51006334741 / X51008142038.
    """
    inv = InvoiceExtraction(
        line_items=[_line(1, "1.80", "1.80"), _line(2, "2.10", "4.20"),
                    _line(1, "0.20", "0.20")],
        tax_amount=ev("0.35"),
        total_amount=ev("6.20"),
        tax_inclusive=True,
    )
    result = InvoiceValidator().validate(inv)
    check = check_by_name(result, "sum_lines_vs_total")
    assert check.status == "PASS", f"variance {check.variance}"
    assert check.calculated_value == Decimal("6.20")


def test_invoice_lines_vs_total_tries_both_readings_when_tax_basis_unstated():
    """
    A receipt that shows tax but never says inclusive or exclusive gives us
    no basis to pick one. Trying both and reporting the reading that
    reconciles beats inventing a fact the document does not state.
    """
    inclusive = InvoiceExtraction(
        line_items=[_line(1, "6.20", "6.20")] + [_line(1, "0.00", "0.00")],
        tax_amount=ev("0.35"),
        total_amount=ev("6.20"),
    )
    assert check_by_name(
        InvoiceValidator().validate(inclusive), "sum_lines_vs_total"
    ).status == "PASS"

    exclusive = InvoiceExtraction(
        line_items=[_line(1, "6.20", "6.20")] + [_line(1, "0.00", "0.00")],
        tax_amount=ev("0.35"),
        total_amount=ev("6.55"),
    )
    assert check_by_name(
        InvoiceValidator().validate(exclusive), "sum_lines_vs_total"
    ).status == "PASS"


def test_subtotal_plus_tax_respects_a_tax_inclusive_total():
    """
    The canonical trap: "Subtotal 29.00, GST 1.64, Total 29.00" — the tax
    is already inside the total (X51005806685, X51006328913). Adding it
    again FAILs a consistent receipt.
    """
    inv = InvoiceExtraction(
        subtotal=ev("29.00"), tax_amount=ev("1.64"),
        total_amount=ev("29.00"), tax_inclusive=True,
    )
    check = check_by_name(
        InvoiceValidator().validate(inv), "subtotal_plus_tax_vs_total"
    )
    assert check.status == "PASS", f"variance {check.variance}"


def test_subtotal_plus_tax_still_validates_an_exclusive_invoice():
    inv = InvoiceExtraction(
        subtotal=ev("100.00"), tax_amount=ev("6.00"),
        total_amount=ev("106.00"), tax_inclusive=False,
    )
    assert check_by_name(
        InvoiceValidator().validate(inv), "subtotal_plus_tax_vs_total"
    ).status == "PASS"


def test_subtotal_plus_tax_fails_when_neither_reading_holds():
    inv = InvoiceExtraction(
        subtotal=ev("100.00"), tax_amount=ev("6.00"), total_amount=ev("999.00"),
    )
    assert check_by_name(
        InvoiceValidator().validate(inv), "subtotal_plus_tax_vs_total"
    ).status == "FAIL"


def test_sum_lines_vs_subtotal_handles_tax_inclusive_line_amounts():
    """
    X51005663293: line amounts carry GST (sum 30.30) while the printed
    subtotal is the net figure 28.58, with GST 1.72 shown separately.
    """
    inv = InvoiceExtraction(
        line_items=[_line(1, "12.50", "12.50"), _line(1, "3.00", "3.00"),
                    _line(1, "4.90", "4.90"), _line(1, "9.90", "9.90")],
        subtotal=ev("28.58"), tax_amount=ev("1.72"),
        total_amount=ev("30.30"), tax_inclusive=True,
    )
    check = check_by_name(InvoiceValidator().validate(inv), "sum_lines_vs_subtotal")
    assert check.status == "PASS", f"variance {check.variance}"


def test_line_discount_is_subtracted_by_magnitude_whatever_its_sign():
    """X00016469619 prints the deduction as -5.59; a discount only reduces."""
    for shown in ("-5.59", "5.59"):
        item = InvoiceLineItem(
            quantity=ev("1"), unit_price=ev("55.90"),
            discount_amount=ev(shown), amount=ev("50.31"),
        )
        result = InvoiceValidator().validate(InvoiceExtraction(line_items=[item]))
        check = check_by_name(result, "line_total_check_1")
        assert check.status == "PASS", f"{shown}: variance {check.variance}"


def test_invoice_lines_vs_total_trusts_arithmetic_over_the_inclusive_label():
    """
    X51005361895 is labelled tax-inclusive yet prints 8.49 + 0.51 = 9.00.
    The document's arithmetic is direct evidence; the flag is a model
    interpretation. A self-consistent receipt must not FAIL on the label.
    """
    inv = InvoiceExtraction(
        line_items=[_line(1, "8.49", "8.49")],
        tax_amount=ev("0.51"),
        total_amount=ev("9.00"),
        tax_inclusive=True,
    )
    result = InvoiceValidator().validate(inv)
    check = check_by_name(result, "sum_lines_vs_total")
    assert check.status == "PASS", f"variance {check.variance}"
    assert check.calculated_value == Decimal("9.00")


def test_invoice_lines_vs_total_still_fails_a_genuine_mismatch():
    """batch3-1495: neither reading reconciles, so it must stay FAIL."""
    inv = InvoiceExtraction(
        line_items=[_line(9, "367", "3303"), _line(9, "1", "9"),
                    _line(1, "110", "110"), _line(6, "4", "24"),
                    _line(2, "17", "34")],
        total_amount=ev("5257"),
    )
    result = InvoiceValidator().validate(inv)
    assert check_by_name(result, "sum_lines_vs_total").status == "FAIL"


def test_invoice_line_discount_is_applied_before_comparing(
):
    """
    A line that prints its own discount reports the NET amount. Comparing
    gross quantity × unit_price against it would FAIL every discounted line
    on an otherwise consistent invoice (Indian GST invoices print this).
    """
    item = InvoiceLineItem(
        quantity=ev("10"), unit_price=ev("100"),
        discount_amount=ev("150"), amount=ev("850"),
    )
    result = InvoiceValidator().validate(InvoiceExtraction(line_items=[item]))
    check = check_by_name(result, "line_total_check_1")
    assert check.status == "PASS"
    assert check.calculated_value == Decimal("850")


def test_invoice_line_discount_rate_is_applied_when_only_a_rate_is_shown():
    item = InvoiceLineItem(
        quantity=ev("10"), unit_price=ev("100"),
        discount_rate=ev("10"), amount=ev("900"),
    )
    result = InvoiceValidator().validate(InvoiceExtraction(line_items=[item]))
    assert check_by_name(result, "line_total_check_1").status == "PASS"


def test_invoice_line_without_discount_still_compares_gross():
    """An absent discount must never be invented as zero-or-something."""
    item = InvoiceLineItem(quantity=ev("2"), unit_price=ev("10"), amount=ev("20"))
    result = InvoiceValidator().validate(InvoiceExtraction(line_items=[item]))
    check = check_by_name(result, "line_total_check_1")
    assert check.status == "PASS"
    assert "discount_amount" not in check.operands


def test_invoice_tax_inclusive_without_confirmed_base_is_not_applicable():
    """
    On a tax-inclusive receipt there is no independent pre-tax base,
    so the rate check must abstain rather than invent one (Trap P6).
    """
    inv = InvoiceExtraction(
        subtotal=ev("100.00"),
        tax_amount=ev("6.00"),
        tax_rate=ev("6"),
        total_amount=ev("100.00"),
        tax_inclusive=True,
        subtotal_confirmed_pretax=False,
    )
    result = InvoiceValidator().validate(inv)
    check = check_by_name(result, "rate_times_base_vs_tax")
    assert check.status == "NOT_APPLICABLE"
    assert "independent taxable base" in check.reason


def test_invoice_tax_inclusive_with_confirmed_base_is_checked():
    inv = InvoiceExtraction(
        subtotal=ev("100.00"),
        tax_amount=ev("6.00"),
        tax_rate=ev("6"),
        total_amount=ev("106.00"),
        tax_inclusive=True,
        subtotal_confirmed_pretax=True,
    )
    result = InvoiceValidator().validate(inv)
    assert check_by_name(result, "rate_times_base_vs_tax").status == "PASS"


def test_invoice_tax_rate_below_one_percent_is_not_misread_as_fraction():
    """
    A 1% rate must mean 1%, not 100%. The rate>1 heuristic breaks here.
    """
    inv = InvoiceExtraction(
        subtotal=ev("1000.00"),
        tax_amount=ev("10.00"),
        tax_rate=ev("1"),
        total_amount=ev("1010.00"),
    )
    result = InvoiceValidator().validate(inv)
    check = check_by_name(result, "rate_times_base_vs_tax")
    assert check.status == "PASS", (
        f"1% of 1000 should be 10, got calculated={check.calculated_value}"
    )


def test_invoice_tax_rate_as_fraction_is_supported():
    """A rate given as 0.06 means 6%."""
    inv = InvoiceExtraction(
        subtotal=ev("100.00"),
        tax_amount=ev("6.00"),
        tax_rate=ev("0.06"),
        total_amount=ev("106.00"),
    )
    result = InvoiceValidator().validate(inv)
    assert check_by_name(result, "rate_times_base_vs_tax").status == "PASS"


def test_invoice_checks_expose_their_operands():
    """A reviewer must be able to see what went into each calculation."""
    inv = InvoiceExtraction(line_items=[_line(3, "10.00", "30.00")])
    result = InvoiceValidator().validate(inv)
    check = check_by_name(result, "line_total_check_1")
    assert check.operands, "operands must record the inputs to the calculation"


# ─────────────────────────────────────────────────────────────────────
# Balance sheet validator
# ─────────────────────────────────────────────────────────────────────

def _bs_period(assets: dict, liabs: dict, total_a, total_l) -> BalanceSheetPeriod:
    return BalanceSheetPeriod(
        assets={k: ev(v) for k, v in assets.items()},
        total_assets=ev(total_a),
        capital_and_liabilities={k: ev(v) for k, v in liabs.items()},
        total_capital_and_liabilities=ev(total_l),
    )


def test_balance_sheet_balances():
    ext = BalanceSheetExtraction(periods={"2021": _bs_period(
        {"Cash": "100", "Investments": "900"},
        {"Capital": "400", "Deposits": "600"},
        "1000", "1000",
    )})
    result = BalanceSheetValidator().validate(ext)
    assert result.overall_status == "PASS"
    assert result.summary["pass"] == 3


def test_balance_sheet_out_of_balance_fails():
    ext = BalanceSheetExtraction(periods={"2021": _bs_period(
        {"Cash": "100", "Investments": "900"},
        {"Capital": "400", "Deposits": "600"},
        "1000", "1100",
    )})
    result = BalanceSheetValidator().validate(ext)
    assert check_by_name(result, "2021_assets_vs_liabilities").status == "FAIL"
    assert result.overall_status == "FAIL"


def test_balance_sheet_each_period_validated_independently():
    """Comparative columns must each get their own checks (Trap D4)."""
    ext = BalanceSheetExtraction(periods={
        "2021": _bs_period({"Cash": "100", "Inv": "900"}, {"Cap": "400", "Dep": "600"}, "1000", "1000"),
        "2020": _bs_period({"Cash": "50", "Inv": "450"}, {"Cap": "200", "Dep": "300"}, "500", "500"),
    })
    result = BalanceSheetValidator().validate(ext)
    assert check_by_name(result, "2021_assets_vs_liabilities") is not None
    assert check_by_name(result, "2020_assets_vs_liabilities") is not None
    assert result.summary["pass"] == 6


def test_balance_sheet_real_2021_figures():
    """Actual HDFC consolidated figures, ₹ in '000 (dataset)."""
    assets = {
        "Cash and balances with Reserve Bank of India": "973703555",
        "Balances with banks and money at call and short notice": "239021709",
        "Investments": "4388231117",
        "Advances": "11852835198",
        "Fixed assets": "50995631",
        "Other assets": "490279232",
    }
    liabs = {
        "Capital": "5512776",
        "Reserves and surplus": "2092589110",
        "Minority interest": "6327647",
        "Deposits": "13337208758",
        "Borrowings": "1776967487",
        "Other liabilities and provisions": "776460664",
    }
    ext = BalanceSheetExtraction(periods={"2021": _bs_period(
        assets, liabs, "17995066442", "17995066442",
    )})
    result = BalanceSheetValidator().validate(ext)
    assert result.overall_status == "PASS"
    assert check_by_name(result, "2021_sum_assets").variance == 0
    assert check_by_name(result, "2021_sum_liabilities").variance == 0


def test_balance_sheet_single_component_is_not_applicable():
    """One component alone cannot meaningfully validate a total."""
    ext = BalanceSheetExtraction(periods={"2021": BalanceSheetPeriod(
        assets={"Cash": ev("1000")},
        total_assets=ev("1000"),
    )})
    result = BalanceSheetValidator().validate(ext)
    assert check_by_name(result, "2021_sum_assets").status == "NOT_APPLICABLE"


def test_balance_sheet_unreadable_component_does_not_become_zero():
    """An unreadable line must not silently drop out and still 'pass'."""
    period = BalanceSheetPeriod(
        assets={"Cash": ev("100"), "Investments": ev_unreadable(), "Advances": ev("900")},
        total_assets=ev("1500"),
        capital_and_liabilities={"Capital": ev("500"), "Deposits": ev("1000")},
        total_capital_and_liabilities=ev("1500"),
    )
    ext = BalanceSheetExtraction(periods={"2021": period})
    result = BalanceSheetValidator().validate(ext)
    check = check_by_name(result, "2021_sum_assets")
    assert check.status != "PASS", "dropping an unreadable component must not yield a pass"


# ─────────────────────────────────────────────────────────────────────
# Profit & loss validator
# ─────────────────────────────────────────────────────────────────────

def test_profit_loss_income_and_expenditure_chain():
    period = ProfitLossPeriod(
        interest_earned=ev("1000"), other_income=ev("200"), total_income=ev("1200"),
        interest_expended=ev("400"), operating_expenses=ev("300"),
        provisions_and_contingencies=ev("100"), total_expenditure=ev("800"),
        consolidated_net_profit=ev("400"),
    )
    result = ProfitLossValidator().validate(ProfitLossExtraction(periods={"2024": period}))
    assert check_by_name(result, "2024_sum_income").status == "PASS"
    assert check_by_name(result, "2024_sum_expenditure").status == "PASS"
    assert check_by_name(result, "2024_income_minus_expenditure").status == "PASS"


def test_profit_loss_net_profit_before_minority_is_checked_separately():
    """
    Income − expenditure gives profit BEFORE minority interest.
    Consolidated profit is that figure adjusted for minority and associates.
    Both equations must be validated, not conflated into one.
    """
    period = ProfitLossPeriod(
        total_income=ev("1200"), total_expenditure=ev("800"),
        net_profit_before_minority=ev("400"),
        minority_interest=ev("30"),
        share_in_profits_of_associates=ev("10"),
        consolidated_net_profit=ev("380"),
    )
    result = ProfitLossValidator().validate(ProfitLossExtraction(periods={"2024": period}))
    assert check_by_name(result, "2024_net_profit_before_minority") is not None, \
        "missing check: total_income − total_expenditure vs net_profit_before_minority"
    assert check_by_name(result, "2024_net_profit_before_minority").status == "PASS"


def test_profit_loss_appropriation_reconciliation():
    period = ProfitLossPeriod(
        total_available_for_appropriation=ev("1000"),
        appropriations={"Transfer to statutory reserve": ev("200"), "Dividend": ev("300")},
        balance_carried_forward=ev("500"),
    )
    result = ProfitLossValidator().validate(ProfitLossExtraction(periods={"2024": period}))
    assert check_by_name(result, "2024_appropriation").status == "PASS"


@pytest.mark.parametrize("carried_label", [
    "Balance carried over to Balance Sheet",
    "Balance carried over to consolidated balance sheet",
    "Balance carried forward",
])
def test_appropriation_excludes_the_carried_forward_residual(carried_label):
    """
    The statements print the residual inside the appropriation block.
    Counting it as an appropriation makes the identity self-cancelling
    (total_available − total_available ≈ 0) and FAILs every real P&L.
    Shape taken from Consolidated Profit & Loss 2024.
    """
    period = ProfitLossPeriod(
        total_available_for_appropriation=ev("188001.49"),
        appropriations={
            "Transfer to Statutory Reserve": ev("15684.72"),
            "Transfer to General Reserve": ev("6081.23"),
            "Transfer to Special Reserve": ev("3000.00"),
            "Transfer to Capital Reserve": ev("4166.41"),
            "Transfer to / (from) Investment Reserve Account (net)": ev("529.42"),
            "Transfer to / (from) Investment Fluctuation Reserve": ev("378.00"),
            "Dividend pertaining to previous year paid during the year": ev("8404.42"),
            "Transfer to / (from) Minority Interest (opening adjustment)": ev("-288.28"),
            carried_label: ev("150045.57"),
        },
        balance_carried_forward=ev("150045.57"),
    )
    result = ProfitLossValidator().validate(ProfitLossExtraction(periods={"2024": period}))
    check = check_by_name(result, "2024_appropriation")
    assert check.status == "PASS", f"variance {check.variance}"
    assert check.calculated_value == Decimal("150045.57")
    # The excluded row is disclosed rather than silently dropped.
    assert carried_label in check.operands["excluded_from_sum"]
    assert carried_label not in [
        k for k in check.operands if k.startswith("Transfer")
    ]


def test_appropriation_keeps_genuine_transfers_in_the_sum():
    """Exclusion is label-scoped: a real transfer must never be dropped."""
    period = ProfitLossPeriod(
        total_available_for_appropriation=ev("1000"),
        appropriations={
            "Transfer to Statutory Reserve": ev("200"),
            "Interim Dividend paid": ev("100"),
        },
        balance_carried_forward=ev("999"),
    )
    result = ProfitLossValidator().validate(ProfitLossExtraction(periods={"2024": period}))
    check = check_by_name(result, "2024_appropriation")
    assert check.status == "FAIL"
    assert check.calculated_value == Decimal("700")


def test_profit_loss_eps_is_not_scaled_by_document_unit():
    """EPS is always ₹ per share even when the statement is in crore (Trap D8)."""
    period = ProfitLossPeriod(basic_eps=ev("79.25"), diluted_eps=ev("78.90"))
    ext = ProfitLossExtraction(unit_code="INR_CRORE", periods={"2024": period})
    assert ext.periods["2024"].basic_eps.value == Decimal("79.25")


# ─────────────────────────────────────────────────────────────────────
# Cash flow validator (Trap D6, D27, D28)
# ─────────────────────────────────────────────────────────────────────

def test_cash_flow_2024_real_figures():
    """HDFC FY2024 consolidated cash flow chain from the dataset."""
    period = CashFlowPeriod(
        net_cash_from_operating=ev("19069.34"),
        net_cash_from_investing=ev("5313.77"),
        net_cash_from_financing=ev("-3988.06"),
        fx_translation_adjustment=ev("94.94"),
        net_increase_in_cash=ev("20489.99"),
        cash_acquired_on_amalgamation=ev("11781.71"),
        amalgamation_position="after_net_increase",
        opening_cash=ev("197147.81"),
        closing_cash=ev("229419.51"),
    )
    result = CashFlowValidator().validate(CashFlowExtraction(periods={"2024": period}))
    assert check_by_name(result, "2024_cash_flow_components").status == "PASS"
    assert check_by_name(result, "2024_cash_reconciliation").status == "PASS"


def test_cash_flow_amalgamation_before_net_increase():
    """When amalgamation sits above the net-increase line it joins equation 1."""
    period = CashFlowPeriod(
        net_cash_from_operating=ev("100"),
        net_cash_from_investing=ev("50"),
        net_cash_from_financing=ev("-30"),
        cash_acquired_on_amalgamation=ev("20"),
        amalgamation_position="before_net_increase",
        net_increase_in_cash=ev("140"),
        opening_cash=ev("500"),
        closing_cash=ev("640"),
    )
    result = CashFlowValidator().validate(CashFlowExtraction(periods={"2024": period}))
    assert check_by_name(result, "2024_cash_flow_components").status == "PASS"
    assert check_by_name(result, "2024_cash_reconciliation").status == "PASS"


def test_cash_flow_amalgamation_not_double_counted():
    """The amalgamation figure must be applied to exactly one equation."""
    period = CashFlowPeriod(
        net_cash_from_operating=ev("100"),
        net_cash_from_investing=ev("50"),
        net_cash_from_financing=ev("-30"),
        cash_acquired_on_amalgamation=ev("20"),
        amalgamation_position="before_net_increase",
        net_increase_in_cash=ev("140"),
        opening_cash=ev("500"),
        closing_cash=ev("640"),
    )
    result = CashFlowValidator().validate(CashFlowExtraction(periods={"2024": period}))
    recon = check_by_name(result, "2024_cash_reconciliation")
    assert recon.calculated_value == Decimal("640"), "amalgamation counted twice"


def test_cash_flow_section_sums_are_validated():
    """
    Each activity section's line items must be reconciled against that
    section's net figure — the same treatment the balance sheet gets.
    """
    period = CashFlowPeriod(
        operating_items={"Profit before tax": ev("800"), "Depreciation": ev("200")},
        net_cash_from_operating=ev("1000"),
        investing_items={"Purchase of fixed assets": ev("-300"), "Sale of investments": ev("100")},
        net_cash_from_investing=ev("-200"),
        financing_items={"Dividends paid": ev("-50"), "Borrowings": ev("150")},
        net_cash_from_financing=ev("100"),
        net_increase_in_cash=ev("900"),
        opening_cash=ev("100"),
        closing_cash=ev("1000"),
    )
    result = CashFlowValidator().validate(CashFlowExtraction(periods={"2024": period}))
    for section in ("operating", "investing", "financing"):
        name = f"2024_sum_{section}_items"
        assert check_by_name(result, name) is not None, f"missing check: {name}"
        assert check_by_name(result, name).status == "PASS"


def test_cash_flow_section_sum_excludes_an_intermediate_subtotal():
    """
    Cash flow sections print running subtotals as real labelled rows —
    "Consolidated profit before income tax and after minority interest"
    is the sum of the two rows above it (CF 2026). Adding it to its own
    components double-counts and the section can never reconcile.
    """
    period = CashFlowPeriod(
        operating_items={
            "Consolidated Profit after tax and minority interest": ev("76025.97"),
            "Add: Provision for income tax": ev("22921.99"),
            "Consolidated profit before income tax": ev("98947.96"),  # subtotal
            "Depreciation on fixed assets": ev("4194.90"),
            "Direct taxes paid (net of refunds)": ev("10363.52"),
        },
        net_cash_from_operating=ev("113506.38"),
    )
    result = CashFlowValidator().validate(CashFlowExtraction(periods={"2026": period}))
    check = check_by_name(result, "2026_sum_operating_items")
    assert check.status == "PASS", f"variance {check.variance}"
    # Excluded, not deleted — the reviewer can see what was left out.
    assert "Consolidated profit before income tax" in check.operands[
        "excluded_intermediate_subtotals"
    ]


def test_cash_flow_section_sum_keeps_a_coincidental_early_row():
    """
    The detector needs two accepted rows before it will call anything a
    subtotal — a second row equal to the first is a real line item, not a
    total of the section.
    """
    period = CashFlowPeriod(
        operating_items={"A": ev("100"), "B": ev("100")},
        net_cash_from_operating=ev("200"),
    )
    result = CashFlowValidator().validate(CashFlowExtraction(periods={"2026": period}))
    check = check_by_name(result, "2026_sum_operating_items")
    assert check.status == "PASS"
    assert check.calculated_value == Decimal("200")
    assert "excluded_intermediate_subtotals" not in check.operands


def test_cash_flow_section_sum_still_fails_a_genuine_mismatch():
    """Exclusion must not become a way of making every section reconcile."""
    period = CashFlowPeriod(
        operating_items={"A": ev("100"), "B": ev("50"), "C": ev("25")},
        net_cash_from_operating=ev("999"),
    )
    result = CashFlowValidator().validate(CashFlowExtraction(periods={"2026": period}))
    check = check_by_name(result, "2026_sum_operating_items")
    assert check.status == "FAIL"
    assert check.calculated_value == Decimal("175")


def test_cash_flow_missing_amalgamation_position_is_not_guessed():
    """With no structural position we must not assume where it belongs."""
    period = CashFlowPeriod(
        net_cash_from_operating=ev("100"),
        net_cash_from_investing=ev("50"),
        net_cash_from_financing=ev("-30"),
        cash_acquired_on_amalgamation=ev("20"),
        amalgamation_position=None,
        net_increase_in_cash=ev("120"),
    )
    result = CashFlowValidator().validate(CashFlowExtraction(periods={"2024": period}))
    check = check_by_name(result, "2024_cash_flow_components")
    assert check.calculated_value == Decimal("120")


# ─────────────────────────────────────────────────────────────────────
# Dispatcher
# ─────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("doc_type", [
    "invoice", "balance_sheet", "profit_and_loss", "cash_flow_statement",
])
def test_service_dispatches_every_document_type(doc_type):
    result = FinancialValidationService().validate(extracted_data={}, document_type=doc_type)
    assert result.overall_status in ("PASS", "FAIL", "NOT_APPLICABLE")


def test_service_returns_empty_result_for_unknown_type():
    result = FinancialValidationService().validate(extracted_data={}, document_type="nope")
    assert result.checks == []
    assert result.overall_status == "NOT_APPLICABLE"


def test_service_round_trips_json_serialized_decimals():
    """
    Extraction is persisted as JSON, so validation must survive
    Decimal -> str -> Decimal without losing precision.
    """
    ext = BalanceSheetExtraction(periods={"2021": _bs_period(
        {"Cash": "973703555.55", "Investments": "4388231117.45"},
        {"Capital": "5512776.10", "Deposits": "5356421896.90"},
        "5361934673.00", "5361934673.00",
    )})
    as_json = ext.model_dump(mode="json")
    result = FinancialValidationService().validate(as_json, "balance_sheet")
    assert result.overall_status == "PASS"
    assert check_by_name(result, "2021_sum_assets").variance == 0


def test_validation_result_overall_status_precedence():
    from backend.app.schemas.validation import ValidationCheck, ValidationResult

    r = ValidationResult(checks=[
        ValidationCheck(name="a", status="PASS"),
        ValidationCheck(name="b", status="FAIL"),
        ValidationCheck(name="c", status="NOT_APPLICABLE"),
    ])
    r.compute_summary()
    assert r.overall_status == "FAIL"

    r = ValidationResult(checks=[ValidationCheck(name="a", status="NOT_APPLICABLE")])
    r.compute_summary()
    assert r.overall_status == "NOT_APPLICABLE"
