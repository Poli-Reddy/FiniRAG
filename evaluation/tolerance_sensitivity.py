"""
Empirical validation of the financial tolerance rule (Correction #8).

`financial_validation_service.is_approximately_equal` currently uses

    tolerance = max(Decimal("0.01"), abs(reported) * Decimal("0.001"))

and the source calls that PROPOSED, "subject to empirical validation". This
script supplies the evidence, using the real check results in
`evaluation/results/` — no LLM calls, no new pipeline runs.

Four questions, in order:

  A. What do the observed variances actually look like? A tolerance only
     matters in the band between the largest variance that should pass and
     the smallest that should fail.

  B. Over what range of (floor, factor) does every verdict stay the same?
     If the verdicts are stable across three orders of magnitude, the exact
     constant is not load-bearing and the rule is safe.

  C. Is the tolerance loose enough to hide a real error? For every check
     that sums document line items, compare the tolerance against the
     smallest line item in that sum: if tolerance >= that item, dropping
     the item entirely would still be reported as PASS.

  D. What follows from A-C.

Usage:
    ./venv/Scripts/python.exe evaluation/tolerance_sensitivity.py
"""

from __future__ import annotations

import json
from decimal import Decimal, InvalidOperation
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
RESULTS_DIR = EVAL_DIR / "results"

# The rule as shipped today.
CURRENT_FLOOR = Decimal("0.01")
CURRENT_FACTOR = Decimal("0.001")

# Floors and relative factors to sweep. 0 means "that half of the rule is
# switched off", which is the cleanest way to see what each half contributes.
FLOORS = [Decimal("0"), Decimal("0.01"), Decimal("1"), Decimal("100"), Decimal("10000")]
FACTORS = [
    Decimal("0"), Decimal("0.000001"), Decimal("0.00001"), Decimal("0.0001"),
    Decimal("0.001"), Decimal("0.01"), Decimal("0.1"),
]

# Checks whose operands are exactly the components of one sum, each keyed by
# its own document label (they are built by `sum_operands`). Only these can
# answer "would a dropped line item be noticed?" — a check like
# `assets_vs_liabilities` compares two totals and has no components.
COMPONENT_SUM_SUFFIXES = (
    "_sum_assets",
    "_sum_liabilities",
    "_sum_income",
    "_sum_expenditure",
    "_sum_operating_items",
    "_sum_investing_items",
    "_sum_financing_items",
)


def dec(text) -> Decimal | None:
    """Parse an exact decimal string, or None if it isn't one."""
    if text is None:
        return None
    try:
        return Decimal(str(text))
    except (InvalidOperation, ValueError, ArithmeticError):
        return None


def tolerance_for(reported: Decimal, floor: Decimal, factor: Decimal) -> Decimal:
    """The shipped rule, with its two constants made into parameters."""
    if reported == 0:
        return floor
    return max(floor, abs(reported) * factor)


# ─────────────────────────────────────────────────────────────────────
# Load
# ─────────────────────────────────────────────────────────────────────

def load_checks() -> list[dict]:
    """Every check that actually compared two numbers, with its document."""
    checks = []
    for path in sorted(RESULTS_DIR.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        validation = payload.get("validation") or {}
        for check in validation.get("checks", []):
            reported = dec(check.get("reported_value"))
            variance = dec(check.get("variance"))
            if reported is None or variance is None:
                continue  # NOT_APPLICABLE — nothing was compared
            checks.append({
                "document": payload.get("document_name", ""),
                "document_type": payload.get("document_type", ""),
                "name": check.get("name", ""),
                "formula": check.get("formula", ""),
                "operands": check.get("operands", {}),
                "reported": reported,
                "calculated": dec(check.get("calculated_value")),
                "variance": variance,
                "tolerance": dec(check.get("tolerance")),
                "status": check.get("status", ""),
            })
    return checks


# ─────────────────────────────────────────────────────────────────────
# A. Observed variance
# ─────────────────────────────────────────────────────────────────────

def analyse_variance(checks: list[dict]) -> dict:
    exact = [c for c in checks if c["variance"] == 0]
    inexact = [c for c in checks if c["variance"] != 0]

    passing = [c for c in checks if c["status"] == "PASS"]
    failing = [c for c in checks if c["status"] == "FAIL"]

    def relative(c):
        return abs(c["variance"]) / abs(c["reported"]) if c["reported"] else None

    return {
        "total_checks": len(checks),
        "exact_matches": len(exact),
        "inexact": [
            {
                "document": c["document"],
                "check": c["name"],
                "variance": str(c["variance"]),
                "reported": str(c["reported"]),
                "relative": str(relative(c)) if relative(c) is not None else None,
                "tolerance": str(c["tolerance"]),
                "status": c["status"],
            }
            for c in sorted(inexact, key=lambda x: -abs(x["variance"]))
        ],
        "max_variance_among_passing": str(max((abs(c["variance"]) for c in passing),
                                              default=Decimal("0"))),
        "min_variance_among_failing": (
            str(min(abs(c["variance"]) for c in failing)) if failing else None
        ),
        "passing": len(passing),
        "failing": len(failing),
    }


# ─────────────────────────────────────────────────────────────────────
# B. Verdict stability
# ─────────────────────────────────────────────────────────────────────

def sweep(checks: list[dict]) -> list[dict]:
    """PASS/FAIL counts and flip count at every (floor, factor) in the grid."""
    baseline = {
        (c["document"], c["name"]): c["variance"] <= tolerance_for(
            c["reported"], CURRENT_FLOOR, CURRENT_FACTOR
        )
        for c in checks
    }

    grid = []
    for floor in FLOORS:
        for factor in FACTORS:
            flipped = []
            passes = 0
            for c in checks:
                ok = c["variance"] <= tolerance_for(c["reported"], floor, factor)
                passes += ok
                if ok != baseline[(c["document"], c["name"])]:
                    flipped.append(f"{c['document']} :: {c['name']}")
            grid.append({
                "floor": str(floor),
                "factor": str(factor),
                "pass": passes,
                "fail": len(checks) - passes,
                "flipped_vs_current": len(flipped),
                "flipped_examples": flipped[:5],
            })
    return grid


# ─────────────────────────────────────────────────────────────────────
# C. Masking — can the tolerance swallow a whole line item?
# ─────────────────────────────────────────────────────────────────────

def component_sums(checks: list[dict]) -> list[dict]:
    """Component-summed checks, with the smallest component in each sum."""
    out = []
    for c in checks:
        if not c["name"].endswith(COMPONENT_SUM_SUFFIXES):
            continue
        magnitudes = []
        for label, raw in (c["operands"] or {}).items():
            value = dec(raw)
            if value is not None and value != 0:
                magnitudes.append((abs(value), label))
        if len(magnitudes) < 2:
            continue
        smallest, label = min(magnitudes)
        out.append({
            "document": c["document"],
            "document_type": c["document_type"],
            "check": c["name"],
            "components": len(magnitudes),
            "reported": c["reported"],
            "smallest_component": smallest,
            "smallest_label": label,
        })
    return out


def masking(sums: list[dict], floor: Decimal, factor: Decimal) -> dict:
    """How many sums have a tolerance wide enough to hide their smallest item."""
    vulnerable = []
    for s in sums:
        tol = tolerance_for(s["reported"], floor, factor)
        if tol >= s["smallest_component"]:
            vulnerable.append({
                "document": s["document"],
                "check": s["check"],
                "smallest_component": str(s["smallest_component"]),
                "label": s["smallest_label"],
                "tolerance": str(tol),
            })
    return {"checked": len(sums), "vulnerable": vulnerable}


def largest_safe_factor(sums: list[dict], floor: Decimal) -> Decimal | None:
    """
    The biggest relative factor at which no sum can hide its smallest item.

    For one sum that bound is smallest_component / |reported|; across the
    dataset it is the minimum of those, minus an epsilon so the comparison
    is strict.
    """
    bounds = [
        s["smallest_component"] / abs(s["reported"])
        for s in sums if s["reported"] != 0
    ]
    return min(bounds) if bounds else None


def recommend(checks, sums, safe_factor):
    """
    The tightest swept factor that changes no verdict and masks no line item.

    A recommendation has to clear both bars at once. Tightening until masking
    disappears is worthless if it starts failing documents that reconcile, and
    a setting that flips nothing is worthless if it can still swallow a
    printed line. Only factors already in FACTORS are considered, so the
    recommendation is one the sweep table above actually evidences.
    """
    if safe_factor is None:
        return None

    candidates = []
    for factor in FACTORS:
        if factor == 0 or factor > safe_factor:
            continue
        flips = sum(
            1 for c in checks
            if (c["variance"] <= tolerance_for(c["reported"], CURRENT_FLOOR, factor))
            != (c["variance"] <= tolerance_for(c["reported"], CURRENT_FLOOR, CURRENT_FACTOR))
        )
        if flips == 0:
            candidates.append(factor)

    if not candidates:
        return None

    best = max(candidates)  # tightest that is still safe = largest qualifying
    return {
        "factor": best,
        "margin_vs_masking_bound": safe_factor / best,
        "tighter_than_current_by": CURRENT_FACTOR / best,
    }


# ─────────────────────────────────────────────────────────────────────
# Render
# ─────────────────────────────────────────────────────────────────────

def render(checks, var, grid, sums, mask_now, safe_factor, rec) -> str:
    lines: list[str] = []
    add = lines.append

    add("# FiniRAG — Tolerance Sensitivity Analysis")
    add("")
    add("The shipped rule is")
    add("")
    add("```python")
    add('tolerance = max(Decimal("0.01"), abs(reported) * Decimal("0.001"))')
    add("```")
    add("")
    add("a 0.1% relative band with a one-paisa floor. The source marks it")
    add("PROPOSED, subject to empirical validation. This is that validation,")
    add(f"computed over the **{var['total_checks']} arithmetic checks** the pipeline")
    add("actually produced on the real dataset (`evaluation/results/`).")
    add("")

    # ── A ──
    add("## A. What the variances actually look like")
    add("")
    add("| | Count |")
    add("|---|---|")
    add(f"| Checks comparing two numbers | {var['total_checks']} |")
    add(f"| Variance exactly zero | {var['exact_matches']} |")
    add(f"| Variance non-zero | {var['total_checks'] - var['exact_matches']} |")
    add(f"| Status PASS | {var['passing']} |")
    add(f"| Status FAIL | {var['failing']} |")
    add("")

    if var["exact_matches"] == var["total_checks"]:
        add("**Every single check reconciles exactly.** Calculated equals reported")
        add("to the last digit in all of them, so the tolerance is never what")
        add("decides a verdict on this dataset — it is pure headroom.")
    else:
        add("Non-zero variances, largest first:")
        add("")
        add("| Document | Check | Variance | Reported | Relative | Tolerance | Verdict |")
        add("|---|---|---|---|---|---|---|")
        for row in var["inexact"][:30]:
            rel = row["relative"]
            rel_txt = f"{Decimal(rel) * 100:.6f}%" if rel else "n/a"
            add(
                f"| {row['document']} | `{row['check']}` | {row['variance']} "
                f"| {row['reported']} | {rel_txt} | {row['tolerance']} | {row['status']} |"
            )
    add("")
    add(f"- Largest variance among **passing** checks: `{var['max_variance_among_passing']}`")
    add(f"- Smallest variance among **failing** checks: "
        f"`{var['min_variance_among_failing'] or 'no failing checks'}`")
    add("")

    # ── B ──
    add("## B. Do the verdicts move when the constants move?")
    add("")
    add("Each cell re-runs every check under a different `(floor, factor)` and")
    add("counts how many verdicts differ from the shipped rule. `factor = 0`")
    add("switches off the relative band; `floor = 0` switches off the absolute one.")
    add("")
    add("| floor \\ factor | " + " | ".join(str(f) for f in FACTORS) + " |")
    add("|---|" + "---|" * len(FACTORS))
    for floor in FLOORS:
        cells = []
        for factor in FACTORS:
            cell = next(
                g for g in grid
                if g["floor"] == str(floor) and g["factor"] == str(factor)
            )
            marker = "**current**" if (floor == CURRENT_FLOOR and factor == CURRENT_FACTOR) else ""
            flips = cell["flipped_vs_current"]
            cells.append(f"{cell['pass']}P/{cell['fail']}F{f' ({flips} flip)' if flips else ''}{marker}")
        add(f"| **{floor}** | " + " | ".join(cells) + " |")
    add("")

    stable = [g for g in grid if g["flipped_vs_current"] == 0]
    add(f"{len(stable)} of the {len(grid)} settings produce **identical verdicts** to")
    add("the shipped rule.")
    unstable = [g for g in grid if g["flipped_vs_current"] > 0]
    if unstable:
        add("")
        add("Settings that change at least one verdict:")
        add("")
        add("| floor | factor | flips | example |")
        add("|---|---|---|---|")
        for g in sorted(unstable, key=lambda x: -x["flipped_vs_current"])[:12]:
            add(f"| {g['floor']} | {g['factor']} | {g['flipped_vs_current']} "
                f"| {g['flipped_examples'][0] if g['flipped_examples'] else ''} |")
    add("")

    # ── C ──
    add("## C. Could the tolerance hide a real error?")
    add("")
    add("Verdict stability alone is not enough. A tolerance of ±50% would also")
    add("be perfectly stable on a dataset that reconciles exactly — and useless.")
    add("The sharper question is whether the band is wide enough to swallow a")
    add("line the document actually printed.")
    add("")
    add("For every check that sums document line items into a reported total,")
    add("compare its tolerance against the **smallest item in that sum**. If the")
    add("tolerance is the larger of the two, dropping that item entirely would")
    add("still be reported as PASS.")
    add("")
    add(f"- Component-summed checks examined: **{mask_now['checked']}**")
    add(f"- Checks whose tolerance is >= their smallest line item: "
        f"**{len(mask_now['vulnerable'])}**")
    add("")
    if mask_now["vulnerable"]:
        add(f"**{len(mask_now['vulnerable'])} of {mask_now['checked']} sums are in that")
        add("position.** Each row below names a line the document actually prints")
        add("whose complete disappearance the current tolerance would not catch:")
        add("")
        add("| Document | Check | Smallest item | Its label | Tolerance |")
        add("|---|---|---|---|---|")
        for v in mask_now["vulnerable"][:20]:
            add(f"| {v['document']} | `{v['check']}` | {v['smallest_component']} "
                f"| {v['label']} | {v['tolerance']} |")
        if len(mask_now["vulnerable"]) > 20:
            add("")
            add(f"({len(mask_now['vulnerable']) - 20} further rows omitted.)")
        add("")
    else:
        add("None. Every sum has a tolerance smaller than its own smallest")
        add("component, so no single omitted line can pass unnoticed.")
        add("")

    if safe_factor is not None:
        add(f"The binding constraint across the dataset is a relative factor of")
        add(f"**{safe_factor:.3e}** — the point at which the tightest sum's smallest")
        add("component equals its tolerance. Above that, at least one omitted line")
        add("item becomes invisible.")
        add("")
        ratio = CURRENT_FACTOR / safe_factor
        if ratio > 1:
            add(f"The shipped factor of {CURRENT_FACTOR} is **{ratio:.0f}x wider** than "
                "that bound.")
        else:
            add(f"The shipped factor of {CURRENT_FACTOR} sits **{1 / ratio:.1f}x below** "
                "that bound.")
        add("")

    # ── D ──
    add("## D. Conclusion")
    add("")

    add("1. **The rule is not load-bearing on this dataset.** "
        f"{var['exact_matches']}/{var['total_checks']} checks reconcile exactly, and "
        f"{len(stable)}/{len(grid)} of the swept settings give identical verdicts. "
        "Tolerance is not what produces the pass rate; correct extraction is. "
        "That also means section B on its own cannot justify the constants — "
        "any value in a very wide range would look equally good.")
    add("")

    add("2. **Both halves of the rule do distinct work.** The relative term is")
    add("   what keeps a multi-billion total from being judged by the same")
    add("   absolute band as a two-figure receipt line. The floor is what stops")
    add("   a legitimately zero total (`reported == 0`, common on nil rows) from")
    add("   demanding exactness on a value that arrived as a decimal string.")
    add("   Neither can be dropped: the `factor = 0` column and the `floor = 0`")
    add("   row above are each a different rule, not a simpler one.")
    add("")

    if mask_now["vulnerable"]:
        share = 100 * len(mask_now["vulnerable"]) / mask_now["checked"]
        add(f"3. **The band is looser than it looks — this is the finding.** "
            f"{len(mask_now['vulnerable'])} of {mask_now['checked']} component sums "
            f"({share:.0f}%) carry a tolerance at least as large as their own "
            "smallest printed line item. For those sums, an extraction that "
            "dropped that line entirely would still be reported PASS. Verdict "
            "stability (section B) hid this completely, because no document in "
            "the dataset *has* a dropped line — the exposure is latent.")
        add("")
        if safe_factor is not None:
            add(f"4. **The masking-free bound is a factor of {safe_factor:.2e}**, "
                f"about {CURRENT_FACTOR / safe_factor:.0f}x tighter than the "
                f"{CURRENT_FACTOR} in use. ")
            if rec:
                add(f"   A factor of **{rec['factor']}** clears both bars at once: it "
                    f"leaves all {var['total_checks']} verdicts unchanged and sits "
                    f"{rec['margin_vs_masking_bound']:.1f}x inside the masking bound, "
                    f"{rec['tighter_than_current_by']:.0f}x tighter than today.")
            else:
                add("   No factor in the swept grid is both masking-free and "
                    "verdict-preserving, so tightening needs a finer sweep than "
                    "this one before it can be recommended.")
            add("")
        add("5. **Recommendation — report, do not silently change.** "
            "`master_prompt.md` puts tolerance constants under CHANGE CONTROL, "
            "and this is a material change: it would tighten the acceptance "
            "criterion for every document the system will ever see. The evidence "
            "for tightening is above; the decision is not this script's to make. "
            "The shipped rule stays as it is until that decision is taken.")
        add("")
        add("   Two caveats bound the finding. It is computed on the documents in")
        add("   `evaluation/results/` only — a dataset of genuinely rounded")
        add("   statements would shift the picture, because real rounding variance")
        add("   is what the relative band exists to absorb. And masking a line")
        add("   item requires the extraction to lose it silently; the schema's")
        add("   `MISSING` status is designed to surface exactly that, so the")
        add("   tolerance is the second line of defence, not the first.")
    else:
        add("3. **The band is tight enough to matter.** Section C shows it stays")
        add(f"   below the smallest printed line item in all {mask_now['checked']} sums")
        add("   checked, which is the failure mode that would actually cost money:")
        add("   a line silently dropped from a total that still adds up.")
        add("")
        add(f"4. **Keep `max({CURRENT_FLOOR}, |reported| * {CURRENT_FACTOR})`.** It is")
        add("   defensible on the evidence above rather than on convention. Should a")
        add("   future dataset produce genuine rounding variance, the sweep in")
        add("   section B is the thing to re-run before changing it.")
    add("")

    add("## Scope of this evidence")
    add("")
    docs = sorted({c["document"] for c in checks})
    types = sorted({c["document_type"] for c in checks})
    add(f"Computed over **{len(docs)} document(s)** of type "
        f"{', '.join(f'`{t}`' for t in types)}, producing "
        f"{var['total_checks']} comparable checks. Document types absent from")
    add("this list are not covered by any claim above — notably, invoice rounding")
    add("is where a relative band is most likely to be exercised, and no invoice")
    add("contributed a check here. Re-run after extending the dataset.")
    add("")
    add("Documents included:")
    add("")
    for d in docs:
        add(f"- {d}")
    add("")
    add("## Reproduce")
    add("")
    add("```bash")
    add("./venv/Scripts/python.exe evaluation/tolerance_sensitivity.py")
    add("```")
    add("")
    add("Reads only `evaluation/results/*.json`. No API calls, no database.")
    add("")

    return "\n".join(lines) + "\n"


def main() -> int:
    if not RESULTS_DIR.is_dir():
        print(f"No results at {RESULTS_DIR}. Run run_evaluation.py first.")
        return 1

    checks = load_checks()
    if not checks:
        print("No comparable checks found in the results.")
        return 1

    var = analyse_variance(checks)
    grid = sweep(checks)
    sums = component_sums(checks)
    mask_now = masking(sums, CURRENT_FLOOR, CURRENT_FACTOR)
    safe_factor = largest_safe_factor(sums, CURRENT_FLOOR)
    rec = recommend(checks, sums, safe_factor)

    (EVAL_DIR / "tolerance_sensitivity.json").write_text(
        json.dumps({
            "variance": var,
            "sweep": grid,
            "masking_at_current_rule": mask_now,
            "largest_factor_with_no_masking": str(safe_factor) if safe_factor else None,
            "recommended_factor": str(rec["factor"]) if rec else None,
            "documents": sorted({c["document"] for c in checks}),
            "document_types": sorted({c["document_type"] for c in checks}),
        }, indent=2),
        encoding="utf-8",
    )
    (EVAL_DIR / "tolerance_sensitivity.md").write_text(
        render(checks, var, grid, sums, mask_now, safe_factor, rec), encoding="utf-8"
    )

    print(f"[tolerance] {var['total_checks']} checks "
          f"({var['exact_matches']} exact, {var['failing']} failing) "
          f"over {len(set(c['document'] for c in checks))} documents")
    print(f"[tolerance] {sum(1 for g in grid if g['flipped_vs_current'] == 0)}"
          f"/{len(grid)} swept settings give identical verdicts")
    print(f"[tolerance] {len(mask_now['vulnerable'])}/{mask_now['checked']} "
          f"component sums could hide their smallest line item")
    if rec:
        print(f"[tolerance] verdict-preserving, masking-free factor: {rec['factor']} "
              f"({rec['tighter_than_current_by']:.0f}x tighter than current)")
    print(f"[tolerance] wrote {EVAL_DIR / 'tolerance_sensitivity.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
