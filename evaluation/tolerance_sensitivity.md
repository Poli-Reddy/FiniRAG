# FiniRAG — Tolerance Sensitivity Analysis

The shipped rule is

```python
tolerance = max(Decimal("0.01"), abs(reported) * Decimal("0.001"))
```

a 0.1% relative band with a one-paisa floor. The source marks it
PROPOSED, subject to empirical validation. This is that validation,
computed over the **360 arithmetic checks** the pipeline
actually produced on the real dataset (`evaluation/results/`).

## A. What the variances actually look like

| | Count |
|---|---|
| Checks comparing two numbers | 360 |
| Variance exactly zero | 343 |
| Variance non-zero | 17 |
| Status PASS | 350 |
| Status FAIL | 10 |

Non-zero variances, largest first:

| Document | Check | Variance | Reported | Relative | Tolerance | Verdict |
|---|---|---|---|---|---|---|
| batch3-1495.jpg | `sum_lines_vs_total` | 1777 | 5257 | 33.802549% | 5.257 | FAIL |
| batch2-0999.jpg | `line_total_check_3` | 680.00 | 136.00 | 500.000000% | 0.13600 | FAIL |
| batch2-0999.jpg | `line_total_check_2` | 678.00 | 226.00 | 300.000000% | 0.22600 | FAIL |
| batch2-0999.jpg | `line_total_check_4` | 672.00 | 96.00 | 700.000000% | 0.09600 | FAIL |
| batch2-0999.jpg | `line_total_check_1` | 346.00 | 346.00 | 100.000000% | 0.34600 | FAIL |
| batch2-0999.jpg | `subtotal_plus_tax_vs_total` | 49.00 | 916.47 | 5.346602% | 0.91647 | FAIL |
| batch2-0499.jpg | `sum_lines_vs_subtotal` | 10.00 | 135.00 | 7.407407% | 0.13500 | FAIL |
| batch2-0499.jpg | `subtotal_plus_tax_vs_total` | 10.00 | 157.48 | 6.350013% | 0.15748 | FAIL |
| batch2-0499.jpg | `rate_times_base_vs_tax` | 1.6800 | 12.48 | 13.461538% | 0.01248 | FAIL |
| batch2-0999.jpg | `rate_times_base_vs_tax` | 0.85 | 63.47 | 1.339215% | 0.06347 | FAIL |
| Consolidated Cash Flow Statement 2022.pdf | `2022_sum_operating_items` | 0.01 | -11959.57 | 0.000084% | 11.95957 | PASS |
| Consolidated Cash Flow Statement 2023.pdf | `2022_sum_operating_items` | 0.01 | -11959.57 | 0.000084% | 11.95957 | PASS |
| X51008099071.jpg | `line_total_check_1` | 0.00600 | 50.00 | 0.012000% | 0.05000 | PASS |
| X51005663293.jpg | `rate_times_base_vs_tax` | 0.0052 | 1.72 | 0.302326% | 0.01 | PASS |
| 20251118_000612.jpg | `line_total_check_5` | 0.0048 | 0.60 | 0.800000% | 0.01 | PASS |
| 20251118_000612.jpg | `line_total_check_2` | 0.0036 | 0.45 | 0.800000% | 0.01 | PASS |
| batch1-1109.jpg | `rate_times_base_vs_tax` | 0.003 | 12.63 | 0.023753% | 0.01263 | PASS |

- Largest variance among **passing** checks: `0.01`
- Smallest variance among **failing** checks: `0.85`

## B. Do the verdicts move when the constants move?

Each cell re-runs every check under a different `(floor, factor)` and
counts how many verdicts differ from the shipped rule. `factor = 0`
switches off the relative band; `floor = 0` switches off the absolute one.

| floor \ factor | 0 | 0.000001 | 0.00001 | 0.0001 | 0.001 | 0.01 | 0.1 |
|---|---|---|---|---|---|---|---|
| **0** | 343P/17F (7 flip) | 345P/15F (5 flip) | 345P/15F (5 flip) | 345P/15F (5 flip) | 347P/13F (3 flip) | 350P/10F | 354P/6F (4 flip) |
| **0.01** | 350P/10F | 350P/10F | 350P/10F | 350P/10F | 350P/10F**current** | 350P/10F | 354P/6F (4 flip) |
| **1** | 351P/9F (1 flip) | 351P/9F (1 flip) | 351P/9F (1 flip) | 351P/9F (1 flip) | 351P/9F (1 flip) | 351P/9F (1 flip) | 354P/6F (4 flip) |
| **100** | 355P/5F (5 flip) | 355P/5F (5 flip) | 355P/5F (5 flip) | 355P/5F (5 flip) | 355P/5F (5 flip) | 355P/5F (5 flip) | 355P/5F (5 flip) |
| **10000** | 360P/0F (10 flip) | 360P/0F (10 flip) | 360P/0F (10 flip) | 360P/0F (10 flip) | 360P/0F (10 flip) | 360P/0F (10 flip) | 360P/0F (10 flip) |

7 of the 35 settings produce **identical verdicts** to
the shipped rule.

Settings that change at least one verdict:

| floor | factor | flips | example |
|---|---|---|---|
| 10000 | 0 | 10 | batch2-0499.jpg :: sum_lines_vs_subtotal |
| 10000 | 0.000001 | 10 | batch2-0499.jpg :: sum_lines_vs_subtotal |
| 10000 | 0.00001 | 10 | batch2-0499.jpg :: sum_lines_vs_subtotal |
| 10000 | 0.0001 | 10 | batch2-0499.jpg :: sum_lines_vs_subtotal |
| 10000 | 0.001 | 10 | batch2-0499.jpg :: sum_lines_vs_subtotal |
| 10000 | 0.01 | 10 | batch2-0499.jpg :: sum_lines_vs_subtotal |
| 10000 | 0.1 | 10 | batch2-0499.jpg :: sum_lines_vs_subtotal |
| 0 | 0 | 7 | 20251118_000612.jpg :: line_total_check_2 |
| 0 | 0.000001 | 5 | 20251118_000612.jpg :: line_total_check_2 |
| 0 | 0.00001 | 5 | 20251118_000612.jpg :: line_total_check_2 |
| 0 | 0.0001 | 5 | 20251118_000612.jpg :: line_total_check_2 |
| 100 | 0 | 5 | batch2-0499.jpg :: sum_lines_vs_subtotal |

## C. Could the tolerance hide a real error?

Verdict stability alone is not enough. A tolerance of ±50% would also
be perfectly stable on a dataset that reconciles exactly — and useless.
The sharper question is whether the band is wide enough to swallow a
line the document actually printed.

For every check that sums document line items into a reported total,
compare its tolerance against the **smallest item in that sum**. If the
tolerance is the larger of the two, dropping that item entirely would
still be reported as PASS.

- Component-summed checks examined: **140**
- Checks whose tolerance is >= their smallest line item: **43**

**43 of 140 sums are in that
position.** Each row below names a line the document actually prints
whose complete disappearance the current tolerance would not catch:

| Document | Check | Smallest item | Its label | Tolerance |
|---|---|---|---|---|
| Consolidated Balance Sheet 2017.pdf | `2017_sum_liabilities` | 2914389 | Minority interest | 8923441.607 |
| Consolidated Balance Sheet 2017.pdf | `2016_sum_liabilities` | 1806228 | Minority interest | 7622123.264 |
| Consolidated Balance Sheet 2018.pdf | `2018_sum_liabilities` | 3563322 | Minority interest | 11031861.695 |
| Consolidated Balance Sheet 2018.pdf | `2017_sum_liabilities` | 2914389 | Minority interest | 8923441.607 |
| Consolidated Balance Sheet 2019.pdf | `2019_sum_liabilities` | 5017945 | Minority interest | 12928057.065 |
| Consolidated Balance Sheet 2019.pdf | `2018_sum_liabilities` | 3563322 | Minority interest | 11031861.695 |
| Consolidated Balance Sheet 2020.pdf | `2020_sum_liabilities` | 5483286 | Capital | 15808304.373 |
| Consolidated Balance Sheet 2020.pdf | `2019_sum_liabilities` | 5017945 | Minority interest | 12928057.065 |
| Consolidated Balance Sheet 2021.pdf | `2021_sum_liabilities` | 5512776 | Capital | 17995066.442 |
| Consolidated Balance Sheet 2021.pdf | `2020_sum_liabilities` | 5483286 | Capital | 15808304.373 |
| Consolidated Balance Sheet 2022.pdf | `2022_sum_assets` | 148.79 | Goodwill on Consolidation | 2122.93430 |
| Consolidated Balance Sheet 2022.pdf | `2022_sum_liabilities` | 554.55 | Capital | 2122.93430 |
| Consolidated Balance Sheet 2022.pdf | `2021_sum_assets` | 148.79 | Goodwill on Consolidation | 1799.50663 |
| Consolidated Balance Sheet 2022.pdf | `2021_sum_liabilities` | 551.28 | Capital | 1799.50663 |
| Consolidated Balance Sheet 2023.pdf | `2023_sum_assets` | 148.79 | Goodwill on Consolidation | 2530.43244 |
| Consolidated Balance Sheet 2023.pdf | `2023_sum_liabilities` | 557.97 | Capital | 2530.43244 |
| Consolidated Balance Sheet 2023.pdf | `2022_sum_assets` | 148.79 | Goodwill on Consolidation | 2122.93430 |
| Consolidated Balance Sheet 2023.pdf | `2022_sum_liabilities` | 554.55 | Capital | 2122.93430 |
| Consolidated Balance Sheet 2024.pdf | `2024_sum_liabilities` | 759.69 | Capital | 4030.19426 |
| Consolidated Balance Sheet 2024.pdf | `2023_sum_assets` | 148.79 | Goodwill on Consolidation | 2530.43244 |

(23 further rows omitted.)

The binding constraint across the dataset is a relative factor of
**3.441e-6** — the point at which the tightest sum's smallest
component equals its tolerance. Above that, at least one omitted line
item becomes invisible.

The shipped factor of 0.001 is **291x wider** than that bound.

## D. Conclusion

1. **The rule is not load-bearing on this dataset.** 343/360 checks reconcile exactly, and 7/35 of the swept settings give identical verdicts. Tolerance is not what produces the pass rate; correct extraction is. That also means section B on its own cannot justify the constants — any value in a very wide range would look equally good.

2. **Both halves of the rule do distinct work.** The relative term is
   what keeps a multi-billion total from being judged by the same
   absolute band as a two-figure receipt line. The floor is what stops
   a legitimately zero total (`reported == 0`, common on nil rows) from
   demanding exactness on a value that arrived as a decimal string.
   Neither can be dropped: the `factor = 0` column and the `floor = 0`
   row above are each a different rule, not a simpler one.

3. **The band is looser than it looks — this is the finding.** 43 of 140 component sums (31%) carry a tolerance at least as large as their own smallest printed line item. For those sums, an extraction that dropped that line entirely would still be reported PASS. Verdict stability (section B) hid this completely, because no document in the dataset *has* a dropped line — the exposure is latent.

4. **The masking-free bound is a factor of 3.44e-6**, about 291x tighter than the 0.001 in use. 
   A factor of **0.000001** clears both bars at once: it leaves all 360 verdicts unchanged and sits 3.4x inside the masking bound, 1000x tighter than today.

5. **Recommendation — report, do not silently change.** `master_prompt.md` puts tolerance constants under CHANGE CONTROL, and this is a material change: it would tighten the acceptance criterion for every document the system will ever see. The evidence for tightening is above; the decision is not this script's to make. The shipped rule stays as it is until that decision is taken.

   Two caveats bound the finding. It is computed on the documents in
   `evaluation/results/` only — a dataset of genuinely rounded
   statements would shift the picture, because real rounding variance
   is what the relative band exists to absorb. And masking a line
   item requires the extraction to lose it silently; the schema's
   `MISSING` status is designed to surface exactly that, so the
   tolerance is the second line of defence, not the first.

## Scope of this evidence

Computed over **50 document(s)** of type `balance_sheet`, `cash_flow_statement`, `invoice`, `profit_and_loss`, producing 360 comparable checks. Document types absent from
this list are not covered by any claim above — notably, invoice rounding
is where a relative band is most likely to be exercised, and no invoice
contributed a check here. Re-run after extending the dataset.

Documents included:

- 20251118_000612.jpg
- Consolidated Balance Sheet 2017.pdf
- Consolidated Balance Sheet 2018.pdf
- Consolidated Balance Sheet 2019.pdf
- Consolidated Balance Sheet 2020.pdf
- Consolidated Balance Sheet 2021.pdf
- Consolidated Balance Sheet 2022.pdf
- Consolidated Balance Sheet 2023.pdf
- Consolidated Balance Sheet 2024.pdf
- Consolidated Balance Sheet 2025.pdf
- Consolidated Balance Sheet 2026.pdf
- Consolidated Cash Flow Statement 2017.pdf
- Consolidated Cash Flow Statement 2018.pdf
- Consolidated Cash Flow Statement 2019.pdf
- Consolidated Cash Flow Statement 2020.pdf
- Consolidated Cash Flow Statement 2021.pdf
- Consolidated Cash Flow Statement 2022.pdf
- Consolidated Cash Flow Statement 2023.pdf
- Consolidated Cash Flow Statement 2024.pdf
- Consolidated Cash Flow Statement 2025.pdf
- Consolidated Cash Flow Statement 2026.pdf
- Consolidated Profit & Loss 2017.pdf
- Consolidated Profit & Loss 2018.pdf
- Consolidated Profit & Loss 2019.pdf
- Consolidated Profit & Loss 2020.pdf
- Consolidated Profit & Loss 2021.pdf
- Consolidated Profit & Loss 2022.pdf
- Consolidated Profit & Loss 2023.pdf
- Consolidated Profit & Loss 2024.pdf
- Consolidated Profit & Loss 2025.pdf
- Consolidated Profit & Loss 2026.pdf
- X00016469619.jpg
- X51005361895.jpg
- X51005663293.jpg
- X51005719883.jpg
- X51005757342.jpg
- X51005806685.jpg
- X51006328913.jpg
- X51006334741.jpg
- X51006387931.jpg
- X51008099043.jpg
- X51008099071.jpg
- X51008142038.jpg
- X51008145450.jpg
- batch1-1109.jpg
- batch2-0499.jpg
- batch2-0999.jpg
- batch3-1354.jpg
- batch3-1445.jpg
- batch3-1495.jpg

## Reproduce

```bash
./venv/Scripts/python.exe evaluation/tolerance_sensitivity.py
```

Reads only `evaluation/results/*.json`. No API calls, no database.

