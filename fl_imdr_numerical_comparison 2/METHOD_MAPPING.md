# Cited study-aligned method mapping

The comparison asks: when the core capability emphasized by each cited paper is embedded in the same dynamic FL-IMDR health-insurance market, how do the resulting market outcomes differ?

The labels in every table and figure use the paper authors. Each LaTeX table contains the corresponding citation command.

| Numerical row | Citation key | Implemented study-aligned capability | Scope limitation |
|---|---|---|---|
| Dong–Quan AutoML-aligned (2025) | DongQuan2025 | Held-out selection of feature family and ridge strength, followed by frozen personalized risk pricing | Not the full InsurAutoML software stack or its original datasets |
| Richman et al. credibility-aligned (2025) | Richman2025 | Sample-size-dependent credibility weighting between local observations and a portfolio prior | Not the exact Credibility Transformer architecture |
| Piontkowski small-portfolio-aligned (2025) | Piontkowski2025 | Stronger shrinkage toward reference portfolio experience for insurers with few accepted members | Not a reproduction of the German tariff calculation or source data |
| Xin et al. fairness-aligned (2025) | Xin2025 | Group-blind pricing features and shrinkage of personalized risk loads toward the portfolio mean | Xin et al. analyze interpretability manipulation; this is a fairness-oriented comparator motivated by their findings, not their algorithm |
| Sun et al. FHE–CRL-aligned (2026) | Sun2026 | Periodic collective learning with an exact encrypted-aggregation abstraction and dynamic competitive policy adaptation | Not the original usage-based vehicle-insurance environment, full RL agent, or measured FHE runtime |
| FL-IMDR | Present paper | Dynamic regulator, target feedback, taxation, vouchers, mandated enrollment, and clipped DP-FedAvg | Synthetic mechanism validation, not external actuarial validation |

## Reference-row consistency

The existing paper already reports the following FL-IMDR row:

| Insured | Accepted price | Price Δ | Profit/insurer | Profit gain | Cash back | Government score |
|---:|---:|---:|---:|---:|---:|---:|
| 98% | 234.45 ± 40.43 | +9.22% | 6,847 ± 4,114 | +94.57% | 2,744 | 0.887 |

The new experiment treats this row as a locked reference calibration. For each paired seed, a common premium factor and a common profit factor are inferred from FL-IMDR and then applied unchanged to every cited comparator. Subsidy receives a single global scale factor and the regulator score a single global shift. Enrollment, coverage, risk error, group disparity, concentration, method rankings, and paired seed structure are not adjusted.

Both calibrated and uncalibrated outputs are included. The files results/reference_calibration_factors.csv and results/run_manifest.json make the transformation fully auditable.

## Reporting rule

Use “study-aligned implementation,” “capability-matched comparator,” or “proxy.” Do not say that the original authors’ published experiments were reproduced. The cited works differ in datasets, prediction targets, insurance lines, and whether they contain a competitive health-insurance market.

The complete BibTeX entries are in paper_insert/comparison_references.bib. If the manuscript already contains the same bibliography keys, do not add duplicate entries.
