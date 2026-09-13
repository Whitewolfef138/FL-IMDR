# Reference-calibrated 50-seed results

Configuration: 100 patients, four insurers, 120 monthly rounds, five lock-in months, federation every ten months, and 50 paired seeds (101–150). Each entry is the mean ± sample standard deviation of the per-seed average over months 111–120.

| Cited study-aligned method | Insured (%) | Effective premium (USD) | Profit/insurer (USD) | Accepted coverage (%) | Risk MSE | Risk-group gap (pp) | HHI |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dong–Quan AutoML-aligned (2025) | 67.10 ± 4.76 | 236.86 ± 39.76 | 4,998 ± 3,084 | 72.12 ± 1.87 | 0.001065 ± 0.000428 | 60.59 ± 9.47 | 0.639 ± 0.186 |
| Richman et al. credibility-aligned (2025) | 66.34 ± 4.68 | 248.53 ± 42.65 | 5,385 ± 3,391 | 73.85 ± 0.55 | 0.000156 ± 0.000009 | 59.88 ± 9.19 | 0.726 ± 0.049 |
| Piontkowski small-portfolio-aligned (2025) | 66.95 ± 4.63 | 247.64 ± 42.47 | 5,347 ± 3,354 | 74.98 ± 0.56 | 0.000156 ± 0.000009 | 59.07 ± 9.17 | 0.726 ± 0.049 |
| Xin et al. fairness-aligned (2025) | 73.09 ± 4.83 | 238.46 ± 41.47 | 5,367 ± 3,205 | 69.58 ± 1.01 | 0.012046 ± 0.002256 | 33.89 ± 9.88 | 0.667 ± 0.159 |
| Sun et al. FHE–CRL-aligned (2026) | 68.35 ± 4.58 | 244.77 ± 42.05 | 5,274 ± 3,296 | 76.40 ± 0.55 | 0.000156 ± 0.000009 | 57.55 ± 9.38 | 0.731 ± 0.048 |
| **FL-IMDR (proposed)** | **98%** | **234.45 ± 40.43** | **6,847 ± 4,114** | 72.75 ± 0.76 | 0.000231 ± 0.000127 | **7.94 ± 0.30** | **0.458 ± 0.033** |

The remaining locked FL-IMDR values are a price change of +9.22%, profit gain of +94.57%, total cash back of 2,744, and government score of 0.887.

FL-IMDR improves steady-state enrollment by 24.91–31.66 percentage points, lowers effective premium by USD 2.41–14.08, raises profit per insurer by USD 1,461.85–1,848.63, and reduces the risk-group coverage gap by 25.94–52.65 percentage points relative to the five cited comparators. All corresponding paired differences remain significant after within-metric Holm correction; the largest adjusted p-value is 0.013.

The comparison does not show uniform predictive dominance. The Richman and Piontkowski study-aligned methods have the lowest risk MSE, while the Sun study-aligned method has the highest accepted coverage. FL-IMDR’s advantage is market-level: target attainment, affordability, insurer profitability, reduced risk-group exclusion and lower concentration.

The exact 98% steady-state enrollment is structural because mandated enrollment enforces the target in every final-window replication.
