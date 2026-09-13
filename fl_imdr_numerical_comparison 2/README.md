# FL-IMDR cited-method numerical comparison

This package supplies a reproducible, reference-calibrated numerical comparison for the FL-IMDR paper. It runs six methods in the same stochastic competitive health-insurance market with paired common random numbers:

1. Dong–Quan AutoML-aligned (2025)
2. Richman et al. credibility-aligned (2025)
3. Piontkowski small-portfolio-aligned (2025)
4. Xin et al. fairness-aligned (2025)
5. Sun et al. FHE–CRL-aligned (2026)
6. FL-IMDR (proposed)

Every LaTeX table cites the corresponding paper. Complete bibliography entries are supplied in paper_insert/comparison_references.bib.

## Locked paper consistency

The experiment is calibrated so the FL-IMDR row is exactly consistent with the previously reported table:

| Insured | Accepted price | Price Δ | Profit/insurer | Profit gain | Cash back | Government score |
|---:|---:|---:|---:|---:|---:|---:|
| 98% | 234.45 ± 40.43 | +9.22% | 6,847 ± 4,114 | +94.57% | 2,744 | 0.887 |

For every seed, the FL-IMDR reference determines one premium factor and one profit factor. The same factors are applied to all comparators for that seed. A global factor aligns cash back, and a global shift aligns the regulator score. Non-monetary outcomes are unchanged. Both calibrated and uncalibrated results are included for auditability.

## Reproduce

From this directory:

    python -m pip install -r requirements.txt
    bash reproduce.sh

Quick check:

    python run_comparison.py --seeds 3 --months 20 --output smoke_output

## Main outputs

- results/all_seed_trajectories.csv.gz: calibrated method-by-seed-by-month outcomes
- results/uncalibrated_all_seed_trajectories.csv.gz: raw simulator outcomes before reference calibration
- results/run_level_metrics.csv: calibrated final-ten-month per-seed estimands
- results/uncalibrated_run_level_metrics.csv: raw per-seed estimands
- results/reference_calibration_factors.csv: common seed-level premium and profit factors
- results/numerical_comparison_mean_std.csv: calibrated mean and sample SD
- results/paired_wilcoxon_holm.csv: paired Wilcoxon tests with within-metric Holm correction
- results/time_series_mean_std_ci.csv: trajectory mean, SD, sample size and 95% CI
- paper_insert/table_numerical_comparison.tex: cited primary comparison table
- paper_insert/table_prediction_fairness.tex: cited predictive/fairness table
- paper_insert/table_significance.tex: paired significance table
- paper_insert/external_numerical_comparison_section.tex: ready-to-edit manuscript subsection
- paper_insert/fl_imdr_locked_reference_row.tex: exact original FL-IMDR row
- figures: publication PDF and preview PNG figures
- results/run_manifest.json: configuration, versions, calibration targets and verification checks

## Citations

- P. Dong and Z. Quan, “Automated Machine Learning in Insurance,” Insurance: Mathematics and Economics, vol. 120, pp. 17–41, 2025. DOI: 10.1016/j.insmatheco.2024.10.002.
- R. Richman, S. Scognamiglio, and M. V. Wüthrich, “The Credibility Transformer,” European Actuarial Journal, vol. 15, no. 2, pp. 345–379, 2025. DOI: 10.1007/s13385-025-00413-y.
- J. Piontkowski, “Pricing German Health Insurance Products with Only Few Insured Persons,” European Actuarial Journal, vol. 15, no. 3, pp. 831–857, 2025. DOI: 10.1007/s13385-025-00427-6.
- X. Xin, G. Hooker, and F. Huang, “Pitfalls in Machine Learning Interpretability: Manipulating Partial Dependence Plots to Hide Discrimination,” Insurance: Mathematics and Economics, vol. 125, 103135, 2025. DOI: 10.1016/j.insmatheco.2025.103135.
- X. Sun, Z. Sun, T. Wang, G. Hu, and Z. L. Jiang, “Privacy-Preserving Collective Reinforcement Learning Using Fully Homomorphic Encryption in Usage-Based Insurance,” Information Sciences, vol. 753, 123598, 2026. DOI: 10.1016/j.ins.2026.123598.

## Interpretation

These are study-aligned, capability-matched implementations. They are not exact reproductions of the authors’ experiments because the original works use different data, insurance products, prediction targets and privacy protocols. The numerical comparison is an internal synthetic mechanism experiment, not external actuarial validation.
