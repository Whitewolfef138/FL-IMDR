# Numerical mechanism-comparison summary

Default experiment: 50 paired seeds, 100 patients, 4 insurers, 120 rounds, final 10 rounds summarized.

| Mechanism | Insured | Effective premium | Coverage | Profit / insurer / round | Subsidy / round | HHI | Override residual |
|---|---:|---:|---:|---:|---:|---:|---:|
| Central welfare planner | 99.74% ± 0.43 | $220.47 ± 1.53 | 0.785 ± 0.002 | $1261.2 ± 6.0 | $166.0 ± 47.4 | 0.391 ± 0.022 | 0.327 ± 0.029 |
| Bayesian direct mechanism | 71.09% ± 2.27 | $214.75 ± 1.56 | 0.793 ± 0.002 | $1026.8 ± 32.7 | $0.0 ± 0.0 | 0.742 ± 0.051 | 0.347 ± 0.030 |
| FL-IMDR restricted | 98.22% ± 1.17 | $219.37 ± 1.41 | 0.757 ± 0.004 | $1662.7 ± 20.7 | $645.4 ± 62.3 | 0.522 ± 0.034 | 0.000 ± 0.000 |

## Interpretation

The central planner reaches the highest coverage because it is allowed to choose market
variables directly. FL-IMDR reaches a similarly high insured fraction in this calibrated
counterfactual while leaving insurer offers and patient assignments as endogenous best
responses. The Bayesian direct comparator has lower participation here because its
allocation probability is report/prior based rather than a universal enrollment target.

The key result for the manuscript is therefore the override residual: FL-IMDR is zero by
construction after the lock-in interval, while both broader mechanisms directly replace or
induce decisions that depart from the decentralized response system.

These numerical comparators are controlled counterfactual implementations of the three
formulations; they are not empirical estimates and the Bayesian comparator is not claimed
to be the globally optimal multidimensional Bayesian mechanism.
