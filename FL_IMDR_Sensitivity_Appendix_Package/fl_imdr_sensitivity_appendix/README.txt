FL-IMDR sensitivity appendix package

Experiment:
- 100 patients, 4 insurers, 120 monthly rounds
- 30 common random seeds per setting
- one-at-a-time sweeps of tau, coverage_target, fl_interval, dp_sigma,
  imitation_weight, and gap_beta
- final metrics averaged over the last 10 rounds
- paired Wilcoxon comparisons against each parameter's baseline with Holm correction

The notebook is self-contained and reproduces the simulation, CSV files,
figures, LaTeX tables, and appendix text. It uses a compact vectorized/Numba
implementation of the manuscript's patient dynamics, decentralized offer
selection, tax-funded vouchers, mandated target response, insurer risk
learning, imitation, and clipped noisy FedAvg. It is a controlled appendix
experiment and should be described as empirical sensitivity within the
simulated market, not as external validation.
