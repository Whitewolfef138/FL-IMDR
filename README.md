# FL-IMDR

### Federated Learning for Insurance Market Modeling with a Dynamic Regulator

[![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python\&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Research%20Code-EE4C2C?logo=pytorch\&logoColor=white)](https://pytorch.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Reproducible%20Notebooks-F37626?logo=jupyter\&logoColor=white)](https://jupyter.org/)

**FL-IMDR** is a research framework for studying a dynamic, data-driven insurance market with interacting **patients, insurers, and a regulator**. The framework combines multi-agent market simulation, insurer-side learning, targeted regulatory intervention, federated aggregation, and optional differential-privacy noise.

Rather than treating insurance pricing as a static prediction problem, FL-IMDR studies the **closed-loop market dynamics** that emerge when learned pricing and coverage decisions affect participation, insurer behavior, fairness, profitability, and future learning.

> **Research use:** this repository contains a synthetic insurance-market simulation and supporting reproducibility experiments. It is not an actuarial pricing system and should not be used for real-world insurance, financial, or clinical decision-making without independent validation.

---

## Key ideas

FL-IMDR models three interacting decision layers:

1. **Patients** evolve over time, evaluate insurer offers, and make coverage decisions under affordability and coverage preferences.
2. **Insurers** learn patient risk and generate personalized price/coverage offers using neural models while adapting to market competition.
3. **The regulator (Governor)** evaluates market outcomes, allocates targeted vouchers/subsidies, and coordinates federated learning of insurer risk models without centralizing insurer data.

The implementation supports:

* dynamic patient attributes and risk;
* heterogeneous insurer behavior;
* learned risk, price, and coverage models;
* market entry, switching, and lock-in dynamics;
* targeted regulatory subsidies/vouchers;
* federated averaging across insurers;
* clipping and Gaussian noise for DP-style federated aggregation;
* coverage, affordability, fairness, profitability, concentration, and prediction metrics;
* sensitivity, privacy-impact, and numerical-comparison experiments.

---

## Repository structure

| Path                                    | Purpose                                                                                                                               |
| --------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| `Update-2.ipynb`                        | Main end-to-end FL-IMDR simulation notebook and primary starting point.                                                               |
| `patient_model.py`                      | Dynamic patient model, observable state history, insurance choice, and evolving risk.                                                 |
| `insurance_model.py`                    | Insurer learning models for risk, coverage, and price, including heterogeneous insurer behavior.                                      |
| `governor_model.py`                     | Dynamic regulator, voucher allocation, regulator-objective learning, FedAvg helpers, clipping, and Gaussian noise.                    |
| `market_model.py`                       | Core insurance-market interaction loop and FL/regulatory coordination.                                                                |
| `market_viz.py`                         | Plotting and diagnostics for market outcomes, losses, calibration, subsidies, and FL behavior.                                        |
| `FL_IMDR_Sensitivity_Appendix_Package/` | Sensitivity-analysis notebook, executed notebook, accelerated Numba experiment, and appendix outputs.                                 |
| `FL_IMDR_DP_Impact_Appendix_Package/`   | Differential-privacy/FedAvg impact notebook, executed notebook, figures, tables, and supporting appendix material.                    |
| `fl_imdr_numerical_comparison 2/`       | Reproducible study-aligned numerical comparison package, statistical tests, figures, manuscript tables, and its own environment file. |

---

## Quick start

### 1. Clone the repository

```bash
git clone https://github.com/Whitewolfef138/FL-IMDR.git
cd FL-IMDR
```

### 2. Create an isolated Python environment

A recent Python 3 environment is recommended.

**macOS / Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows PowerShell**

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install the core dependencies

```bash
python -m pip install --upgrade pip
python -m pip install numpy pandas matplotlib seaborn scipy numba jupyter torch
```

For CUDA/GPU-specific PyTorch builds, use the installation command recommended for your platform by PyTorch.

### 4. Run the main experiment

```bash
jupyter lab Update-2.ipynb
```

Then run the notebook from top to bottom.

`Update-2.ipynb` is the primary end-to-end reference for the complete FL-IMDR experiment configuration and simulation workflow.

---

## Using the modular implementation

The major components can also be imported independently:

```python
from patient_model import Patient
from insurance_model import Insurance
from governor_model import Governor
from market_model import InsuranceMarket
```

The main notebook demonstrates how these components are instantiated and connected.

At the market level, `InsuranceMarket` coordinates:

```text
Patient state evolution
        ↓
Insurer risk prediction
        ↓
Price / coverage offers
        ↓
Patient choice
        ↓
Market outcomes
        ↓
Dynamic regulatory intervention
        ↓
Federated model aggregation
        ↓
Next market round
```

Important market parameters include:

| Parameter         | Meaning                                                                    |
| ----------------- | -------------------------------------------------------------------------- |
| `history_len`     | Number of recent observations supplied to temporal patient/insurer models. |
| `lock_in_rounds`  | Initial rounds during which seeded insurance assignments remain locked.    |
| `coverage_target` | Regulatory target for insured-population coverage.                         |
| `cash_back_rate`  | Regulator subsidy/cash-back control parameter.                             |
| `fl_every_rounds` | Number of market rounds between federated aggregations.                    |
| `clip_C`          | Client-update clipping bound before aggregation.                           |
| `dp_sigma`        | Gaussian-noise scale used in DP-style server aggregation.                  |
| `switch_prob`     | Probability that an insured patient considers switching after lock-in.     |

The experiment notebooks may override these defaults depending on the study being reproduced.

---

# Reproducing the experiments

The repository contains several reproducibility paths. They address different parts of the manuscript and should not be assumed to use identical experimental settings.

## 1. Main FL-IMDR simulation

Run:

```bash
jupyter lab Update-2.ipynb
```

This notebook is the best starting point for understanding the complete interaction among:

* patients;
* insurers;
* the market;
* the regulator;
* federated aggregation;
* differential-privacy noise;
* market diagnostics.

---

## 2. Numerical comparison

The study-aligned numerical-comparison package has its own dependency file and reproducibility script.

```bash
cd "fl_imdr_numerical_comparison 2"
python -m pip install -r requirements.txt
bash reproduce.sh
```

The default reproduction script runs:

```bash
python run_comparison.py \
  --seeds 50 \
  --start-seed 101 \
  --months 120 \
  --patients 100 \
  --insurers 4 \
  --output .
```

For a faster smoke test:

```bash
python run_comparison.py \
  --seeds 3 \
  --months 20 \
  --output smoke_output
```

The package produces, among other outputs:

* seed-level trajectories;
* final-window run metrics;
* mean and standard-deviation summaries;
* paired statistical tests;
* Holm-adjusted significance results;
* time-series confidence intervals;
* publication figures;
* manuscript-ready LaTeX tables;
* run manifests and calibration information.

See:

```text
fl_imdr_numerical_comparison 2/README.md
```

for the full methodology, output list, assumptions, and interpretation notes.

---

## 3. Sensitivity analysis

Sensitivity-analysis material is located in:

```text
FL_IMDR_Sensitivity_Appendix_Package/
```

Recommended entry point:

```text
FL_IMDR_Sensitivity_Appendix.ipynb
```

An executed version is also provided:

```text
FL_IMDR_Sensitivity_Appendix_executed.ipynb
```

The package additionally contains:

```text
sensitivity_experiment_numba.py
```

which implements an accelerated batch sensitivity study.

The current study varies quantities including:

* tax/cash-back rate;
* coverage target;
* federation interval;
* DP noise scale;
* imitation weight;
* coverage-gap gain.

The analysis tracks outcomes including:

* insured percentage;
* effective accepted premium;
* accepted coverage;
* insurer profit;
* subsidy expenditure;
* risk-prediction MSE;
* market concentration / HHI;
* coverage disparity;
* regulator goal score;
* insurer prediction disparity.

---

## 4. Differential privacy and FedAvg impact

The privacy-impact experiments are located in:

```text
FL_IMDR_DP_Impact_Appendix_Package/
```

Recommended entry point:

```text
FL_IMDR_DP_Impact_Appendix.ipynb
```

An executed notebook is also available:

```text
FL_IMDR_DP_Impact_Appendix_executed.ipynb
```

The package contains generated:

* convergence figures;
* prediction/accuracy results;
* fairness analyses;
* per-seed summaries;
* paired statistical tests;
* manuscript-ready appendix material.

These experiments are intended to study how clipping and Gaussian perturbation during federated aggregation influence learning and downstream market outcomes.

---

# Reproducibility guidelines

FL-IMDR is stochastic. When comparing mechanisms or reproducing manuscript results:

1. Keep the same patient and insurer populations.
2. Keep the same number of market rounds.
3. Fix random seeds.
4. Prefer matched seeds/common random numbers across competing methods.
5. Record the federation interval.
6. Record the clipping threshold.
7. Record the DP noise scale.
8. Record the regulatory target and subsidy parameters.
9. Specify which experiment package generated each reported result.

Because the repository contains multiple experimental packages developed for different manuscript analyses, **do not assume every subdirectory uses the same defaults**.

For manuscript verification, prefer the provided:

* executed notebooks;
* generated CSV files;
* run manifests;
* statistical-test outputs;
* manuscript-ready tables.

---

# Core model components

## Patients

`patient_model.py` represents patients with dynamic observable characteristics and evolving insurance risk.

Patient variables include quantities such as:

```text
age
income
credit score
health status
location
desired coverage
insurance budget
actual risk
insurance status
```

Recent observations are retained so insurer models can operate on temporal patient information rather than only the instantaneous state.

---

## Insurers

`insurance_model.py` contains insurer-side neural models for:

```text
patient history → risk estimate
patient history + risk → coverage offer
patient history + risk + coverage → price offer
```

The model additionally supports heterogeneous insurer behavior through parameters controlling pricing margins, coverage margins, risk attitudes, exploration, and imitation.

---

## Dynamic regulator

`governor_model.py` represents the regulatory layer.

Its roles include:

* learning a regulator/goal score;
* evaluating patient-offer outcomes;
* allocating vouchers to improve market participation;
* coordinating insurer risk-model aggregation;
* clipping insurer updates;
* optionally perturbing aggregated weights with Gaussian noise.

The Governor therefore acts both as a **market regulator** and as the **federated-learning coordinator**.

---

## Insurance market

`market_model.py` coordinates the complete market interaction.

Each market round includes combinations of:

1. insurer offer generation;
2. patient evaluation and choice;
3. optional patient switching;
4. insurer learning;
5. regulatory intervention;
6. subsidy allocation;
7. market-statistic collection;
8. periodic federated aggregation.

This closed-loop interaction is central to FL-IMDR: learned decisions affect the market state, which subsequently changes the data available for future learning.

---

# Visualization and diagnostics

`market_viz.py` provides utilities for inspecting outcomes including:

* insurer market share over time;
* insurer profit;
* loss ratios;
* insured-population trajectory;
* nominal accepted premiums;
* effective accepted premiums after intervention;
* accepted coverage;
* model-loss trajectories;
* risk calibration;
* subsidy efficiency;
* federated-aggregation events;
* regulator alignment;
* subsidy dynamics.

These plots are intended both for debugging and for inspecting the economic behavior of the learned market.

---

# Research questions supported by FL-IMDR

The framework can be used to investigate questions such as:

* Can dynamic regulation prevent learned pricing from producing exclusionary market outcomes?
* Can insurers improve shared risk learning without directly pooling proprietary patient data?
* How does federation frequency affect learning and market performance?
* How does privacy noise affect prediction, fairness, and regulatory performance?
* How do regulatory subsidies affect insurer incentives?
* How does market competition interact with personalized learned pricing?
* When does insurer adaptation produce cherry-picking or market segmentation?
* What trade-offs emerge among privacy, predictive quality, profitability, affordability, and coverage?
* How sensitive are conclusions to regulator and market hyperparameters?

---

# Extending the framework

The implementation is intentionally modular.

Possible extensions include replacing or modifying:

```text
Patient dynamics
       │
       ├── demographic / behavioral models
       ├── alternative risk processes
       └── empirical patient data

Insurer learning
       │
       ├── alternative neural architectures
       ├── Bayesian models
       ├── reinforcement learning
       └── strategic/game-theoretic policies

Federated learning
       │
       ├── FedAvg variants
       ├── personalization
       ├── robust aggregation
       └── alternative privacy mechanisms

Regulation
       │
       ├── alternative subsidy rules
       ├── constrained optimization
       ├── welfare objectives
       ├── fairness constraints
       └── mechanism-design policies
```

Researchers are encouraged to preserve matched-seed evaluation when comparing extensions against the baseline framework.

---

# Troubleshooting

### `ModuleNotFoundError`

Verify that your environment is active and install the research dependencies:

```bash
python -m pip install numpy pandas matplotlib seaborn scipy numba jupyter torch
```

### PyTorch / CUDA problems

The core experiments can run on CPU.

If you require GPU execution, install the appropriate PyTorch build for your operating system and CUDA version.

### Imports cannot find project modules

Start Jupyter from the repository root:

```bash
cd FL-IMDR
jupyter lab Update-2.ipynb
```

rather than launching the notebook from an unrelated working directory.

### Results differ between runs

The simulator contains stochastic patient dynamics, insurer exploration, initialization, and learning.

Check:

```text
random seeds
number of patients
number of insurers
simulation horizon
FL interval
DP sigma
clipping threshold
regulatory parameters
```

before comparing results.

---

# Citation

If you use **FL-IMDR**, its source code, or its experimental design in academic work, please cite the accompanying **FL-IMDR manuscript** and link to this repository.

Formal publication metadata and a BibTeX entry will be added here when the final bibliographic information becomes available.

---

# Contributing and questions

Reproducibility fixes, documentation improvements, and research extensions are welcome.

For problems reproducing an experiment, please open a GitHub issue and include:

* the notebook or experiment you ran;
* your Python version;
* relevant package versions;
* the random seed/configuration;
* the full traceback or unexpected output;
* your operating system;
* whether PyTorch is running on CPU or GPU.

---

**Repository:** `Whitewolfef138/FL-IMDR`
