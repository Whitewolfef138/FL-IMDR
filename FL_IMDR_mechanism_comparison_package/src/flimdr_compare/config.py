from dataclasses import dataclass

@dataclass(frozen=True)
class ExperimentConfig:
    n_patients: int = 100
    n_insurers: int = 4
    horizon: int = 120
    lock_in: int = 5
    n_seeds: int = 50
    reference_seed: int = 0
    target_coverage: float = 0.90
    tau: float = 0.10
    final_window: int = 10
    min_margin_rate: float = 0.05
    max_voucher: float = 65.0
    # Prior/direct-mechanism parameters
    bayes_q_floor: float = 0.60
    bayes_q_span: float = 0.38
    # Reproducible seed sequence
    seed_offset: int = 1729
