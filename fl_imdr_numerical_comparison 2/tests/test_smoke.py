import numpy as np

from fl_imdr_benchmark.config import METHODS, SimulationConfig
from fl_imdr_benchmark.calibration import calibrate_to_paper_reference
from fl_imdr_benchmark.simulation import METRICS, run_experiment


def test_smoke_run_is_complete_and_finite():
    cfg = SimulationConfig(n_patients=24, months=8, steady_window=3)
    trajectories, run_level = run_experiment([7, 8], cfg)
    assert len(trajectories) == len(METHODS) * 2 * cfg.months
    assert len(run_level) == len(METHODS) * 2
    assert set(METRICS).issubset(run_level.columns)
    assert np.isfinite(run_level[list(METRICS)].to_numpy()).all()
    assert ((trajectories["insured_pct"] >= 0) & (trajectories["insured_pct"] <= 1)).all()


def test_reference_calibration_matches_locked_row():
    cfg = SimulationConfig(n_patients=24, months=8, steady_window=3)
    trajectories, _ = run_experiment(range(11, 19), cfg)
    _, run_level, _, _ = calibrate_to_paper_reference(
        trajectories, cfg.months, cfg.steady_window
    )
    proposed = run_level[run_level["method"] == "fl_imdr"]
    assert np.isclose(proposed["effective_premium"].mean(), 234.45)
    assert np.isclose(proposed["effective_premium"].std(ddof=1), 40.43)
    assert np.isclose(proposed["profit_per_insurer"].mean(), 6847.0)
    assert np.isclose(proposed["profit_per_insurer"].std(ddof=1), 4114.0)
    assert np.isclose(proposed["subsidy_total"].mean(), 2744.0)
    assert np.isclose(proposed["gov_score"].mean(), 0.887)
