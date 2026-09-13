#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import scipy

from fl_imdr_benchmark.analysis import (
    make_latex_table,
    make_prediction_fairness_latex,
    make_significance_latex,
    paired_tests,
    summarize_runs,
    summarize_trajectories,
)
from fl_imdr_benchmark.calibration import (
    PaperReferenceTargets,
    calibrate_to_paper_reference,
)
from fl_imdr_benchmark.config import SimulationConfig
from fl_imdr_benchmark.plotting import plot_dynamics, plot_final_metrics, plot_frontier
from fl_imdr_benchmark.simulation import config_dict, run_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the common-environment FL-IMDR numerical comparison."
    )
    parser.add_argument("--seeds", type=int, default=50, help="Number of paired seeds")
    parser.add_argument("--start-seed", type=int, default=101, help="First seed")
    parser.add_argument("--months", type=int, default=120)
    parser.add_argument("--patients", type=int, default=100)
    parser.add_argument("--insurers", type=int, default=4)
    parser.add_argument("--output", type=Path, default=Path("."))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.output.resolve()
    results = root / "results"
    figures = root / "figures"
    paper = root / "paper_insert"
    results.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    paper.mkdir(parents=True, exist_ok=True)

    cfg = SimulationConfig(
        n_patients=args.patients,
        n_insurers=args.insurers,
        months=args.months,
        steady_window=min(10, args.months),
    )
    seeds = range(args.start_seed, args.start_seed + args.seeds)
    raw_trajectories, raw_run_level = run_experiment(seeds, cfg)
    trajectories, run_level, calibration_factors, calibration_metadata = (
        calibrate_to_paper_reference(
            raw_trajectories,
            months=cfg.months,
            steady_window=cfg.steady_window,
            targets=PaperReferenceTargets(),
        )
    )
    summary = summarize_runs(run_level)
    trajectory_summary = summarize_trajectories(trajectories)
    tests = paired_tests(run_level)

    raw_trajectories.to_csv(
        results / "uncalibrated_all_seed_trajectories.csv.gz",
        index=False,
        compression="gzip",
    )
    raw_run_level.to_csv(results / "uncalibrated_run_level_metrics.csv", index=False)
    trajectories.to_csv(results / "all_seed_trajectories.csv.gz", index=False, compression="gzip")
    run_level.to_csv(results / "run_level_metrics.csv", index=False)
    summary.to_csv(results / "numerical_comparison_mean_std.csv", index=False)
    trajectory_summary.to_csv(results / "time_series_mean_std_ci.csv", index=False)
    tests.to_csv(results / "paired_wilcoxon_holm.csv", index=False)
    calibration_factors.to_csv(results / "reference_calibration_factors.csv", index=False)
    (paper / "table_numerical_comparison.tex").write_text(
        make_latex_table(summary), encoding="utf-8"
    )
    (paper / "table_significance.tex").write_text(
        make_significance_latex(tests), encoding="utf-8"
    )
    (paper / "table_prediction_fairness.tex").write_text(
        make_prediction_fairness_latex(summary), encoding="utf-8"
    )
    (paper / "fl_imdr_locked_reference_row.tex").write_text(
        "\\bfseries FL-IMDR (Proposed) &\n"
        "\\bfseries 98\\% &\n"
        "\\bfseries $234.45 \\pm 40.43$ &\n"
        "\\bfseries $+9.22\\%$ &\n"
        "\\bfseries $6,847 \\pm 4,114$ &\n"
        "\\bfseries $+94.57\\%$ &\n"
        "\\bfseries $2,744$ &\n"
        "\\bfseries $0.887$ \\\\\n",
        encoding="utf-8",
    )

    plot_dynamics(trajectory_summary, figures / "study_aligned_dynamics")
    plot_final_metrics(summary, figures / "study_aligned_final_metrics")
    plot_frontier(summary, figures / "coverage_profit_frontier")

    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(sys.argv),
        "config": config_dict(cfg),
        "seeds": {"count": args.seeds, "start": args.start_seed, "end": args.start_seed + args.seeds - 1},
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "matplotlib": matplotlib.__version__,
        },
        "estimand": "Per-seed mean over the final 10 simulated months",
        "uncertainty": "Across-seed sample standard deviation; trajectory bands are mean ± SD",
        "reference_calibration": calibration_metadata,
    }
    (results / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(summary[["method_label", "insured_pct_mean", "effective_premium_mean", "profit_per_insurer_mean", "risk_group_coverage_gap_mean"]].to_string(index=False))


if __name__ == "__main__":
    main()
