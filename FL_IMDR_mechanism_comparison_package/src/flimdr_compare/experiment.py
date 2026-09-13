from pathlib import Path
import numpy as np
import pandas as pd

from .market import generate_market_path
from .mechanisms import FLIMDRRestricted, CentralWelfarePlanner, BayesianDirectMechanism

MECHANISM_ORDER = [
    "Central welfare planner",
    "Bayesian direct mechanism",
    "FL-IMDR restricted",
]

def run_one_seed(cfg, seed: int):
    path = generate_market_path(cfg, seed)
    mechs = [
        CentralWelfarePlanner(cfg),
        BayesianDirectMechanism(cfg, path),
        FLIMDRRestricted(cfg),
    ]
    rows = []
    for mech in mechs:
        for t in range(cfg.horizon):
            res = mech.step(path, t)
            row = {
                "seed": seed,
                "round": t + 1,
                "mechanism": mech.name,
            }
            row.update(res.metrics)
            rows.append(row)
    return pd.DataFrame(rows)

def summarize_final_window(df, cfg):
    final = df[df["round"] > cfg.horizon - cfg.final_window].copy()
    numeric = [
        "insured_pct", "accepted_premium", "accepted_coverage",
        "profit_per_insurer", "subsidy_total", "hhi",
        "override_residual", "assignment_override_rate",
        "premium_override", "coverage_override",
    ]
    by_seed = final.groupby(["seed", "mechanism"], as_index=False)[numeric].mean()

    agg_rows = []
    for mech in MECHANISM_ORDER:
        g = by_seed[by_seed["mechanism"] == mech]
        row = {"mechanism": mech, "n_seeds": int(g["seed"].nunique())}
        for col in numeric:
            vals = g[col].dropna().to_numpy()
            row[f"{col}_mean"] = float(np.mean(vals)) if vals.size else np.nan
            row[f"{col}_sd"] = float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0
        agg_rows.append(row)
    return by_seed, pd.DataFrame(agg_rows)

def trajectory_summary(df):
    rows = []
    for (mech, rnd), g in df.groupby(["mechanism", "round"]):
        row = {"mechanism": mech, "round": rnd, "n": g["seed"].nunique()}
        for col in ["insured_pct", "override_residual"]:
            vals = g[col].to_numpy(dtype=float)
            mean = np.nanmean(vals)
            sd = np.nanstd(vals, ddof=1)
            ci = 1.96 * sd / np.sqrt(max(np.sum(np.isfinite(vals)), 1))
            row[f"{col}_mean"] = mean
            row[f"{col}_ci95"] = ci
        rows.append(row)
    return pd.DataFrame(rows)

def run_experiment(cfg, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dfs = [run_one_seed(cfg, s) for s in range(cfg.n_seeds)]
    all_df = pd.concat(dfs, ignore_index=True)
    all_df.to_csv(output_dir / "all_seed_round_metrics.csv", index=False)

    by_seed, agg = summarize_final_window(all_df, cfg)
    by_seed.to_csv(output_dir / "per_seed_final_window.csv", index=False)
    agg.to_csv(output_dir / "aggregate_results.csv", index=False)

    ref = by_seed[by_seed["seed"] == cfg.reference_seed].copy()
    ref.to_csv(output_dir / "reference_seed_results.csv", index=False)

    traj = trajectory_summary(all_df)
    traj.to_csv(output_dir / "trajectory_summary.csv", index=False)

    return all_df, by_seed, agg, ref, traj
