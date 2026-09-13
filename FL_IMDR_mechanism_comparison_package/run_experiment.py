import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from flimdr_compare.config import ExperimentConfig
from flimdr_compare.experiment import run_experiment
from flimdr_compare.plots import plot_insured_trajectory, plot_override_residual
from flimdr_compare.tables import write_latex_table

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=50)
    p.add_argument("--reference-seed", type=int, default=0)
    p.add_argument("--output-dir", default="outputs")
    args = p.parse_args()

    cfg = ExperimentConfig(n_seeds=args.seeds, reference_seed=args.reference_seed)
    out = ROOT / args.output_dir

    all_df, by_seed, agg, ref, traj = run_experiment(cfg, out)
    plot_insured_trajectory(traj, out, cfg.lock_in)
    plot_override_residual(traj, out, cfg.lock_in)
    write_latex_table(agg, out / "appendix_results_table.tex")

    print("\nAggregate final-window results:\n")
    cols = [
        "mechanism", "insured_pct_mean", "accepted_premium_mean",
        "accepted_coverage_mean", "profit_per_insurer_mean",
        "hhi_mean", "override_residual_mean"
    ]
    print(agg[cols].to_string(index=False))
    print(f"\nOutputs written to: {out}")

if __name__ == "__main__":
    main()
