from pathlib import Path
import matplotlib.pyplot as plt

ORDER = [
    "Central welfare planner",
    "Bayesian direct mechanism",
    "FL-IMDR restricted",
]

def plot_insured_trajectory(traj, output_dir, lock_in):
    output_dir = Path(output_dir)
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    for mech in ORDER:
        g = traj[traj["mechanism"] == mech].sort_values("round")
        x = g["round"].to_numpy()
        y = 100.0 * g["insured_pct_mean"].to_numpy()
        ci = 100.0 * g["insured_pct_ci95"].to_numpy()
        line, = ax.plot(x, y, label=mech, linewidth=2.0)
        ax.fill_between(x, y-ci, y+ci, alpha=0.18)
    ax.axvspan(1, lock_in, alpha=0.08)
    ax.set_xlabel("Market round (month)")
    ax.set_ylabel("Insured patients (%)")
    ax.set_ylim(0, 103)
    ax.legend(frameon=False)
    ax.grid(alpha=0.22)
    fig.tight_layout()
    fig.savefig(output_dir / "figure_1_insured_trajectory.pdf", bbox_inches="tight")
    fig.savefig(output_dir / "figure_1_insured_trajectory.png", dpi=600, bbox_inches="tight")
    plt.close(fig)

def plot_override_residual(traj, output_dir, lock_in):
    output_dir = Path(output_dir)
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    for mech in ORDER:
        g = traj[traj["mechanism"] == mech].sort_values("round")
        x = g["round"].to_numpy()
        y = g["override_residual_mean"].to_numpy()
        ci = g["override_residual_ci95"].to_numpy()
        line, = ax.plot(x, y, label=mech, linewidth=2.0)
        ax.fill_between(x, y-ci, y+ci, alpha=0.18)
    ax.axhline(0.0, linewidth=1.0)
    ax.axvspan(1, lock_in, alpha=0.08)
    ax.set_xlabel("Market round (month)")
    ax.set_ylabel("Endogenous-decision override residual")
    ax.legend(frameon=False)
    ax.grid(alpha=0.22)
    fig.tight_layout()
    fig.savefig(output_dir / "figure_2_override_residual.pdf", bbox_inches="tight")
    fig.savefig(output_dir / "figure_2_override_residual.png", dpi=600, bbox_inches="tight")
    plt.close(fig)
