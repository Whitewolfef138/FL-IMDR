from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .analysis import DISPLAY_METRICS
from .config import METHODS


COLORS = {
    "dong_quan_automl": "#7A7A7A",
    "richman_credibility": "#E69F00",
    "piontkowski_small_portfolio": "#D55E00",
    "xin_fairness": "#8E63CE",
    "sun_fhe_crl": "#009E73",
    "fl_imdr": "#0072B2",
}


def _style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 140,
            "savefig.bbox": "tight",
        }
    )


def plot_dynamics(trajectory_summary: pd.DataFrame, output: Path) -> None:
    _style()
    panels = [
        ("insured_pct", 100.0, "Insured population (%)"),
        ("effective_premium", 1.0, "Effective premium (USD)"),
        ("profit_per_insurer", 1.0, "Profit per insurer (USD)"),
        ("risk_mse", 1.0, "Risk-prediction MSE"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(7.15, 5.35), sharex=True)
    for ax, (metric, scale, ylabel) in zip(axes.flat, panels):
        data = trajectory_summary[trajectory_summary["metric"] == metric]
        for spec in METHODS:
            d = data[data["method"] == spec.key].sort_values("month")
            x = d["month"].to_numpy(float)
            mean = d["mean"].to_numpy(float) * scale
            std = d["std"].fillna(0.0).to_numpy(float) * scale
            ax.plot(x, mean, color=COLORS[spec.key], lw=2.0 if spec.key == "fl_imdr" else 1.25, label=spec.short_label)
            ax.fill_between(x, mean - std, mean + std, color=COLORS[spec.key], alpha=0.10)
        ax.axvspan(1, 5, color="#F2C14E", alpha=0.12, lw=0)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.22, lw=0.55)
        if metric == "insured_pct":
            ax.axhline(98.0, color="#333333", ls="--", lw=0.8, alpha=0.7)
            ax.set_ylim(0, 103)
    axes[1, 0].set_xlabel("Month")
    axes[1, 1].set_xlabel("Month")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.01))
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    for suffix in ("pdf", "png"):
        fig.savefig(output.with_suffix(f".{suffix}"), dpi=300)
    plt.close(fig)


def plot_final_metrics(summary: pd.DataFrame, output: Path) -> None:
    _style()
    panels = [
        ("insured_pct", 100.0, "Insured population (%)"),
        ("accepted_coverage", 100.0, "Accepted coverage (%)"),
        ("profit_per_insurer", 1.0, "Profit per insurer (USD)"),
        ("risk_group_coverage_gap", 100.0, "Risk-group coverage gap (pp)"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(7.15, 5.25))
    x = np.arange(len(METHODS))
    for ax, (metric, scale, ylabel) in zip(axes.flat, panels):
        values = []
        errors = []
        for spec in METHODS:
            row = summary[summary["method"] == spec.key].iloc[0]
            values.append(row[f"{metric}_mean"] * scale)
            errors.append(row[f"{metric}_std"] * scale)
        ax.bar(x, values, yerr=errors, capsize=2.5, color=[COLORS[m.key] for m in METHODS], alpha=0.88, linewidth=0)
        ax.set_ylabel(ylabel)
        ax.set_xticks(x, [m.short_label for m in METHODS], rotation=24, ha="right")
        ax.grid(axis="y", alpha=0.22, lw=0.55)
    fig.tight_layout()
    for suffix in ("pdf", "png"):
        fig.savefig(output.with_suffix(f".{suffix}"), dpi=300)
    plt.close(fig)


def plot_frontier(summary: pd.DataFrame, output: Path) -> None:
    _style()
    fig, ax = plt.subplots(figsize=(5.2, 3.65))
    for spec in METHODS:
        row = summary[summary["method"] == spec.key].iloc[0]
        x = row["insured_pct_mean"] * 100.0
        y = row["profit_per_insurer_mean"]
        ax.errorbar(
            x,
            y,
            xerr=row["insured_pct_std"] * 100.0,
            yerr=row["profit_per_insurer_std"],
            fmt="o",
            ms=7.5 if spec.key == "fl_imdr" else 6.0,
            color=COLORS[spec.key],
            capsize=2.5,
            label=spec.short_label,
        )
    ax.set_xlabel("Insured population (%)")
    ax.set_ylabel("Profit per insurer (USD)")
    ax.grid(alpha=0.25, lw=0.6)
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    for suffix in ("pdf", "png"):
        fig.savefig(output.with_suffix(f".{suffix}"), dpi=300)
    plt.close(fig)
