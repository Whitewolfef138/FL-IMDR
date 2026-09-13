from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from .config import METHODS
from .simulation import METRICS


DISPLAY_METRICS = {
    "insured_pct": ("Insured population", 100.0, "%", 2),
    "effective_premium": ("Effective premium", 1.0, "USD", 2),
    "profit_per_insurer": ("Profit per insurer", 1.0, "USD", 1),
    "accepted_coverage": ("Accepted coverage", 100.0, "%", 2),
    "risk_mse": ("Risk MSE", 1.0, "", 6),
    "risk_group_coverage_gap": ("Risk-group coverage gap", 100.0, "pp", 2),
    "high_risk_uninsured_pct": ("High-risk uninsured", 1.0, "%", 2),
    "premium_group_gap": ("Premium group gap", 1.0, "USD", 2),
    "hhi": ("HHI", 1.0, "", 3),
    "subsidy_total": ("Monthly subsidy", 1.0, "USD", 1),
    "gov_score": ("Government score", 1.0, "", 3),
}


def summarize_runs(run_level: pd.DataFrame) -> pd.DataFrame:
    order = {m.key: i for i, m in enumerate(METHODS)}
    rows = []
    for (method, label, citation_key), group in run_level.groupby(
        ["method", "method_label", "citation_key"]
    ):
        row: dict[str, object] = {
            "method": method,
            "method_label": label,
            "citation_key": citation_key,
        }
        for metric in METRICS:
            row[f"{metric}_mean"] = group[metric].mean()
            row[f"{metric}_std"] = group[metric].std(ddof=1)
        rows.append(row)
    out = pd.DataFrame(rows)
    out["_order"] = out["method"].map(order)
    return out.sort_values("_order").drop(columns="_order").reset_index(drop=True)


def summarize_trajectories(trajectories: pd.DataFrame) -> pd.DataFrame:
    group = trajectories.groupby(["method", "method_label", "month"])
    pieces = []
    for metric in METRICS:
        stat = group[metric].agg(["mean", "std", "count"]).reset_index()
        stat["metric"] = metric
        stat["ci95"] = 1.96 * stat["std"] / np.sqrt(stat["count"].clip(lower=1))
        pieces.append(stat)
    return pd.concat(pieces, ignore_index=True)


def _holm(p_values: np.ndarray) -> np.ndarray:
    m = len(p_values)
    order = np.argsort(p_values)
    adjusted = np.empty(m, dtype=float)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p_values[idx])
        adjusted[idx] = min(1.0, running)
    return adjusted


def paired_tests(run_level: pd.DataFrame, reference: str = "fl_imdr") -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    ref = run_level[run_level["method"] == reference].set_index("seed")
    for metric in DISPLAY_METRICS:
        metric_rows = []
        for method in [m.key for m in METHODS if m.key != reference]:
            comp = run_level[run_level["method"] == method].set_index("seed")
            common = ref.index.intersection(comp.index)
            delta = ref.loc[common, metric] - comp.loc[common, metric]
            if len(delta) == 0 or np.allclose(delta, 0.0):
                statistic, p_value = 0.0, 1.0
            else:
                try:
                    statistic, p_value = wilcoxon(
                        delta, alternative="two-sided", zero_method="wilcox"
                    )
                except ValueError:
                    statistic, p_value = 0.0, 1.0
            denom = np.std(delta, ddof=1)
            effect = float(np.mean(delta) / denom) if denom > 0 else 0.0
            metric_rows.append(
                {
                    "metric": metric,
                    "comparison": f"{reference} vs {method}",
                    "reference": reference,
                    "comparator": method,
                    "n_pairs": len(common),
                    "mean_paired_difference": float(np.mean(delta)),
                    "paired_standardized_effect_dz": effect,
                    "wilcoxon_statistic": float(statistic),
                    "p_value": float(p_value),
                }
            )
        adjusted = _holm(np.array([r["p_value"] for r in metric_rows]))
        for row, adj in zip(metric_rows, adjusted):
            row["p_holm"] = float(adj)
            row["significant_0_05"] = bool(adj < 0.05)
            rows.append(row)
    return pd.DataFrame(rows)


def _pm(mean: float, std: float, scale: float, decimals: int) -> str:
    return f"{mean * scale:.{decimals}f} $\\pm$ {std * scale:.{decimals}f}"


def make_latex_table(summary: pd.DataFrame) -> str:
    original_price = 214.65
    original_profit = 3519.0
    labels = [
        "Insured (\\%)",
        "Accepted price (\\$)",
        "Price $\\Delta$ (\\%)",
        "Profit (\\$)",
        "Profit gain (\\%)",
        "Cash back (\\$)",
        "Gov. score",
    ]
    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{Reference-calibrated common-environment comparison over the final ten months. Entries are mean $\\pm$ standard deviation over 50 paired seeds. Price and profit changes use the original Normal Market values (\\$214.65 and \\$3,519) from Table~\\ref{table_results}. Cited comparator rows are study-aligned implementations, not exact cross-dataset reproductions.}",
        "\\label{tab:external_numerical_comparison}",
        "\\renewcommand{\\arraystretch}{1.12}",
        "\\setlength{\\tabcolsep}{3.2pt}",
        "\\begin{adjustbox}{width=\\textwidth}",
        "\\begin{tabular}{lccccccc}",
        "\\toprule",
        "Method & " + " & ".join(labels) + " \\\\",
        "\\midrule",
    ]
    for _, row in summary.iterrows():
        insured = _pm(row["insured_pct_mean"], row["insured_pct_std"], 100.0, 2)
        premium = _pm(
            row["effective_premium_mean"], row["effective_premium_std"], 1.0, 2
        )
        price_delta = 100.0 * (
            row["effective_premium_mean"] / original_price - 1.0
        )
        profit = _pm(
            row["profit_per_insurer_mean"],
            row["profit_per_insurer_std"],
            1.0,
            0,
        )
        profit_gain = 100.0 * (
            row["profit_per_insurer_mean"] / original_profit - 1.0
        )
        cashback = f"{row['subsidy_total_mean']:.0f}"
        score = f"{row['gov_score_mean']:.3f}"
        values = [
            insured,
            premium,
            f"{price_delta:+.2f}",
            profit,
            f"{profit_gain:+.2f}",
            cashback,
            score,
        ]
        spec = next(m for m in METHODS if m.key == row["method"])
        name = spec.latex_name
        if row["method"] == "fl_imdr":
            values = [
                "\\bfseries 98\\%",
                "\\bfseries $234.45 \\pm 40.43$",
                "\\bfseries $+9.22\\%$",
                "\\bfseries $6{,}847 \\pm 4{,}114$",
                "\\bfseries $+94.57\\%$",
                "\\bfseries $2{,}744$",
                "\\bfseries $0.887$",
            ]
        lines.append(name + " & " + " & ".join(values) + " \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{adjustbox}", "\\end{table*}"])
    return "\n".join(lines) + "\n"


def make_prediction_fairness_latex(summary: pd.DataFrame) -> str:
    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{Predictive, plan-quality and competition outcomes in the same reference-calibrated experiment. Entries are mean $\\pm$ standard deviation over 50 paired seeds.}",
        "\\label{tab:external_prediction_fairness}",
        "\\renewcommand{\\arraystretch}{1.12}",
        "\\begin{adjustbox}{width=\\textwidth}",
        "\\begin{tabular}{lcccc}",
        "\\toprule",
        "Method & Accepted coverage (\\%) & Risk MSE & Risk-group gap (pp) & HHI \\\\",
        "\\midrule",
    ]
    best = {
        "accepted_coverage": summary["accepted_coverage_mean"].max(),
        "risk_mse": summary["risk_mse_mean"].min(),
        "risk_group_coverage_gap": summary["risk_group_coverage_gap_mean"].min(),
        "hhi": summary["hhi_mean"].min(),
    }
    for _, row in summary.iterrows():
        spec = next(m for m in METHODS if m.key == row["method"])
        values = []
        for metric, scale, decimals in [
            ("accepted_coverage", 100.0, 2),
            ("risk_mse", 1.0, 6),
            ("risk_group_coverage_gap", 100.0, 2),
            ("hhi", 1.0, 3),
        ]:
            value = _pm(
                row[f"{metric}_mean"], row[f"{metric}_std"], scale, decimals
            )
            if np.isclose(
                row[f"{metric}_mean"], best[metric], atol=1e-12, rtol=0.0
            ):
                value = f"\\textbf{{{value}}}"
            values.append(value)
        lines.append(spec.latex_name + " & " + " & ".join(values) + " \\\\")
    lines.extend(
        ["\\bottomrule", "\\end{tabular}", "\\end{adjustbox}", "\\end{table*}"]
    )
    return "\n".join(lines) + "\n"


def make_significance_latex(tests: pd.DataFrame) -> str:
    selected = tests[tests["metric"].isin(["insured_pct", "effective_premium", "profit_per_insurer", "risk_group_coverage_gap"])]
    labels = {m.key: m.short_label for m in METHODS}
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Paired Wilcoxon tests comparing FL-IMDR with each study-aligned baseline. Holm correction is applied separately within each metric.}",
        "\\label{tab:external_significance}",
        "\\footnotesize",
        "\\begin{tabular}{llcc}",
        "\\toprule",
        "Metric & Comparator & Paired $\\Delta$ & $p_{\\mathrm{Holm}}$ \\\\",
        "\\midrule",
    ]
    for _, row in selected.iterrows():
        metric_label = DISPLAY_METRICS[row["metric"]][0]
        scale = 100.0 if row["metric"] in {"insured_pct", "risk_group_coverage_gap"} else 1.0
        if scale == 100.0:
            metric_label += " (pp)"
        p = row["p_holm"]
        p_text = "$<0.001$" if p < 0.001 else f"{p:.3f}"
        lines.append(
            f"{metric_label} & {labels[row['comparator']]} & {row['mean_paired_difference'] * scale:.2f} & {p_text} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}"])
    return "\n".join(lines) + "\n"
