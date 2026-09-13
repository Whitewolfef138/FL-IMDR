from pathlib import Path
import numpy as np

ORDER = [
    "Central welfare planner",
    "Bayesian direct mechanism",
    "FL-IMDR restricted",
]
DISPLAY = {
    "Central welfare planner": "Central welfare",
    "Bayesian direct mechanism": "Bayesian direct",
    "FL-IMDR restricted": r"\textbf{FL-IMDR}",
}

def _pm(mean, sd, digits=2, scale=1.0):
    if not np.isfinite(mean):
        return "--"
    return f"{mean*scale:.{digits}f} $\\pm$ {sd*scale:.{digits}f}"

def write_latex_table(agg, outpath):
    agg = agg.set_index("mechanism")
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Counterfactual mechanism comparison on the original manuscript-sized market ($N=100$, $M=4$, $T=120$). Values are means over the final ten rounds, reported as mean $\pm$ standard deviation across paired market seeds. The override residual measures direct displacement of decentralized insurer offers and patient best responses; lower values indicate narrower intervention authority.}",
        r"\label{tab:mechanism_numerical_comparison}",
        r"\small",
        r"\renewcommand{\arraystretch}{1.25}",
        r"\begin{tabular}{lccccccc}",
        r"\toprule",
        r"Mechanism & Insured (\%) & Eff. premium (\$) & Coverage & Profit/insurer (\$) & Subsidy (\$) & HHI & Override residual \\",
        r"\midrule",
    ]
    for mech in ORDER:
        r = agg.loc[mech]
        lines.append(
            f"{DISPLAY[mech]} & "
            f"{_pm(r['insured_pct_mean'], r['insured_pct_sd'], 2, 100)} & "
            f"{_pm(r['accepted_premium_mean'], r['accepted_premium_sd'], 2)} & "
            f"{_pm(r['accepted_coverage_mean'], r['accepted_coverage_sd'], 3)} & "
            f"{_pm(r['profit_per_insurer_mean'], r['profit_per_insurer_sd'], 1)} & "
            f"{_pm(r['subsidy_total_mean'], r['subsidy_total_sd'], 1)} & "
            f"{_pm(r['hhi_mean'], r['hhi_sd'], 3)} & "
            f"{_pm(r['override_residual_mean'], r['override_residual_sd'], 3)} \\\\"
        )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
    ]
    Path(outpath).write_text("\n".join(lines), encoding="utf-8")
