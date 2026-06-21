# market_viz.py
import numpy as np
import matplotlib.pyplot as plt

__all__ = [
    "_moving_avg",
    "plot_market_statistics",
    "plot_model_losses",
    "plot_risk_calibration",
    "plot_price_views",
    "plot_subsidy_efficiency",
    "plot_fl_aggregation_flags",
    "plot_governor_alignment_vs_subsidy",
]

# -------- helpers --------
def _moving_avg(x, k=1):
    x = np.array(x, dtype=float)
    if k is None or k <= 1 or x.size < 2:
        return x
    k = int(k)
    pad = k // 2
    xpad = np.pad(x, (pad, pad), mode='edge')
    ker = np.ones(k) / k
    y = np.convolve(xpad, ker, mode='valid')
    if y.size > x.size:  # guard when k is even
        y = y[:x.size]
    return y

def _get_stat(e, iid, key_final, key_pre_fallback):
    """Fetch insurer stat, preferring final (post-voucher) then pre-voucher."""
    st = e['insurer_stats'][iid]
    if key_final in st and not (isinstance(st[key_final], float) and np.isnan(st[key_final])):
        return st[key_final]
    return st.get(key_pre_fallback, np.nan)

def _nan_polyfit(x, y, deg=1):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < deg + 1:
        return None  # not enough points
    return np.polyfit(x[m], y[m], deg)

# -------- main plots --------
def plot_market_statistics(market_history, smooth=1):
    if not market_history:
        print("No history.")
        return

    rounds = [e['round'] for e in market_history]
    insurer_ids = list(market_history[0]['insurer_stats'].keys())

    # Market share (final if present)
    plt.figure(figsize=(10, 5))
    for iid in insurer_ids:
        vals = [_get_stat(e, iid, 'market_share_final', 'market_share') for e in market_history]
        plt.plot(rounds, _moving_avg(vals, smooth), label=f'Insurer {iid}')
    plt.title('Market Share Over Time'); plt.xlabel('Round'); plt.ylabel('Market Share')
    plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout(); plt.show()

    # Profit over time (final if present)
    plt.figure(figsize=(10, 5))
    for iid in insurer_ids:
        vals = [_get_stat(e, iid, 'profit_final', 'profit_pre_voucher') for e in market_history]
        plt.plot(rounds, _moving_avg(vals, smooth), label=f'Insurer {iid}')
    plt.yscale('symlog', linthresh=1.0)
    plt.title('Profit Over Time (symlog)'); plt.xlabel('Round'); plt.ylabel('Profit')
    plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout(); plt.show()

    # Loss ratio over time (final if present)
    plt.figure(figsize=(10, 5))
    for iid in insurer_ids:
        vals = [_get_stat(e, iid, 'loss_ratio_final', 'loss_ratio_pre_voucher') for e in market_history]
        plt.plot(rounds, _moving_avg(vals, smooth), label=f'Insurer {iid}')
    plt.title('Loss Ratio Over Time'); plt.xlabel('Round'); plt.ylabel('Loss Ratio')
    plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout(); plt.show()

    # Insured %
    insured_pct = [e.get('insured_pct', np.nan) for e in market_history]
    plt.figure(figsize=(10, 5))
    plt.plot(rounds, _moving_avg(insured_pct, smooth))
    plt.title('Insured Percentage Over Time'); plt.xlabel('Round'); plt.ylabel('Insured %')
    plt.grid(True, alpha=0.3); plt.tight_layout(); plt.show()

    # Mean accepted price (nominal & effective) and coverage
    mean_nom, mean_eff, mean_covs = [], [], []
    for e in market_history:
        po = e['patient_outcomes']
        pn = po.get('accepted_prices', [])
        pe = po.get('accepted_prices_effective', [])
        cv = po.get('accepted_coverages', [])
        mean_nom.append(float(np.nanmean(pn)) if len(pn) else np.nan)
        mean_eff.append(float(np.nanmean(pe)) if len(pe) else np.nan)
        mean_covs.append(float(np.nanmean(cv)) if len(cv) else np.nan)

    plt.figure(figsize=(10, 5))
    plt.plot(rounds, _moving_avg(mean_nom, smooth), label='Mean Accepted Price (Nominal)')
    plt.plot(rounds, _moving_avg(mean_eff, smooth), label='Mean Accepted Price (Effective)', linestyle='--')
    plt.title('Accepted Prices Over Time'); plt.xlabel('Round'); plt.ylabel('Price')
    plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout(); plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(rounds, _moving_avg(mean_covs, smooth), label='Mean Accepted Coverage')
    plt.title('Mean Accepted Coverage'); plt.xlabel('Round'); plt.ylabel('Coverage')
    plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout(); plt.show()

    # Governor metrics (loss & alignment score)
    if 'governor_loss' in market_history[0] or 'goal_score' in market_history[0]:
        gov_loss = [float(e.get('governor_loss', np.nan)) for e in market_history]
        gov_score = [float(e.get('goal_score', np.nan)) for e in market_history]

        plt.figure(figsize=(10, 5))
        plt.plot(rounds, _moving_avg(gov_loss, smooth), label='Governor Loss')
        plt.title('Governor Loss Over Time'); plt.xlabel('Round'); plt.ylabel('Loss (MSE, mean)')
        plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout(); plt.show()

        plt.figure(figsize=(10, 5))
        plt.plot(rounds, _moving_avg(gov_score, smooth), label='Governor Goal Alignment (R²)', linestyle='--')
        plt.title('Governor Goal Alignment Over Time'); plt.xlabel('Round'); plt.ylabel('R²')
        plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout(); plt.show()


def plot_model_losses(market_history, smooth=1, logy=False):
    if not market_history:
        print("No history.")
        return
    rounds = [e['round'] for e in market_history]
    insurer_ids = list(market_history[0]['insurer_stats'].keys())

    # Insurer training loss
    plt.figure(figsize=(10,5))
    for iid in insurer_ids:
        raw = [e['insurer_stats'][iid].get('loss',
               e['insurer_stats'][iid].get('avg_train_loss', np.nan)) for e in market_history]
        vals = _moving_avg(raw, smooth)
        plt.plot(rounds, vals, label=f'Insurer {iid}')
    if logy: plt.yscale('log')
    plt.xlabel("Round"); plt.ylabel("Insurer Loss" + (" (log)" if logy else ""))
    plt.title("Insurer Model Losses Over Time")
    plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout(); plt.show()

    # Governor loss + goal score (R²)
    gov_losses = [float(e.get('governor_loss', np.nan)) for e in market_history]
    gov_scores = [float(e.get('goal_score', np.nan)) for e in market_history]
    fig, ax1 = plt.subplots(figsize=(10,5))
    ax1.plot(rounds, _moving_avg(gov_losses, smooth), label='Governor Loss')
    if logy: ax1.set_yscale('log')
    ax1.set_xlabel("Round"); ax1.set_ylabel("Governor Loss" + (" (log)" if logy else ""))
    ax1.grid(True, alpha=0.3)
    ax2 = ax1.twinx()
    ax2.plot(rounds, _moving_avg(gov_scores, smooth), linestyle='--', label='Goal Alignment (R²)')
    ax2.set_ylabel("Goal Alignment (R²)")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')
    plt.title("Governor Loss and Alignment Over Time")
    fig.tight_layout(); plt.show()


def plot_risk_calibration(market_history):
    preds_all, actual_all = [], []
    for e in market_history:
        po = e.get('patient_outcomes', {})
        pr = po.get('predicted_risks', None)
        ar = po.get('actual_risks', None)
        if pr is not None and ar is not None and len(pr) == len(ar) and len(pr) > 0:
            preds_all.append(np.asarray(pr, dtype=float))
            actual_all.append(np.asarray(ar, dtype=float))
    if not preds_all:
        print("No accepted risk pairs to plot.")
        return
    preds = np.concatenate(preds_all)
    actual = np.concatenate(actual_all)

    # 2D density comparison (kernel density estimate) to visualize distribution alignment
    plt.figure(figsize=(7, 6))
    try:
        from scipy.stats import gaussian_kde
        xy = np.vstack([preds, actual])
        z = gaussian_kde(xy)(xy)
        idx = np.argsort(z)
        plt.scatter(preds[idx], actual[idx], c=z[idx], s=8, cmap="viridis", alpha=0.6)
    except Exception:
        plt.scatter(preds, actual, s=8, alpha=0.4)

    plt.xlabel('Predicted Risk')
    plt.ylabel('Actual Risk')
    plt.title('Risk Calibration: Pred vs Actual (density view)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Marginal distributions side by side
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.hist(preds, bins=40, color="tab:blue", alpha=0.7, density=True)
    plt.title("Predicted Risk Distribution")
    plt.xlabel("Predicted Risk"); plt.ylabel("Density"); plt.grid(alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.hist(actual, bins=40, color="tab:orange", alpha=0.7, density=True)
    plt.title("Actual Risk Distribution")
    plt.xlabel("Actual Risk"); plt.ylabel("Density"); plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_price_views(market_history, smooth=1):
    if not market_history:
        print("No history.")
        return
    rounds = [e['round'] for e in market_history]
    mean_nom, mean_eff = [], []
    for e in market_history:
        po = e.get('patient_outcomes', {})
        pn = po.get('accepted_prices', [])
        pe = po.get('accepted_prices_effective', [])
        mean_nom.append(float(np.nanmean(pn)) if len(pn) else np.nan)
        mean_eff.append(float(np.nanmean(pe)) if len(pe) else np.nan)
    plt.figure(figsize=(10,5))
    plt.plot(rounds, _moving_avg(mean_nom, smooth), label='Nominal Price (mean)')
    plt.plot(rounds, _moving_avg(mean_eff, smooth), label='Effective Price (mean)', linestyle='--')
    plt.title('Nominal vs Effective Accepted Prices')
    plt.xlabel('Round'); plt.ylabel('Price')
    plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout(); plt.show()


def plot_subsidy_efficiency(market_history, smooth=1):
    if not market_history:
        print("No history.")
        return
    rounds = [e['round'] for e in market_history]
    insured = [e.get('insured_pct', np.nan) for e in market_history]
    subsidy = [e.get('patient_outcomes', {}).get('gov_subsidy_total', np.nan) for e in market_history]
    round_pool = [e.get('cash_back_history', [{}])[-1].get('total_pool', np.nan) if e.get('cash_back_history') else np.nan for e in market_history]

    # Combined view: insured % vs subsidies
    fig, ax1 = plt.subplots(figsize=(10,5))
    ax1.plot(rounds, _moving_avg(insured, smooth), label='Insured %')
    ax1.set_xlabel('Round'); ax1.set_ylabel('Insured %')
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(rounds, _moving_avg(subsidy, smooth), label='Gov Subsidy (paid)', linestyle='--')
    ax2.plot(rounds, _moving_avg(round_pool, smooth), label='Gov Subsidy Pool (per round)', linestyle=':')
    ax2.set_ylabel('Subsidy')

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')
    plt.title('Coverage vs Subsidy Over Time')
    fig.tight_layout(); plt.show()

    # Separate view: subsidy components per round
    plt.figure(figsize=(10, 4))
    plt.plot(rounds, _moving_avg(subsidy, smooth), label='Gov Subsidy (paid)', linestyle='--')
    plt.plot(rounds, _moving_avg(round_pool, smooth), label='Gov Subsidy Pool (per round)', linestyle=':')
    plt.xlabel('Round'); plt.ylabel('Subsidy')
    plt.title('Government Subsidy Per Round')
    plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout(); plt.show()

    # Cumulative view
    cum_paid = np.cumsum(np.nan_to_num(subsidy, nan=0.0))
    cum_pool = np.cumsum(np.nan_to_num(round_pool, nan=0.0))
    plt.figure(figsize=(10, 4))
    plt.plot(rounds, cum_paid, label='Cumulative Subsidy Paid', linestyle='--')
    plt.plot(rounds, cum_pool, label='Cumulative Subsidy Pool', linestyle=':')
    plt.xlabel('Round'); plt.ylabel('Cumulative Subsidy')
    plt.title('Cumulative Government Subsidy')
    plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout(); plt.show()


def plot_fl_aggregation_flags(history):
    if not history:
        print("No history.")
        return
    rounds = [h.get('round', i) for i, h in enumerate(history)]
    flags = [1 if h.get('fl_aggregated', False) else 0 for h in history]
    plt.figure(figsize=(8, 2.5))
    plt.stem(rounds, flags)  # Matplotlib >= 3.6
    plt.yticks([0, 1], ['No', 'Yes'])
    plt.xlabel('Round'); plt.title('Was DP-FedAvg (Risk) Run?')
    plt.tight_layout(); plt.show()


def plot_governor_alignment_vs_subsidy(market_history, smooth=1, population_size=None, show_corr=True):
    """
    Plots governor alignment (R^2 = goal_score) against subsidy.
    If population_size (N) is given, also plots 'per-insured subsidy' using N * insured_pct as denominator.
    """
    if not market_history:
        print("No history.")
        return

    rounds   = [e.get('round', i) for i, e in enumerate(market_history)]
    r2       = np.array([float(e.get('goal_score', np.nan)) for e in market_history], dtype=float)
    subsidy  = np.array([float(e.get('patient_outcomes', {}).get('gov_subsidy_total', np.nan))
                         for e in market_history], dtype=float)
    insuredp = np.array([float(e.get('insured_pct', np.nan)) for e in market_history], dtype=float)

    # Optional: per-insured subsidy proxy
    per_insured = None
    if population_size is not None and population_size > 0:
        denom = np.maximum(insuredp * float(population_size), 1.0)
        per_insured = subsidy / denom

    # 1) Scatter: Total subsidy vs R^2
    plt.figure(figsize=(7.5, 6))
    plt.scatter(subsidy, r2, alpha=0.6)
    coef = _nan_polyfit(subsidy, r2, deg=1)
    if coef is not None:
        xs = np.linspace(np.nanmin(subsidy), np.nanmax(subsidy), 100)
        ys = np.polyval(coef, xs)
        plt.plot(xs, ys, linewidth=2)
        label = f"Trend: R² ≈ {coef[0]:.3g}·Subsidy + {coef[1]:.3g}"
        plt.legend([label], loc='best')
    plt.xlabel("Total Subsidy (this round)")
    plt.ylabel("Governor Goal Alignment (R²)")
    plt.title("Alignment vs Total Subsidy")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # 2) Time series overlay: R^2 and subsidy
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(rounds, _moving_avg(r2, smooth), label='Goal Alignment (R²)')
    ax1.set_xlabel("Round")
    ax1.set_ylabel("Goal Alignment (R²)")
    ax1.grid(True, alpha=0.3)
    ax2 = ax1.twinx()
    ax2.plot(rounds, _moving_avg(subsidy, smooth), color='tab:orange', linestyle='--', label='Total Subsidy')
    ax2.set_ylabel("Total Subsidy")
    l1, lab1 = ax1.get_legend_handles_labels()
    l2, lab2 = ax2.get_legend_handles_labels()
    ax1.legend(l1 + l2, lab1 + lab2, loc='best')
    plt.title("Governor Alignment and Subsidy Over Time")
    fig.tight_layout()
    plt.show()

    # 3) Optional: per-insured subsidy vs R^2
    if per_insured is not None:
        plt.figure(figsize=(7.5, 6))
        plt.scatter(per_insured, r2, alpha=0.6)
        coef2 = _nan_polyfit(per_insured, r2, deg=1)
        if coef2 is not None:
            xs = np.linspace(np.nanmin(per_insured), np.nanmax(per_insured), 100)
            ys = np.polyval(coef2, xs)
            plt.plot(xs, ys, linewidth=2)
            label = f"Trend: R² ≈ {coef2[0]:.3g}·(Subsidy/Insured) + {coef2[1]:.3g}"
            plt.legend([label], loc='best')
        plt.xlabel("Per-Insured Subsidy (proxy)")
        plt.ylabel("Governor Goal Alignment (R²)")
        plt.title("Alignment vs Per-Insured Subsidy")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    # Optional correlation printout
    if show_corr:
        def _corr(a, b):
            m = np.isfinite(a) & np.isfinite(b)
            if m.sum() < 3:
                return np.nan
            a0 = a[m] - np.nanmean(a[m]); b0 = b[m] - np.nanmean(b[m])
            denom = np.sqrt((a0*a0).sum() * (b0*b0).sum()) + 1e-12
            return float((a0*b0).sum() / denom)

        print("Correlation (R² vs total subsidy):", _corr(r2, subsidy))
        if per_insured is not None:
            print("Correlation (R² vs subsidy per insured):", _corr(r2, per_insured))
