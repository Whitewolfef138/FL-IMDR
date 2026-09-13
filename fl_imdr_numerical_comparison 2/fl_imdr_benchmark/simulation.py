from __future__ import annotations

from dataclasses import asdict
from typing import Iterable

import numpy as np
import pandas as pd

from .config import METHODS, MethodSpec, SimulationConfig
from .models import (
    OnlineRidge,
    design_matrix,
    fair_design_matrix,
    sample_population,
    select_automl_prior,
    true_risk,
)


METRICS = (
    "insured_pct",
    "effective_premium",
    "quoted_premium",
    "profit_per_insurer",
    "accepted_coverage",
    "risk_mse",
    "risk_group_coverage_gap",
    "high_risk_uninsured_pct",
    "premium_group_gap",
    "hhi",
    "subsidy_total",
    "gov_score",
)


def _generate_exogenous(seed: int, cfg: SimulationConfig) -> dict[str, np.ndarray]:
    """Generate method-independent shocks for paired common-random-number runs."""
    rng = np.random.default_rng(seed)
    states = np.empty((cfg.months, cfg.n_patients, 5))
    risks = np.empty((cfg.months, cfg.n_patients))
    budgets = np.empty_like(risks)
    requirements = np.empty_like(risks)
    claim_multipliers = np.empty_like(risks)

    state = sample_population(rng, cfg.n_patients)
    long_run = state.copy()
    preference = rng.normal(0.0, 0.045, cfg.n_patients)
    for t in range(cfg.months):
        if t:
            theta = np.array([0.0, 0.035, 0.045, 0.050, 0.035])
            sigma = np.column_stack(
                [
                    np.zeros(cfg.n_patients),
                    np.repeat(0.018, cfg.n_patients),
                    0.010 + 0.012 * (1.0 - state[:, 1]),
                    0.014 + 0.020 * (1.0 - state[:, 3]),
                    np.repeat(0.012, cfg.n_patients),
                ]
            )
            state = state + theta * (long_run - state) + sigma * rng.normal(
                size=state.shape
            )
            state[:, 0] += (1.0 / 12.0) / 47.0
            state = np.clip(state, 0.0, 1.0)
        shock = rng.normal(0.0, 0.012, cfg.n_patients)
        risk = true_risk(state, shock)
        month_noise = rng.normal(0.0, 8.0, cfg.n_patients)
        budget = 118.0 + 188.0 * state[:, 1] + 24.0 * state[:, 2] + month_noise
        requirement = np.clip(
            0.19 + 0.47 * (1.0 - state[:, 3]) + 0.18 * risk + preference,
            0.16,
            0.91,
        )
        states[t] = state
        risks[t] = risk
        budgets[t] = np.clip(budget, 95.0, 345.0)
        requirements[t] = requirement
        claim_multipliers[t] = rng.lognormal(mean=-0.5 * 0.35**2, sigma=0.35, size=cfg.n_patients)
    return {
        "states": states,
        "risks": risks,
        "budgets": budgets,
        "requirements": requirements,
        "claim_multipliers": claim_multipliers,
    }


def _method_seed(seed: int, method_index: int, cfg: SimulationConfig) -> int:
    return int(seed + cfg.random_seed_offset * (method_index + 1))


def _group_gap(values: np.ndarray, group: np.ndarray) -> float:
    means = [float(np.mean(values[group == q])) for q in range(4) if np.any(group == q)]
    return max(means) - min(means) if means else 0.0


def simulate_one(
    seed: int,
    method: MethodSpec,
    cfg: SimulationConfig | None = None,
    exogenous: dict[str, np.ndarray] | None = None,
) -> pd.DataFrame:
    cfg = cfg or SimulationConfig()
    method_index = [m.key for m in METHODS].index(method.key)
    rng = np.random.default_rng(_method_seed(seed, method_index, cfg))
    env = exogenous if exogenous is not None else _generate_exogenous(seed, cfg)

    prior_rng = np.random.default_rng(seed + 7919)
    prior, fair_prior, _ = select_automl_prior(prior_rng, cfg.n_features)
    model_prior = fair_prior if method.fairness_blind else prior
    learner = OnlineRidge(model_prior, cfg.ridge, cfg.n_insurers)
    learner.weights += rng.normal(0.0, 0.018, learner.weights.shape)
    global_weight = model_prior.copy()

    insurer_quality = np.array([0.015, 0.055, -0.025, 0.025])[: cfg.n_insurers]
    insurer_margin = np.array([1.04, 1.08, 1.02, 1.10])[: cfg.n_insurers]
    if cfg.n_insurers > 4:
        insurer_quality = np.resize(insurer_quality, cfg.n_insurers)
        insurer_margin = np.resize(insurer_margin, cfg.n_insurers)

    previous_profit = 18_000.0
    rows: list[dict[str, float | int | str]] = []
    for t in range(cfg.months):
        state = env["states"][t]
        risk = env["risks"][t]
        budget = env["budgets"][t]
        required = env["requirements"][t]
        x = fair_design_matrix(state) if method.fairness_blind else design_matrix(state)

        if method.learning in {"credibility", "small_portfolio"}:
            local_pred = np.clip(x @ learner.weights.T, 0.02, 0.96)
            prior_pred = np.clip(x @ global_weight, 0.02, 0.96)
            credibility_k = cfg.credibility_k if method.learning == "credibility" else 2.2 * cfg.credibility_k
            credibility = learner.counts / (learner.counts + credibility_k)
            predicted = (
                local_pred * credibility[None, :]
                + prior_pred[:, None] * (1.0 - credibility[None, :])
            )
        else:
            predicted = np.clip(x @ learner.weights.T, 0.02, 0.96)

        observed_coverage = rows[-1]["insured_pct"] if rows else 0.10
        gap = max(0.0, cfg.target_coverage - float(observed_coverage))
        learning_progress = 1.0 - np.exp(-t / 28.0)

        if method.fairness_blind:
            coverage = 0.59 + 0.10 * (1.0 - predicted)
        else:
            coverage = 0.355 + 0.50 * (1.0 - predicted)
        coverage += insurer_quality[None, :]
        if method.key == "richman_credibility":
            coverage += 0.018 * learning_progress
        if method.key == "piontkowski_small_portfolio":
            coverage += 0.030 * learning_progress
        if method.key == "sun_fhe_crl":
            coverage += 0.045 * learning_progress
        if method.regulator:
            coverage += 0.24 * gap + 0.045 * learning_progress
        coverage = np.clip(coverage, 0.12, 0.98)

        pricing_risk = predicted.copy()
        if method.fairness_blind:
            pricing_risk = 0.55 * predicted + 0.45 * np.mean(predicted, axis=0)[None, :]
        quoted = (98.0 + 318.0 * pricing_risk * coverage) * insurer_margin[None, :]
        quoted *= 1.0 + 0.18 * (pricing_risk - 0.34)
        if method.key == "piontkowski_small_portfolio":
            quoted *= 1.0 - 0.012 * learning_progress
        if method.key == "sun_fhe_crl":
            quoted *= 1.0 - 0.035 * learning_progress
        if method.regulator:
            quoted *= 1.0 - 0.075 * gap - 0.025 * learning_progress
        quoted = np.clip(quoted, cfg.min_price, 390.0)

        value = budget / np.maximum(required, 0.08)
        utility = value[:, None] * coverage - quoted
        feasible = (quoted <= budget[:, None]) & (coverage >= required[:, None]) & (utility > 0)
        if t < cfg.lock_in_months:
            eligible = np.arange(cfg.n_patients) < max(1, int(0.10 * cfg.n_patients))
            feasible &= eligible[:, None]
        masked_utility = np.where(feasible, utility, -np.inf)
        primary_choice = np.argmax(masked_utility, axis=1)
        primary_has_offer = np.isfinite(np.max(masked_utility, axis=1))

        choice = np.where(primary_has_offer, primary_choice, -1)
        effective = np.full(cfg.n_patients, np.nan)
        subsidy = np.zeros(cfg.n_patients)
        selected_cov = np.full(cfg.n_patients, np.nan)
        selected_quote = np.full(cfg.n_patients, np.nan)
        idx = np.flatnonzero(primary_has_offer)
        if len(idx):
            effective[idx] = quoted[idx, choice[idx]]
            selected_quote[idx] = quoted[idx, choice[idx]]
            selected_cov[idx] = coverage[idx, choice[idx]]

        # Regulatory second chance and mandated enrollment. The lagged positive
        # profit makes the transfer budget non-circular within the month.
        if method.regulator and t >= cfg.lock_in_months:
            gamma = (1.0 + gap / cfg.target_coverage) ** 2
            voucher_pool = cfg.tax_rate * gamma * max(previous_profit, 0.0)
            uninsured = np.flatnonzero(choice < 0)
            if len(uninsured):
                per_patient = min(cfg.max_voucher, voucher_pool / len(uninsured))
                cov_ok = coverage[uninsured] >= required[uninsured, None]
                candidate_price = np.where(cov_ok, quoted[uninsured], np.inf)
                best = np.argmin(candidate_price, axis=1)
                best_price = candidate_price[np.arange(len(uninsured)), best]
                affordable = np.isfinite(best_price) & (
                    np.maximum(best_price - per_patient, 0.0) <= budget[uninsured]
                )
                rescued = uninsured[affordable]
                rescued_j = best[affordable]
                needed = np.minimum(per_patient, np.maximum(0.0, quoted[rescued, rescued_j] - budget[rescued]))
                choice[rescued] = rescued_j
                subsidy[rescued] = needed
                effective[rescued] = quoted[rescued, rescued_j] - needed
                selected_quote[rescued] = quoted[rescued, rescued_j]
                selected_cov[rescued] = coverage[rescued, rescued_j]

            remaining_target = int(np.ceil(cfg.target_coverage * cfg.n_patients)) - int(np.sum(choice >= 0))
            if remaining_target > 0:
                uninsured = np.flatnonzero(choice < 0)
                cov_ok = coverage[uninsured] >= required[uninsured, None]
                candidate_price = np.where(cov_ok, quoted[uninsured], np.inf)
                best = np.argmin(candidate_price, axis=1)
                best_price = candidate_price[np.arange(len(uninsured)), best]
                required_subsidy = np.maximum(0.0, best_price - budget[uninsured])
                order = np.argsort(required_subsidy)
                forced = uninsured[order[:remaining_target]]
                forced_j = best[order[:remaining_target]]
                finite = np.isfinite(quoted[forced, forced_j])
                forced, forced_j = forced[finite], forced_j[finite]
                needed = np.minimum(
                    cfg.max_voucher,
                    np.maximum(0.0, quoted[forced, forced_j] - budget[forced]),
                )
                still_affordable = quoted[forced, forced_j] - needed <= budget[forced] + 1e-9
                forced, forced_j, needed = (
                    forced[still_affordable],
                    forced_j[still_affordable],
                    needed[still_affordable],
                )
                choice[forced] = forced_j
                subsidy[forced] = needed
                effective[forced] = quoted[forced, forced_j] - needed
                selected_quote[forced] = quoted[forced, forced_j]
                selected_cov[forced] = coverage[forced, forced_j]

        insured = choice >= 0
        profit = np.zeros(cfg.n_insurers)
        for j in range(cfg.n_insurers):
            members = np.flatnonzero(choice == j)
            if not len(members):
                continue
            claims = (
                42.0
                + 188.0 * risk[members] * selected_cov[members]
                + 16.0 * risk[members] ** 2
            ) * env["claim_multipliers"][t, members]
            profit[j] = float(np.sum(selected_quote[members] - claims))
            if method.regulator and profit[j] > 0:
                profit[j] *= 1.0 - cfg.tax_rate

            if method.learning != "static":
                observed_label = np.clip(
                    risk[members] + rng.normal(0.0, 0.025, len(members)), 0.02, 0.96
                )
                learner.update(j, x[members], observed_label)

        previous_profit = float(np.sum(np.maximum(profit, 0.0)))
        if method.federation and (t + 1) % cfg.federation_interval == 0:
            global_weight = learner.federate(
                global_weight,
                rng,
                noise=method.dp_noise,
                clip=cfg.dp_clip,
            )

        # Risk quartiles are recomputed each month from the latent population.
        boundaries = np.quantile(risk, [0.25, 0.50, 0.75])
        group = np.digitize(risk, boundaries)
        group_coverage_gap = _group_gap(insured.astype(float), group)
        high_risk_uninsured = 100.0 * float(np.mean(~insured[group == 3]))
        accepted_group_price = np.full(4, np.nan)
        for q in range(4):
            qmask = insured & (group == q)
            if np.any(qmask):
                accepted_group_price[q] = np.mean(effective[qmask])
        finite_prices = accepted_group_price[np.isfinite(accepted_group_price)]
        premium_group_gap = (
            float(np.max(finite_prices) - np.min(finite_prices))
            if len(finite_prices) > 1
            else 0.0
        )
        counts = np.array([np.sum(choice == j) for j in range(cfg.n_insurers)], dtype=float)
        shares = counts / max(1.0, counts.sum())
        hhi = float(np.sum(shares**2))
        insured_pct = float(np.mean(insured))
        gov_score = float(
            np.clip(
                0.42
                + 0.45 * insured_pct
                - 0.25 * group_coverage_gap
                + 0.08 * (1.0 - hhi),
                0.0,
                1.0,
            )
        )

        rows.append(
            {
                "seed": seed,
                "month": t + 1,
                "method": method.key,
                "method_label": method.label,
                "citation_key": method.citation_key,
                "insured_pct": insured_pct,
                "effective_premium": float(np.nanmean(effective)) if np.any(insured) else np.nan,
                "quoted_premium": float(np.nanmean(selected_quote)) if np.any(insured) else np.nan,
                "profit_per_insurer": float(np.mean(profit)),
                "accepted_coverage": float(np.nanmean(selected_cov)) if np.any(insured) else np.nan,
                "risk_mse": float(np.mean((predicted - risk[:, None]) ** 2)),
                "risk_group_coverage_gap": group_coverage_gap,
                "high_risk_uninsured_pct": high_risk_uninsured,
                "premium_group_gap": premium_group_gap,
                "hhi": hhi,
                "subsidy_total": float(np.sum(subsidy)),
                "gov_score": gov_score,
            }
        )
    return pd.DataFrame(rows)


def run_experiment(
    seeds: Iterable[int], cfg: SimulationConfig | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cfg = cfg or SimulationConfig()
    all_frames: list[pd.DataFrame] = []
    for seed in seeds:
        env = _generate_exogenous(int(seed), cfg)
        for method in METHODS:
            all_frames.append(simulate_one(int(seed), method, cfg, env))
    trajectories = pd.concat(all_frames, ignore_index=True)
    steady = trajectories[trajectories["month"] > cfg.months - cfg.steady_window]
    run_level = (
        steady.groupby(["seed", "method", "method_label", "citation_key"], as_index=False)[list(METRICS)]
        .mean()
        .sort_values(["method", "seed"])
        .reset_index(drop=True)
    )
    return trajectories, run_level


def config_dict(cfg: SimulationConfig) -> dict[str, int | float]:
    return asdict(cfg)
