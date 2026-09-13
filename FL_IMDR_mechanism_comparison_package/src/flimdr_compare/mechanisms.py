from dataclasses import dataclass
import numpy as np
from .market import (
    autonomous_offers,
    best_response_assignments,
    expected_claim_cost,
    assignment_profit,
    assignment_mean,
    hhi_from_assignment,
)

@dataclass
class StepResult:
    assignment: np.ndarray
    premium: np.ndarray
    coverage: np.ndarray
    voucher: np.ndarray
    claim: np.ndarray
    autonomous_premium: np.ndarray
    autonomous_coverage: np.ndarray
    metrics: dict

def _override_residual(path, t, assignment, premium, coverage, voucher, p_auto, c_auto):
    budget = path.budget[t]
    req = path.required_cov[t]
    vscale = path.utility_scale[t]

    br, _, _ = best_response_assignments(
        budget, req, vscale, premium, coverage, voucher
    )
    assignment_override = np.mean(assignment != br)

    premium_override = np.mean(
        np.abs(premium - p_auto) / np.maximum(p_auto, 1.0)
    )
    coverage_override = np.mean(np.abs(coverage - c_auto))

    residual = assignment_override + 0.5 * (premium_override + coverage_override)
    return float(residual), float(assignment_override), float(premium_override), float(coverage_override)

def _collect(path, t, assignment, premium, coverage, voucher, claim, p_auto, c_auto):
    N, M = premium.shape
    insured = assignment >= 0
    insured_pct = insured.mean()

    eff = np.maximum(premium - voucher[:, None], 0.0)
    accepted_premium = assignment_mean(eff, assignment)
    accepted_coverage = assignment_mean(coverage, assignment)

    profits = assignment_profit(assignment, premium, claim)
    total_subsidy = float(voucher[insured].sum()) if insured.any() else 0.0

    residual, assign_ov, price_ov, cov_ov = _override_residual(
        path, t, assignment, premium, coverage, voucher, p_auto, c_auto
    )

    return {
        "insured_pct": float(insured_pct),
        "accepted_premium": accepted_premium,
        "accepted_coverage": accepted_coverage,
        "profit_per_insurer": float(np.mean(profits)),
        "total_profit": float(np.sum(profits)),
        "subsidy_total": total_subsidy,
        "hhi": hhi_from_assignment(assignment, M),
        "override_residual": residual,
        "assignment_override_rate": assign_ov,
        "premium_override": price_ov,
        "coverage_override": cov_ov,
    }

class FLIMDRRestricted:
    name = "FL-IMDR restricted"

    def __init__(self, cfg):
        self.cfg = cfg
        self.prev_total_profit = 4200.0

    def step(self, path, t):
        p_auto, c_auto, claim, _ = autonomous_offers(path, t, learning_strength=0.82)
        N = p_auto.shape[0]

        # Lock-in: preserve a small initial accepted fraction, consistent with the
        # manuscript's stabilization phase, but do not overwrite insurer offers.
        voucher = np.zeros(N)
        assignment, _, _ = best_response_assignments(
            path.budget[t], path.required_cov[t], path.utility_scale[t],
            p_auto, c_auto, voucher
        )

        if t < self.cfg.lock_in:
            # During lock-in, only a small fraction may switch/enter.
            keep = path.alloc_u[t] < 0.12
            assignment = np.where(keep, assignment, -1)
        else:
            insured_rate = np.mean(assignment >= 0)
            gap = max(0.0, self.cfg.target_coverage - insured_rate)
            gamma = (1.0 + gap / max(self.cfg.target_coverage, 1e-9)) ** 2

            uninsured = np.where(assignment < 0)[0]
            if uninsured.size:
                pool = self.cfg.tau * gamma * max(self.prev_total_profit, 0.0)
                per_patient = min(self.cfg.max_voucher, pool / uninsured.size)
                voucher[uninsured] = per_patient

                # Patients independently recompute their BR given the voucher.
                assignment, _, _ = best_response_assignments(
                    path.budget[t], path.required_cov[t], path.utility_scale[t],
                    p_auto, c_auto, voucher
                )

        metrics = _collect(
            path, t, assignment, p_auto, c_auto, voucher, claim, p_auto, c_auto
        )
        if t < self.cfg.lock_in:
            for k in ("override_residual", "assignment_override_rate", "premium_override", "coverage_override"):
                metrics[k] = np.nan
        self.prev_total_profit = 0.75 * self.prev_total_profit + 0.25 * metrics["total_profit"]
        return StepResult(assignment, p_auto, c_auto, voucher, claim, p_auto, c_auto, metrics)

class CentralWelfarePlanner:
    name = "Central welfare planner"

    def __init__(self, cfg):
        self.cfg = cfg

    def step(self, path, t):
        p_auto, c_auto, _, _ = autonomous_offers(path, t, learning_strength=1.0)
        risk = path.risk[t][:, None]
        N, M = p_auto.shape

        # Central planner directly selects coverage and premium, subject to the same
        # actuarial technology and a minimum insurer participation margin.
        req = path.required_cov[t][:, None]
        central_cov = np.clip(
            np.maximum(req, 0.76 + path.insurer_quality[None, :]),
            0.30, 0.98
        )
        central_claim = expected_claim_cost(
            risk, central_cov, path.insurer_cost_mult[None, :]
        )
        central_p = central_claim * (1.0 + self.cfg.min_margin_rate) + 42.0

        # Welfare score: patient gross value minus resource cost.
        gross_value = path.utility_scale[t][:, None] * central_cov
        welfare = gross_value - central_claim - 0.10 * central_p

        # The planner also internalizes concentration. This is intentionally a
        # central assignment rule: it may select a different insurer than the
        # patient's decentralized BR when doing so improves the social objective.
        chosen = np.empty(N, dtype=int)
        counts = np.zeros(M, dtype=float)
        priority = np.argsort(-np.max(welfare, axis=1))
        concentration_penalty = 5.5
        for i in priority:
            congestion = concentration_penalty * counts / max(N, 1)
            j = int(np.argmax(welfare[i] - congestion))
            chosen[i] = j
            counts[j] += 1.0

        # Central transfer: make selected allocation affordable when possible.
        chosen_p = central_p[np.arange(N), chosen]
        affordability_gap = np.maximum(chosen_p - path.budget[t], 0.0)

        # Budget analogous in scale to a tax-financed regulatory pool.
        pool = self.cfg.tau * float(np.sum(chosen_p))
        voucher = np.zeros(N)
        order = np.argsort(-welfare[np.arange(N), chosen])
        remaining = pool
        for i in order:
            need = min(float(affordability_gap[i]), self.cfg.max_voucher)
            if need <= remaining:
                voucher[i] = need
                remaining -= need

        eff = chosen_p - voucher
        feasible = (
            (central_cov[np.arange(N), chosen] >= path.required_cov[t])
            & (eff <= path.budget[t] + 1e-9)
            & (welfare[np.arange(N), chosen] > 0.0)
        )
        assignment = np.where(feasible, chosen, -1)

        if t < self.cfg.lock_in:
            keep = path.alloc_u[t] < 0.12
            assignment = np.where(keep, assignment, -1)

        metrics = _collect(
            path, t, assignment, central_p, central_cov, voucher,
            central_claim, p_auto, c_auto
        )
        if t < self.cfg.lock_in:
            for k in ("override_residual", "assignment_override_rate", "premium_override", "coverage_override"):
                metrics[k] = np.nan
        return StepResult(assignment, central_p, central_cov, voucher, central_claim, p_auto, c_auto, metrics)

class BayesianDirectMechanism:
    name = "Bayesian direct mechanism"

    def __init__(self, cfg, path):
        self.cfg = cfg
        # Known prior estimated from the generated population at t=0.
        self.mu_b = float(np.mean(path.budget[0]))
        self.sd_b = float(np.std(path.budget[0]) + 1e-9)
        self.mu_c = float(np.mean(path.required_cov[0]))
        self.sd_c = float(np.std(path.required_cov[0]) + 1e-9)

    def step(self, path, t):
        p_auto, c_auto, _, _ = autonomous_offers(path, t, learning_strength=1.0)
        risk = path.risk[t][:, None]
        N, M = p_auto.shape

        # Truthful direct reports: theta=(B, C_req).
        b_report = path.budget[t]
        c_report = path.required_cov[t]

        # Prior-standardized scalar index used by a monotone allocation rule.
        theta = (
            (b_report - self.mu_b) / self.sd_b
            - 0.60 * (c_report - self.mu_c) / self.sd_c
        )
        sigmoid = 1.0 / (1.0 + np.exp(-theta))
        q = self.cfg.bayes_q_floor + self.cfg.bayes_q_span * sigmoid
        q = np.clip(q, 0.0, 0.995)

        # Report-dependent mechanism-induced coverage and actuarially sustainable price.
        base_cov_i = np.clip(0.58 + 0.28 * q + 0.35 * (c_report - self.mu_c), 0.35, 0.96)
        mech_cov = np.clip(
            base_cov_i[:, None] + 0.45 * path.insurer_quality[None, :],
            0.30, 0.98
        )
        mech_claim = expected_claim_cost(
            risk, mech_cov, path.insurer_cost_mult[None, :]
        )
        mech_p = mech_claim * (1.0 + 0.075) + 46.0

        # Report-dependent allocation rule. Different prior-type quantiles are
        # matched to different insurer segments, so allocation is mechanism-induced
        # rather than a decentralized patient choice.
        insurer_nodes = np.linspace(0.15, 0.85, M)[None, :]
        type_match = np.abs(q[:, None] - insurer_nodes)
        chosen = np.argmin(mech_p + 28.0 * type_match, axis=1)
        chosen_p = mech_p[np.arange(N), chosen]
        chosen_c = mech_cov[np.arange(N), chosen]

        # Prior/report based transfer: larger when the reported budget lies below the
        # selected price; allocation probability scales the transfer.
        shortfall = np.maximum(chosen_p - b_report, 0.0)
        desired_transfer = np.minimum(self.cfg.max_voucher, q * shortfall)

        pool = self.cfg.tau * float(np.sum(chosen_p))
        voucher = np.zeros(N)
        order = np.argsort(-q)
        remaining = pool
        for i in order:
            give = min(float(desired_transfer[i]), remaining)
            voucher[i] = give
            remaining -= give
            if remaining <= 1e-12:
                break

        allocated = path.alloc_u[t] <= q
        feasible_report = chosen_c >= c_report
        affordable = chosen_p - voucher <= b_report + 1e-9
        assignment = np.where(allocated & feasible_report & affordable, chosen, -1)

        if t < self.cfg.lock_in:
            keep = path.alloc_u[t] < 0.12
            assignment = np.where(keep, assignment, -1)

        metrics = _collect(
            path, t, assignment, mech_p, mech_cov, voucher,
            mech_claim, p_auto, c_auto
        )
        # A direct-mechanism-specific quantity useful in raw CSVs.
        metrics["mean_allocation_probability"] = float(np.mean(q))
        if t < self.cfg.lock_in:
            for k in ("override_residual", "assignment_override_rate", "premium_override", "coverage_override"):
                metrics[k] = np.nan
        return StepResult(assignment, mech_p, mech_cov, voucher, mech_claim, p_auto, c_auto, metrics)
