from dataclasses import dataclass
import numpy as np

@dataclass
class MarketPath:
    risk: np.ndarray          # [T, N]
    budget: np.ndarray        # [T, N]
    required_cov: np.ndarray  # [T, N]
    utility_scale: np.ndarray # [T, N]
    offer_noise: np.ndarray   # [T, N, M]
    alloc_u: np.ndarray       # [T, N]
    insurer_quality: np.ndarray
    insurer_margin: np.ndarray
    insurer_cost_mult: np.ndarray

def generate_market_path(cfg, seed: int) -> MarketPath:
    rng = np.random.default_rng(cfg.seed_offset + seed)
    T, N, M = cfg.horizon, cfg.n_patients, cfg.n_insurers

    age = rng.uniform(18, 65, N)
    age_n = (age - 18.0) / 47.0
    income_n = rng.uniform(0.0, 1.0, N)
    credit_n = rng.uniform(0.0, 1.0, N)
    health = rng.beta(4.0, 2.2, N)
    location = rng.uniform(0.0, 1.0, N)

    base_risk = np.clip(
        0.08
        + 0.34 * (1.0 - health)
        + 0.20 * age_n
        + 0.17 * (1.0 - income_n)
        + 0.13 * (1.0 - credit_n)
        + 0.08 * location,
        0.02, 0.95
    )

    base_budget = (
        230.0
        + 48.0 * income_n
        + 18.0 * credit_n
        - 14.0 * base_risk
        + rng.normal(0, 13, N)
    )
    base_budget = np.clip(base_budget, 155.0, 315.0)

    base_req = np.clip(
        0.40 + 0.30 * base_risk + rng.normal(0, 0.08, N),
        0.28, 0.92
    )

    risk = np.empty((T, N))
    budget = np.empty((T, N))
    req = np.empty((T, N))
    risk[0] = base_risk
    budget[0] = base_budget
    req[0] = base_req

    for t in range(1, T):
        # Mean-reverting, bounded dynamic market path.
        risk[t] = np.clip(
            risk[t-1]
            + 0.07 * (base_risk - risk[t-1])
            + rng.normal(0, 0.025, N),
            0.02, 0.98
        )
        budget[t] = np.clip(
            budget[t-1]
            + 0.05 * (base_budget - budget[t-1])
            + rng.normal(0, 3.5, N),
            145.0, 330.0
        )
        req[t] = np.clip(
            req[t-1]
            + 0.07 * (base_req - req[t-1])
            + 0.015 * (risk[t] - base_risk)
            + rng.normal(0, 0.012, N),
            0.25, 0.95
        )

    utility_scale = budget / np.maximum(req, 0.15)
    offer_noise = rng.normal(0.0, 1.0, size=(T, N, M))
    alloc_u = rng.uniform(0.0, 1.0, size=(T, N))

    insurer_quality = np.linspace(-0.035, 0.045, M)
    insurer_margin = np.linspace(0.08, 0.20, M)
    insurer_cost_mult = np.linspace(0.96, 1.07, M)

    return MarketPath(
        risk=risk,
        budget=budget,
        required_cov=req,
        utility_scale=utility_scale,
        offer_noise=offer_noise,
        alloc_u=alloc_u,
        insurer_quality=insurer_quality,
        insurer_margin=insurer_margin,
        insurer_cost_mult=insurer_cost_mult,
    )

def expected_claim_cost(risk, coverage, cost_mult):
    # A simple actuarial technology shared by all mechanisms. The scale is
    # chosen to keep accepted premiums in the same order as the manuscript.
    return cost_mult * (100.0 + 170.0 * risk) * (0.55 + 0.45 * coverage)

def autonomous_offers(path: MarketPath, t: int, learning_strength: float = 1.0):
    risk = path.risk[t][:, None]
    M = path.insurer_quality.size

    # Decaying prediction noise: later market experience improves local risk estimates.
    noise_scale = (0.105 / np.sqrt(1.0 + t / 10.0) + 0.018) * learning_strength
    risk_hat = np.clip(
        risk + noise_scale * path.offer_noise[t] + np.linspace(-0.025, 0.025, M)[None, :],
        0.0, 1.0
    )

    coverage = np.clip(
        0.63
        + 0.27 * (1.0 - risk_hat)
        + path.insurer_quality[None, :],
        0.30, 0.98
    )

    claim = expected_claim_cost(
        risk,
        coverage,
        path.insurer_cost_mult[None, :]
    )
    premium = claim * (1.0 + path.insurer_margin[None, :]) + 52.0
    return premium, coverage, claim, risk_hat

def best_response_assignments(budget, req, utility_scale, premium, coverage, voucher):
    eff = np.maximum(premium - voucher[:, None], 0.0)
    feasible = (eff <= budget[:, None]) & (coverage >= req[:, None])
    utility = utility_scale[:, None] * coverage - eff
    utility = np.where(feasible, utility, -np.inf)

    best = np.argmax(utility, axis=1)
    best_u = utility[np.arange(len(budget)), best]
    assigned = np.where(np.isfinite(best_u), best, -1)
    return assigned, eff, utility

def assignment_profit(assignment, premium, claim):
    M = premium.shape[1]
    profits = np.zeros(M)
    for j in range(M):
        idx = np.where(assignment == j)[0]
        if idx.size:
            profits[j] = np.sum(premium[idx, j] - claim[idx, j])
    return profits

def hhi_from_assignment(assignment, M):
    insured = assignment >= 0
    n = int(insured.sum())
    if n == 0:
        return 0.0
    shares = np.array([(assignment == j).sum() / n for j in range(M)])
    return float(np.sum(shares ** 2))

def assignment_mean(x, assignment):
    idx = np.where(assignment >= 0)[0]
    if idx.size == 0:
        return np.nan
    return float(np.mean(x[idx, assignment[idx]]))
