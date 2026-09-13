from __future__ import annotations

import numpy as np


def design_matrix(state: np.ndarray) -> np.ndarray:
    """Nonlinear additive/interaction basis on normalized patient state.

    State columns: age, income, credit, health, and location, all in [0, 1].
    """
    age, income, credit, health, location = state.T
    return np.column_stack(
        [
            np.ones(len(state)),
            age,
            1.0 - income,
            1.0 - credit,
            1.0 - health,
            location,
            age * (1.0 - health),
            (1.0 - income) * (1.0 - health),
            age**2,
            (1.0 - health) ** 2,
        ]
    )


def fair_design_matrix(state: np.ndarray) -> np.ndarray:
    """Group-blind basis: suppress direct age and health terms."""
    x = design_matrix(state).copy()
    x[:, [1, 4, 6, 7, 8, 9]] = 0.0
    return x


def true_risk(state: np.ndarray, shock: np.ndarray | float = 0.0) -> np.ndarray:
    age, income, credit, health, location = state.T
    risk = (
        0.055
        + 0.125 * age
        + 0.090 * (1.0 - income)
        + 0.080 * (1.0 - credit)
        + 0.310 * (1.0 - health)
        + 0.035 * location
        + 0.170 * age * (1.0 - health)
        + 0.090 * (1.0 - income) * (1.0 - health)
        + shock
    )
    return np.clip(risk, 0.025, 0.94)


def ridge_fit(x: np.ndarray, y: np.ndarray, ridge: float) -> np.ndarray:
    penalty = np.eye(x.shape[1]) * ridge
    penalty[0, 0] = ridge * 0.05
    return np.linalg.solve(x.T @ x + penalty, x.T @ y)


def select_automl_prior(
    rng: np.random.Generator, n_features: int
) -> tuple[np.ndarray, np.ndarray, float]:
    """Select a compact pricing learner on an independent calibration panel.

    This is deliberately a small, dependency-free AutoML abstraction: it chooses
    the feature family and ridge strength using held-out MSE. It is not the
    InsurAutoML package or an exact reproduction of Dong and Quan (2025).
    """
    n = 1800
    state = sample_population(rng, n)
    y = true_risk(state, rng.normal(0.0, 0.018, n))
    idx = rng.permutation(n)
    train, valid = idx[:1300], idx[1300:]
    candidates: list[tuple[bool, float, np.ndarray, float]] = []
    for fair in (False, True):
        x = fair_design_matrix(state) if fair else design_matrix(state)
        for lam in (0.05, 0.2, 1.0, 5.0, 20.0):
            w = ridge_fit(x[train], y[train], lam)
            mse = float(np.mean((x[valid] @ w - y[valid]) ** 2))
            candidates.append((fair, lam, w, mse))
    fair, lam, w, _ = min(candidates, key=lambda item: item[3])

    # A separate group-blind prior is retained for the fairness-aligned baseline.
    xf = fair_design_matrix(state)
    wf = ridge_fit(xf[train], y[train], 1.0)
    if len(w) != n_features:
        raise RuntimeError("Unexpected design-matrix width")
    return w, wf, lam


def sample_population(rng: np.random.Generator, n: int) -> np.ndarray:
    age = rng.uniform(18.0, 65.0, n)
    income = rng.uniform(20_000.0, 120_000.0, n)
    credit = rng.uniform(300.0, 850.0, n)
    health = rng.beta(4.0, 2.0, n)
    location = rng.uniform(0.0, 1.0, n)
    return np.column_stack(
        [
            (age - 18.0) / 47.0,
            (income - 20_000.0) / 100_000.0,
            (credit - 300.0) / 550.0,
            health,
            location,
        ]
    )


class OnlineRidge:
    def __init__(self, prior: np.ndarray, ridge: float, n_models: int):
        self.prior = prior.copy()
        self.ridge = ridge
        self.xtx = np.repeat((np.eye(len(prior)) * ridge)[None, :, :], n_models, axis=0)
        self.xty = np.repeat((prior * ridge)[None, :], n_models, axis=0)
        self.counts = np.zeros(n_models, dtype=float)
        self.weights = np.repeat(prior[None, :], n_models, axis=0)

    def update(self, insurer: int, x: np.ndarray, y: np.ndarray) -> None:
        if len(y) == 0:
            return
        self.xtx[insurer] += x.T @ x
        self.xty[insurer] += x.T @ y
        self.counts[insurer] += len(y)
        self.weights[insurer] = np.linalg.solve(self.xtx[insurer], self.xty[insurer])

    def federate(
        self,
        global_weight: np.ndarray,
        rng: np.random.Generator,
        noise: float,
        clip: float,
    ) -> np.ndarray:
        total = self.counts.sum()
        alpha = (
            np.repeat(1.0 / len(self.counts), len(self.counts))
            if total <= 0
            else self.counts / total
        )
        deltas = self.weights - global_weight
        norms = np.linalg.norm(deltas, axis=1)
        scales = np.minimum(1.0, clip / np.maximum(norms, 1e-12))
        clipped = deltas * scales[:, None]
        update = np.sum(alpha[:, None] * clipped, axis=0)
        if noise > 0:
            update += rng.normal(0.0, noise, len(global_weight))
        new_global = global_weight + update
        # Broadcast while retaining a small local component.
        self.weights = 0.85 * new_global[None, :] + 0.15 * self.weights
        return new_global
