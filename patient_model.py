import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
import pandas as pd

import numpy as np

class Patient:
    """
    Patient with stochastic attributes evolving as mean-reverting processes
    + population-level random drift for non-stationary mean behavior.
    """

    def __init__(self, dt=1.0):
        # Initialize state
        self.age = np.random.randint(18, 70)
        self.income = np.random.normal(20_000, 120_000)
        self.credit_score = np.random.uniform(300, 850)
        self.health_status = np.random.beta(4, 2)
        self.location = np.random.uniform(0, 1)

        self.wanted_coverage = np.random.uniform(0.1, 0.90)
        self.maximum_insurance_spending = self.income * np.random.uniform(0.07, 0.2)
        self.actual_risk = self.compute_risk()

        # Process parameters
        self.dt = dt

        # History + behavioral settings
        self.max_obs_history = 10
        self.obs_history = []
        self.accept_softness = 5.0  # steeper -> more deterministic acceptance

        # Mean-reversion parameters (Ornstein–Uhlenbeck style)
        self.k_income = 1/80.0
        self.k_credit = 1/120.0
        self.k_health = 1/60.0

        self.sigma_income = 6000.0
        self.sigma_credit = 25.0
        self.sigma_health = 0.03

        # Global (population-wide) random drift parameters
        self.global_drift_scale_income = 2000.0
        self.global_drift_scale_credit = 10.0

        # Insurance state
        self.has_insurance = False
        self.insurer_id = None
        self.current_cost = 0.0
        self.current_coverage = 0.0
        self.effective_cost = 0.0
        self.gov_subsidy = 0.0

        # seed observation history with initial state
        self._record_observation()

    # ----- Risk definition -----
    def compute_risk(self):
        """
        Nonlinear stochastic risk depending on health, credit, age, and income-based social vulnerability.
        Includes nonlinear saturation, credit penalties, and idiosyncratic noise.
        """
        # Normalize attributes
        h = np.clip(self.health_status, 0.0, 1.0)
        c = np.clip(self.credit_score / 850.0, 0.0, 1.0)
        a = np.clip(self.age / 100.0, 0.0, 1.0)
        y = np.clip(self.income / 120_000.0, 0.0, 1.0)

        # 1. Health effect: exponential risk growth when health deteriorates
        health_factor = np.exp(-3.0 * h)

        # 2. Credit effect: quadratic penalty for poor credit
        credit_factor = (1.0 - c) ** 2.0

        # 3. Age effect: superlinear growth after mid-age
        age_factor = a ** 2.0 + 0.2 * a

        # 4. Social vulnerability (income gradient)
        lambda_social = 0.2  # strength of income-health gradient
        social_factor = 1.0 - lambda_social * y  # lower income → higher risk multiplier

        # Combine effects
        base_risk = social_factor * health_factor * (0.5 * credit_factor + 0.5 * age_factor)

        # 5. Idiosyncratic stochastic noise (lifestyle / shocks)
        epsilon = np.random.normal(0.0, 0.05)
        risk = np.clip(base_risk * (1.0 + epsilon), 0.0, 1.0)

        return risk
    # ----- Observable info -----
    def get_observable_info(self):
        return {
            "age": self.age / 70,
            "income": self.income / 120_000,
            "credit_score": self.credit_score / 850,
            "health_status": self.health_status,
            "location": self.location,
        }

    def get_temporal_observable(self, history_len: int = 10):
        """
        Flattened sequence of the last `history_len` normalized observations.
        Pads with zeros if history is shorter than requested.
        """
        h = int(max(1, history_len))
        seq = self.obs_history[-h:]
        if len(seq) < h:
            pad = [[0.0] * 5] * (h - len(seq))
            seq = pad + seq
        return np.array(seq, dtype=np.float32).flatten()

    # ----- Dynamics -----
    def update_state(self):
        """
        Evolve patient attributes as a *mild* mean-reverting random process.
        - Income, health, and credit fluctuate around their personal baselines
        with small noise.
        - This keeps the population distribution stable instead of exploding.
        """

        # --- initialize personal baselines once ---
        if not hasattr(self, "base_income"):
            self.base_income  = float(self.income)
            self.base_health  = float(self.health_status)
            self.base_credit  = float(self.credit_score)

        # time step (one "round")
        dt = 1.0

        # mean-reversion strengths (smaller = slower move toward baseline)
        alpha_inc   = 0.05    # income mean-reversion
        alpha_h     = 0.05    # health
        alpha_cred  = 0.05    # credit

        # relative noise levels (as fraction of baseline)
        sigma_inc_rel  = 0.05   # 5% of baseline income
        sigma_h_abs    = 0.05   # absolute noise on health
        sigma_cred_rel = 0.03   # 3% of baseline credit

        # --- Income: OU-like process around base_income ---
        eps_y = np.random.randn()
        inc_drift = alpha_inc * (self.base_income - self.income) * dt
        inc_noise = sigma_inc_rel * self.base_income * np.sqrt(dt) * eps_y
        new_income = self.income + inc_drift + inc_noise

        # Clamp to reasonable range
        self.income = float(np.clip(new_income, 10_000, 200_000))

        # --- Health: bounded [0,1], mean-reverts to base_health ---
        eps_h = np.random.randn()
        h_drift = alpha_h * (self.base_health - self.health_status) * dt
        h_noise = sigma_h_abs * np.sqrt(dt) * eps_h
        new_health = self.health_status + h_drift + h_noise
        self.health_status = float(np.clip(new_health, 0.0, 1.0))

        # --- Credit score: bounded [300, 850], mean-reverts to base_credit ---
        eps_s = np.random.randn()
        cred_drift = alpha_cred * (self.base_credit - self.credit_score) * dt
        cred_noise = sigma_cred_rel * self.base_credit * np.sqrt(dt) * eps_s
        new_credit = self.credit_score + cred_drift + cred_noise
        self.credit_score = float(np.clip(new_credit, 300.0, 850.0))

        # --- Age: slow, deterministic increase ---
        self.age = min(self.age + 1, 100)

        # --- Spending ability: tied to updated income but with modest randomness ---
        rate = np.random.uniform(0.07, 0.20)
        self.maximum_insurance_spending = float(self.income * rate)

        # --- Recompute risk with your risk function ---
        self.actual_risk = float(self.compute_risk())

        # record the new observation for temporal models
        self._record_observation()

    # ----- Claim -----
    def claim_amount(self, n_mc: int = 5, as_paid: bool = False):
        """
        Monte Carlo claim:
        • If insured, paid claim can be a fraction of the incurred medical loss.
        • If uninsured, incurred loss is still simulated (paid=0 unless as_paid=False).
        Args:
        n_mc   : MC samples to smooth variance.
        as_paid: if True, return insurer-paid component; else return incurred medical loss.
        """
        # If the patient has a policy, we can use the policy "scale" too.
        if self.current_coverage > 0.0 and self.current_cost > 0.0:
            # use policy scale as one proxy for exposure
            exposure = max(self.current_coverage * self.current_cost, 1.0)
        else:
            # uninsured exposure proxy: fraction of income (adjust as you like)
            exposure = max(0.08 * self.income, 1.0)  # 8% of income baseline medical exposure

        p_event = float(np.clip(self.actual_risk, 0.0, 1.0))

        total = 0.0
        for _ in range(max(1, n_mc)):
            # event occurs with prob = risk
            if np.random.rand() < p_event:
                # severity ~ Gamma(k=2, theta=exposure/2) -> mean ≈ exposure
                severity = np.random.gamma(shape=2.0, scale=exposure / 2.0)
            else:
                severity = 0.0

            if as_paid and self.current_coverage > 0.0:
                # insurer pays covered fraction of incurred loss
                paid = self.current_coverage * severity
                total += paid
            else:
                # return incurred medical loss (regardless of insurance)
                total += severity

        return float(total / max(1, n_mc))
    
    def offer_insurance(self,
                        price: float,
                        coverage: float,
                        insurer_id: int,
                        forced: bool = False,
                        gov_incentive: float = 0.0) -> bool:
        """
        Decide whether to accept an insurance offer.

        Parameters
        ----------
        price : float
            The (effective) price the patient is asked to pay.
        coverage : float
            Coverage fraction offered.
        insurer_id : int
            ID of the insurer making the offer.
        forced : bool, optional
            If True, accept regardless of utility (used for seeding / hard policy).
        gov_incentive : float, optional
            Extra 'utility' coming from the government (bonuses, nudges, penalties for
            remaining uninsured, etc.). Positive values make acceptance more likely.

        Logic
        -----
        - Base economic utility: U_base = v_i * coverage - price,
        where v_i = maximum_insurance_spending / wanted_coverage.
        - Total utility: U_total = U_base + gov_incentive.
        - Accept if forced or U_total >= 0.  (You can tighten/loosen this threshold.)
        """

        # Avoid divide-by-zero if wanted_coverage is tiny
        desired = max(1e-8, float(self.wanted_coverage))
        v_i = float(self.maximum_insurance_spending) / desired  # value per unit coverage

        # Base utility without government influence
        U_base = v_i * float(coverage) - float(price)

        # Government incentive shifts the utility (scale to spending so it matters across income levels)
        U_total = U_base + float(gov_incentive) * float(self.maximum_insurance_spending)

        # Soft acceptance: probability via logistic on normalized utility
        util_norm = U_total / max(1e-6, float(self.maximum_insurance_spending))
        logits = np.clip(self.accept_softness * util_norm, -30.0, 30.0)
        accept_prob = 1.0 / (1.0 + np.exp(-logits))
        accept = forced or (np.random.rand() < accept_prob)

        if accept:
            self.has_insurance = True
            self.insurer_id = insurer_id
            self.current_cost = float(price)
            self.current_coverage = float(coverage)
            return True

        return False

    # -------- internal helpers --------
    def _record_observation(self):
        vec = [
            self.age / 70,
            self.income / 120_000,
            self.credit_score / 850,
            self.health_status,
            self.location,
        ]
        self.obs_history.append(vec)
        if len(self.obs_history) > self.max_obs_history:
            self.obs_history = self.obs_history[-self.max_obs_history:]
    
    

        
# ===============================================================
# === POPULATION SIMULATION & VISUALIZATION =====================
# ===============================================================

def simulate_population(n_patients=1000, n_rounds=50, seed=42):
    """
    Run a Monte Carlo simulation of n_patients over n_rounds.
    Returns: list of dicts (each patient’s time series).
    """
    np.random.seed(seed)
    patients = [Patient() for _ in range(n_patients)]

    history = []
    for t in range(n_rounds):
        for p in patients:
            p.update_state()
        # record stats snapshot
        snapshot = {
            "round": t,
            "mean_income": np.mean([p.income for p in patients]),
            "mean_health": np.mean([p.health_status for p in patients]),
            "mean_credit": np.mean([p.credit_score for p in patients]),
            "mean_risk": np.mean([p.actual_risk for p in patients]),
            "var_risk": np.var([p.actual_risk for p in patients]),
        }
        history.append(snapshot)
    return patients, pd.DataFrame(history)


def plot_population_trends(df):
    """
    Plot population-level averages over time in four aligned subplots.
    Each subplot shows one variable (income, health, credit, risk).
    """
    variables = ["mean_income", "mean_health", "mean_credit", "mean_risk"]
    titles = ["Income", "Health", "Credit Score", "Risk"]

    fig, axs = plt.subplots(1, 5, figsize=(16, 4), sharex=True)
    for i, (ax, var, title) in enumerate(zip(axs, variables, titles)):
        ax.plot(df["round"], df[var], color="tab:blue", linewidth=2)
        ax.set_title(title)
        ax.set_xlabel("Round")
        ax.set_ylabel("Mean Value" if i == 0 else "")
        ax.grid(alpha=0.3)

    fig.suptitle("Population Averages Over Time", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


def plot_distribution_snapshots(patients, title_suffix=""):
    """Show histograms for patient states at final round."""
    incomes = [p.income for p in patients]
    healths = [p.health_status for p in patients]
    credits = [p.credit_score for p in patients]
    risks = [p.actual_risk for p in patients]

    fig, axs = plt.subplots(2,2, figsize=(10,8))
    sns.histplot(incomes, bins=40, ax=axs[0,0], color="tab:blue")
    axs[0,0].set_title("Income Distribution" + title_suffix)

    sns.histplot(healths, bins=40, ax=axs[0,1], color="tab:green")
    axs[0,1].set_title("Health Distribution" + title_suffix)

    sns.histplot(credits, bins=40, ax=axs[1,0], color="tab:orange")
    axs[1,0].set_title("Credit Score Distribution" + title_suffix)

    sns.histplot(risks, bins=40, ax=axs[1,1], color="tab:red")
    axs[1,1].set_title("Risk Distribution" + title_suffix)

    for ax in axs.flat:
        ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_correlations(patients):
    """Visualize correlation matrix between patient attributes."""
    data = {
        "income": [p.income for p in patients],
        "health": [p.health_status for p in patients],
        "credit": [p.credit_score for p in patients],
        "risk": [p.actual_risk for p in patients],
        "age": [p.age for p in patients],
    }
    df = pd.DataFrame(data)
    plt.figure(figsize=(6,5))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", vmin=-1, vmax=1)
    plt.title("Correlation Matrix of Patient Attributes")
    plt.tight_layout()
    plt.show()

def plot_scatter_relations(patients):
    """Show key nonlinear relationships (income-risk, health-risk, credit-risk)."""
    data = {
        "income": [p.income for p in patients],
        "health": [p.health_status for p in patients],
        "credit": [p.credit_score for p in patients],
        "risk": [p.actual_risk for p in patients],
    }
    df = pd.DataFrame(data)

    fig, axs = plt.subplots(1,3, figsize=(15,5))
    sns.scatterplot(df, x="income", y="risk", ax=axs[0], alpha=0.5)
    axs[0].set_title("Income vs Risk")

    sns.scatterplot(df, x="health", y="risk", ax=axs[1], alpha=0.5)
    axs[1].set_title("Health vs Risk")

    sns.scatterplot(df, x="credit", y="risk", ax=axs[2], alpha=0.5)
    axs[2].set_title("Credit vs Risk")

    for ax in axs: ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def simulate_claims_over_time(n_patients=1000, n_rounds=50, seed=42, paid=False):
    np.random.seed(seed)
    patients = [Patient() for _ in range(n_patients)]

    rows = []
    for t in range(n_rounds):
        claims = []
        for p in patients:
            p.update_state()
            c = p.claim_amount(n_mc=5, as_paid=paid)  # <-- choose incurred (False) or paid (True)
            claims.append(c)

        rows.append({
            "round": t,
            "mean_claim": float(np.mean(claims)),
            "std_claim": float(np.std(claims)),
            "total_claims": float(np.sum(claims)),
            "mean_risk": float(np.mean([q.actual_risk for q in patients])),
            "mean_income": float(np.mean([q.income for q in patients])),
            "mean_health": float(np.mean([q.health_status for q in patients])),
        })
    return patients, pd.DataFrame(rows)


def plot_claims_evolution(df):
    """
    Plot claim statistics (mean, std, total) across simulation rounds.
    """
    fig, axs = plt.subplots(1, 3, figsize=(15, 4))

    axs[0].plot(df["round"], df["mean_claim"], label="Mean Claim", color="tab:blue")
    axs[0].set_title("Mean Claim per Patient")
    axs[0].set_xlabel("Round"); axs[0].set_ylabel("Claim Amount"); axs[0].grid(alpha=0.3)

    axs[1].plot(df["round"], df["std_claim"], label="Std. Dev", color="tab:orange")
    axs[1].set_title("Claim Variability")
    axs[1].set_xlabel("Round"); axs[1].set_ylabel("Standard Deviation"); axs[1].grid(alpha=0.3)

    axs[2].plot(df["round"], df["total_claims"], label="Total Claims", color="tab:red")
    axs[2].set_title("Aggregate Claims (Population)")
    axs[2].set_xlabel("Round"); axs[2].set_ylabel("Total Claim Amount"); axs[2].grid(alpha=0.3)

    for ax in axs: ax.legend()
    plt.tight_layout(); plt.show()


# ---------- Uninsured market simulation (TAM) ----------
def simulate_uninsured_market(n_patients=1000, n_rounds=50, seed=42, load=0.20):
    """
    Simulate a population with NO insurance and compute:
      - total incurred medical losses (TAM proxy),
      - potential gross premium at loading 'load',
      - potential profit (upper bound) = load * incurred.
    Assumptions: full uptake if market exists, perfect pricing, ignore lapses/expenses > load.

    Returns
    -------
    patients : list[Patient]
    df       : pd.DataFrame with columns:
               ['round','incurred_total','incurred_mean','premium_total','profit_total',
                'cum_incurred','cum_premium','cum_profit',
                'mean_risk','mean_income','mean_health']
    """
    rng = np.random.default_rng(seed)
    np.random.seed(seed)
    patients = [Patient() for _ in range(n_patients)]

    records = []
    for t in range(n_rounds):
        incurred_list = []
        for p in patients:
            # evolve the stochastic state
            p.update_state()
            # incurred medical loss even if uninsured
            loss = p.claim_amount(n_mc=5, as_paid=False)
            incurred_list.append(loss)

        incurred_total = float(np.sum(incurred_list))
        premium_total  = float((1.0 + load) * incurred_total)
        profit_total   = float(load * incurred_total)

        records.append({
            "round": t,
            "incurred_total": incurred_total,
            "incurred_mean": float(np.mean(incurred_list)),
            "premium_total": premium_total,
            "profit_total": profit_total,
            "mean_risk": float(np.mean([q.actual_risk for q in patients])),
            "mean_income": float(np.mean([q.income for q in patients])),
            "mean_health": float(np.mean([q.health_status for q in patients])),
        })

    df = pd.DataFrame(records)
    df["cum_incurred"] = df["incurred_total"].cumsum()
    df["cum_premium"]  = df["premium_total"].cumsum()
    df["cum_profit"]   = df["profit_total"].cumsum()
    df.attrs["load"]   = load
    return patients, df

# ---------- Plots: market money opportunity ----------
def plot_uninsured_market_opportunity(df):
    """
    Three panels: total incurred (TAM), potential gross premium, and potential profit.
    Also plots cumulative curves below.
    """
    rounds = df["round"].values
    load = df.attrs.get("load", None)

    # Top row: per-round flows
    fig, axs = plt.subplots(1, 3, figsize=(18, 4))
    axs[0].plot(rounds, df["incurred_total"], label="Incurred (TAM)", lw=2)
    axs[0].set_title("Total Incurred Medical Loss (Uninsured)")
    axs[0].set_xlabel("Round"); axs[0].set_ylabel("Amount"); axs[0].grid(alpha=0.3); axs[0].legend()

    axs[1].plot(rounds, df["premium_total"], label="Potential Premium", lw=2, color="tab:orange")
    if load is not None:
        axs[1].set_title(f"Potential Gross Premium  (load = {load:.0%})")
    else:
        axs[1].set_title("Potential Gross Premium")
    axs[1].set_xlabel("Round"); axs[1].grid(alpha=0.3); axs[1].legend()

    axs[2].plot(rounds, df["profit_total"], label="Potential Profit", lw=2, color="tab:green")
    axs[2].set_title("Potential Operating Profit (Upper Bound)")
    axs[2].set_xlabel("Round"); axs[2].grid(alpha=0.3); axs[2].legend()
    fig.tight_layout(); plt.show()

    # Bottom row: cumulative totals
    fig, axs = plt.subplots(1, 3, figsize=(18, 4))
    axs[0].plot(rounds, df["cum_incurred"], label="Cum Incurred", lw=2)
    axs[0].set_title("Cumulative Incurred"); axs[0].set_xlabel("Round")
    axs[0].set_ylabel("Cumulative Amount"); axs[0].grid(alpha=0.3); axs[0].legend()

    axs[1].plot(rounds, df["cum_premium"], label="Cum Premium", lw=2, color="tab:orange")
    axs[1].set_title("Cumulative Potential Premium"); axs[1].set_xlabel("Round")
    axs[1].grid(alpha=0.3); axs[1].legend()

    axs[2].plot(rounds, df["cum_profit"], label="Cum Profit", lw=2, color="tab:green")
    axs[2].set_title("Cumulative Potential Profit"); axs[2].set_xlabel("Round")
    axs[2].grid(alpha=0.3); axs[2].legend()
    fig.tight_layout(); plt.show()

def simulate_budget_capacity_market(n_patients=1000, n_rounds=50, seed=42, takeup=1.0):
    """
    Uninsured economy: track how much money could be collected if insurers sold
    to a fraction `takeup` of the population at their maximum spendable budgets M_i.
    Also simulate incurred medical losses for context.

    Returns
    -------
    patients : list[Patient]
    df       : DataFrame with per-round and cumulative metrics.
    """
    rng = np.random.default_rng(seed)
    np.random.seed(seed)
    patients = [Patient() for _ in range(n_patients)]

    rows = []
    for t in range(n_rounds):
        cap_list = []     # M_i
        loss_list = []    # incurred losses (no insurance)
        for p in patients:
            p.update_state()
            # capacity the person could pay this round
            cap_list.append(float(p.maximum_insurance_spending))
            # incurred loss even if uninsured (expected realized via MC)
            loss_list.append(float(p.claim_amount(n_mc=3, as_paid=False)))

        cap_total = takeup * float(np.sum(cap_list))         # “money that could be made”
        incurred_total = float(np.sum(loss_list))
        profit_cap = max(cap_total - incurred_total, 0.0)    # optimistic upper bound

        rows.append({
            "round": t,
            "cap_total": cap_total,
            "cap_mean": float(np.mean(cap_list)),
            "incurred_total": incurred_total,
            "incurred_mean": float(np.mean(loss_list)),
            "profit_cap": profit_cap,
        })

    import pandas as pd
    df = pd.DataFrame(rows)
    for c in ["cap_total", "incurred_total", "profit_cap"]:
        df[f"cum_{c}"] = df[c].cumsum()
    df.attrs["takeup"] = takeup
    return patients, df

def plot_budget_capacity_market(df):
    """
    Panel A: per-round money-from-budgets (cap_total), incurred losses, and profit_cap.
    Panel B: cumulative versions.
    """
    import matplotlib.pyplot as plt

    rounds = df["round"].values
    takeup = df.attrs.get("takeup", 1.0)

    # Per-round flows
    fig, axs = plt.subplots(1, 3, figsize=(18, 4))
    axs[0].plot(rounds, df["cap_total"], lw=2, label=f"Budget Capacity (takeup={takeup:.0%})")
    axs[0].set_title("Money Collectible from Budgets (Per Round)")
    axs[0].set_xlabel("Round"); axs[0].set_ylabel("Amount"); axs[0].grid(alpha=0.3); axs[0].legend()

    axs[1].plot(rounds, df["incurred_total"], lw=2, color="tab:orange", label="Incurred Medical Loss")
    axs[1].set_title("Incurred Medical Loss (Per Round)")
    axs[1].set_xlabel("Round"); axs[1].grid(alpha=0.3); axs[1].legend()

    axs[2].plot(rounds, df["profit_cap"], lw=2, color="tab:green", label="Profit Upper Bound")
    axs[2].set_title("Optimistic Profit Upper Bound (Per Round)")
    axs[2].set_xlabel("Round"); axs[2].grid(alpha=0.3); axs[2].legend()
    fig.tight_layout(); plt.show()

    # Cumulative totals
    fig, axs = plt.subplots(1, 3, figsize=(18, 4))
    axs[0].plot(rounds, df["cum_cap_total"], lw=2, label="Cum Budget Capacity")
    axs[0].set_title("Cumulative Money Collectible"); axs[0].set_xlabel("Round")
    axs[0].set_ylabel("Cumulative Amount"); axs[0].grid(alpha=0.3); axs[0].legend()

    axs[1].plot(rounds, df["cum_incurred_total"], lw=2, color="tab:orange", label="Cum Incurred Loss")
    axs[1].set_title("Cumulative Incurred Loss"); axs[1].set_xlabel("Round")
    axs[1].grid(alpha=0.3); axs[1].legend()

    axs[2].plot(rounds, df["cum_profit_cap"], lw=2, color="tab:green", label="Cum Profit Upper Bound")
    axs[2].set_title("Cumulative Profit Upper Bound"); axs[2].set_xlabel("Round")
    axs[2].grid(alpha=0.3); axs[2].legend()
    fig.tight_layout(); plt.show()
