# market_model.py
from typing import Optional, Dict, List, Tuple
import numpy as np
import torch

__all__ = [
    "ensure_patient_subsidy_fields",
    "patient_obs_tensor",
    "patient_value_per_coverage",
    "InsuranceMarket",
    "FEATURE_KEYS",
]

FEATURE_KEYS = ['age', 'income', 'credit_score', 'health_status', 'location']


# ---------- helpers ----------
def ensure_patient_subsidy_fields(p) -> None:
    if not hasattr(p, 'effective_cost'):
        p.effective_cost = float(getattr(p, 'current_cost', 0.0))
    if not hasattr(p, 'gov_subsidy'):
        p.gov_subsidy = 0.0


def patient_obs_tensor(p, history_len: int = 10) -> torch.Tensor:
    if hasattr(p, "get_temporal_observable"):
        obs = p.get_temporal_observable(history_len=history_len)
    else:
        obs = [p.get_observable_info()[k] for k in FEATURE_KEYS]
    return torch.tensor(obs, dtype=torch.float32).view(1, -1)


def patient_value_per_coverage(p) -> float:
    wanted = max(1e-8, float(getattr(p, 'wanted_coverage', 0.2)))
    return float(p.maximum_insurance_spending) / wanted


# ---------- main class ----------
class InsuranceMarket:
    def __init__(self, patients, insurers, governor,
                 initial_assign_pct: float = 0.8, lock_in_rounds: int = 3, cash_back_rate: float = 0.10,
                 fl_every_rounds: int = 10, dp_sigma: float = 0.05, clip_C: Optional[float] = 1.0,
                 history_len: int = 10, coverage_target: float = 0.90,
                 coverage_gap_beta: float = 3.0, min_cash_back: float = 150_000.0,
                 enforce_target: bool = True,
                 force_subsidy_cap: float = 120_000.0,
                 force_step_pct: float = 0.05,
                 switch_prob: float = 0.5):
        self.patients = patients
        self.insurers = insurers
        self.governor = governor
        self.history_len = int(max(1, history_len))
        self.coverage_target = float(coverage_target)
        self.coverage_gap_beta = float(max(0.0, coverage_gap_beta))
        self.min_cash_back = float(max(0.0, min_cash_back))
        self.enforce_target = bool(enforce_target)
        self.force_subsidy_cap = float(max(0.0, force_subsidy_cap))
        self.force_step_pct = float(max(0.0, min(1.0, force_step_pct)))
        self.switch_prob = float(max(0.0, min(1.0, switch_prob)))
        self.lock_in_rounds = int(lock_in_rounds)
        self.history: List[Dict] = []
        self.round_id = 0
        self.cash_back_rate = float(cash_back_rate)
        self.cash_back_history: List[Dict] = []

        # FL params
        self.fl_every_rounds = int(fl_every_rounds)
        self.dp_sigma = float(dp_sigma)
        self.clip_C = float(clip_C) if clip_C is not None else None

        # Seed assignment (forced, no vouchers)
        for p in self.patients:
            # ensure patients keep at least this many steps of observation history
            try:
                p.max_obs_history = max(int(getattr(p, "max_obs_history", 1)), self.history_len)
            except Exception:
                p.max_obs_history = self.history_len
        n_assign = int(len(patients) * float(initial_assign_pct))
        indices = np.random.permutation(len(patients))
        for i in range(n_assign):
            p = patients[indices[i]]
            ins = np.random.choice(insurers)
            price, cov, _ = ins.random_initial_offer(p)
            p.offer_insurance(price, cov, ins.id, forced=True)
            p.current_cost = float(price)
            p.current_coverage = float(cov)
            ensure_patient_subsidy_fields(p)
            p.effective_cost = float(price)
            p.gov_subsidy = 0.0

    def run_round(self, round_id: Optional[int] = None) -> Dict:
        if round_id is None:
            round_id = self.round_id

        # Lock-in flag: during lock-in we do not add new sales; only seed stays
        lock_in_active = round_id < self.lock_in_rounds
        allow_switch = (not lock_in_active)

        # --- initialize lists ONCE and never rebind them later ---
        offers: Dict[int, List] = {ins.id: [] for ins in self.insurers}
        predicted_risks: List[float] = []
        actual_risks: List[float] = []

        # Unlock switching after lock-in (keep existing coverage; allow optional switching)
        for p in self.patients:
            ensure_patient_subsidy_fields(p)

        # gap_scale initialized using current insured share (updated later after baseline)
        insured_pct_now = sum(p.has_insurance for p in self.patients) / len(self.patients)
        coverage_gap = max(0.0, self.coverage_target - insured_pct_now)
        gap_scale = 1.0 + self.coverage_gap_beta * (coverage_gap / max(self.coverage_target, 1e-6))
        gap_scale = gap_scale * gap_scale

        # Prefill offers with existing insured so stats/training account for current book
        for p in self.patients:
            if getattr(p, "has_insurance", False) and p.insurer_id is not None:
                offers[p.insurer_id].append(p)
                predicted_risks.append(float('nan'))
                actual_risks.append(float(p.actual_risk))

        # 1) Baseline offers (no vouchers) & choices
        if not lock_in_active:
            for p in self.patients:
                if p.has_insurance:
                    # allow switching with some probability
                    if np.random.rand() > self.switch_prob:
                        continue
                    old_insurer = p.insurer_id
                else:
                    old_insurer = None

                best_offer: Optional[Tuple[float, float, int]] = None
                best_util = -np.inf
                best_risk: Optional[float] = None
                x = patient_obs_tensor(p, history_len=getattr(self, "history_len", 10))
                for ins in self.insurers:
                    price, cov, rhat = ins.offer_to_patient(x, add_noise=True)
                    v = patient_value_per_coverage(p)
                    util = v * cov - price
                    # softer penalties for affordability/coverage gaps instead of hard filters
                    util -= 0.10 * max(0.0, price - p.maximum_insurance_spending)
                    util -= 0.10 * v * max(0.0, p.wanted_coverage - cov)
                    if util > best_util:
                        best_util, best_offer, best_risk = util, (price, cov, ins.id), rhat

                if best_offer and p.offer_insurance(*best_offer, gov_incentive=0.05 * gap_scale):
                    p.effective_cost = float(p.current_cost)
                    p.gov_subsidy = 0.0
                    offers[best_offer[2]].append(p)
                    # --- defensive append ---
                    predicted_risks.append(float(best_risk) if best_risk is not None else float('nan'))
                    actual_risks.append(float(p.actual_risk))
                    # if switched, remove from old insurer book
                    if (old_insurer is not None) and (old_insurer != best_offer[2]):
                        try:
                            offers[old_insurer].remove(p)
                        except ValueError:
                            pass

        # 2) Pre-voucher stats & training
        insured_pct_pre_voucher = sum(p.has_insurance for p in self.patients) / len(self.patients)
        round_stats: Dict = {
            'round': round_id,
            'insurer_stats': {},
            'insured_pct': insured_pct_pre_voucher,
            'insured_pct_pre_voucher': insured_pct_pre_voucher,
        }
        coverage_gap = max(0.0, self.coverage_target - insured_pct_pre_voucher)
        gap_scale = 1.0 + self.coverage_gap_beta * (coverage_gap / max(self.coverage_target, 1e-6))
        gap_scale = gap_scale * gap_scale  # amplify incentives when coverage is far below target
        raw_profits: Dict[int, float] = {}
        client_sizes_for_fl: Dict[int, int] = {ins.id: len(offers[ins.id]) for ins in self.insurers}

        for ins in self.insurers:
            accepted = offers[ins.id]
            lambda_gov = min(0.20, 0.05 * gap_scale)  # stronger alignment when coverage is low
            losses = [ins.train_on_patient(p, governor=self.governor, lambda_gov=lambda_gov) for p in accepted]
            total_prem = float(sum(p.current_cost for p in accepted))
            total_claims = float(sum(p.claim_amount() for p in accepted))
            profit = total_prem - total_claims
            raw_profits[ins.id] = max(0.0, profit)
            round_stats['insurer_stats'][ins.id] = {
                'market_share': len(accepted) / len(self.patients),
                'avg_price': float(np.mean([p.current_cost for p in accepted])) if accepted else 0.0,
                'avg_coverage': float(np.mean([p.current_coverage for p in accepted])) if accepted else 0.0,
                'loss': float(np.mean(losses)) if losses else 0.0,
                'premium_pre_voucher': total_prem,
                'claims_pre_voucher': total_claims,
                'profit_pre_voucher': profit,
                'loss_ratio_pre_voucher': float((total_claims / total_prem) if total_prem > 1e-9 else 0.0),
            }

        # 3) Build per-patient offers (no noise) for UNINSURED patients
        uninsured_idx: List[int] = []
        offers_by_patient: Dict[int, Dict[int, Tuple[float, float]]] = {}
        cached_offers_detail: Dict[int, Dict[int, Tuple[float, float, float]]] = {}
        if not lock_in_active:
            uninsured_idx = [i for i, p in enumerate(self.patients) if not p.has_insurance]

            for i in uninsured_idx:
                p = self.patients[i]
                x = patient_obs_tensor(p, history_len=getattr(self, "history_len", 10))
                offers_by_patient[i] = {}
                cached_offers_detail[i] = {}

                for ins in self.insurers:
                    price, cov, rhat = ins.offer_to_patient(x, add_noise=False)

                    # governor sees (price, coverage)
                    offers_by_patient[i][ins.id] = (float(price), float(cov))

                    # market still keeps (price, coverage, risk_hat) for later use
                    cached_offers_detail[i][ins.id] = (
                        float(price),
                        float(cov),
                        float(rhat),
                    )
        # 4) Pool & per-patient voucher allocation (give voucher to lowest-price offer per patient)
        if not lock_in_active:
            profit_pool = float(self.cash_back_rate * sum(raw_profits.values()))
            gap = coverage_gap
            floor_pool = float(self.min_cash_back * (gap / max(self.coverage_target, 1e-6)))
            total_cash_back = float((profit_pool + floor_pool) * gap_scale)
            per_patient_voucher = self.governor.allocate_cash_back_per_patient(
                self.insurers,
                self.patients,
                cash_back_pool=total_cash_back,
                max_cash_per_patient=50000.0,
                offers_by_patient=offers_by_patient
            )
        else:
            total_cash_back = 0.0
            per_patient_voucher = {}
        self.cash_back_history.append({
            'basis': 'per_patient_lowest_price',
            'per_patient_voucher': {k: (v[0], float(v[1])) for k, v in per_patient_voucher.items()},
            'total_pool': total_cash_back,
            'components': {
                'profit_pool': float(self.cash_back_rate * sum(raw_profits.values())) if not lock_in_active else 0.0,
                'floor_pool': float(self.min_cash_back * (coverage_gap / max(self.coverage_target, 1e-6))) if not lock_in_active else 0.0,
                'gap_scale': float(gap_scale)
            }
        })

        # 5) Second-chance acceptances with vouchers
        if not lock_in_active:
            for i in uninsured_idx:
                p = self.patients[i]
                tpl = per_patient_voucher.get(i)
                if tpl is None:
                    continue

                j_star, voucher = tpl
                if j_star == -1 or voucher <= 0.0:
                    continue
                if j_star not in cached_offers_detail[i]:
                    continue

                # Cached nominal offer from insurer j_star to patient i
                price, cov, rhat = cached_offers_detail[i][j_star]
                price = float(price)
                cov   = float(cov)
                rhat  = float(rhat)

                # Effective price after voucher (what the patient actually feels)
                eff_price = max(price - voucher, 0.0)

                v = patient_value_per_coverage(p)
                util = v * cov - eff_price
                # --- build obs_vec (normalized features) ---
                obs_vals = p.get_temporal_observable(history_len=getattr(self, "history_len", 10))
                obs_vec = torch.tensor(obs_vals, dtype=torch.float32)
                gov_score_ij = self.governor.score_offer(
                    obs_vec=obs_vec,
                    risk=rhat,                # predicted risk from insurer j*
                    coverage=cov,
                    effective_price=eff_price,
                    voucher=voucher
                )
                # Use effective price for acceptance logic
                gov_incentive = 0.3 * gap_scale * np.tanh(gov_score_ij)   # stronger incentive when coverage is low
                accepted = p.offer_insurance(eff_price, cov, j_star, forced=False, gov_incentive=gov_incentive)
                if not accepted:
                    continue

                # Overwrite fields to distinguish nominal vs effective for accounting
                # - current_cost: nominal premium billed to insurer accounts
                # - effective_cost: what the patient actually pays
                # - gov_subsidy: what the governor pays
                p.current_cost   = float(price)
                p.effective_cost = float(eff_price)
                p.gov_subsidy    = float(min(voucher, price))

                offers[j_star].append(p)
                predicted_risks.append(float(rhat))
                actual_risks.append(float(p.actual_risk))
                client_sizes_for_fl[j_star] = client_sizes_for_fl.get(j_star, 0) + 1

        # Recompute insured percentage after voucher-stage acceptances
        insured_pct_post_voucher = sum(p.has_insurance for p in self.patients) / len(self.patients)
        round_stats['insured_pct_post_voucher'] = insured_pct_post_voucher
        round_stats['insured_pct'] = insured_pct_post_voucher

        # 5b) Forced enrollment toward target (coverage mandate) if enabled and not in lock-in
        forced_additions = 0
        if self.enforce_target and (not lock_in_active) and (insured_pct_post_voucher < self.coverage_target):
            remaining_uninsured = [i for i, p in enumerate(self.patients) if not p.has_insurance]
            insured_count = sum(p.has_insurance for p in self.patients)
            target_count = int(np.ceil(self.coverage_target * len(self.patients)))
            needed = max(0, target_count - insured_count)
            if needed > 0:
                # cap forced additions per round to allow dynamics (proportional to gap and population)
                allowed_force = min(
                    needed,
                    max(1, int(np.ceil(self.force_step_pct * len(self.patients) * (1.0 + coverage_gap)))))
            # collect cheapest offers across insurers for each uninsured
            candidate_rows: List[Tuple[float, int, int]] = []  # (price, patient_idx, insurer_id)
            for i in remaining_uninsured:
                offers_i = cached_offers_detail.get(i, {})
                if not offers_i:
                    continue
                j_star, tpl = min(offers_i.items(), key=lambda kv: kv[1][0])
                candidate_rows.append((float(tpl[0]), i, int(j_star)))
            candidate_rows.sort(key=lambda t: t[0])
            for price, i, j_star in candidate_rows[:allowed_force]:
                if j_star not in cached_offers_detail.get(i, {}):
                    continue
                price, cov, rhat = cached_offers_detail[i][j_star]
                voucher = min(self.force_subsidy_cap, float(price))
                eff_price = max(float(price) - voucher, 0.0)
                p = self.patients[i]
                accepted = p.offer_insurance(eff_price, cov, j_star, forced=True, gov_incentive=0.0)
                if not accepted:
                    continue
                p.current_cost = float(price)
                p.effective_cost = float(eff_price)
                p.gov_subsidy = float(voucher)
                offers[j_star].append(p)
                predicted_risks.append(float(rhat))
                actual_risks.append(float(p.actual_risk))
                client_sizes_for_fl[j_star] = client_sizes_for_fl.get(j_star, 0) + 1
                forced_additions += 1

            insured_pct_post_force = sum(p.has_insurance for p in self.patients) / len(self.patients)
            round_stats['insured_pct_post_force'] = insured_pct_post_force
            round_stats['insured_pct'] = insured_pct_post_force
            round_stats['forced_enrollments'] = forced_additions
        # 6) Patient outcomes
        accepted_prices_nom = [float(p.current_cost) for p in self.patients if p.has_insurance]
        accepted_prices_eff = [float(p.effective_cost) for p in self.patients if p.has_insurance]
        accepted_coverages  = [float(p.current_coverage) for p in self.patients if p.has_insurance]
        total_gov_subsidy   = float(sum(p.gov_subsidy for p in self.patients if p.has_insurance))

        round_stats['patient_outcomes'] = {
            'accepted_prices': accepted_prices_nom,
            'accepted_prices_effective': accepted_prices_eff,
            'accepted_coverages': accepted_coverages,
            'actual_risks': np.array(actual_risks, dtype=np.float32),
            'predicted_risks': np.array(predicted_risks, dtype=np.float32),
            'gov_subsidy_total': total_gov_subsidy,
        }

        # 7) Final stats AFTER voucher-stage acceptances
        for ins in self.insurers:
            accepted_final = offers[ins.id]
            total_prem = float(sum(p.current_cost for p in accepted_final))
            total_claims = float(sum(p.claim_amount() for p in accepted_final))
            profit = total_prem - total_claims
            round_stats['insurer_stats'][ins.id].update({
                'market_share_final': len(accepted_final) / len(self.patients),
                'premium_final': total_prem,
                'claims_final': total_claims,
                'profit_final': profit,
                'loss_ratio_final': float((total_claims / total_prem) if total_prem > 1e-9 else 0.0),
            })

        # 8) Governor update (effective-price-aware)
        self.governor.update_insurers(
            self.insurers,
            self.patients,
            kappa=10.0,
            price_floor=100.0,
            per_patient_voucher=per_patient_voucher
        )
        corr = float(getattr(self.governor, 'mean_corr', getattr(self.governor, 'mean_r2', 0.0)))
        corr_pos = 0.5 * (corr + 1.0)  # map [-1,1] to [0,1]
        align_cov = float(round_stats.get('insured_pct', insured_pct_post_voucher))
        round_stats['goal_score'] = float(0.5 * corr_pos + 0.5 * align_cov)
        round_stats['governor_loss'] = float(getattr(self.governor, 'last_loss', 0.0))

        # 9) Cross-learning
        for ins in self.insurers:
            others = [o for o in self.insurers if o.id != ins.id]
            ins.learn_from_others(others)

        # 10) DP-FL (risk only) every K rounds
        if ((round_id + 1) % self.fl_every_rounds) == 0:
            self.governor.federated_round_risk(
                insurers=self.insurers,
                client_sizes=client_sizes_for_fl,
                dp_sigma=self.dp_sigma,
                clip_C=self.clip_C
            )
            round_stats['fl_aggregated'] = True
        else:
            round_stats['fl_aggregated'] = False

        self.history.append(round_stats)
        self.round_id += 1
        return round_stats
