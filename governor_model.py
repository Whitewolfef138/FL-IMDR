# governor_model.py
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn

# Keep key order consistent with Patient.get_observable_info()
FEATURE_KEYS = ['age', 'income', 'credit_score', 'health_status', 'location']

__all__ = [
    "FEATURE_KEYS",
    "Governor",
    "_avg_state_dict",
    "_clip_update",
    "_add_gaussian_noise",
]

# ----------------- FedAvg helpers -----------------
def _avg_state_dict(state_dicts: List[Dict[str, torch.Tensor]],
                    weights: Optional[List[float]] = None) -> Dict[str, torch.Tensor]:
    """Weighted average of PyTorch state_dicts (same keys)."""
    if not state_dicts:
        raise ValueError("state_dicts must be a non-empty list.")
    keys = list(state_dicts[0].keys())
    out = {k: torch.zeros_like(state_dicts[0][k]) for k in keys}
    if weights is None:
        w = 1.0 / len(state_dicts)
        weights = [w] * len(state_dicts)
    for sd, alpha in zip(state_dicts, weights):
        for k in keys:
            out[k] += alpha * sd[k]
    return out

def _clip_update(sd: Dict[str, torch.Tensor],
                 base_sd: Dict[str, torch.Tensor],
                 C: float) -> Dict[str, torch.Tensor]:
    """Clip client update (sd - base_sd) to L2 norm C, return clipped weights."""
    if C is None or C <= 0:
        return sd
    flat = []
    for k in sd.keys():
        flat.append((sd[k] - base_sd[k]).reshape(-1))
    upd = torch.cat(flat, dim=0)
    norm = torch.norm(upd) + 1e-12
    if norm.item() <= C:
        return sd
    scale = C / norm
    out = {}
    idx = 0
    for k in sd.keys():
        n = sd[k].numel()
        delta = (upd[idx:idx+n] * scale).view_as(sd[k])
        out[k] = base_sd[k] + delta
        idx += n
    return out

def _add_gaussian_noise(sd: Dict[str, torch.Tensor], sigma: float) -> Dict[str, torch.Tensor]:
    """Add isotropic Gaussian noise to state dict (DP on server)."""
    if sigma is None or sigma <= 0:
        return sd
    noisy = {}
    for k, v in sd.items():
        noisy[k] = v + torch.randn_like(v) * sigma
    return noisy



# ----------------- Governor -----------------
class Governor:
    """
    Governor:
      • Learns a goal/utility score g_θ(obs, risk, cov, effective_price, voucher)
      • Allocates per-patient vouchers to the lowest-price insurer for each uninsured patient
      • Acts as FL server for risk-model FedAvg (optional DP)
    """
    def __init__(self,
                 feature_dim: Optional[int] = None,
                 history_len: int = 10,
                 hidden: int = 64,
                 lr: float = 1e-2,
                 device: Optional[torch.device] = None):
        # goal input: obs(5) + risk(1) + cov(1) + eff_price(1) + voucher(1) = feature_dim + 4
        self.device = device if device is not None else torch.device("cpu")
        self.history_len = int(max(1, history_len))
        base_step_dim = feature_dim if feature_dim is not None else len(FEATURE_KEYS)
        base_dim = base_step_dim * self.history_len
        self.input_dim = base_dim + 4
        self.feature_dim = base_dim

        self.global_goal = nn.Sequential(
            nn.Linear(self.input_dim, hidden),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(hidden, hidden*2),
            nn.ReLU(),
            nn.Linear(hidden*2, hidden),
            nn.Dropout(p=0.1),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        ).to(self.device)

        self.optimizer = torch.optim.Adam(self.global_goal.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss(reduction='mean')

        self.goal_scores: Dict[Tuple[int, int], float] = {}
        self.cash_back_history: List[Dict] = []
        self.last_loss: float = 0.0
        self.mean_r2: float = 0.0

        # FL server (risk-only)
        self.global_risk_weights: Optional[Dict[str, torch.Tensor]] = None

    # --------- Scoring ----------
    @torch.no_grad()
    def score_offer(self,
                    obs_vec,
                    risk: float,
                    coverage: float,
                    effective_price: float,
                    voucher: float = 0.0) -> float:
        """Score a hypothetical offer using the SAME inputs as training."""
        if not torch.is_tensor(obs_vec):
            obs_vec = torch.tensor(obs_vec, dtype=torch.float32)
        if obs_vec.ndim == 1:
            obs_vec = obs_vec.view(1, -1)
        obs_vec = obs_vec.to(self.device)

        def _to_1x1(z):
            if torch.is_tensor(z):
                z = z.to(self.device)
                return z.view(1, 1) if z.ndim == 0 else z
            return torch.tensor([[float(z)]], dtype=torch.float32, device=self.device)

        r = _to_1x1(risk)
        c = _to_1x1(coverage)
        ep = _to_1x1(effective_price)
        vo = _to_1x1(voucher)
        x = torch.cat([obs_vec, r, c, ep, vo], dim=1)
        s = self.global_goal(x)
        return float(s.detach().cpu().item())

    # --------- Training g_θ on realized outcomes ----------
    def update_insurers(self,
                        insurers,
                        patients,
                        kappa: float = 10.0,
                        price_floor: float = 100.0,
                        per_insurer_voucher: Optional[Dict[int, float]] = None,
                        per_patient_voucher: Optional[Dict[int, Tuple[int, float]]] = None) -> None:
        """
        Train g_θ on realized outcomes using a numerically stable target based on EFFECTIVE price.
        Inputs per insured patient: [obs(5), risk, cov, effective_price, voucher]
        Voucher precedence:
          - If per_patient_voucher is provided: use that (patient-indexed).
          - Else if per_insurer_voucher is provided: use insurer-level voucher.
          - Else: voucher = 0.
        Target: y = log(1 + kappa * cov / max(effective_price, price_floor))
        """
        self.goal_scores = {}
        self.optimizer.zero_grad()

        preds, targets, losses = [], [], []

        for p_idx, p in enumerate(patients):
            if not getattr(p, 'has_insurance', False):
                continue

            if hasattr(p, "get_temporal_observable"):
                obs_vals = p.get_temporal_observable(history_len=self.history_len)
            else:
                obs_vals = [p.get_observable_info()[k] for k in FEATURE_KEYS]
            obs = torch.tensor(obs_vals, dtype=torch.float32, device=self.device).view(1, -1)
            risk = torch.tensor([[float(p.actual_risk)]], dtype=torch.float32, device=self.device)
            cov  = torch.tensor([[float(p.current_coverage)]], dtype=torch.float32, device=self.device)
            price_nom = torch.tensor([[float(p.current_cost)]], dtype=torch.float32, device=self.device)

            # pick voucher
            voucher_val = 0.0
            if per_patient_voucher is not None and p_idx in per_patient_voucher:
                voucher_val = float(per_patient_voucher[p_idx][1])
            elif per_insurer_voucher is not None:
                voucher_val = float(per_insurer_voucher.get(p.insurer_id, 0.0))

            eff_price = torch.clamp(price_nom - voucher_val, min=0.0)
            x = torch.cat([
                obs, risk, cov,
                eff_price,
                torch.tensor([[voucher_val]], dtype=torch.float32, device=self.device)
            ], dim=1)

            # redesigned target: reward high coverage relative to want, and affordability
            wanted_cov = torch.tensor([[float(getattr(p, "wanted_coverage", 0.5))]],
                                      dtype=torch.float32, device=self.device)
            max_spend = torch.tensor([[float(getattr(p, "maximum_insurance_spending", price_floor))]],
                                     dtype=torch.float32, device=self.device)

            cov_ratio = torch.clamp(cov / torch.clamp(wanted_cov, min=1e-3), 0.0, 2.0)
            afford = torch.clamp(1.0 - eff_price / torch.clamp(max_spend, min=price_floor), -1.0, 1.0)
            # weighted blend: emphasize coverage but keep prices affordable
            target = 0.6 * cov_ratio + 0.4 * afford

            out = self.global_goal(x)
            loss = self.loss_fn(out, target) + 1e-4 * nn.L1Loss()(out, target)
            losses.append(loss)

            self.goal_scores[(int(p.insurer_id), int(p_idx))] = float(out.detach().cpu().item())
            preds.append(float(out.detach().cpu().item()))
            targets.append(float(target.detach().cpu().item()))

        if losses:
            total = torch.stack(losses, dim=0).mean()
            total.backward()
            self.optimizer.step()
            self.last_loss = float(total.detach().cpu().item())
        else:
            self.last_loss = 0.0

        if preds and targets:
            y = np.asarray(targets, dtype=np.float64)
            yhat = np.asarray(preds, dtype=np.float64)
            ss_res = np.sum((y - yhat) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2) + 1e-12
            r2 = 1.0 - ss_res / ss_tot
            # correlation is more stable and interpretable; use it for goal_score
            y_center = y - np.mean(y)
            yhat_center = yhat - np.mean(yhat)
            denom = (np.linalg.norm(y_center) * np.linalg.norm(yhat_center) + 1e-12)
            corr = float(np.dot(y_center, yhat_center) / denom)
            self.mean_r2 = float(r2)
            self.mean_corr = float(corr)
        else:
            self.mean_r2 = 0.0
            self.mean_corr = 0.0

    # --------- Offer book builder (no-noise price quotes) ----------
    @torch.no_grad()
    def build_offers_by_patient(self,
                                insurers,
                                patients) -> Dict[int, Dict[int, Tuple[float, float]]]:
        """
        Construct offers_by_patient for allocation:
          offers_by_patient[i][j] = (price, coverage) that insurer j would offer to patient i
        Uses add_noise=False to avoid stochastic tie-breaking.
        """
        offers_by_patient: Dict[int, Dict[int, Tuple[float, float]]] = {}
        for i, p in enumerate(patients):
            if hasattr(p, "get_temporal_observable"):
                obs_vals = p.get_temporal_observable(history_len=self.history_len)
            else:
                obs_vals = [p.get_observable_info()[k] for k in FEATURE_KEYS]
            x = torch.tensor(obs_vals, dtype=torch.float32, device=self.device).view(1, -1)
            offers_by_patient[i] = {}
            for ins in insurers:
                # risk, cov predicted; record both price and coverage for selection
                price, cov, rhat = ins.offer_to_patient(x, add_noise=False)
                offers_by_patient[i][int(ins.id)] = (float(price), float(cov))
        return offers_by_patient

    # ---- Per-patient voucher: assign to lowest-price insurer for each uninsured patient ----

    def allocate_cash_back_per_patient(self,
                                       insurers,
                                       patients,
                                       cash_back_pool: float,
                                       max_cash_per_patient: float,
                                       offers_by_patient: Dict[int, Dict[int, Tuple[float, float]]]
                                       ) -> Dict[int, Tuple[int, float]]:
        """
        Inputs
        ------
        - offers_by_patient[i][j] = (price_ij, cov_ij) offered by insurer j to patient i
          (built in the market or via build_offers_by_patient with add_noise=False).

        Returns
        -------
        - per_patient_voucher: {i: (j_star, voucher_amount)} where j_star is the chosen insurer.

        Policy
        ------
        - Consider only offers with coverage >= wanted_coverage_i as 'feasible'.
        - Among feasible offers, pick the one with minimum price.
        - If no feasible offers, pick minimum price among all offers.
        - If there are U uninsured patients, base voucher per patient is
              v0 = min(max_cash_per_patient, cash_back_pool / U).
        - Iterate patients in some order; if pool remains, give v = min(v0, remaining_pool),
          otherwise give 0.
        """
        uninsured_idx = [
            i for i, p in enumerate(patients)
            if not getattr(p, 'has_insurance', False)
        ]
        U = len(uninsured_idx)

        per_patient_voucher: Dict[int, Tuple[int, float]] = {}
        if U == 0 or cash_back_pool <= 0.0:
            self.cash_back_history.append({
                'total_pool': float(cash_back_pool),
                'basis': 'per_patient_lowest_price_cov',
                'per_patient': {}
            })
            return per_patient_voucher

        v0 = min(max_cash_per_patient, cash_back_pool / U if U > 0 else 0.0)
        remaining = float(cash_back_pool)
        log_map: Dict[int, Dict[str, float]] = {}

        for i in uninsured_idx:
            p = patients[i]
            offers_i = offers_by_patient.get(i, {})

            if not offers_i:
                # no insurer offered anything to this patient
                per_patient_voucher[i] = (-1, 0.0)
                log_map[i] = {
                    'winner': -1,
                    'voucher': 0.0,
                    'reason': 'no_offers'
                }
                continue

            # Split offers into feasible (good coverage) and all
            feasible: List[Tuple[int, float, float]] = []
            all_offers: List[Tuple[int, float, float]] = []
            wanted_cov_i = float(getattr(p, 'wanted_coverage', 0.0))

            for j, (price_ij, cov_ij) in offers_i.items():
                j_int = int(j)
                price_f = float(price_ij)
                cov_f = float(cov_ij)
                all_offers.append((j_int, price_f, cov_f))
                if cov_f >= wanted_cov_i:
                    feasible.append((j_int, price_f, cov_f))

            if feasible:
                # among coverage-feasible offers, pick lowest price
                j_star, _, _ = min(feasible, key=lambda t: t[1])
                base_reason = 'min_price_coverage_ok'
            else:
                # fallback: lowest price among *all* offers
                j_star, _, _ = min(all_offers, key=lambda t: t[1])
                base_reason = 'min_price_any_coverage'

            if remaining <= 1e-12:
                per_patient_voucher[i] = (j_star, 0.0)
                log_map[i] = {
                    'winner': j_star,
                    'voucher': 0.0,
                    'reason': base_reason + '_budget_depleted'
                }
                continue

            v = float(min(v0, remaining))
            per_patient_voucher[i] = (j_star, v)
            remaining -= v
            log_map[i] = {
                'winner': j_star,
                'voucher': v,
                'reason': base_reason
            }

        self.cash_back_history.append({
            'total_pool': float(cash_back_pool),
            'basis': 'per_patient_lowest_price_cov',
            'per_patient': log_map,
            'v0': float(v0),
            'remaining_after_allocation': float(remaining)
        })
        return per_patient_voucher
    # ---------- DP-FL server: risk-only, FedAvg every K rounds ----------
    def federated_round_risk(self,
                             insurers: List['Insurance'],
                             client_sizes: Dict[int, int],
                             dp_sigma: float = 0.05,
                             clip_C: Optional[float] = 1.0) -> None:
        """
        Risk-only FedAvg with optional clipping + Gaussian noise (server-side DP).
        Shares ONLY risk-model weights.
        """
        client_sds, weights = [], []
        for ins in insurers:
            sd = ins.get_risk_weights()
            client_sds.append(sd)
            n = max(1, int(client_sizes.get(ins.id, 1)))
            weights.append(n)
        wsum = float(sum(weights))
        weights = [w / wsum for w in weights]

        # initialize global if needed
        if self.global_risk_weights is None:
            base = _avg_state_dict(client_sds, weights=weights)
            self.global_risk_weights = {k: v.clone() for k, v in base.items()}
        base = self.global_risk_weights

        # clip client deltas wrt base
        if clip_C is not None:
            client_sds = [_clip_update(sd, base, C=clip_C) for sd in client_sds]

        # FedAvg
        agg = _avg_state_dict(client_sds, weights=weights)

        # DP noise on server (optional)
        if dp_sigma and dp_sigma > 0.0:
            agg = _add_gaussian_noise(agg, sigma=dp_sigma)

        # update/broadcast
        self.global_risk_weights = {k: v.clone() for k, v in agg.items()}
        for ins in insurers:
            ins.set_risk_weights(self.global_risk_weights)
    
