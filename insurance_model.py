# insurance_model.py
from typing import Optional, List, Dict
import numpy as np
import torch
import torch.nn as nn

# Keep consistent with your Patient.get_observable_info() order
FEATURE_KEYS = ['age', 'income', 'credit_score', 'health_status', 'location']

__all__ = [
    "FEATURE_KEYS",
    "Insurance",
    "_cpu_state_dict",
    "_load_state_dict",
]

def _cpu_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

def _load_state_dict(model: nn.Module, sd: Dict[str, torch.Tensor]) -> None:
    model.load_state_dict(sd, strict=True)

class Insurance:
    def __init__(
        self,
        id,
        input_dim: Optional[int] = None,
        history_len: int = 10,
        lr: float = 0.01,
        device: Optional[torch.device] = None,
        max_offer_history: int = 10,
        # --- Heterogeneity knobs (defaults ≈ original behavior) ---
        margin_price: float = 0.0,         # φ_p : baseline markup/discount on price
        margin_cov: float = 0.0,           # φ_c : baseline shift on coverage
        risk_load_price: float = 0.0,      # k_p : >0 risk-averse; <0 risk-seeking
        risk_load_cov: float = 0.0,        # k_c : >0 risk-averse; <0 risk-seeking
        risk_anchor: float = 0.5,          # r0  : reference risk level
        temp_explore: float = 1.0,         # τ   : scales exploration noise
        imitation_weight: float = 1.0,     # ω   : strength of imitation learning
        price_min: float = 200.0,
        price_max: float = 120_000.0,
        coverage_min: float = 0.1,
        coverage_max: float = 1.0,
    ):
        self.id = id
        self.history_len = int(max(1, history_len))
        base_dim = int(input_dim) if input_dim is not None else len(FEATURE_KEYS)
        self.input_dim = base_dim * self.history_len
        self.device = device if device is not None else torch.device("cpu")
        self.max_offer_history = int(max_offer_history)

        # Heterogeneity parameters
        self.margin_price = float(margin_price)
        self.margin_cov = float(margin_cov)
        self.risk_load_price = float(risk_load_price)
        self.risk_load_cov = float(risk_load_cov)
        self.risk_anchor = float(risk_anchor)
        self.temp_explore = float(temp_explore)
        self.imitation_weight = float(np.clip(imitation_weight, 0.0, None))
        self.price_min = float(price_min)
        self.price_max = float(price_max)
        self.coverage_min = float(coverage_min)
        self.coverage_max = float(coverage_max)

        # ----- Improved models for dynamic patient processes -----
        hidden_risk = 32
        hidden_cov_price = 48
        dropout_p = 0.15  # helps generalization under non-stationary distributions

        # Risk model — deeper nonlinear mapping with dropout
        self.risk_model = nn.Sequential(
            nn.Linear(self.input_dim, hidden_risk),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(hidden_risk, hidden_risk),
            nn.ReLU(),
            nn.Linear(hidden_risk, 1), # may guess it to be negative for it self
        ).to(self.device)

        # Coverage model — conditioned on predicted risk
        self.coverage_model = nn.Sequential(
            nn.Linear(self.input_dim + 1, hidden_cov_price),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(hidden_cov_price, hidden_cov_price),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(hidden_cov_price, hidden_cov_price),
            nn.ReLU(),
            nn.Linear(hidden_cov_price, 1),
        ).to(self.device)

        # Price model — conditioned on (x, risk, coverage), uses Softplus for strictly positive output
        self.price_model = nn.Sequential(
            nn.Linear(self.input_dim + 2, hidden_cov_price),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(hidden_cov_price, hidden_cov_price),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(hidden_cov_price, hidden_cov_price),
            nn.ReLU(),
            nn.Linear(hidden_cov_price, 1),
        ).to(self.device)

        # Joint optimizer for all networks
        self.optimizer = torch.optim.Adam(
            list(self.risk_model.parameters())
            + list(self.coverage_model.parameters())
            + list(self.price_model.parameters()),
            lr=lr,
            betas=(0.9, 0.999),
            weight_decay=1e-5   # mild L2 regularization for stability
        )

        # Loss: mean-squared error (shared for simplicity)
        self.loss_fn = nn.MSELoss(reduction='mean')
        self.offer_history: List[Dict] = []  # used by imitation

    # -------- convenience: quick presets for attitudes --------
    def set_attitude_profile(self, preset: str):
        """
        Quick presets for heterogeneity.
        'averse'  : higher price for risky, lower coverage for risky
        'neutral' : close to original behavior
        'seeking' : lower price for risky, higher coverage for risky
        """
        p = preset.lower()
        if p in ('averse', 'risk_averse'):
            self.margin_price = 0.05
            self.margin_cov = -0.05
            self.risk_load_price = 0.8
            self.risk_load_cov = 0.8
            self.temp_explore = 0.8
            self.imitation_weight = 0.6
        elif p in ('neutral', 'risk_neutral'):
            self.margin_price = 0.0
            self.margin_cov = 0.0
            self.risk_load_price = 0.0
            self.risk_load_cov = 0.0
            self.temp_explore = 1.0
            self.imitation_weight = 1.0
        elif p in ('seeking', 'risk_seeking'):
            self.margin_price = -0.05
            self.margin_cov = 0.05
            self.risk_load_price = -0.6
            self.risk_load_cov = -0.6
            self.temp_explore = 1.2
            self.imitation_weight = 0.8
        else:
            raise ValueError(f"Unknown preset: {preset}")
        return self

    # ----- FL helpers: risk-only -----
    def get_risk_weights(self) -> Dict[str, torch.Tensor]:
        return _cpu_state_dict(self.risk_model)

    def set_risk_weights(self, sd: Dict[str, torch.Tensor]) -> None:
        _load_state_dict(self.risk_model, sd)

    # ----- Offers & training -----
    def random_initial_offer(self, patient) -> (float, float, float):
        coverage = np.clip(np.random.beta(2, 2), self.coverage_min, self.coverage_max) # roughly centered
        price = np.clip(
            patient.maximum_insurance_spending * np.random.uniform(0.5, 1.0),
            self.price_min, self.price_max
        ) # random between 50%-100% of max willing to pay
        risk = patient.actual_risk
        return float(price), float(coverage), float(risk)

    def _apply_attitude_transforms(self, price_t: torch.Tensor, cov_t: torch.Tensor, risk_t: torch.Tensor):
        """
        Post-network transforms to induce heterogeneity:
          p' = clip((1+φ_p)*p + k_p*(r - r0))
          c' = clip((1+φ_c)*c - k_c*(r - r0))
        """
        r_centered = risk_t - self.risk_anchor
        price_adj = (1.0 + self.margin_price) * price_t + self.risk_load_price * r_centered
        cov_adj   = (1.0 + self.margin_cov)   * cov_t   - self.risk_load_cov  * r_centered
        price_adj = torch.clamp(price_adj, self.price_min, self.price_max)
        cov_adj   = torch.clamp(cov_adj,   self.coverage_min, self.coverage_max)
        return price_adj, cov_adj

    @torch.no_grad()
    def offer_to_patient(self, patient_obs_tensor: torch.Tensor, add_noise: bool = True):
        x = patient_obs_tensor.to(self.device)
        risk = self.risk_model(x)
        xr = torch.cat((x, risk), dim=1)
        cov = self.coverage_model(xr)
        xrc = torch.cat((xr, cov), dim=1)
        price = self.price_model(xrc)

        # --- apply heterogeneity ---
        price, cov = self._apply_attitude_transforms(price, cov, risk)

        # exploration (scaled by per-insurer temperature τ)
        if add_noise:
            if self.temp_explore != 1.0:
                price = price + torch.randn_like(price) * 0.05 * self.temp_explore * (price + 1e-6)
                cov   = cov   + torch.randn_like(cov)   * 0.05 * self.temp_explore
            else:
                price = price + torch.randn_like(price) * 0.05 * (price + 1e-6)
                cov   = cov   + torch.randn_like(cov)   * 0.05

        # bounds after noise
        price = torch.clamp(price, self.price_min, self.price_max)
        cov   = torch.clamp(cov,   self.coverage_min, self.coverage_max)

        # save for imitation (cap history so it doesn't grow forever)
        self.offer_history.append({
            'obs': x.detach().cpu().numpy().flatten(),
            'risk': float(risk.item()),
            'coverage': float(cov.item()),
            'price': float(price.item()),
        })
        if len(self.offer_history) > self.max_offer_history:
            self.offer_history = self.offer_history[-self.max_offer_history:]

        return float(price.item()), float(cov.item()), float(risk.item())

    def train_on_patient(self, patient, governor=None, lambda_gov: float = 0.0):
        """
        Supervised proxy updates on realized outcomes (risk, coverage, price).

        If a Governor instance is passed with lambda_gov > 0, we add an extra
        regularization term that encourages this insurer's predicted
        (risk, coverage, price) triple to receive a high score from the governor.
        """
        self.optimizer.zero_grad()

        # -------- base supervised losses (as before) --------
        if hasattr(patient, "get_temporal_observable"):
            obs_vec = patient.get_temporal_observable(history_len=self.history_len)
        else:
            obs_vec = [patient.get_observable_info()[k] for k in FEATURE_KEYS]
        obs = torch.tensor(obs_vec, dtype=torch.float32, device=self.device).view(1, -1)

        # risk
        target_risk = torch.tensor([[patient.actual_risk]],
                                   dtype=torch.float32, device=self.device)
        risk_pred = self.risk_model(obs)
        loss_risk = self.loss_fn(risk_pred, target_risk)

        # coverage
        xr = torch.cat((obs, target_risk), dim=1)
        target_cov = torch.tensor([[patient.current_coverage]],
                                  dtype=torch.float32, device=self.device)
        cov_pred = self.coverage_model(xr)
        loss_cov = self.loss_fn(cov_pred, target_cov)

        # price
        xrc = torch.cat((xr, target_cov), dim=1)
        target_price = torch.tensor([[patient.current_cost]],
                                    dtype=torch.float32, device=self.device)
        price_pred = self.price_model(xrc)
        loss_price = self.loss_fn(price_pred, target_price)

        total_loss = loss_risk + loss_cov + loss_price

        # -------- optional governor-alignment term --------
        if (governor is not None) and (lambda_gov > 0.0):
            # voucher used for this patient (0 if field not present)
            voucher_scalar = float(getattr(patient, "gov_subsidy", 0.0))
            v_t = torch.tensor([[voucher_scalar]],
                               dtype=torch.float32, device=self.device)

            # effective price according to current prediction
            eff_price_pred = torch.clamp(price_pred - v_t, min=0.0)

            # governor network (assumed already on the right device)
            gov_net = governor.global_goal

            # input matches what the governor sees during its own training:
            # [obs, risk, coverage, effective_price, voucher]
            gov_input = torch.cat([obs, risk_pred, cov_pred, eff_price_pred, v_t],
                                  dim=1)

            # higher score is better, so we *minimize* negative score
            gov_score_pred = gov_net(gov_input)          # shape [1,1]
            loss_gov = -gov_score_pred.mean()

            total_loss = total_loss + lambda_gov * loss_gov

        # -------- backprop + step --------
        total_loss.backward()
        self.optimizer.step()
        return float(total_loss.detach().cpu().item())
    
    def learn_from_others(self, other_insurers: List['Insurance']) -> None:
        """Imitate last ~10 offers of others; strength scaled by imitation_weight (ω)."""
        "Mean-Field style imitation learning."
        if self.imitation_weight <= 0.0:
            return
        for other in other_insurers:
            if other.id == self.id:
                continue
            # making it mean field, so average per all accepted offers
            if not other.offer_history:
                continue
            obs_list = []
            risk_list = []
            cov_list = []
            price_list = []
            for offer in other.offer_history:
                obs_list.append(torch.tensor(offer['obs'], dtype=torch.float32, device=self.device).view(1, -1))
                risk_list.append(torch.tensor([[offer['risk']]], dtype=torch.float32, device=self.device))
                cov_list.append(torch.tensor([[offer['coverage']]], dtype=torch.float32, device=self.device))
                price_list.append(torch.tensor([[offer['price']]], dtype=torch.float32, device=self.device))

            obs = torch.stack(obs_list, dim=0).mean(dim=0)
            target_risk = torch.stack(risk_list, dim=0).mean(dim=0)
            target_cov = torch.stack(cov_list, dim=0).mean(dim=0)
            target_price = torch.stack(price_list, dim=0).mean(dim=0)

            risk_pred = self.risk_model(obs)
            loss_r = self.loss_fn(risk_pred, target_risk)

            cov_pred = self.coverage_model(torch.cat((obs, target_risk), dim=1))
            loss_c = self.loss_fn(cov_pred, target_cov)

            price_pred = self.price_model(torch.cat((obs, target_risk, target_cov), dim=1))
            loss_p = self.loss_fn(price_pred, target_price)

            total = self.imitation_weight * (loss_r + loss_c + loss_p)
            self.optimizer.zero_grad()
            total.backward()
            self.optimizer.step()
