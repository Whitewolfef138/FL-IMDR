from dataclasses import dataclass


@dataclass(frozen=True)
class MethodSpec:
    key: str
    label: str
    short_label: str
    learning: str
    regulator: bool = False
    federation: bool = False
    dp_noise: float = 0.0
    fairness_blind: bool = False
    encrypted_aggregation: bool = False
    citation_key: str = ""
    latex_name: str = ""


METHODS = (
    MethodSpec(
        "dong_quan_automl",
        "Dong–Quan AutoML-aligned (2025)",
        "Dong–Quan",
        learning="static",
        citation_key="DongQuan2025",
        latex_name=r"Dong--Quan AutoML-aligned~\cite{DongQuan2025}",
    ),
    MethodSpec(
        "richman_credibility",
        "Richman et al. credibility-aligned (2025)",
        "Richman et al.",
        learning="credibility",
        citation_key="Richman2025",
        latex_name=r"Richman \emph{et al.} credibility-aligned~\cite{Richman2025}",
    ),
    MethodSpec(
        "piontkowski_small_portfolio",
        "Piontkowski small-portfolio-aligned (2025)",
        "Piontkowski",
        learning="small_portfolio",
        citation_key="Piontkowski2025",
        latex_name=r"Piontkowski small-portfolio-aligned~\cite{Piontkowski2025}",
    ),
    MethodSpec(
        "xin_fairness",
        "Xin et al. fairness-aligned (2025)",
        "Xin et al.",
        learning="static",
        fairness_blind=True,
        citation_key="Xin2025",
        latex_name=r"Xin \emph{et al.} fairness-aligned~\cite{Xin2025}",
    ),
    MethodSpec(
        "sun_fhe_crl",
        "Sun et al. FHE-CRL-aligned (2026)",
        "Sun et al.",
        learning="online",
        federation=True,
        encrypted_aggregation=True,
        citation_key="Sun2026",
        latex_name=r"Sun \emph{et al.} FHE--CRL-aligned~\cite{Sun2026}",
    ),
    MethodSpec(
        "fl_imdr",
        "FL-IMDR (proposed)",
        "FL-IMDR",
        learning="online",
        regulator=True,
        federation=True,
        dp_noise=0.012,
        latex_name=r"\textbf{FL-IMDR (proposed)}",
    ),
)


@dataclass
class SimulationConfig:
    n_patients: int = 100
    n_insurers: int = 4
    months: int = 120
    steady_window: int = 10
    federation_interval: int = 10
    lock_in_months: int = 5
    target_coverage: float = 0.98
    tax_rate: float = 0.10
    ridge: float = 1.0
    credibility_k: float = 35.0
    dp_clip: float = 0.12
    max_voucher: float = 180.0
    min_price: float = 55.0
    random_seed_offset: int = 104729

    @property
    def n_features(self) -> int:
        return 10
