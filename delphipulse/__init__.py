"""DelphiPulse - Autonomous Evidence Mining & LMSR Information-Gain Engine for Delphi / Gensyn.

Track 4797: Evidence Mining Agents.
"""

from delphipulse.config import (
    COMPETITION_PROFILE,
    MAINNET_PROFILE,
    DelphiPulseConfig,
    NetworkProfile,
)
from delphipulse.lmsr import (
    brier_score_improvement,
    estimate_evidence_bounty,
    kl_divergence_bernoulli,
    lmsr_cost,
    lmsr_max_loss,
    lmsr_prices,
    lmsr_trade_cost,
    posterior_probability,
)

__version__ = "1.0.0"

__all__ = [
    "COMPETITION_PROFILE",
    "MAINNET_PROFILE",
    "DelphiPulseConfig",
    "NetworkProfile",
    "lmsr_cost",
    "lmsr_prices",
    "lmsr_trade_cost",
    "lmsr_max_loss",
    "posterior_probability",
    "kl_divergence_bernoulli",
    "brier_score_improvement",
    "estimate_evidence_bounty",
]
