"""Hanson LMSR & Bayesian Information-Gain Mathematical Engine.

Implements numerically stable Logarithmic Market Scoring Rule (LMSR)
cost functions, exact pricing, trade calculations, and Bayesian
evidence shift metrics for prediction market automated market makers.
"""

from __future__ import annotations

import math
from typing import Sequence


def lmsr_cost(quantities: Sequence[float], b: float) -> float:
    """Calculate the total LMSR cost for outstanding shares: C(q) = b * ln(sum_i exp(q_i / b)).

    Uses numerically stable log-sum-exp trick:
        m = max_i(q_i / b)
        C(q) = b * (m + ln(sum_i exp(q_i / b - m)))

    Args:
        quantities: Sequence of outstanding share quantities for each outcome.
        b: Liquidity parameter (b > 0).

    Returns:
        Total cost in market collateral units (e.g., USDC.e).

    Raises:
        ValueError: If b <= 0 or quantities is empty.
    """
    if b <= 0:
        raise ValueError(f"LMSR liquidity parameter 'b' must be > 0, got {b}")
    if not quantities:
        raise ValueError("Quantities sequence cannot be empty")

    scaled = [q / b for q in quantities]
    m = max(scaled)
    sum_exp = sum(math.exp(s - m) for s in scaled)
    return b * (m + math.log(sum_exp))


def lmsr_prices(quantities: Sequence[float], b: float) -> list[float]:
    """Calculate instantaneous prices (implied probabilities) for all outcomes.

    P_i = exp(q_i / b) / sum_j exp(q_j / b)
    Guaranteed sum == 1.0 within 1e-12, overflow-safe even for large q_i.

    Args:
        quantities: Outstanding shares for each outcome.
        b: Liquidity parameter (b > 0).

    Returns:
        List of instantaneous prices summing to 1.0.
    """
    if b <= 0:
        raise ValueError(f"LMSR liquidity parameter 'b' must be > 0, got {b}")
    if not quantities:
        raise ValueError("Quantities sequence cannot be empty")

    scaled = [q / b for q in quantities]
    m = max(scaled)
    exps = [math.exp(s - m) for s in scaled]
    sum_exps = sum(exps)
    if sum_exps <= 0:
        uniform = 1.0 / len(quantities)
        return [uniform for _ in quantities]

    prices = [e / sum_exps for e in exps]
    # Normalize strictly to ensure precision sum == 1.0
    total = sum(prices)
    return [p / total for p in prices]


def lmsr_trade_cost(
    quantities: Sequence[float],
    b: float,
    outcome_idx: int,
    delta_shares: float,
) -> float:
    """Calculate the cost to purchase (or sell if negative) delta_shares of outcome_idx.

    Cost = C(q + delta) - C(q)

    Args:
        quantities: Current outstanding shares.
        b: Liquidity parameter.
        outcome_idx: Index of the outcome traded (0-indexed).
        delta_shares: Shares bought (>0) or sold (<0).

    Returns:
        Cost difference. Positive means trader pays, negative means trader receives collateral.
    """
    if outcome_idx < 0 or outcome_idx >= len(quantities):
        raise IndexError(f"Outcome index {outcome_idx} out of range [0, {len(quantities) - 1}]")

    initial_cost = lmsr_cost(quantities, b)
    new_quantities = list(quantities)
    new_quantities[outcome_idx] += delta_shares
    new_cost = lmsr_cost(new_quantities, b)
    return new_cost - initial_cost


def lmsr_max_loss(num_outcomes: int, b: float) -> float:
    """Maximum possible subsidy / worst-case loss for an LMSR market maker: b * ln(num_outcomes).

    Args:
        num_outcomes: Number of mutually exclusive outcomes (>= 1).
        b: Liquidity parameter (b > 0).

    Returns:
        Theoretical maximum loss of the automated market maker.
    """
    if num_outcomes < 1:
        raise ValueError("Number of outcomes must be >= 1")
    if b <= 0:
        raise ValueError("Liquidity parameter b must be > 0")
    return b * math.log(num_outcomes)


def posterior_probability(prior_yes: float, likelihood_ratio: float) -> float:
    """Update prior Yes probability using Bayesian log-odds shift from evidence likelihood ratio.

    odds_prior = p / (1 - p)
    odds_post = odds_prior * LR
    p_post = odds_post / (1 + odds_post)

    Safe under extreme odds with clipping within [1e-9, 1 - 1e-9].

    Args:
        prior_yes: Prior probability of YES (0 < prior_yes < 1).
        likelihood_ratio: P(Evidence | YES) / P(Evidence | NO) (> 0).

    Returns:
        Posterior probability of YES.
    """
    if prior_yes <= 0.0 or prior_yes >= 1.0:
        raise ValueError(f"Prior probability must be in (0, 1), got {prior_yes}")
    if likelihood_ratio <= 0.0:
        raise ValueError(f"Likelihood ratio must be > 0, got {likelihood_ratio}")

    clamped_prior = min(max(prior_yes, 1e-9), 1.0 - 1e-9)
    log_odds_prior = math.log(clamped_prior / (1.0 - clamped_prior))
    log_odds_post = log_odds_prior + math.log(likelihood_ratio)

    # Sigmoid function for numerical stability
    if log_odds_post > 35.0:
        return 1.0 - 1e-9
    if log_odds_post < -35.0:
        return 1e-9

    odds_post = math.exp(log_odds_post)
    p_post = odds_post / (1.0 + odds_post)
    return min(max(p_post, 1e-9), 1.0 - 1e-9)


def kl_divergence_bernoulli(p_post: float, p_prior: float) -> float:
    """Kullback-Leibler divergence D_KL(P_post || P_prior) in bits (base 2).

    Measures pure informational update introduced by the mined evidence:
    D_KL = p_post * log2(p_post / p_prior) + (1 - p_post) * log2((1 - p_post) / (1 - p_prior))
    """
    p = min(max(p_post, 1e-12), 1.0 - 1e-12)
    q = min(max(p_prior, 1e-12), 1.0 - 1e-12)

    term1 = p * math.log2(p / q)
    term2 = (1.0 - p) * math.log2((1.0 - p) / (1.0 - q))
    return max(0.0, term1 + term2)


def brier_score_improvement(p_prior: float, p_post: float, assumed_true_outcome: int) -> float:
    """Calculates improvement in Brier Score: Brier(prior) - Brier(posterior).

    assumed_true_outcome: 1 for YES, 0 for NO.
    Positive value means posterior is more accurate than prior.
    """
    if assumed_true_outcome not in (0, 1):
        raise ValueError("assumed_true_outcome must be 0 or 1")
    brier_prior = (p_prior - assumed_true_outcome) ** 2
    brier_post = (p_post - assumed_true_outcome) ** 2
    return brier_prior - brier_post


def estimate_evidence_bounty(
    reward_pot_usdc: float,
    p_prior: float,
    p_post: float,
    prior_submissions_count: int = 0,
) -> float:
    """Estimate expected USDC.e judgment / settlement reward share for a mined evidence submission.

    Delphi rewards evidence submissions based on information gain (KL divergence),
    current uncertainty entropy, and earlier submission saturation:
        reward = reward_pot * (info_weight / (total_info_weight + epsilon)) * decay_factor
    """
    if reward_pot_usdc <= 0.0:
        return 0.0

    kl_bits = kl_divergence_bernoulli(p_post, p_prior)
    if kl_bits <= 1e-6:
        return 0.0

    # Diminishing return factor for late submissions on the same market
    saturation_decay = 1.0 / (1.0 + 0.35 * max(0, prior_submissions_count))
    # Information gain scaled reward share (capped at 40% of market pot per submission)
    raw_fraction = min(0.40, 0.15 * math.tanh(kl_bits * 1.5))
    expected_payout = reward_pot_usdc * raw_fraction * saturation_decay
    return round(expected_payout, 4)
