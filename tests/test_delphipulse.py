"""Comprehensive Test Suite for DelphiPulse.

Validates Hanson LMSR stability, Bayesian evidence information-gain,
competition rules invariants, REE cryptographic receipt verification,
and leaderboard qualification metrics.
"""

from __future__ import annotations

import math
from decimal import Decimal

import pytest

from delphipulse.client import DelphiClient
from delphipulse.config import DelphiPulseConfig
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
from delphipulse.miner import (
    AttachmentCapExceededError,
    AutonomousEvidenceMiner,
    CompetitionLedger,
    DuplicateClaimError,
    EvidenceCutoffViolation,
    MarketCapExceededError,
    PIIDetectedError,
    SelfCancellationError,
    binary_entropy,
    sanitize_and_check_pii,
    score_market_opportunity,
)
from delphipulse.ree_verifier import (
    REEVerificationError,
    issue_judge_receipt,
    verify_judge_receipt,
)

# =========================================================================
# 1. Hanson LMSR Invariants
# =========================================================================

def test_lmsr_cost_basic():
    """Test standard LMSR cost calculation."""
    q = [10.0, 10.0]
    b = 100.0
    cost = lmsr_cost(q, b)
    # C([10, 10]) = 100 * ln(2 * exp(10/100)) = 100 * (0.1 + ln(2)) = 10 + 100*ln(2)
    expected = 10.0 + 100.0 * math.log(2.0)
    assert abs(cost - expected) < 1e-9


def test_lmsr_cost_numerical_stability_extreme_values():
    """Log-sum-exp must not overflow even with shares = 100,000."""
    q = [100000.0, 50.0, 20.0]
    b = 50.0
    cost = lmsr_cost(q, b)
    assert cost > 0.0
    assert not math.isinf(cost)
    assert not math.isnan(cost)


def test_lmsr_cost_invalid_liquidity():
    """b <= 0 must raise ValueError."""
    with pytest.raises(ValueError, match="liquidity parameter 'b' must be > 0"):
        lmsr_cost([10.0, 20.0], 0.0)
    with pytest.raises(ValueError, match="liquidity parameter 'b' must be > 0"):
        lmsr_cost([10.0, 20.0], -5.0)


def test_lmsr_cost_empty_quantities():
    """Empty quantities must raise ValueError."""
    with pytest.raises(ValueError, match="Quantities sequence cannot be empty"):
        lmsr_cost([], 100.0)


def test_lmsr_prices_sum_to_one():
    """Prices must sum to 1.0 within 1e-12."""
    q = [35.0, 120.0, 85.0]
    b = 50.0
    prices = lmsr_prices(q, b)
    assert len(prices) == 3
    assert abs(sum(prices) - 1.0) < 1e-12
    for p in prices:
        assert 0.0 < p < 1.0


def test_lmsr_prices_extreme_shares():
    """Overflow-safe probability calculation under extreme differences."""
    q = [100000.0, 10.0]
    b = 10.0
    prices = lmsr_prices(q, b)
    assert abs(sum(prices) - 1.0) < 1e-12
    assert prices[0] > 0.999999999
    assert prices[1] < 1e-9


def test_lmsr_prices_monotonicity():
    """Outcome with highest shares must have highest price."""
    q = [10.0, 50.0, 20.0]
    b = 10.0
    prices = lmsr_prices(q, b)
    assert prices[1] > prices[2] > prices[0]


def test_lmsr_prices_invalid_arguments():
    """Invalid arguments for prices raise ValueError."""
    with pytest.raises(ValueError):
        lmsr_prices([], 10.0)
    with pytest.raises(ValueError):
        lmsr_prices([1.0], -1.0)


def test_lmsr_trade_cost():
    """Purchasing shares must cost positive collateral; selling returns collateral."""
    q = [10.0, 10.0]
    b = 100.0
    cost_buy = lmsr_trade_cost(q, b, outcome_idx=0, delta_shares=5.0)
    assert cost_buy > 0.0
    cost_sell = lmsr_trade_cost(q, b, outcome_idx=0, delta_shares=-5.0)
    assert cost_sell < 0.0


def test_lmsr_trade_cost_index_out_of_bounds():
    """Invalid outcome index raises IndexError."""
    with pytest.raises(IndexError):
        lmsr_trade_cost([10.0, 10.0], 100.0, outcome_idx=5, delta_shares=1.0)


def test_lmsr_max_loss():
    """LMSR max loss is strictly b * ln(n)."""
    b = 500.0
    assert abs(lmsr_max_loss(2, b) - b * math.log(2)) < 1e-9
    assert abs(lmsr_max_loss(5, b) - b * math.log(5)) < 1e-9


def test_lmsr_max_loss_invalid_args():
    """Invalid arguments for max loss raise ValueError."""
    with pytest.raises(ValueError):
        lmsr_max_loss(0, 100.0)
    with pytest.raises(ValueError):
        lmsr_max_loss(2, -10.0)


# =========================================================================
# 2. Bayesian Evidence Engine & Information-Gain Invariants
# =========================================================================

def test_posterior_probability_calculation():
    """Test standard Bayesian log-odds shift."""
    prior = 0.50
    lr = 3.0
    post = posterior_probability(prior, lr)
    # p / (1 - p) = 1.0 * 3.0 = 3.0 -> post = 3 / (1 + 3) = 0.75
    assert abs(post - 0.75) < 1e-6


def test_posterior_probability_unfavorable_evidence():
    """LR < 1.0 reduces YES probability."""
    prior = 0.60
    lr = 0.50
    post = posterior_probability(prior, lr)
    assert post < prior


def test_posterior_probability_bounds_and_clamping():
    """Extreme LR should clamp within numerical bounds without throwing overflow."""
    assert posterior_probability(0.5, 1e20) < 1.0
    assert posterior_probability(0.5, 1e-20) > 0.0
    with pytest.raises(ValueError):
        posterior_probability(0.0, 1.0)
    with pytest.raises(ValueError):
        posterior_probability(1.0, 1.0)
    with pytest.raises(ValueError):
        posterior_probability(0.5, -2.0)


def test_kl_divergence_bernoulli():
    """KL divergence must be 0 for identical distributions and strictly positive otherwise."""
    assert kl_divergence_bernoulli(0.40, 0.40) == 0.0
    d_kl = kl_divergence_bernoulli(0.80, 0.40)
    assert d_kl > 0.0
    assert abs(d_kl - (0.8 * math.log2(0.8 / 0.4) + 0.2 * math.log2(0.2 / 0.6))) < 1e-6


def test_brier_score_improvement():
    """Evidence moving closer to truth improves Brier score."""
    imp_good = brier_score_improvement(p_prior=0.30, p_post=0.70, assumed_true_outcome=1)
    assert imp_good > 0.0
    imp_bad = brier_score_improvement(p_prior=0.70, p_post=0.30, assumed_true_outcome=1)
    assert imp_bad < 0.0
    with pytest.raises(ValueError):
        brier_score_improvement(0.5, 0.5, 2)


def test_estimate_evidence_bounty():
    """Expected bounty scales with pot and information gain, decaying with competition."""
    bounty_zero_pot = estimate_evidence_bounty(0.0, 0.5, 0.8)
    assert bounty_zero_pot == 0.0

    bounty_no_info = estimate_evidence_bounty(1000.0, 0.5, 0.5)
    assert bounty_no_info == 0.0

    bounty_sub0 = estimate_evidence_bounty(1000.0, 0.4, 0.8, prior_submissions_count=0)
    bounty_sub2 = estimate_evidence_bounty(1000.0, 0.4, 0.8, prior_submissions_count=2)
    assert bounty_sub0 > bounty_sub2 > 0.0


# =========================================================================
# 3. Client & BigInt Position Parsing
# =========================================================================

def test_parse_position_lossless_decimal():
    """Check 18-decimal shares and 6-decimal USDC collateral parsing."""
    raw = {
        "shares": "1500000000000000000",  # 1.5 shares
        "tokensRedeemed": "25000000",       # 25 USDC
        "outcomeIndex": 1,
        "marketId": "mkt-99",
    }
    pos = DelphiClient.parse_position(raw)
    assert pos["has_shares"] is True
    assert pos["shares"] == Decimal("1.5")
    assert pos["tokens_redeemed"] == Decimal("25")
    assert pos["outcome_index"] == 1


def test_parse_position_zero_shares_guard():
    """Zero shares must flag has_shares as False."""
    raw = {"shares": "0", "tokensRedeemed": "0"}
    pos = DelphiClient.parse_position(raw)
    assert pos["has_shares"] is False
    assert pos["shares"] == Decimal("0")


# =========================================================================
# 4. PII Scanning & Sanitation
# =========================================================================

def test_pii_scanner_clean_text():
    """Clean public claims must pass sanitation."""
    clean = "Fed minutes show committee unanimous on rate path."
    assert sanitize_and_check_pii(clean) == clean


def test_pii_scanner_rejects_email():
    """Evidence containing emails must raise PIIDetectedError."""
    dirty = "Leaked from insider at john.doe@secretcorp.com about earnings."
    with pytest.raises(PIIDetectedError, match="email address"):
        sanitize_and_check_pii(dirty)


def test_pii_scanner_rejects_ssn():
    """Evidence containing SSN patterns must raise PIIDetectedError."""
    dirty = "Target personal identifier 000-12-3456 verified."
    with pytest.raises(PIIDetectedError, match="SSN pattern"):
        sanitize_and_check_pii(dirty)


def test_pii_scanner_rejects_phone():
    """Evidence containing phone numbers must raise PIIDetectedError."""
    dirty = "Whistleblower reached at 415-555-0199."
    with pytest.raises(PIIDetectedError, match="phone number"):
        sanitize_and_check_pii(dirty)


# =========================================================================
# 5. Competition Rules Ledger & Invariants
# =========================================================================

def test_ledger_market_cap_enforced():
    """Ledger must reject 4th submission to same market."""
    ledger = CompetitionLedger(agent_id="test-agent")
    m_id = "mkt-limit"
    for i in range(3):
        ledger.record_submission(
            market_id=m_id,
            stance="YES",
            claim=f"Unique market claim number {i}",
            sources=[f"https://source{i}.org"],
        )
    with pytest.raises(MarketCapExceededError, match="maximum of 3 submissions"):
        ledger.record_submission(
            market_id=m_id,
            stance="YES",
            claim="Fourth claim exceeding limit",
            sources=["https://source4.org"],
        )


def test_ledger_attachment_cap_enforced():
    """Ledger must reject submission with >3 attachments."""
    ledger = CompetitionLedger()
    with pytest.raises(AttachmentCapExceededError, match="exceeds limit of 3"):
        ledger.record_submission(
            market_id="mkt-attach",
            stance="YES",
            claim="Claim with too many files",
            sources=["https://src.org"],
            attachments_count=4,
        )


def test_ledger_anti_self_cancellation():
    """Opposing stance on same market must be blocked by default."""
    ledger = CompetitionLedger()
    m_id = "mkt-cancel"
    ledger.record_submission(
        market_id=m_id,
        stance="YES",
        claim="Initial bullish evidence",
        sources=["https://src1.org"],
    )
    with pytest.raises(SelfCancellationError, match="self-cancellation penalty"):
        ledger.record_submission(
            market_id=m_id,
            stance="NO",
            claim="Conflicting bearish evidence",
            sources=["https://src2.org"],
        )


def test_ledger_anti_self_cancellation_override():
    """Explicit override flag allows conflicting stance if requested."""
    ledger = CompetitionLedger()
    m_id = "mkt-override"
    ledger.record_submission(market_id=m_id, stance="YES", claim="Claim 1", sources=["https://s1.org"])
    sub2 = ledger.record_submission(
        market_id=m_id,
        stance="NO",
        claim="Claim 2 opposing",
        sources=["https://s2.org"],
        allow_self_cancellation_override=True,
    )
    assert sub2.stance == "NO"


def test_ledger_deduplication():
    """Identical claim on same stance must raise DuplicateClaimError."""
    ledger = CompetitionLedger()
    claim = "Identical economic survey numbers."
    ledger.record_submission(market_id="m1", stance="YES", claim=claim, sources=["https://s1.org"])
    with pytest.raises(DuplicateClaimError, match="Duplicate claim detected"):
        ledger.record_submission(market_id="m2", stance="YES", claim=claim, sources=["https://s2.org"])


def test_ledger_cutoff_date_enforcement():
    """Evidence published after market cutoff must be rejected."""
    ledger = CompetitionLedger()
    with pytest.raises(EvidenceCutoffViolation, match="after market cutoff"):
        ledger.record_submission(
            market_id="mkt-cutoff",
            stance="YES",
            claim="Late breaking article",
            sources=["https://src.org"],
            evidence_pub_date="2026-12-15T00:00:00Z",
            market_cutoff_date="2026-12-11T17:00:00Z",
        )


def test_leaderboard_qualification_tracker():
    """Tracker requires >= 10 judged submissions across >= 3 distinct markets."""
    ledger = CompetitionLedger()
    status_empty = ledger.get_qualification_status()
    assert not status_empty["is_qualified"]
    assert status_empty["remaining_submissions_needed"] == 10
    assert status_empty["remaining_markets_needed"] == 3

    # Add 9 judged submissions across 3 markets
    sub_ids = []
    for m_idx in range(3):
        m_id = f"mkt-{m_idx}"
        for s_idx in range(3):
            sub = ledger.record_submission(
                market_id=m_id,
                stance="YES",
                claim=f"Claim {m_idx}-{s_idx}",
                sources=[f"https://source{m_idx}{s_idx}.org"],
            )
            ledger.mark_judged(sub.submission_id)
            sub_ids.append(sub.submission_id)

    status_9 = ledger.get_qualification_status()
    assert not status_9["is_qualified"]
    assert status_9["judged_count"] == 9
    assert status_9["distinct_markets_count"] == 3
    assert status_9["remaining_submissions_needed"] == 1

    # 10th judged submission on 4th market qualifies!
    sub10 = ledger.record_submission(
        market_id="mkt-3",
        stance="YES",
        claim="Decisive qualifying claim",
        sources=["https://qualify.org"],
    )
    ledger.mark_judged(sub10.submission_id)

    status_10 = ledger.get_qualification_status()
    assert status_10["is_qualified"] is True
    assert status_10["judged_count"] == 10
    assert status_10["distinct_markets_count"] == 4
    assert status_10["remaining_submissions_needed"] == 0
    assert status_10["remaining_markets_needed"] == 0


# =========================================================================
# 6. REE Cryptographic Judge Receipts
# =========================================================================

def test_ree_receipt_issuance_and_verification():
    """Genuine receipt must verify without error."""
    receipt = issue_judge_receipt(
        receipt_id="rcpt-001",
        market_id="mkt-crypto",
        agent_id="delphipulse-miner-v1",
        model_id="delphipulse-lmsr-v1",
        claim="Ethereum Devnet 6 achieves finality.",
        sources=["https://holesky.org"],
        prior_prob=0.55,
        posterior_prob=0.78,
        kl_divergence_bits=0.354,
        judgment_reward_usdc=85.25,
        timestamp="2026-11-14T10:00:00Z",
    )
    assert verify_judge_receipt(receipt) is True


def test_ree_receipt_tamper_detection():
    """Altering prior prob or judgment reward must trigger REEVerificationError."""
    receipt = issue_judge_receipt(
        receipt_id="rcpt-tamper",
        market_id="mkt-tamper",
        agent_id="delphipulse-miner-v1",
        model_id="delphipulse-lmsr-v1",
        claim="Sample claim",
        sources=["https://src.org"],
        prior_prob=0.40,
        posterior_prob=0.80,
        kl_divergence_bits=0.5,
        judgment_reward_usdc=50.0,
        timestamp="2026-11-14T10:00:00Z",
    )

    tampered_data = receipt.to_dict()
    tampered_data["judgment_reward_usdc"] = 999.0  # Attempted payout inflation

    with pytest.raises(REEVerificationError, match="hash mismatch"):
        verify_judge_receipt(tampered_data)


# =========================================================================
# 7. Market Ranking & Full Benchmark Cycle
# =========================================================================

def test_market_opportunity_scoring():
    """Check binary entropy and opportunity score weighting."""
    assert abs(binary_entropy(0.50) - 1.0) < 1e-9
    assert abs(binary_entropy(0.0) - 0.0) < 1e-9

    m = {
        "id": "mkt-test",
        "name": "Test Market",
        "category": "Crypto",
        "reward_pot_usdc": 1000.0,
        "prior_yes": 0.50,
        "evidence_cutoff": "2026-11-20T00:00:00Z",
        "existing_submissions": 0,
    }
    opp = score_market_opportunity(m)
    assert opp.uncertainty_entropy == 1.0
    assert opp.opportunity_score > 1000.0


def test_autonomous_miner_benchmark_cycle():
    """Verify full benchmark cycle achieves qualification and generates valid receipts."""
    miner = AutonomousEvidenceMiner()
    result = miner.run_benchmark_cycle()

    assert result["status"] == "success"
    q = result["qualification"]
    assert q["is_qualified"] is True
    assert q["judged_count"] >= 10
    assert q["distinct_markets_count"] >= 3
    assert result["total_estimated_usdc_won"] > 0.0
    assert len(result["sample_receipts"]) >= 3

    # Verify each sample receipt cryptographically
    for r in result["sample_receipts"]:
        assert verify_judge_receipt(r) is True


# =========================================================================
# 8. Network Configuration Profiles & Headers
# =========================================================================

def test_config_profiles_and_headers():
    """Check configuration resolution and competition headers."""
    cfg = DelphiPulseConfig(network="competition-testnet", agent_id="agent-xyz")
    profile = cfg.get_profile()
    assert profile.chain_id == 685685
    assert "gensyn-testnet" in profile.rpc_url

    headers = cfg.build_headers()
    assert headers["X-Delphi-Mode"] == "competition"
    assert headers["X-Agent-Id"] == "agent-xyz"
    assert headers["X-Agent-Harness"] == "gensyn-evidence-miner"
