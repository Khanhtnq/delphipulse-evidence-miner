"""DelphiPulse Command-Line Interface.

Provides tools for benchmark simulations, Bayesian information-gain evaluations,
and live network connectivity probing to Gensyn and Delphi endpoints.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from delphipulse.client import DelphiAPIError, DelphiClient, DelphiNetworkError
from delphipulse.config import DelphiPulseConfig
from delphipulse.lmsr import (
    estimate_evidence_bounty,
    kl_divergence_bernoulli,
    lmsr_prices,
    posterior_probability,
)
from delphipulse.miner import AutonomousEvidenceMiner


def cmd_benchmark(args: argparse.Namespace) -> int:
    """Run full autonomous mining benchmark simulation across competition markets."""
    miner = AutonomousEvidenceMiner()
    result = miner.run_benchmark_cycle()

    if getattr(args, "json", False):
        print(json.dumps(result, indent=2))
        return 0

    print("=" * 72)
    print(" DELPHIPULSE: AUTONOMOUS EVIDENCE MINING BENCHMARK")
    print("=" * 72)
    print(f"Agent ID: {result['agent_id']}")
    q = result["qualification"]
    print(f"Leaderboard Qualified: {q['is_qualified']} (Status: {'PASS' if q['is_qualified'] else 'FAIL'})")
    print(f"Judged Submissions:    {q['judged_count']} / {q['min_judged_required']} required")
    print(f"Distinct Markets:      {q['distinct_markets_count']} / {q['min_markets_required']} required")
    print(f"Total USDC.e Earned:   ${result['total_estimated_usdc_won']:,.2f}")
    print("\nDistinct Markets Covered:")
    for m in q["distinct_markets"]:
        print(f"  - {m}")

    print("\nSample Gensyn REE Judge Receipts:")
    for r in result["sample_receipts"]:
        print(f"  [Receipt {r['receipt_id']}]")
        print(f"    Market:      {r['market_id']}")
        print(f"    Shift:       P(YES) {r['prior_prob']:.2f} -> {r['posterior_prob']:.2f}")
        print(f"    Information: {r['kl_divergence_bits']:.4f} bits KL-divergence")
        print(f"    Reward:      ${r['judgment_reward_usdc']:.2f} USDC.e")
        print(f"    Seal Hash:   {r['receipt_hash'][:24]}...")
    print("=" * 72)
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    """Calculate Bayesian shift, KL-divergence, and expected USDC.e bounty from parameters."""
    prior = float(args.prior)
    lr = float(args.lr)
    pot = float(args.pot)
    subs = int(args.submissions)

    post = posterior_probability(prior, lr)
    kl_bits = kl_divergence_bernoulli(post, prior)
    bounty = estimate_evidence_bounty(pot, prior, post, subs)

    # Compute mock LMSR share quantities corresponding to probabilities with b=100
    b = 100.0
    q_prior = [b * prior, b * (1.0 - prior)]
    prices_prior = lmsr_prices(q_prior, b)

    res: dict[str, Any] = {
        "prior_probability": round(prior, 6),
        "likelihood_ratio": round(lr, 4),
        "posterior_probability": round(post, 6),
        "kl_divergence_bits": round(kl_bits, 6),
        "reward_pot_usdc": pot,
        "existing_submissions_count": subs,
        "estimated_judgment_bounty_usdc": bounty,
        "lmsr_implied_prior_yes": round(prices_prior[0], 6),
    }

    if getattr(args, "json", False):
        print(json.dumps(res, indent=2))
    else:
        print("=" * 60)
        print(" DELPHIPULSE: BAYESIAN EVIDENCE EVALUATOR")
        print("=" * 60)
        print(f"Prior Probability P(YES):     {prior:.4f}")
        print(f"Evidence Likelihood Ratio:    {lr:.4f}")
        print(f"Posterior Probability P(YES): {post:.4f}")
        print(f"KL Information Gain:          {kl_bits:.4f} bits")
        print(f"Market Collateral Pot:        ${pot:,.2f} USDC.e")
        print(f"Expected Judgment Bounty:     ${bounty:,.2f} USDC.e")
        print("=" * 60)
    return 0


def cmd_live_probe(args: argparse.Namespace) -> int:
    """Probe live Gensyn RPC, Delphi REST API health, and Goldsky Subgraph endpoint."""
    config = DelphiPulseConfig.from_env()
    client = DelphiClient(config=config)
    profile = config.get_profile()

    print("=" * 72)
    print(" DELPHIPULSE: LIVE PROTOCOL & NETWORK CONNECTIVITY PROBE")
    print("=" * 72)
    print(f"Active Network: {profile.name} (Chain ID: {profile.chain_id})")
    print(f"RPC Endpoint:   {profile.rpc_url}")
    print(f"API Base URL:   {profile.api_base_url}")
    print(f"Subgraph URL:   {profile.subgraph_url[:60]}...")
    print("-" * 72)

    # 1. Probe JSON-RPC eth_chainId
    rpc_ok = False
    try:
        live_chain_id = client.check_rpc_chain_id()
        rpc_ok = (live_chain_id == profile.chain_id)
        status_rpc = f"OK (eth_chainId = {live_chain_id})" if rpc_ok else f"MISMATCH ({live_chain_id})"
    except (DelphiNetworkError, DelphiAPIError, Exception) as err:
        status_rpc = f"OFFLINE / UNREACHABLE ({err})"

    print(f"1. Gensyn JSON-RPC Check:       {status_rpc}")

    # 2. Probe Delphi REST API Health
    api_ok = False
    try:
        health_resp = client.check_api_health()
        api_ok = True
        status_api = f"OK ({health_resp})"
    except (DelphiNetworkError, DelphiAPIError, Exception) as err:
        status_api = f"OFFLINE / UNREACHABLE ({err})"

    print(f"2. Delphi REST API Health:     {status_api}")

    # 3. Probe Goldsky Subgraph Endpoint
    subgraph_ok = False
    try:
        subgraph_res = client.query_subgraph_markets(limit=1)
        subgraph_ok = True
        status_subgraph = f"OK (Introspected {len(subgraph_res)} entities/blocks)"
    except (DelphiNetworkError, DelphiAPIError, Exception) as err:
        status_subgraph = f"OFFLINE / UNREACHABLE ({err})"

    print(f"3. Goldsky Subgraph GraphQL:    {status_subgraph}")
    print("-" * 72)
    print(f"Summary: RPC={rpc_ok}, API={api_ok}, Subgraph={subgraph_ok}")
    print("(DelphiPulse operates seamlessly in offline deterministic simulation mode)")
    print("=" * 72)
    return 0


def main(argv: list[str] | None = None) -> int:
    """CLI Entry point."""
    parser = argparse.ArgumentParser(
        prog="delphipulse",
        description="DelphiPulse - Autonomous Evidence Miner for Delphi / Gensyn",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # benchmark
    p_bench = subparsers.add_parser("benchmark", help="Run full mining benchmark")
    p_bench.add_argument("--json", action="store_true", help="Output JSON format")
    p_bench.set_defaults(func=cmd_benchmark)

    # evaluate
    p_eval = subparsers.add_parser("evaluate", help="Evaluate single Bayesian evidence shift")
    p_eval.add_argument("--prior", type=float, default=0.50, help="Prior probability P(YES)")
    p_eval.add_argument("--lr", type=float, default=2.0, help="Likelihood ratio")
    p_eval.add_argument("--pot", type=float, default=500.0, help="Market reward pot USDC")
    p_eval.add_argument("--submissions", type=int, default=0, help="Existing submissions count")
    p_eval.add_argument("--json", action="store_true", help="Output JSON format")
    p_eval.set_defaults(func=cmd_evaluate)

    # live-probe
    p_probe = subparsers.add_parser("live-probe", help="Probe live RPC, API, and Subgraph")
    p_probe.set_defaults(func=cmd_live_probe)

    parsed = parser.parse_args(argv)
    return parsed.func(parsed)


if __name__ == "__main__":
    sys.exit(main())
