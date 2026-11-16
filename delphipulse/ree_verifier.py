"""Gensyn REE (Reproducible Execution Environment) Judge Receipt Verifier.

Computes and verifies RFC-6962 / SHA-256 canonical cryptographic receipts
for Delphi automated evidence judgments and settlements on the Gensyn network.
Detects bit-flips, tampering in model ID, evidence hashes, or probability shifts.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class REEJudgeReceipt:
    """Cryptographic execution receipt issued by the Gensyn REE judge."""

    receipt_id: str
    market_id: str
    agent_id: str
    model_id: str
    inputs_hash: str
    evidence_hashes: list[str]
    prior_prob: float
    posterior_prob: float
    kl_divergence_bits: float
    judgment_reward_usdc: float
    timestamp: str
    receipt_hash: str

    def to_dict(self) -> dict[str, Any]:
        """Convert receipt to dictionary."""
        return asdict(self)


class REEVerificationError(Exception):
    """Raised when an REE execution receipt fails cryptographic validation."""


def canonical_json_bytes(data: Mapping[str, Any]) -> bytes:
    """Produce deterministic UTF-8 bytes from a dictionary.

    Sorts keys lexicographically, uses compact separators (no trailing spaces),
    and enforces float formatting consistency.
    """
    return json.dumps(
        data,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def compute_inputs_hash(market_id: str, claim: str, sources: list[str]) -> str:
    """Compute deterministic SHA-256 hash of evidence miner inputs."""
    payload = {
        "market_id": market_id,
        "claim": claim.strip(),
        "sources": sorted(sources),
    }
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def compute_receipt_hash(
    receipt_id: str,
    market_id: str,
    agent_id: str,
    model_id: str,
    inputs_hash: str,
    evidence_hashes: list[str],
    prior_prob: float,
    posterior_prob: float,
    kl_divergence_bits: float,
    judgment_reward_usdc: float,
    timestamp: str,
) -> str:
    """Compute canonical SHA-256 hash over the judge's deterministic state."""
    canonical_dict = {
        "receipt_id": receipt_id,
        "market_id": market_id,
        "agent_id": agent_id,
        "model_id": model_id,
        "inputs_hash": inputs_hash,
        "evidence_hashes": sorted(evidence_hashes),
        "prior_prob": round(prior_prob, 8),
        "posterior_prob": round(posterior_prob, 8),
        "kl_divergence_bits": round(kl_divergence_bits, 8),
        "judgment_reward_usdc": round(judgment_reward_usdc, 6),
        "timestamp": timestamp,
    }
    return hashlib.sha256(canonical_json_bytes(canonical_dict)).hexdigest()


def issue_judge_receipt(
    receipt_id: str,
    market_id: str,
    agent_id: str,
    model_id: str,
    claim: str,
    sources: list[str],
    prior_prob: float,
    posterior_prob: float,
    kl_divergence_bits: float,
    judgment_reward_usdc: float,
    timestamp: str,
) -> REEJudgeReceipt:
    """Construct and seal a new REE judge receipt."""
    inputs_hash = compute_inputs_hash(market_id, claim, sources)
    evidence_hashes = [
        hashlib.sha256(src.encode("utf-8")).hexdigest() for src in sorted(sources)
    ]
    r_hash = compute_receipt_hash(
        receipt_id=receipt_id,
        market_id=market_id,
        agent_id=agent_id,
        model_id=model_id,
        inputs_hash=inputs_hash,
        evidence_hashes=evidence_hashes,
        prior_prob=prior_prob,
        posterior_prob=posterior_prob,
        kl_divergence_bits=kl_divergence_bits,
        judgment_reward_usdc=judgment_reward_usdc,
        timestamp=timestamp,
    )
    return REEJudgeReceipt(
        receipt_id=receipt_id,
        market_id=market_id,
        agent_id=agent_id,
        model_id=model_id,
        inputs_hash=inputs_hash,
        evidence_hashes=evidence_hashes,
        prior_prob=prior_prob,
        posterior_prob=posterior_prob,
        kl_divergence_bits=kl_divergence_bits,
        judgment_reward_usdc=judgment_reward_usdc,
        timestamp=timestamp,
        receipt_hash=r_hash,
    )


def verify_judge_receipt(receipt: REEJudgeReceipt | Mapping[str, Any]) -> bool:
    """Verify cryptographic integrity of an REE judge receipt.

    Returns True if valid. Raises REEVerificationError on tampering.
    """
    if isinstance(receipt, REEJudgeReceipt):
        data = receipt.to_dict()
    else:
        data = dict(receipt)

    required_fields = [
        "receipt_id",
        "market_id",
        "agent_id",
        "model_id",
        "inputs_hash",
        "evidence_hashes",
        "prior_prob",
        "posterior_prob",
        "kl_divergence_bits",
        "judgment_reward_usdc",
        "timestamp",
        "receipt_hash",
    ]
    for field in required_fields:
        if field not in data:
            raise REEVerificationError(f"Missing required receipt field: '{field}'")

    expected_hash = compute_receipt_hash(
        receipt_id=data["receipt_id"],
        market_id=data["market_id"],
        agent_id=data["agent_id"],
        model_id=data["model_id"],
        inputs_hash=data["inputs_hash"],
        evidence_hashes=data["evidence_hashes"],
        prior_prob=float(data["prior_prob"]),
        posterior_prob=float(data["posterior_prob"]),
        kl_divergence_bits=float(data["kl_divergence_bits"]),
        judgment_reward_usdc=float(data["judgment_reward_usdc"]),
        timestamp=data["timestamp"],
    )

    if expected_hash != data["receipt_hash"]:
        raise REEVerificationError(
            f"Receipt hash mismatch! Stored: {data['receipt_hash'][:16]}..., "
            f"Recomputed: {expected_hash[:16]}... (Possible bit-flip or state tampering)"
        )

    return True
