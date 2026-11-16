"""DelphiPulse Configuration & Network Registry.

Official Delphi / Gensyn protocol constants verified from docs.delphi.fyi and
agent-competition.gensyn.ai.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class NetworkProfile:
    """Delphi / Gensyn network connection profile."""

    name: str
    chain_id: int
    rpc_url: str
    api_base_url: str
    app_url: str
    subgraph_url: str
    mode_header: str | None = None


# Official Network Configurations
MAINNET_PROFILE = NetworkProfile(
    name="mainnet",
    chain_id=685689,
    rpc_url="https://gensyn-mainnet.g.alchemy.com/public",
    api_base_url="https://api.delphi.fyi",
    app_url="https://app.delphi.fyi",
    subgraph_url=(
        "https://api.goldsky.com/api/public/project_cmnoqdag1obop01z3efnu8ssq/subgraphs/delphi-mainnet-autoset/1.0.0/gn"
    ),
    mode_header=None,
)

COMPETITION_PROFILE = NetworkProfile(
    name="competition-testnet",
    chain_id=685685,
    rpc_url="https://gensyn-testnet.g.alchemy.com/public",
    api_base_url="https://delphi-api.gensyn.ai",
    app_url="https://agent-competition.gensyn.ai",
    subgraph_url=(
        "https://api.goldsky.com/api/public/project_cmnoqdag1obop01z3efnu8ssq/subgraphs/delphi-agent-competition/1.0.0/gn"
    ),
    mode_header="competition",
)

PROFILES: Mapping[str, NetworkProfile] = {
    "mainnet": MAINNET_PROFILE,
    "competition-testnet": COMPETITION_PROFILE,
    "competition": COMPETITION_PROFILE,
    "testnet": COMPETITION_PROFILE,
}


@dataclass
class DelphiPulseConfig:
    """Runtime configuration for DelphiPulse agent miner."""

    network: str = "competition-testnet"
    agent_id: str = "delphipulse-miner-v1"
    harness: str = "gensyn-evidence-miner"
    model: str = "delphipulse-lmsr-v1"
    api_key: str = ""
    wallet_address: str = ""
    gateway_contract: str = ""
    request_timeout_seconds: float = 12.0
    evidence_cutoff_safety_seconds: int = 1800

    @classmethod
    def from_env(cls) -> DelphiPulseConfig:
        """Load configuration from environment variables."""
        network = os.getenv("DELPHI_NETWORK", "competition-testnet")
        return cls(
            network=network,
            agent_id=os.getenv("DELPHI_AGENT_ID", "delphipulse-miner-v1"),
            harness=os.getenv("DELPHI_AGENT_HARNESS", "gensyn-evidence-miner"),
            model=os.getenv("DELPHI_AGENT_MODEL", "delphipulse-lmsr-v1"),
            api_key=os.getenv("DELPHI_API_ACCESS_KEY", ""),
            wallet_address=os.getenv("DELPHI_WALLET_ADDRESS", ""),
            gateway_contract=os.getenv("DELPHI_GATEWAY_CONTRACT", ""),
            request_timeout_seconds=float(os.getenv("DELPHI_REQUEST_TIMEOUT", "12.0")),
            evidence_cutoff_safety_seconds=int(os.getenv("DELPHI_CUTOFF_SAFETY_SEC", "1800")),
        )

    def get_profile(self) -> NetworkProfile:
        """Resolve active network profile."""
        if self.network not in PROFILES:
            raise ValueError(f"Unknown network profile '{self.network}'. Supported: {list(PROFILES.keys())}")
        return PROFILES[self.network]

    def build_headers(self) -> dict[str, str]:
        """Construct official Delphi REST API headers for agent competition."""
        profile = self.get_profile()
        headers: dict[str, str] = {
            "User-Agent": f"DelphiPulse-Agent/{self.agent_id}",
            "Accept": "application/json",
            "Content-Type": "application/json",
            "X-Agent-Id": self.agent_id,
            "X-Agent-Harness": self.harness,
            "X-Agent-Model": self.model,
        }
        if profile.mode_header:
            headers["X-Delphi-Mode"] = profile.mode_header
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers
