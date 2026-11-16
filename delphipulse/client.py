"""Live Delphi REST, Goldsky Subgraph GraphQL & Gensyn JSON-RPC Client.

Uses standard Python urllib with typed error handling, competition header
injection, and lossless Decimal math for Delphi 18-decimal shares and
6-decimal USDC collateral strings.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from decimal import Decimal
from typing import Any, Mapping

from delphipulse.config import DelphiPulseConfig


class DelphiClientError(Exception):
    """Base exception for Delphi client communication failures."""


class DelphiNetworkError(DelphiClientError):
    """Network connection or DNS failure."""


class DelphiAPIError(DelphiClientError):
    """Delphi REST or RPC API returned an error response."""

    def __init__(self, status_code: int, message: str, payload: Any = None):
        super().__init__(f"HTTP {status_code}: {message}")
        self.status_code = status_code
        self.message = message
        self.payload = payload


class DelphiClient:
    """Production client for Delphi Protocol and Gensyn network."""

    def __init__(self, config: DelphiPulseConfig | None = None) -> None:
        self.config = config or DelphiPulseConfig.from_env()
        self.profile = self.config.get_profile()

    def _request(
        self,
        url: str,
        method: str = "GET",
        headers: Mapping[str, str] | None = None,
        data: bytes | None = None,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        """Execute HTTP request with strict validation and error wrapping."""
        req_timeout = timeout or self.config.request_timeout_seconds
        req_headers = dict(headers or {})

        parsed = urllib.parse.urlparse(url)
        if parsed.scheme not in ("http", "https"):
            raise ValueError(f"Invalid URL scheme '{parsed.scheme}'. Only HTTP/HTTPS allowed.")

        req = urllib.request.Request(url, data=data, headers=req_headers, method=method)

        try:
            with urllib.request.urlopen(req, timeout=req_timeout) as resp:
                raw = resp.read().decode("utf-8")
                if not raw.strip():
                    return {}
                return json.loads(raw)
        except urllib.error.HTTPError as err:
            err_body = ""
            try:
                err_body = err.read().decode("utf-8")
                parsed_err = json.loads(err_body)
            except Exception:
                parsed_err = err_body
            raise DelphiAPIError(err.code, str(err.reason), parsed_err) from err
        except (urllib.error.URLError, TimeoutError, OSError) as err:
            raise DelphiNetworkError(f"Connection failed for {url}: {err}") from err

    def check_rpc_chain_id(self) -> int:
        """Call Gensyn JSON-RPC `eth_chainId` on the configured RPC endpoint.

        Returns:
            The integer chain ID (e.g., 685685 for competition testnet).
        """
        payload = json.dumps({
            "jsonrpc": "2.0",
            "method": "eth_chainId",
            "params": [],
            "id": 1,
        }).encode("utf-8")

        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": f"DelphiPulse/{self.config.agent_id}",
        }

        resp = self._request(self.profile.rpc_url, method="POST", headers=headers, data=payload)
        if "error" in resp:
            raise DelphiAPIError(500, f"RPC Error: {resp['error']}")
        hex_id = resp.get("result", "0x0")
        return int(hex_id, 16)

    def check_api_health(self) -> dict[str, Any]:
        """Check the health status of the Delphi REST API backend.

        Queries `GET {api_base_url}/health`.
        """
        url = f"{self.profile.api_base_url.rstrip('/')}/health"
        headers = self.config.build_headers()
        return self._request(url, method="GET", headers=headers)

    def query_subgraph_markets(self, limit: int = 10) -> list[dict[str, Any]]:
        """Send a real GraphQL query to the Goldsky Subgraph endpoint.

        Performs introspection and queries recent markets or block meta.
        """
        query = {
            "query": """
            query GetDelphiMarkets($first: Int!) {
              _meta {
                block {
                  number
                }
              }
              markets(first: $first, orderBy: timestamp, orderDirection: desc) {
                id
                name
                description
                status
                collateralToken
                evidenceCutoff
                resolutionTimestamp
              }
            }
            """,
            "variables": {"first": limit},
        }

        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": f"DelphiPulse/{self.config.agent_id}",
        }

        payload = json.dumps(query).encode("utf-8")
        resp = self._request(self.profile.subgraph_url, method="POST", headers=headers, data=payload)

        if "errors" in resp:
            # Fallback for simpler introspection if market entity schema differs
            query_fallback = {"query": "{ _meta { block { number } } }"}
            resp_fallback = self._request(
                self.profile.subgraph_url,
                method="POST",
                headers=headers,
                data=json.dumps(query_fallback).encode("utf-8"),
            )
            return [resp_fallback.get("data", {})]

        data = resp.get("data", {})
        return data.get("markets", [])

    def list_markets(self, status: str = "open", limit: int = 20) -> list[dict[str, Any]]:
        """List markets from Delphi REST API endpoint."""
        query_params = urllib.parse.urlencode({"status": status, "limit": limit})
        url = f"{self.profile.api_base_url.rstrip('/')}/v1/markets?{query_params}"
        headers = self.config.build_headers()
        resp = self._request(url, method="GET", headers=headers)
        return resp.get("markets", resp if isinstance(resp, list) else [])

    def get_market(self, market_id: str) -> dict[str, Any]:
        """Fetch full market details by market ID."""
        url = f"{self.profile.api_base_url.rstrip('/')}/v1/markets/{urllib.parse.quote(market_id)}"
        headers = self.config.build_headers()
        return self._request(url, method="GET", headers=headers)

    def submit_evidence(
        self,
        market_id: str,
        directional_stance: str,
        evidence_text: str,
        sources: list[str],
        attachments: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Submit mined evidence for a market to Delphi REST API."""
        url = f"{self.profile.api_base_url.rstrip('/')}/v1/markets/{urllib.parse.quote(market_id)}/evidence"
        headers = self.config.build_headers()
        payload = {
            "agent_id": self.config.agent_id,
            "harness": self.config.harness,
            "model": self.config.model,
            "stance": directional_stance.upper(),
            "claim": evidence_text,
            "sources": sources,
            "attachments": attachments or [],
        }
        return self._request(
            url,
            method="POST",
            headers=headers,
            data=json.dumps(payload).encode("utf-8"),
        )

    def submit_feedback(
        self,
        category: str,
        title: str,
        detailed_feedback: str,
        suggested_fix: str = "",
    ) -> dict[str, Any]:
        """Submit feedback for the Delphi competition (eligible for 5 x $250 feedback bounties)."""
        url = f"{self.profile.api_base_url.rstrip('/')}/v1/competition/feedback"
        headers = self.config.build_headers()
        payload = {
            "agent_id": self.config.agent_id,
            "category": category,
            "title": title,
            "feedback": detailed_feedback,
            "suggested_fix": suggested_fix,
        }
        return self._request(
            url,
            method="POST",
            headers=headers,
            data=json.dumps(payload).encode("utf-8"),
        )

    @staticmethod
    def parse_position(raw: Mapping[str, Any]) -> dict[str, Any]:
        """Parse raw on-chain position dictionaries without floating-point loss.

        Per official Delphi SDK docs:
        - `shares` is an 18-decimal bigint string. If shares == "0", trader has no position.
        - `tokensRedeemed` is a 6-decimal USDC collateral bigint string.
        """
        raw_shares = str(raw.get("shares", "0"))
        raw_redeemed = str(raw.get("tokensRedeemed", "0"))

        has_shares = raw_shares != "0"
        shares_decimal = Decimal(raw_shares) / Decimal(10**18)
        redeemed_decimal = Decimal(raw_redeemed) / Decimal(10**6)

        return {
            "has_shares": has_shares,
            "shares_raw": raw_shares,
            "shares": shares_decimal,
            "shares_float": float(shares_decimal),
            "tokens_redeemed_raw": raw_redeemed,
            "tokens_redeemed": redeemed_decimal,
            "tokens_redeemed_float": float(redeemed_decimal),
            "outcome_index": int(raw.get("outcomeIndex", 0)),
            "market_id": raw.get("marketId", ""),
        }
