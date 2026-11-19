# Gensyn Delphi Protocol Feedback & Systems Engineering Report

> **Target Competition**: DoraHacks ID `2479` — Gensyn Delphi Agent Competition 2 ([Competition Detail](https://dorahacks.io/hackathon/delphi-agent-competition2/detail))  
> **Eligibility**: Protocol Feedback Bounties (`5 x $250 USDC`) & Track `4797` (`$7,250` Evidence Mining Agents)  
> **Submitted By**: DelphiPulse Engineering (`delphipulse-miner-v1`)  
> **Repository**: [`Khanhtnq/delphipulse-evidence-miner`](https://github.com/Khanhtnq/delphipulse-evidence-miner)  
> **Live Cockpit**: [`https://khanhtnq.github.io/delphipulse-evidence-miner/`](https://khanhtnq.github.io/delphipulse-evidence-miner/)  
> **Payout Wallet**: `0xcbEDbeCC127a32217075Ff40f16Ec804aa1E0533`

---

## Executive Summary

While building **DelphiPulse** and executing live telemetry probes against the Gensyn Competition Testnet (`Chain ID 685685`), Delphi REST API (`https://delphi-api.gensyn.ai`), and Goldsky Subgraph (`_meta` introspection), we audited the protocol's market scoring rules, evidence reward mechanics, Reproducible Execution Environment (REE) verification pipeline, and anti-gaming invariants.

This report delivers **five concrete, quantitatively verified systems engineering recommendations** with reproducible edge-case measurements and drop-in architectural fixes.

---

## Feedback 1: Hanson LMSR Liquidity `b` Sensitivity & Price-Impact Collapse Near Extreme Probabilities ($p \to 0 / 1$)

### Observation & Mathematical Diagnosis
In Hanson's Logarithmic Market Scoring Rule (LMSR), the binary cost and instantaneous price functions are:

$$C(q) = b \cdot \ln\!\left(e^{q_0 / b} + e^{q_1 / b}\right), \qquad P_1(q) = \frac{e^{q_1 / b}}{e^{q_0 / b} + e^{q_1 / b}}$$

Taking the partial derivative of market probability with respect to outstanding outcome shares yields:

$$\frac{\partial P_1}{\partial q_1} = \frac{P_1 (1 - P_1)}{b}$$

Near extreme probabilities ($p \to 0$ or $p \to 1$), $P_1(1 - P_1)$ collapses **quadratically** relative to a balanced market ($p = 0.50$). Conversely, shifting the market by a fixed Bayesian likelihood ratio ($\text{LR} = e$, i.e., $+1.0$ logit) requires $\Delta q = b \cdot \Delta \text{logit}(P)$ shares, forcing share quantities $q_i / b$ to diverge logarithmically and triggering IEEE-754 `exp()` overflow unless every smart-contract and off-chain pricing node applies Log-Sum-Exp shifting ($m = \max_i(q_i / b)$).

### Quantitative Measurements (`delphipulse.lmsr`)
Reproduced directly via `lmsr_prices()` and `lmsr_trade_cost()` for a fixed $+10$-share evidence impulse across $b \in \{10, 100, 1000\}$:

| Liquidity `b` | Prior $P_{\text{YES}}$ | Sensitivity $\frac{\partial P}{\partial q}$ | Post-$+10$ Shares $P'_{\text{YES}}$ | Collateral Cost $\Delta C$ (USDC.e) | Shares for $\text{LR}=e$ |
|---:|---:|---:|---:|---:|---:|
| `10.0` | `0.500` | `0.025000` | `0.731059` | `6.2011` | `10.0` |
| `10.0` | `0.900` | `0.009000` | `0.960730` | `9.3470` | `10.0` |
| `10.0` | `0.990` | `0.000990` | `0.996298` | `9.9366` | `10.0` |
| `10.0` | `0.999` | `0.000100` | `0.999632` | `9.9937` | `10.0` |
| `100.0` | `0.500` | `0.002500` | `0.524979` | `5.1249` | `100.0` |
| `100.0` | `0.990` | `0.000099` | `0.990943` | `9.9048` | `100.0` |
| `100.0` | `0.999` | `0.000010` | `0.999095` | `9.9905` | `100.0` |
| `1000.0` | `0.500` | `0.000250` | `0.502500` | `5.0125` | `1000.0` |
| `1000.0` | `0.999` | `0.000001` | `0.999010` | `9.9900` | `1000.0` |

**Key Risk**: At $b = 100$, $\frac{\partial P}{\partial q}$ drops by **$250\times$** between $p = 0.50$ (`0.002500`) and $p = 0.999$ (`0.000010`). Meanwhile, on thin testnet markets ($b = 10$), a single noisy `+10`-share impulse at $p = 0.50$ swings probability by $+23.1\%$ (`0.500 -> 0.731`), creating high KL-divergence reward volatility for low-conviction claims.

### Actionable Protocol Recommendation
1. **Liquidity-Sensitive Automated Market Maker (LS-LMSR)**: Scale $b(q) = \alpha \sum_i |q_i|$ with a floor $b_{\min}$ calibrated to market pot size ($b_{\min} \ge 0.25 \cdot \text{pot}_{\text{USDC}}$), preventing thin-market manipulation near $p=0.50$ while preserving depth as volume grows.
2. **Probability Clamping & Logit Bounds**: Enforce canonical probability bounds $p \in [10^{-6}, 1 - 10^{-6}]$ at the judge scoring layer and mandate Log-Sum-Exp stabilization (`m = max_i(q_i / b)`) in reference Solidity/Python pricing implementations.

---

## Feedback 2: KL-Divergence Reward Scaling vs. Wash-Evidence Splitting & Parallel Stale-Prior Exploits

### Observation & Mathematical Diagnosis
Delphi rewards evidence submissions using Bayesian information gain $D_{KL}(P_{\text{post}} \parallel P_{\text{prior}})$. We analyzed two distinct adversarial strategies around an authoritative evidence shift $P_0 = 0.50 \to P_n = 0.90$ on a $\$1,000$ USDC.e market pot:

1. **Sequential Wash-Evidence Splitting (Single-Agent Multi-Step)**: Splitting $0.50 \to 0.90$ into three sequential sub-claims ($0.50 \to 0.65 \to 0.80 \to 0.90$) where the judge updates the prior after each step.
2. **Parallel Stale-Prior Sybil Replication (Multi-Agent Race)**: Submitting $k=3$ semantically equivalent claims concurrently against the un-updated prior $P_0 = 0.50$ before state finality.

### Quantitative Measurements
Computed via `kl_divergence_bernoulli()` and `estimate_evidence_bounty()` (`pot = $1,000` USDC.e):

| Strategy | Path | Total $D_{KL}$ (bits) | Estimated Payout (USDC.e) | Relative Extraction |
|---|---|---:|---:|---:|
| **Honest Single Submission** | `0.50 -> 0.90` | `0.531004` | `$99.31` | `1.00x` (Baseline) |
| **Sequential 3-Step Split** | `0.50 -> 0.65 -> 0.80 -> 0.90` | `0.197042` | `$34.75` | `0.35x` (Penalized) |
| **Parallel Stale-Prior (3 Sybils)** | `3 x (0.50 -> 0.90)` | `1.593013` | `$297.94` | **`3.00x` (Exploit)** |
| **Unverified Extreme Jump** | `0.50 -> 0.999` | `0.988592` | `$135.30` | `1.36x` (Over-confidence risk) |

**Why This Matters**:
- By the Bregman divergence / Pythagorean discrepancy identity for KL divergence, for any monotonic path $P_0 < P_1 < \dots < P_n$:
  $$D_{KL}(P_n \parallel P_0) - \sum_{i=1}^{n} D_{KL}(P_i \parallel P_{i-1}) = \sum_{i=1}^{n-1} (P_n - P_i) \ln\!\left(\frac{P_i(1 - P_{i-1})}{(1 - P_i)P_{i-1}}\right) > 0$$
  Sequential splitting is strictly **disincentivized** ($0.1970 < 0.5310$ bits) *only if* prior updates are strictly serialized.
- **However**, if concurrent submissions read a stale prior $P_0$ within the same block or judge batch window, $k$ Sybil agents extract $k \times$ the full KL payout (`$297.94` vs `$99.31`). Furthermore, because $D_{KL}(P_{\text{post}} \parallel P_{\text{prior}})$ is measured **ex-ante** (without conditioning on the eventual ground-truth resolution $Y \in \{0, 1\}$), an aggressive over-confident claim pushing $0.50 \to 0.999$ captures `0.9886` bits even if wrong!

### Actionable Protocol Recommendation
1. **Serial Epoch Ordering + Semantic Embedding Deduplication**: Score submissions sequentially in deterministic timestamp/nonce order per block, and apply cosine-similarity / MinHash clustering over canonical claim text + normalized URL domains so near-duplicate parallel claims receive `0` marginal KL credit.
2. **Resolution-Conditioned Escrow Holdback (Brier / Log-Score Delta)**: Release at most $25\%$ of the $D_{KL}$ judgment reward immediately and escrow $75\%$ until market resolution $Y$, scaling final settlement by $\Delta \text{LogScore}(Y) = \ln P_{\text{post}}(Y) - \ln P_{\text{prior}}(Y)$ so confident false evidence yields negative net payout.

---

## Feedback 3: Gensyn REE Deterministic JSON Canonicalization & Cross-Runtime Hashing Edge Cases

### Observation & Engineering Diagnosis
Gensyn's Reproducible Execution Environment (REE) validates off-chain ML judge outputs by recomputing cryptographic SHA-256 hashes over serialized JSON receipts (`inputs_hash`, `receipt_hash`). While implementing `delphipulse/ree_verifier.py` (Python) and mirroring it in `docs/index.html` (JavaScript), we identified **four silent cross-runtime hash-mismatch vectors**:

1. **Signed Zero (`-0.0` vs `0.0`) & Float Representation**: In Python `json.dumps({"x": -0.0})` outputs `{"x": -0.0}`, whereas ECMAScript `JSON.stringify({x: -0.0})` outputs `{"x":0}`. Rounding near zero can produce `-0.0` in IEEE-754 arithmetic.
2. **Unicode Normalization (NFC vs NFD) & ASCII Escaping**: Evidence claims scraped from web sources (e.g., PDFs vs HTML) frequently mix NFC (`\u00e9`) and NFD (`e\u0301`) sequences, or toggle `ensure_ascii=True` vs `False`.
3. **Half-to-Even vs Round-Half-Away-From-Zero**: Python `round(x, 8)` uses banker's rounding (round half to even), whereas JavaScript `Math.round(x * 1e8) / 1e8` rounds half toward $+\infty$—causing 1-ULP divergence on boundary floats like `0.123456785`.
4. **Array Ordering & Duplicate Sources**: Sorting `sources` without deduplicating (`sorted(set(sources))`) allows identical URLs repeated twice to alter `inputs_hash` without adding new information.

### Reproducible SHA-256 Divergence Proof
Measured directly with Python `hashlib.sha256`:

| Edge Case Pair | Payload A | Payload B | SHA-256 Prefix A | SHA-256 Prefix B | Match? |
|---|---|---|---|---|---|
| Signed Zero | `{"x": 0.0}` | `{"x": -0.0}` | `b89cd506a17a4dc0` | `3e00e84fd9d396f6` | **MISMATCH** |
| Unicode NFC vs NFD | `{"c": "caf\u00e9"}` | `{"c": "cafe\u0301"}` | `9ca7f76e5c57628c` | `89ebe37408d456dc` | **MISMATCH** |
| UTF-8 vs Escaped ASCII | `ensure_ascii=False` | `ensure_ascii=True` | `9ca7f76e5c57628c` | `7325344e791098f6` | **MISMATCH** |

### Actionable Protocol Recommendation
1. **Adopt RFC 8785 (JSON Canonicalization Scheme — JCS)** across all REE judge workers and client SDKs, plus mandatory **Unicode NFC normalization** (`unicodedata.normalize("NFC", text.strip())`) prior to hashing.
2. **Ban Raw Floats in Receipt Preimages**: Encode probabilities and KL divergence in receipts as **fixed-point integer strings** (e.g., `10^8` micro-prob or `10^18` wei-scaled strings) and USDC.e rewards as `10^6` integer strings—completely eliminating cross-language IEEE-754 rounding drift.
3. **Canonical Source Set**: Deduplicate and normalize URLs (`sorted(set(normalized_urls))`) before hashing `inputs_hash`.

---

## Feedback 4: Anti-Self-Cancellation Cooldown & Honest Bayesian Belief-Revision Window

### Observation & Incentive Analysis
Under current competition rules, submitting an opposing directional stance (`YES` vs `NO`) on a market where the agent previously submitted evidence triggers a **binary, permanent self-cancellation penalty** (blocking leaderboard eligibility). While this successfully deters naive wash-hedging (submitting both sides to guarantee one winning stance), it introduces an unintended negative side effect:

- **Honest Belief-Revision Trap**: In real-world prediction markets (e.g., macro releases, protocol governance votes, software release freezes), new primary-source evidence frequently invalidates an earlier signal **before** `evidenceCutoff`. Under a permanent stance lock, a disciplined Bayesian agent cannot publish the corrective counter-evidence without forfeiting its leaderboard eligibility, degrading overall market calibration.

### Actionable Protocol Recommendation
Replace the permanent binary lock with a **Cooldown-Gated Explicit Retraction & Net-Brier Rule**:
1. **Minimum Cooldown Window ($\tau_c = 6\text{ hours}$)**: Reject rapid opposing flips (`Δt < 6h`) unconditionally as wash-hedging.
2. **Explicit `supersedes_submission_id` Flag**: Allow an agent to submit an opposing stance after $\tau_c$ **only** by forfeiting any unearned KL reward from its prior submission on that market and incurring a small revision friction fee ($\delta = 10\%$ of prior claimed share) if the flip occurs within the final 24h before cutoff.
3. **Net Trajectory Scoring**: Score the agent on the **net** movement from market prior $P_0$ to its final stance at cutoff against true outcome $Y$, making two-sided hedging strictly zero-sum or negative-EV while preserving honest Bayesian error correction.

---

## Feedback 5: Live Endpoint & SDK Ergonomics (18-Decimal `shares` Parsing, CORS Headers, & Subgraph Discovery)

### Observation from Live Protocol Probes
Verified against live endpoints via `python -m delphipulse.cli live-probe`:
- **Gensyn Testnet JSON-RPC** (`https://gensyn-testnet.g.alchemy.com/public`): `eth_chainId = 685685` (`OK`)
- **Delphi Competition REST API** (`https://delphi-api.gensyn.ai/health`): `{"status": "ok"}` (`OK`)
- **Goldsky Subgraph GraphQL**: Responds cleanly to `_meta { block { number } }`, while market entity field names can evolve between testnet deployments.

### Three High-Impact Developer Experience Fixes
1. **Lossless Fixed-Point Position Parsing**: Delphi positions return `shares` as 18-decimal strings (`"0"` = empty position) and `tokensRedeemed` as 6-decimal strings. Standard JS `Number()` / Python `float()` parsing corrupts precision above $2^{53}$ (`~9e15`, i.e., `< 0.01` 18-decimal shares!). Official docs and starter kits should enforce `BigInt` / `Decimal` parsing everywhere (implemented in `DelphiClient.parse_position()`).
2. **CORS Pre-Flight Whitelisting for Competition Headers**: Competition endpoints require `X-Delphi-Mode: competition`, `X-Agent-Id`, `X-Agent-Harness`, and `X-Agent-Model`. Adding all four explicitly to `Access-Control-Allow-Headers` on `delphi-api.gensyn.ai` enables browser-native agent cockpits (such as GitHub Pages SPAs) to interact directly without needing a proxy.
3. **Versioned Subgraph Meta Endpoint (`/v1/subgraph/meta`)**: Exposing active entity names and current indexed block height via REST allows autonomous miners to adapt automatically across subgraph schema upgrades without broken GraphQL queries.

---

## Verification & Reproduction

All quantitative metrics and protocol invariants in this report are directly reproducible using the **DelphiPulse** codebase:

```bash
python -m pytest -v
python -m delphipulse.cli benchmark
python -m delphipulse.cli evaluate --prior 0.35 --lr 3.2 --pot 500
python -m delphipulse.cli live-probe
```
