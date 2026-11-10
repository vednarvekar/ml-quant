# 13 - Risks & open questions

## Top risks
| Risk | Impact | Mitigation |
|---|---|---|
| No tradable edge exists in our data/horizon | Project yields no profit | Many families, honest gates, treat as research; platform still reusable |
| Overfitting / fooling ourselves | Real losses | CPCV, DSR, holdout, trial log, paper trading |
| Cost drag (STT, slippage) | Edge vanishes | Cost-first design, low turnover, liquid names |
| Short intraday history (minutes from 2022) | Few regimes | Daily-horizon families, longer daily data, caution on intraday |
| Alpha decay / regime change | Live decay | Monitoring, retraining, demotion rules, diversification |
| Bugs in engine | Wrong/duplicate orders | Risk layer, idempotent OMS, reconciliation, kill switch, small size |
| API/regulatory change | Access lost | Track SEBI/Upstox notices; compliance checklist |
| Token/connection failure with open positions | Unmanaged risk | Broker-side stop orders, alerts, flatten procedure |
| Scope creep (too many models) | Never ships | Phase gates, priority order in doc 03 |
| Emotional override / overtrading | Losses | Rules + semi-auto discipline |
| Data licensing/ToS | Legal | Personal use only, check sources |

## Decisions (from user, 2026-11-09)
| Topic | Decision |
|---|---|
| Hardware | Intel i5-1135G7 (4C/8T), 16 GB RAM, Iris Xe (no CUDA GPU). **CPU-only local.** Fine for data, GBDT, linear, HMM/GARCH, nets up to a few M params. |
| Colab | Used only for heavy training (Google AI Pro; confirm it includes Colab Pro/TPU access). **Never for live trading.** |
| Capital | Not decided; irrelevant until paper-trading phase. |
| Horizon | Intraday + swing. |
| Universe | NSE/BSE, Nifty 50/100/200 (flexible; start Nifty 100 for data size). |
| Time | Not a constraint; phases are gated by evidence, not dates. |
| Language | Python for research; Rust engine later (Python sidecar first). |
| Direction | Long and short (futures / MIS shorts). |
| Broker | Upstox demat account exists, API access via it. |
| Existing code | **Discard old ml-quant code; start from scratch.** Old log kept only as lessons (doc 01). |

## Still open (need you or a check)
1. **Static IP (decided: no paid options).** Live engine runs on the user's Ubuntu Dell server. Plan: log public IP daily for a week; if stable, whitelist in Upstox My Apps (changeable once/week) with an IP-change watchdog that halts trading on mismatch. Fallback if unstable/CGNAT: free-tier cloud VM in India with a fixed IP [verify terms] - to be tested with sandbox orders.
2. Does your Google AI Pro plan include Colab Pro / TPU v5e-1 compute units? (check Colab UI)
3. Can you create a new Upstox API app today? (check developer console)
4. Futures shorting needs F&O segment activated on Upstox: is it enabled?
5. Where the live engine will run (this laptop vs always-on box)?

## Verification log (2026-11-09)
Done: Upstox rate limits, historical limits, websocket caps, token life, sandbox scope, static IP + 10 OPS rule; Kronos sizes/licence/framework; STT 2026 rates (multi-source); Colab limits (official: unpublished, 24h needs Pro+); `ort` is 2.0.0-rc (release candidate, MSRV 1.81).
Still open:
- SEBI circular primary text (feb-2025 + later) and NSE/INVG/73992 (30 Apr 2026): not retrieved
- Upstox "May 2026" algo-app requirements (e.g. X-Algo-Name) and exact brokerage charges
- Whether new Upstox API apps can be created today
- candle / burn / polars-rs maturity (not searched successfully)
- Colab real TPU compute-unit burn (check in UI)
- Tax rules (CA)

## New risks from verification
- Static IP required: engine cannot run from Colab/most laptops on dynamic IP; need static IP or India VPS.
- Daily 3:30 AM token expiry: needs a morning login routine; failure = no trading that day.
- Kronos is CUDA/PyTorch only: TPU plan applies to our own JAX models, not Kronos.
- Only ~4.8y of intraday history: favour daily-horizon families first.
