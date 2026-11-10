# 07 - Backtesting, costs, risk

## 1. Simulator requirements
- Event-driven, bar-based (1m/5m/1d), deterministic, same strategy code path as paper/live where possible.
- Order at bar t executes at t+1 open (or VWAP of next bar) - never at the close that generated the signal.
- Fill model: market orders pay half-spread + impact; limit orders fill only if price trades through; partial fills for illiquid names.
- Slippage model: base bps by liquidity bucket + volatility-scaled component; calibrate later with real paper fills.
- Constraints: circuit limits, lot sizes (F&O), margin, MIS square-off (~15:15-15:20 **[verify]**), no overnight shorts in cash.
- Corporate actions applied correctly; delistings handled.
- Portfolio-level: cash, leverage cap, concentration caps, sector caps.
- Determinism: same config + seed -> identical result.

## 2. India cost model (verify rates against broker contract notes before trusting)
| Item | Value |
|---|---|
| STT equity delivery | 0.1% buy & sell |
| STT equity intraday | 0.025% sell side |
| STT futures | 0.05% sell side (raised from 0.02% on 1 Apr 2026) **[verify]** |
| STT options | 0.15% of premium on sell side (raised) **[verify]** |
| Brokerage (discount broker typical) | Rs 0 delivery; ~Rs 20 flat or 0.03% intraday/F&O - **Upstox exact: verify** |
| Exchange txn charges, SEBI fee, stamp duty (buy), GST on brokerage+txn | Add per contract note (small but not zero) |
| **Implemented:** `src/mlq/costs.py` (`leg_cost`, `round_trip`) | Rs 1 lakh flat-price round trip, 5 bps slippage per fill: intraday ~18 bps (10 slippage + ~8 fees), delivery ~39 bps (10 slippage + ~29 fees). A strategy must beat this per trade. Delivery shorts are rejected. Brokerage, exchange, DP rates marked [verify] in code. |
| DP charges on delivery sell | per-scrip flat fee |
| Slippage | Our model, 2-10 bps typical large-cap, more mid-cap |
Rule of thumb: an intraday round trip costs roughly 5-15 bps all-in on liquid large caps. A signal must beat that *consistently*.

## 3. Validation of the simulator itself
- Null tests: random entries -> expected PnL ~ -costs.
- Known-answer tests: hand-computed trades match.
- Replay paper-trade fills against simulator predictions to calibrate slippage.
- No-lookahead test: shifting features by +1 bar must not improve results.

## 4. Metrics & promotion gates (proposed, tune later)
| Stage | Gate |
|---|---|
| Research -> candidate | Net Sharpe > 1.0 in walk-forward, DSR p-value acceptable, positive in >= 70% of CPCV paths, max DD < 20%, turnover within budget |
| Candidate -> paper | Passes holdout once, stress tests, code review of engine path |
| Paper -> tiny live | >= 40 trading days, net Sharpe within ~50% of backtest, slippage within model, zero risk-limit breaches, no ops incidents |
| Tiny live -> scale | >= 3 months live, DD within plan, human sign-off; scale in steps (2x max) |
| Kill/demote | Rolling Sharpe < 0 over N days, DD > limit, IC collapse, live-vs-sim gap > threshold |

## 5. Stress & robustness tests
- Parameter sensitivity (+/-20%), cost x2, slippage x2, delay +1 bar.
- Subperiod and regime splits (2022 chop, 2024 election volatility, 2025-26 etc.).
- Crash days (e.g. circuit days, budget day, election result day), gap opens.
- Universe perturbation (drop top contributors).
- Capacity: scale capital 10x and watch decay.

## 6. Risk layer (deterministic code, outside models)
Position/portfolio:
- Max risk per trade (e.g. 0.25-0.5% of equity), max position % of equity, max sector %, max gross/net exposure.
- Volatility-targeted sizing; fractional Kelly (<= 1/4 Kelly) at most.
- Max order value, max qty vs ADV, price-band / fat-finger checks, duplicate-order guard.
Account:
- Daily loss limit (halt trading), weekly loss limit, max drawdown stop -> require human restart.
- Max open orders, max orders/min (stay far below the VERIFIED Upstox limit of 10 orders/sec, 500/min, 2000 per 30 min).
System:
- Global kill switch (UI + hotkey + remote), flatten-all command.
- Heartbeats on feed/broker; stale data => no new orders.
- Position reconciliation with broker every N seconds; mismatch => halt.
- Market-hours and holiday guard; MIS auto square-off before cut-off.
- Every decision logged with inputs (audit trail).
