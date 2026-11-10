# 17 - Phase 2 + 3 results: harness gate and baseline leaderboard (2026-11-10)

Everything below is reproducible: `python -m mlq.rebuild`, `python -m mlq.swing`, `python -m mlq.intraday_strats`. All runs are logged in `data/experiments.jsonl` (the registry), so Deflated Sharpe counts every try.

## Phase 2 gate: PASSED (daily + 5-minute engines)
| Requirement | Evidence |
|---|---|
| Null test (random strategy ~ -costs) | `test_backtest`, `test_intraday`: random signals lose money, Sharpe < 0.5 |
| No-lookahead test | legit signal earns ~nothing, a peeking signal earns a lot on both engines (proves leaks would show) |
| Known-answer / execution timing | trap-price tests: fills happen at the NEXT bar/day open; intraday is flat by 15:15 |
| Reproducible | deterministic tests; one-command rebuild; registry log |
| Costs | `costs.py`: delivery / intraday / futures, slippage on every fill |
| Validation tools | purged walk-forward, CPCV, Deflated Sharpe (`validation.py`), experiment registry (`registry.py`) |
| F&O facts | `futures.py`: 218 F&O stocks (96 of Nifty 100), current lot sizes (e.g. RELIANCE 500 = ~Rs 7 lakh/lot, SBIN 750, TCS 225), 20% margin placeholder |
Still approximate: lot sizes are today's (history not available), margin is a flat 20%, circuit limits not modelled, MIS square-off assumed 15:15.

## Phase 3: swing leaderboard (daily engine, monthly rebalance, costs + 5 bps slippage)
Clean window = 2022-01 -> 2026-11 (point-in-time universe, 96% coverage). Long window = 2012-2021 (only 78% of the true top-100 had data -> **survivor-biased, optimistic**).

| Strategy | Clean CAGR | Clean Sharpe | Clean DD | Long Sharpe (biased) |
|---|---|---|---|---|
| EW top-100 | 10.0% | 0.64 | -23% | 0.77 |
| Trend200 + EW | 7.7% | 0.61 | -17% | 0.43 |
| Momentum 12-1, top 20 | 12.0% | 0.57 | -37% | 0.88 |
| Momentum + hysteresis (keep 40) | 8.7% | 0.46 | -41% | 0.99 |
| Low-vol 60d, 20 names | 6.5% | 0.57 | -19% | 0.99 |
| Weekly 5-day reversal | 0.9% | 0.16 | -43% | 0.23 |
| Combo mom+lowvol+trend | 9.0% | 0.66 | -19% | 0.87 |
| **Long/short momentum, F&O futures** | 7.5% | **0.71** | **-17%** | 0.39 |
| (Nifty 50 buy & hold, no costs) | 7.6% | 0.60 | -16% | - |

Read-out:
- Factor baselines give Sharpe ~0.5-0.7 net of costs. **None is statistically convincing** (DSR 0.5-0.7 on the clean window; need > 0.95). Four years is too short to tell.
- Weekly reversal dies on costs (93x turnover). Cost-aware design matters more than the signal.
- Market-neutral long/short (futures) has the best clean-window risk profile with tiny fees, but its long window is weak (0.39): regime-dependent. Lot sizes make it capital-hungry (one RELIANCE lot ~ Rs 7 lakh notional).
- The old "survivor" test shows the bias: with 78% coverage the same strategies look 20-40% better.

## Phase 3: intraday leaderboard (5-min engine, 40 liquid names, 2022-2026, MIS, K=5 positions)
Rules: opening-range breakout, gap fade, 30-min momentum, VWAP fade (14 variants). At 5 bps slippage per fill **every variant loses money** (net -7 to -21 bps per trade; Sharpe -0.7 to -7.6). Anchored walk-forward "pick the best past variant" gives OOS Sharpe -1.0, DSR 0.0.

But gross edge exists before costs (+6 to +9 bps/trade, mom30 and gap fade; VWAP fade is negative, VWAP *follow* is the mirror image). Sensitivity (`data/clean/intraday_sensitivity.csv`):
- Fewer, higher-conviction trades earn more per trade: gap fade (>1%) gross 16 bps at K=1 vs 8 bps at K=5; mom30 12.6 vs 8.8.
- Best simple case (gap fade >1%, 1 position, 2 bps slippage): net +8 bps/trade, Sharpe 0.43. At 5 bps slippage it is ~0.
- Fees alone are ~6 bps round trip, slippage 2 x slip. Break-even needs either gross > ~16 bps or passive/limit fills (<= 1-2 bps slippage).

## What this means for Phase 4 (the plan we derive)
1. **Intraday:** simple rules are not enough, but conviction matters, so a model that *ranks* trades (to lift gross per trade from ~8 to > 20 bps) is the right ML target. Features: gap, first-30-min return/volume, VWAP distance, VIX level/change, index return, sector index move, time of day, participant OI. Also test limit-order entry to cut slippage.
2. **Swing:** cross-sectional ranking models (LightGBM) on momentum/low-vol/reversal/volume/VIX features, long/short via futures for F&O names; hold 5-20 days, trade rarely; must beat Sharpe ~0.7 net.
3. **Risk/regime:** VIX- and index-trend-based exposure scaling (Trend200 cut drawdown from -23% to -17%).
4. Anything promoted must also survive: CPCV paths >= 70% positive, DSR > 0.95 after counting every logged trial, cost x2 and delay +1 bar.

## Phase 4, first ML experiments (2026-11-10, same harness, all trials logged)
**1. Intraday trade-ranking (`mlq.ml_intraday`, LightGBM, 80 liquid names, walk-forward every 63 days).** Features known at 09:45 (gap, first-30-min return/range/volume, previous-day returns, VIX). Out-of-sample rank IC **0.038 (t = 7.5)**: a real but small signal. Long top-k / short bottom-k from 09:50 to 15:15: gross +7.6 bps/trade at k=1, falling to 1.7 at k=3. At 5 bps slippage Sharpe -0.7; at 1 bp slippage Sharpe +0.12. **Not tradable at retail costs.** Intraday needs either a much sharper model or passive (limit) execution.

**2. Swing ranking (`mlq.ml_swing`, LightGBM, 21 daily features, point-in-time top-150 universe, retrained every 126 days, weekly rebalance).** Weekly rank IC **0.039 in the clean 2022+ window (t = 4.3)**, 0.053 in the survivor-biased 2016-2021 window. Most important features: VIX level, 20/60-day returns, market trend.

| Portfolio (2022+ clean, net of costs) | Slippage | CAGR | Sharpe | Max DD |
|---|---|---|---|---|
| Long-only top-20, delivery | 5 bps | 14.3% | **0.73** | -27% |
| Long-only top-20, delivery | 2 bps | 16.5% | **0.82** | -27% |
| Long/short 10+10, futures | 5 bps | -0.7% | -0.02 | -23% |
| Long/short 10+10, futures | 2 bps | 1.6% | 0.20 | -20% |
(Baselines to beat: EW top-100 Sharpe 0.64 / 10%; L/S momentum rule 0.71.)

Read-out: the model has real skill (IC), the long-only version already edges the best baseline, but **turnover is 60-70x per year** (weekly full reshuffle): fees were ~Rs 7.6 lakh on a Rs 10 lakh account over 4.8 years. The market-neutral version loses its edge to that turnover. DSR is still < 0.95, so this is not yet proven.

## Next experiments (cheap, high value)
1. Cut turnover: rank-hysteresis, trade only when predicted-rank change is large, 10-20 day holds, cost-aware portfolio optimiser.
2. Longer-horizon labels (10/20 days) so the signal per trade is bigger than the round-trip cost.
3. Add sector-index, participant-OI, USDINR/Brent/Nasdaq (lagged) features; try ranking-loss objective (LambdaRank) and an ensemble of LightGBM + linear.
4. Market-regime gating from VIX / Nifty trend for exposure.
5. Intraday: only trade the highest-conviction 1 stock/day, test limit-order fills.

## Phase 4, round 2: cutting turnover, horizons, futures (2026-11-10)
Predictions are saved per label horizon (`ml_swing_pred_h{5,10,20}.npy`); `mlq.swing_portfolio` tries 24 portfolio constructions (slower rebalance, rank hysteresis, score smoothing, partial rebalancing, trend gating) in both windows. All 48 runs are in the registry. 2022+ clean window, long-only delivery, 5 bps slippage:

| Variant | CAGR | Sharpe | Max DD | Turnover/yr | Fees (Rs) |
|---|---|---|---|---|---|
| Weekly top-20 (reference) | 18.0% | 0.90 | -28% | 59x | 7.8 lakh |
| Weekly + keep-while-rank<40 (hysteresis) | 19.5% | 0.98 | -25% | 35x | 5.6 lakh |
| Hysteresis + trend gate | 16.4% | 1.00 | -26% | 32x | 4.9 lakh |
| Rebalance every 20 days, top-20 | 19.3% | 0.95 | -20% | 19x | 4.0 lakh |
| (EW top-100 baseline / Nifty 50) | 10% / 7.6% | 0.64 / 0.60 | -23% / -16% | | |

**Honesty check:** the *same* weekly strategy scored Sharpe 0.73 in round 1 and 0.90 here, only because rebalances started on a different weekday. So Sharpe differences below ~0.15 between variants are noise, and DSR stays ~0.55 (not significant) after ~150 logged trials. What *is* consistent: every ML variant beats the equal-weight and index baselines in the 2022+ window, and longer horizons / slower rebalancing / hysteresis reduce cost without hurting.

**Biggest cost lever = STT.** Delivery STT is 0.1% each side (about 20 bps round trip). Same portfolios priced as stock futures (STT 0.05% sell only), F&O names only:

| Variant (2022+, 5 bps slippage) | Delivery Sharpe / fees | Futures Sharpe / fees |
|---|---|---|
| Weekly top-20 | 0.90 / 7.8 lakh | **1.36** / 2.8 lakh |
| Weekly hysteresis | 0.98 / 5.6 lakh | **1.31** / 1.7 lakh |
| 20-day top-20 | 0.95 / 4.0 lakh | **1.15** / 0.9 lakh |

Caveats before believing this: lot sizes are not enforced (holding 20 names needs roughly Rs 1 crore notional, ~Rs 20 lakh margin), no futures roll/basis cost, only today's F&O list, one 4.8-year window. Next: lot-aware sizing with a capital parameter, then paper trading of the best candidate.

## Phase 4, round 3: lot-aware futures (`mlq.futures_bt`, `mlq.futures_run`)
Whole lots (today's lot sizes), real capital, 20% margin cap, monthly roll round-trip cost, 5 bps slippage, F&O names only, ML h5 predictions, 2022+ window. `exposure` = notional / equity.

| Strategy | Capital | Exposure | CAGR | Sharpe | Max DD | Avg names held |
|---|---|---|---|---|---|---|
| top-5 weekly | Rs 10L | 1x | 34% | 1.10 | -36% | 4.1 |
| top-10 weekly hysteresis | Rs 20L | 1x | 32% | 1.24 | -28% | 7.9 |
| top-10 weekly hysteresis | Rs 50L | 1x | 23% | 1.03 | -26% | 9.8 |
| top-20 weekly hysteresis | Rs 50L | 1x | 23% | 1.01 | -29% | 18.0 |
| top-20 weekly hysteresis | Rs 1 crore | 1x | 22% | 1.06 | -22% | 19.7 |
| top-20, 20-day rebalance | Rs 1 crore | 1x | 22% | 1.08 | -24% | 19.8 |
| any of the above | any | 2x | 40-70% | 1.0-1.4 | **-43% to -59%** | |

Read-out:
- **Lot sizes decide the minimum account:** below ~Rs 20 lakh a 20-name portfolio holds 0-1 names (one lot is Rs 3-10 lakh). Roughly: Rs 10L -> 5 names (concentrated, DD -36%); Rs 20L -> 10 names; Rs 50L+ -> 20 names.
- The edge survives lots, margin and roll: Sharpe ~1.0-1.2 and 22-32% CAGR at 1x across account sizes where it can actually hold positions. Fees are 8-20% of capital over 4.8 years, and slippage is a separate cost on top.
- 2x leverage doubles return **and** drawdown (about -45%); not recommended before paper trading proves the live edge.
- Still unproven: DSR below 0.95 after ~200 trials, one 4.8-year window, today's lot sizes used for history, no futures basis modelled, ML trained on a 150-name liquidity universe.
- A bug found and fixed along the way: non-F&O names had a missing lot size, which silently disabled the margin cap (test added).

## CORRECTION (2026-11-10): universe look-ahead fixed, futures numbers revised down
Found while building paper trading: the point-in-time universe is stored as month-END snapshots, and the backtests applied each snapshot to the whole of that month (a small look-ahead in who was in the top-150/100). Fixed in `swing.Ctx` and `baselines`: day t now uses the previous month's snapshot. Also fixed in `futures_bt`: lots are no longer silently re-sized every day (trades only when the target weights change, as a real account would), plus a 5% margin buffer. Everything from "Phase 4, first ML experiments" onward was re-run. The old registry is archived as `data/experiments_pre_leakfix.jsonl`.

What changed (2022+ clean window):
- ML signal is intact: weekly rank IC **0.047 (t = 4.8)**, vs 0.039 before.
- Baselines about the same (EW top-100 Sharpe 0.61, Combo 0.70, L/S momentum rule 0.74).
- Best delivery ML portfolios: Sharpe 0.9-1.0, CAGR 18-20%, max DD -21% to -28%, DSR 0.6-0.65 (still not significant).
- **Lot-aware futures, revised (1x exposure, 5 bps slippage, monthly roll):**

| Strategy | Capital | CAGR | Sharpe | Max DD | Names held |
|---|---|---|---|---|---|
| top-5 weekly | Rs 10L | 33% | 1.17 | -26% | 3.5 |
| top-10 weekly hyst | Rs 20L | 25% | 1.00 | -24% | 7.2 |
| top-10 weekly hyst | Rs 50L | 13% | 0.67 | -25% | 9.8 |
| top-20 weekly hyst | Rs 50L | 21% | 0.95 | -26% | 17.9 |
| top-20 weekly hyst | Rs 1 crore | 16% | 0.86 | -23% | 19.8 |
| top-20, 20-day rebalance | Rs 1 crore | 22% | 1.09 | -23% | 19.8 |

Honest summary: roughly **Sharpe 0.8-1.0 and 15-25% a year** across account sizes that can hold positions (the old 1.0-1.2 / 22-32% was inflated by the leak). Variation between near-identical configurations is large (e.g. 13% vs 21% CAGR for k=10 vs k=20 at Rs 50L), which is what noise looks like on a 4.8-year window. Do not read precision into any single row.

## Paper trading (built 2026-11-10): `mlq.paper`
- `broker.py`: simulated futures broker (lots, margin, slippage, fees, monthly roll), state saved in `data/paper/state.json`; there is no connection to any real order API.
- `signals.py`: same features, LightGBM model and hysteresis selection as the backtest; retrained at least every 126 trading days.
- `update.py`: incremental Upstox daily candles (read-only token, 1 request/s) + NSE bhavcopy + offline rebuild.
- `run.py`: daily runner. Processes new days, decides every 5 trading days (executes at the next open), writes `data/paper/{equity.csv, trades.csv, decisions.jsonl, report.md}`. Guards: KILL file, stale data, feature mismatch, >15% moves in a held name.
- **Self-test:** `python -m mlq.paper.run --replay 2022-01-03` pushes 4.8 years through the live code path and ends at exactly the same equity as the backtest engine (Rs 58,78,708 vs Rs 58,78,708).
Daily use: `uv run python -m mlq.paper.run --update` after market close (about 10 minutes the first time).
