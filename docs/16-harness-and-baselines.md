# 16 - Backtest harness + first baselines (2026-11-10)

## Harness (Phase 2, daily engine) - `src/mlq/backtest.py`, `src/mlq/costs.py`
- Signal at close of day t -> trade at OPEN of t+1. Every fill pays slippage (5 bps default) + full India fees. Delivery = long only; futures kind allows shorts.
- Tests (`tests/test_backtest.py`): known-answer (trap close price proves next-open execution), delivery short rejected, **null test** (random stock picks lose money, Sharpe < 0.5), **no-lookahead test** (a legit same-day-close signal earns ~nothing; a peeking signal earns a lot, so leaks would show), determinism.
- **Validation tools done** (`src/mlq/validation.py`, `tests/test_validation.py`): `purged_walk_forward` (train only on the past, label-overlap purge + embargo), `cpcv_splits` (combinatorial purged CV; 6 groups, 2 test = 15 splits, 5 paths), `deflated_sharpe` (penalises the number of variants tried). Tests prove no train/test overlap, purge/embargo gaps, and that DSR drops with more trials and rejects best-of-200 noise.
- NOT done yet: 5-minute intraday engine, experiment registry (log every trial so DSR knows N), futures margin/lots, circuit limits. Gate for Phase 2 is only half met (daily engine + null/lookahead tests).

## Baselines (Phase 3, first run) - `python -m mlq.baselines`
Window 2022-01 -> 2026-11, Rs 10 lakh, monthly rebalance, delivery, costs + 5 bps slippage. Universe = point-in-time liquidity rank (coverage of names we hold data for: 95-96%).

| Strategy | CAGR | Sharpe | Max DD | Turnover/yr | Fees (Rs) |
|---|---|---|---|---|---|
| EW top-50 | 7.4% | 0.50 | -24% | 4.0x | 1.49 lakh |
| EW top-100 | 9.7% | 0.64 | -23% | 3.3x | 1.40 lakh |
| Momentum 12-1 (20 of top-100) | 12.6% | 0.60 | -36% | 9.6x | 2.84 lakh |
| Nifty 50 buy&hold (no costs) | 7.6% | 0.60 | -16% | 0 | 0 |

## Read-out
- Nothing beats the index on risk-adjusted terms yet (gate: any net Sharpe > 0.5 -> yes, EW top-100 / momentum / Nifty are all ~0.5-0.65, so the bar for ML is ~0.65).
- Monthly full rebalance costs 1.4-2.8 lakh over 4.8 years: **turnover is the enemy**. Next baselines should rebalance only on drift/threshold.
- 4.8 years only, one regime: treat as a sanity check, not evidence.
