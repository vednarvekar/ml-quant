# 03 - Strategy families (all tried in parallel on one scoreboard)

Each family = hypothesis + horizon + universe + models + known risks. IDs are used in the experiment registry (doc 11).

| ID | Family | Horizon | Universe | Core idea | Models | Main risk |
|---|---|---|---|---|---|---|
| **S1** | Cross-sectional daily ranking | 1-5 days | Nifty 200 (liquid) | Rank stocks by predicted forward return (market-neutral-ish: long top decile; short via index/stock futures hedge later) | Ridge, LightGBM, small MLP, Alpha101/158 features | Crowded factors, turnover cost |
| **S2** | Trend / momentum + vol targeting | days-weeks | Index + top stocks | AQR/AHL-style diversified trend, size by inverse vol | Rules + vol forecaster (HAR/GARCH/GBM) | Whipsaw in sideways markets |
| **S3** | Intraday primary rules + meta-labelling | minutes-hours | ~20-30 most liquid stocks | Simple primary signals (ORB, VWAP reversion, momentum) -> model filters which to take + sizes | LightGBM / small TCN on triple-barrier labels | Costs, slippage, intraday short only via MIS |
| **S4** | Regime detection overlay | any | Index + breadth | Detect trend/chop/high-vol regimes; switch or scale S1-S3 | HMM/GMM, GBM classifier | Regime labels lag |
| **S5** | Volatility forecasting for sizing | 1-20 days | All | Predict next-period vol; size positions, set stops | HAR-RV, GARCH, LightGBM | Mostly a helper, not an alpha |
| **S6** | Event / news driven | hours-days | Nifty 200 | Earnings/results dates, corporate announcements (BSE/NSE), headline sentiment drift | FinBERT-India, small LLM, GBM on event features | Data collection effort, latency, thin evidence |
| **S7** | Stat-arb / pairs | days | Same-sector pairs | Cointegrated pairs mean reversion | Kalman filter, z-score rules + GBM filter | Break of relationship, shorting constraints |
| **S8** | Foundation-model features | any | Nifty 200 | Kronos embeddings / forecasts as features into S1/S3 | Kronos (mini/small/base) fine-tune | Likely weak zero-shot |
| **S9** | Options-derived signals | days | Index + liquid F&O | PCR, OI build-up, IV skew/term as features (not selling options) | GBM | Data availability via API, noise |
| **S10** | RL for sizing/execution | n/a | Sim env | Learn sizing or entry timing given signals | PPO/SAC small | Overfitting; last priority |

## Priority order (by expected robustness / cost)
1. S1, S2 (daily; low cost drag, lots of literature) - **baseline strategies**
2. S5, S4 (helpers that improve all others)
3. S3 (intraday meta-labelling - what the old project wanted, done right)
4. S6, S9, S8 (new information sources)
5. S7, S10 (later)

## Scope decisions
Horizons: intraday + swing. Both long and short (stock/index futures, MIS shorts). Universe: NSE (BSE optional later), start Nifty 100. Priorities above unchanged: daily/swing families first because intraday history is only ~4.8 years.

## Rules for every strategy
- Entry/exit/sizing/stop all defined *before* testing.
- Must beat simple baselines: buy&hold Nifty, equal-weight Nifty 200, 12-1 momentum, random with same turnover.
- Must be tested across regimes (2022 bear/chop, 2023-24 bull, 2025-26 etc.).
- Capacity check: order size <= 1% of average traded value in window.
- Combine survivors through a portfolio layer (doc 07) - low correlation matters more than any single Sharpe.

## India-specific constraints that shape the families
- Cash market: cannot hold short overnight. Shorts only intraday (MIS) or via futures.
- Stock futures: lot sizes, margin, STT 0.05% sell side since Apr 2026 **[verify]**, monthly expiry rollover.
- Circuit limits, F&O ban list, T2F lists, corporate-action adjustments, 9:15-15:30 IST session (+ pre-open), weekly/monthly expiry effects.
