# 19 - Response to the audit: fixes, honest re-test, decisions (2026-11-11)

Audit: [18-audit-2026-11-10.md](18-audit-2026-11-10.md). It said the platform is good but the edge is unproven. We agree, fixed what could be fixed with free data, and re-tested everything the same strict way.

## What we changed (audit step -> what was done)
| Audit item | Done |
|---|---|
| 1. Fixed measuring stick | `evaluate.py`/`compare*.py`: every result is averaged over all rebalance start phases, compared with random-score, 5-day-reversal and universe baselines plus Nifty 50 / Midcap 100 / Smallcap / Nifty 500 index buy&hold, with a factor regression (alpha t-stat vs Nifty, Midcap-Nifty, Small-Mid) |
| 2. Futures biases | Downloaded the NSE **F&O bhavcopy** 2012-2026 (3,671 days, `nse_fo.py`). Backtests now use point-in-time F&O membership, historical lot sizes (inferred from open interest, = gcd of OI numbers; RELIANCE 2013 = 250 correct), ex-dividend prices, roll 2 trading days before each expiry, and positions tracked in shares |
| 3. Universe holes | Model universe cut to the point-in-time **top-100** (coverage 96%) |
| 4. Re-target model | Residual (beta-neutral) label tried; new features: futures OI build-up, OI vs history, futures volume, basis, beta, size regime, and **delivery %** (new download `nse_mto.py`, 3,672 days) |
| 5. Event/text model | **Not done.** The announcement sources (BSE/NSE APIs) block scripts; we will not circumvent that. Needs a licensed or manual feed. |
| Plumbing #1 (critical) | Paper broker now rolls 2 days before expiry (stock futures are physically settled) instead of the 1st of the month |
| Plumbing #6 | Daily run moved to 19:30; paper trader now uses model v3 and pulls futures + delivery data daily; a feature mismatch triggers an automatic retrain |
Also found and fixed in our own engine: lots were counted instead of shares, so a noisy inferred lot size created phantom P&L (equity below zero). Live paper code still matches the backtest engine to the rupee (self-test), 46 tests pass.

## How much the old numbers were inflated
Same old model, same 2022+ window, biases removed: Rs 20L top-10 futures Sharpe **1.11 -> 0.71**; Rs 1 crore top-20 **0.8-1.1 -> 0.81**. Random-score picks with the same setup score 0.23-0.48.

## Honest results, 2022-01 -> 2026-11, 5 bps slippage, whole lots, margin cap, roll costs, phase-averaged
Signal skill (weekly rank IC, 2022+): old model 0.046 (t 4.8) | v2 (top-100, futures features) 0.045 (t 4.2) | **v3 (+ delivery %) 0.058 (t 5.5)**.

**Long-only stock futures**
| Strategy | Sharpe mean [min, max] | CAGR | Max DD | Alpha t (vs Nifty/size factors) |
|---|---|---|---|---|
| v3, Rs 1 crore, top-20 | **0.88** [0.81, 0.93] | 16.9% | -29% | 0.65 |
| v3, Rs 20L, top-10 | 0.57 [0.33, 0.98] | 4.5% | -13% | 0.37 |
| v2, Rs 1 crore top-20 | 0.60 / 0.69 | 11% | -30% | -0.8 / -0.2 |
| random picks, Rs 1 crore top-20 | 0.40 [0.18, 0.66] | 6.2% | -35% | -2.1 |
| random picks, Rs 20L top-10 | 0.35 [-0.16, 0.71] | 2.8% | -17% | -0.3 |
| 5-day reversal rule, Rs 1 crore | 0.48 | 8.6% | -32% | -1.3 |
| **Midcap 100 index fund** | **0.91** | 15.7% | -21% | |
| Nifty 500 / Nifty 50 index | 0.73 / 0.62 | 9.9% / 8.0% | -19% / -17% | |

**Hedged with Nifty futures (long stocks, short index):** v3 Rs 1 crore **0.51**; Rs 20L **-0.79** (the long book is mostly market + size beta).

**Market-neutral long/short stock futures (Rs 5 crore, equal weights, k per side)**
| Model | k, rebalance | Sharpe mean [min, max] | CAGR on capital | Max DD | Alpha/yr, t |
|---|---|---|---|---|---|
| **v3 (delivery %)** | 10, 5-day | **0.70** [0.56, 0.85] | 5.9% | -11.5% | 6.5%, 1.66 |
| **v3** | 15, 5-day | **0.67** [0.53, 0.83] | 4.7% | -8.5% | 5.4%, 1.64 |
| v3 | 15, 10-day | -0.08 | -0.8% | -17% | 0, 0.0 |
| v2 raw / v2 resid | 15, 5-day | 0.09 / -0.30 | ~0 / -2.6% | | |
| random L/S | 15, 5-day | -1.15 | | | |
| 5-day-reversal L/S | 15, 5-day | -0.45 | -4% | -34% | -1.2 |

## Verdict
- **Real but modest skill.** Delivery % clearly helped (IC 0.045 -> 0.058). The v3 market-neutral book is positive in every start phase and beats random/reversal L/S by a wide margin, with small drawdowns. Alpha is about 5-6% a year on gross capital with t about 1.7: suggestive, not proven (needs t > 2 and holds only at 5-day rebalance, failing at 10-day).
- **Long-only does not beat a Midcap 100 index fund** (0.88 vs 0.91). For an ordinary investor the index fund is the better, cheaper choice; the model's value, if any, is the market-neutral part.
- **Capital matters:** every name is a whole lot. Long/short needs about Rs 5 crore; long-only top-20 needs about Rs 1 crore; Rs 20L gives a concentrated, weaker book. We do not have evidence for small accounts.
- **Not ready for real money.** Paper trading is a plumbing and fill-quality test, not proof of edge.

## Still open (next)
1. Fixed baselines in `compare*.py` are the standard from now on; add DSR across all logged trials.
2. Try longer horizons (10-20 day) with delivery/OI features and a sector-neutral label; test L/S with beta-neutral hedge for lower variance.
3. Real futures prices/basis (we use spot ex-dividend; carry not modelled), real margins (flat 20%), F&O ban list.
4. Event model needs a legitimate announcements feed (not scraped from blocked APIs).
5. Incremental feature build to make the daily run faster (currently ~13 min, mostly downloads).

## Round 2 (2026-11-11): longer horizons and a luck check
- **Longer horizons do not help.** Delivery/OI model trained for 10 and 20 day labels: weekly rank IC (2022+) 0.058 (5d) -> 0.041 (10d) -> 0.020 (20d). Market-neutral L/S, Rs 5 crore: 10-day model best case Sharpe 0.37 (k=10, rebalance 5d), 0.08 at 10d rebalance; **20-day model negative (-0.3 to -0.5)**. The signal lives only at the 5-day horizon, which is also the horizon where costs bite hardest.
- **Luck check (Deflated Sharpe, rough).** About 35 distinct ML portfolio configurations were evaluated in this audit round (several models x portfolio types x k x rebalance x capital). Their Sharpes have mean 0.09, std 0.55, max 0.88, so the best Sharpe expected from luck alone is about **1.2**. Our best market-neutral result is 0.70, giving a Deflated Sharpe probability of about **0.15** (we want > 0.95). In plain terms: after counting every try, the 0.70 cannot be told apart from luck. The estimate is crude (it mixes very different strategy types) but the direction is clear.
- **Therefore:** keep the v3 5-day market-neutral book as the single candidate and judge it only on **new, untouched data** (paper trading from 2026-11-09 on), not on more re-tuning of the 2022-2026 window. Further tuning on the same window only deepens the multiple-testing problem.
