# 00 - Vision & principles

## Goal
A semi-automated (later maybe fully automated) trading system for Indian markets via Upstox, built from **small, specialised, well-validated models** orchestrated by a rule/risk layer, running on a lightweight Rust engine.

Same idea as `jitmo` (small purpose-built models + deterministic engine around them), applied to markets.

## What "success" means
Not accuracy. Success = **positive net-of-cost, out-of-sample, risk-adjusted return** that survives:
1. purged walk-forward / CPCV validation
2. realistic India cost model (STT, brokerage, slippage)
3. 1-2+ months of paper trading
4. a tiny-capital live period

Honest expectation: most retail algo traders lose. A realistic win is a modest edge (net Sharpe ~0.7-1.5) with controlled drawdown, not "10x". Preserving capital is the first win.

## Principles (never break)
1. **Plan before code.** Each phase has a written gate.
2. **Backtester before models.** A model is only as trustworthy as the harness that grades it.
3. **Null-test everything.** Random signals must lose money after costs in our simulator. If not, the simulator is wrong.
4. **Many approaches, one fair scoreboard.** Never marry one idea; every approach goes through the same harness (doc 11).
5. **Log every trial.** Count how many things we tried (needed for Deflated Sharpe). No silent cherry-picking.
6. **Risk layer is deterministic code, not a model.** Models propose, rules dispose. Kill switch always works.
7. **Human in the loop first.** Semi-auto (approve each trade) -> limited auto -> full auto, only by passing gates.
8. **Small models first.** Bigger only if smaller ones show signal; data/labels beat parameters.
9. **Document decisions** (decision log, like jitmo's).
10. **Never risk money we can't lose.** Start tiny.

## Non-goals (for now)
- HFT / latency arbitrage (retail REST API can't compete, see doc 02)
- Options selling strategies with unbounded risk
- Managing other people's money (would need SEBI registration, see doc 10)
- Crypto, forex
