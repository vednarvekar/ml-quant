# ml-quant

Research platform for systematic trading of Indian equities (NSE) with small, specialised ML models, built around the Upstox API. Python for research now, Rust execution engine later.

**Status:** research phase. Data foundation, backtest harness and first baselines/ML experiments are done. Nothing here is a proven money-maker and no live trading exists yet. See [docs/17-baseline-results.md](docs/17-baseline-results.md) for the honest numbers.

## What is here
- `docs/` - full plan (vision, data, strategies, models, backtest/costs/risk, phases) and findings. Start with [docs/README.md](docs/README.md).
- `src/mlq/upstox/` - Upstox V3 historical downloader (daily from 2005, 5-minute from 2022), rate-limit safe and resumable.
- `src/mlq/` - cleaning, corporate-action/adjustment checks, point-in-time liquidity universe, context data (India VIX, indices, FRED, participant OI), India cost model, daily and 5-minute backtest engines, purged walk-forward / CPCV / Deflated Sharpe, experiment registry, baselines, and the first LightGBM experiments.
- `tests/` - unit tests (null-strategy and no-lookahead tests for both engines).

## Quick start
```bash
uv sync
cp .env.example .env        # add your Upstox analytics token (read-only market data)
uv run python -m mlq.upstox.bulk            # download Nifty 100 history
uv run python -m mlq.rebuild                # clean -> adjust -> universe -> calendar
uv run pytest
uv run python -m mlq.swing                  # swing baselines
uv run python -m mlq.ml_swing               # ML swing experiment
```
Data, tokens and `.env` are git-ignored; never commit them.

## Principles
Cost-first (India STT, brokerage, slippage on every fill), no look-ahead (signals trade at the next open), every trial logged so Deflated Sharpe is honest, and a new idea must beat simple baselines before it earns more complexity.
