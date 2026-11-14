# ml-quant

Research platform for systematic trading of Indian equities (NSE) with small ML models, built around the Upstox API. Python for research now, Rust execution engine later.

**Status:** research + paper trading (pretend money, no real orders). Nothing here is a proven money-maker yet. Honest numbers: [docs/19-audit-response.md](docs/19-audit-response.md), [docs/20-holdout-2013-2021.md](docs/20-holdout-2013-2021.md).

## Layout
- `docs/` - plan and findings. Start with [docs/README.md](docs/README.md).
- `src/mlq/upstox/` - Upstox V3 historical downloader (daily 2005+, 5-minute 2022+).
- `src/mlq/` data: `nse_bhav`, `nse_fo`, `nse_mto` (NSE archives), `clean`, `adjust`, `adjust_5m`, `review`, `universe`, `context`, `rebuild`.
- `src/mlq/` research: `costs`, `backtest`, `intraday`, `futures`, `futures_bt`, `validation`, `registry`, `evaluate`, `swing`, `swing_portfolio`, `baselines`, `ml_swing`, `ml_v2` (current model), `compare_v2`, `compare_ls`, `exec_test`, `holdout`.
- `src/mlq/paper/` - daily paper trader (`run`, `status`).
- `scripts/paper_daily.ps1` - Windows scheduled-task wrapper.
- `tests/` - unit tests.

## Quick start
```bash
uv sync
cp .env.example .env                        # Upstox analytics token (read-only)
uv run python -m mlq.upstox.bulk            # download history
uv run python -m mlq.rebuild                # clean -> adjust -> universe -> calendar
uv run pytest
uv run python -m mlq.paper.run --update     # one paper-trading day
uv run python -m mlq.paper.status           # plain-English status
```
Data, tokens and `.env` are git-ignored; never commit them.

Removed research dead-ends (intraday rule families, intraday ML, old v1 comparison and futures runner) live in git history before commit `abca7e8`.
