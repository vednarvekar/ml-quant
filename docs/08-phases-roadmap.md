# 08 - Phases & roadmap

Each phase: goal, deliverables, **exit gate**, **kill/pivot criteria**. Times are rough part-time estimates.

## Phase 0 - Planning & research (now, ~1-2 wks)
- Deliverables: these docs; verified facts for all **[verify]** items; decisions on capital, hardware, universe; environment checklist.
- Gate: user sign-off on docs; open questions in doc 13 answered.

## Phase 1 - Data foundation (~2-3 wks)
Fresh start: new repo layout, no old code/data reused (old repo only informs lessons in doc 01). Start with Nifty 100 to fit 16 GB RAM; use Parquet + DuckDB, process per-symbol/lazy.
- Upstox auth flow + ingest (daily + intraday), NSE bhavcopy, index constituents, corporate actions, India VIX, FII/DII.
- Convert existing JSON to Parquet; clean/adjust; data quality report; trading calendar.
- Gate: Upstox vs bhavcopy reconciliation passes; one command rebuilds all datasets; coverage report for Nifty 200.
- Kill: data can't be sourced for >= 100 liquid symbols with acceptable quality -> shrink universe.

## Phase 2 - Backtest & validation harness (~2-3 wks)
- Event-driven simulator, India cost model, purged walk-forward/CPCV, DSR, metrics, experiment registry.
- Gate: null tests pass (random strategy ~ -costs), no-lookahead test passes, results reproducible.
- Nothing proceeds until this passes.

## Phase 3 - Baselines (~2 wks)
- Buy&hold Nifty, equal-weight, 12-1 momentum, simple trend + vol targeting (S2), linear and LightGBM on S1 features.
- Re-run the old CNN idea under the new harness for an honest comparison.
- Gate: a documented baseline leaderboard. Any baseline net Sharpe > 0.5? If none, widen universe/horizon before going deeper.

## Phase 4 - Multi-approach tournament (~4-8 wks)
- Run families S1, S3, S4, S5 (then S6, S9, S8) through the same harness. Model zoo from doc 05; Colab TPU for heavy ones.
- Weekly review of leaderboard; keep failures documented.
- Gate: >= 2 candidate strategies pass the research->candidate gate with low mutual correlation.
- Pivot: if nothing passes, revisit data (more history, alt data), horizon, or stop live plans and keep as research.

## Phase 5 - Portfolio, meta-labelling, risk (~2-3 wks)
- Combine candidates (rank avg / stacking / HRP), meta-labeller, vol targeting, risk layer spec + tests.
- Gate: combined portfolio beats best single strategy on risk-adjusted basis; risk limits unit-tested.

## Phase 6 - Engine MVP in Rust + paper trading (~4-6 wks)
- Crates: feed, broker-upstox, strategy-host, risk, oms, store, ipc, ui (see doc 09). Model inference via Python sidecar first.
- Paper mode: real data, simulated fills (Upstox sandbox if available **[verify]**), compare with sim.
- Gate: >= 40 trading days paper; behaviour matches gates in doc 07; zero unhandled errors.

## Phase 7 - Semi-automated live, tiny capital (~3 months)
- Signals -> approval UI -> orders. Capital: an amount we can fully lose (decide in doc 13).
- Complete SEBI/Upstox algo requirements (registration, static IP, algo ID) **[verify]**.
- Gate: 3 months, DD within plan, live-vs-sim gap small, no ops incidents.

## Phase 8 - Scale & automate (ongoing)
- Gradual automation, capital steps, retraining pipeline, drift monitors, new strategy research continuing in parallel.
- Stop rules reviewed monthly.

## Parallel tracks
- Documentation/decision log (every phase).
- Security (secrets, tokens), backups.
- Learning plan: time-series CV, portfolio theory, Rust async, Upstox API.

## Rough timeline
~6-8 months to tiny live if everything goes well. Failure to find an edge at Phase 4 is a valid outcome: we keep the platform and iterate on new ideas.
