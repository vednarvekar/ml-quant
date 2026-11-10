# ml-quant docs

Planning docs for the trading system. **No implementation starts until Phase 0 gate is signed off** (see `08-phases-roadmap.md`).

Status: DRAFT v0.2 (2026-11-09) - facts verified, user decisions recorded in doc 13. Fresh start (old code discarded). Facts marked **[verify]** came from secondary sources or memory and must be checked against primary sources before we rely on them.

| # | Doc | What it answers |
|---|---|---|
| 00 | [Vision & principles](00-vision-and-principles.md) | What are we building, what are the rules we never break |
| 01 | [Evidence & reality check](01-evidence-and-reality-check.md) | What research says works/doesn't, why the old CNN failed |
| 02 | [Big orgs & tools comparison](02-big-orgs-and-tools.md) | What Qlib, WorldQuant, Renaissance, AQR, Nautilus etc. do, what we copy |
| 03 | [Strategy families](03-strategy-families.md) | Every approach we will try in parallel |
| 04 | [Data plan](04-data.md) | Sources, schema, features, labels, pitfalls |
| 05 | [Model zoo](05-models.md) | Every model, its job, size, how it works |
| 06 | [Training & validation](06-training-and-validation.md) | How we train (local / Colab TPU), split, tune, avoid fooling ourselves |
| 07 | [Backtesting, costs, risk](07-backtest-costs-risk.md) | Simulator, India cost model, risk layer, metrics, gates |
| 08 | [Phases & roadmap](08-phases-roadmap.md) | Step-by-step phases with entry/exit gates and kill criteria |
| 09 | [System architecture](09-architecture.md) | Rust engine + Python research, components, safety |
| 10 | [Broker, regulation, tax](10-broker-regulation-tax.md) | Upstox API, SEBI algo rules, costs, tax |
| 11 | [Experiment framework](11-experiment-framework.md) | How we run many approaches and compare them fairly |
| 12 | [Full inventory](12-inventory.md) | Everything we need: tools, data, compute, skills, time, money |
| 13 | [Risks & open questions](13-risks-and-open-questions.md) | What can kill the project, what we still need to decide |
| 14 | [Sources](14-sources.md) | Links used |
| 15 | [Data findings](15-data-findings.md) | What we found in the real downloaded data (Phase 1) |
| 16 | [Harness and baselines](16-harness-and-baselines.md) | Backtest engine + validation tools (Phase 2) |
| 17 | [Baseline results](17-baseline-results.md) | Phase 2 gate, swing + intraday leaderboards, Phase 4 plan |
