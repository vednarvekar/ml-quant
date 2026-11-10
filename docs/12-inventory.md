# 12 - Full inventory: everything on our plate

## A. Data
- [ ] Upstox historical (daily + intraday) for Nifty 200 + indices
- [ ] NSE bhavcopy history, index constituents history, corporate actions
- [ ] India VIX, sector indices, FII/DII, USDINR, crude, global index data
- [ ] Option chain / OI (S9)
- [ ] News + announcements feed (S6), results calendar
- [ ] Trading calendar/holidays
- [ ] Storage: Parquet lake, DuckDB, DVC/LFS, backups (est. tens of GB)

## B. Software - research
- [ ] Python 3.11+, uv/conda env, Docker image
- [ ] polars, duckdb, pyarrow, numpy, scipy, statsmodels, arch, hmmlearn
- [ ] lightgbm, xgboost, catboost, scikit-learn, optuna
- [ ] pytorch, (jax/flax for TPU), transformers, kronos repo
- [ ] experiment registry (SQLite/MLflow), reporting (notebooks, markdown)
- [ ] own libs: labels (triple barrier), purged CV/CPCV, DSR/PBO, cost model, simulator, metrics

## C. Software - engine
- [ ] Rust toolchain, workspace per doc 09
- [ ] tokio, reqwest, tokio-tungstenite, prost (protobuf), serde, sqlx/rusqlite, tracing
- [ ] ort (ONNX) or Python sidecar, polars-rs/arrow
- [ ] Tauri + React UI (reuse jitmo patterns), alerts (Telegram)
- [ ] Test harness: replay recorded days, property tests on risk limits

## D. Compute
- [x] Local PC: i5-1135G7 / 16 GB / Iris Xe (CPU-only) for data + GBDT + small NNs
- [ ] Live host: user's Ubuntu Dell laptop server (always-on). Static-IP requirement handled free via weekly My Apps update + IP watchdog; free-cloud-VM fallback (no paid VPS)
- [ ] Colab Pro + TPU v5e-1 for heavy jobs; Drive for checkpoints
- [ ] Always-on host for live (local first; VPS/home server later)
- [ ] Stable internet + backup (mobile hotspot), UPS if running at home

## E. Accounts / access
- [ ] Upstox account with API app approved under SEBI rules, static IP if required
- [ ] Google AI Pro / Colab Pro, GitHub, HF account (models), data providers if any
- [ ] Secrets management (OS credential store)

## F. Knowledge to build
- [ ] Time-series CV (purging/embargo), meta-labelling, DSR (Lopez de Prado book chapters 3, 7, 12, 14)
- [ ] Portfolio theory: vol targeting, Kelly, HRP
- [ ] Indian market microstructure, F&O mechanics, STT/tax
- [ ] Rust async + trading engine design (read Nautilus/barter source)
- [ ] Qlib docs + Alpha158; WorldQuant 101 paper

## G. Process
- [ ] Decision log, experiment cards, weekly review
- [ ] Code review checklist for anything touching orders
- [ ] Incident log and runbook (disconnects, mismatch, token expiry)
- [ ] Compliance checklist (SEBI/Upstox), tax records

## H. Money & time
- Capital for live: TBD (only what can be lost). Running costs: Colab/AI plan (have), data (mostly free), VPS later (~small).
- Time: ~6-8 months part-time to tiny-live; ongoing maintenance after.

## I. Deliverables by phase
P1 data lake | P2 simulator+validator | P3 baseline leaderboard | P4 candidate strategies | P5 portfolio+risk spec | P6 Rust engine+paper | P7 live tiny | P8 scale
