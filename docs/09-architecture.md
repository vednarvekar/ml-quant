# 09 - System architecture

Mirror `jitmo`'s structure: Rust workspace, separate model process, IPC, SQLite, config-driven plug-ins ("skills" -> "strategies"), small UI.

## 1. Two worlds
- **Research (Python)**: Polars/DuckDB, LightGBM, PyTorch/JAX, notebooks, experiment registry. Slow, flexible.
- **Engine (Rust)**: live data, risk, orders, state, UI. Small, reliable, always-on.
- Contract between them: versioned feature spec + model artifact (ONNX/pickle) + strategy config (YAML).

## 2. Rust workspace (proposed)
| Crate | Job |
|---|---|
| `feed` | Upstox websocket (protobuf) -> normalised bars/quotes; reconnect, heartbeat |
| `broker-upstox` | REST: auth/token refresh, orders, positions, funds, instruments; rate-limit guard |
| `features` | Same feature calc as research (parity tests vs Python) |
| `model-worker` | Separate process serving predictions (start: Python sidecar; later ONNX via `ort`) |
| `strategy-host` | Loads strategy YAML, combines signals, meta-label filter |
| `risk` | Deterministic limits, kill switch, reconciliation |
| `oms` | Order state machine, idempotent order IDs, retries |
| `store` | SQLite (orders, fills, decisions, audit log, model versions) |
| `ipc` | Messages between components/UI |
| `sim` | Shared fill/cost model for paper mode and backtest parity |
| `ui` | Tauri/React dashboard: signals, approve/reject, positions, PnL, kill switch |
| `testset-runner` | Replay recorded days, assert behaviour (like jitmo's) |

## 3. Honest note on Rust
Upstox REST/websocket latency is tens-hundreds of ms; Rust won't give a trading speed edge here. Value of Rust = **safety, reliability, low memory, single binary**. Python is fine for research and an early sidecar. Don't let Rust slow the research phases.

## 4. Data flow (live)
Feed -> bar builder -> features -> model-worker -> strategy-host -> risk -> (semi-auto: UI approval) -> oms -> broker -> fills -> store -> monitoring.

## 5. Modes
Backtest (sim) | Replay | Paper (live data, simulated fills) | Semi-auto live | Auto live. Same code path, mode switch only changes broker adapter and approval step.

## 6. Operational concerns
- Upstox access tokens expire 3:30 AM daily (VERIFIED) -> daily login step before market open (OAuth+2FA; TOTP endpoint exists but revokes the old token); never store passwords in code; secrets in OS credential manager.
- **Static IP mandatory** for order APIs, whitelisted in My Apps, changeable once a week (VERIFIED) -> engine must run from a fixed IP (home static IP or India VPS). Cloud/Colab can never place orders.
- Sandbox covers order endpoints only (VERIFIED) -> paper mode = live feed + our simulated fills.
- Time sync (NTP), IST handling, market calendar service.
- Logging + metrics + alerting (Telegram/push) on risk breaches, disconnects, mismatches.
- Backups of DB and models; disaster plan: if engine dies with open positions -> alert + broker-side stop orders.
- Where it runs: local PC at first; later a small always-on VPS/home server in India **[decision]**.

## 7. Tech choices to compare (decision log)
| Decision | Options |
|---|---|
| Backtest engine | Own Rust/Python sim vs Nautilus vs vectorbt (screening) |
| Model serving | Python sidecar vs ONNX in Rust (`ort`) vs candle/burn |
| Experiment tracking | MLflow local vs W&B vs SQLite+Parquet |
| Data versioning | DVC vs content-hash manifests |
| UI | Tauri (reuse jitmo) vs terminal TUI first |
