# 06 - Training & validation

## 1. Splits (never random)
- **Walk-forward**: train on [t0, t1], validate [t1+embargo, t2], roll forward. Retrain cadence matches live plan (e.g. monthly).
- **Purged K-fold + embargo**: remove train samples whose label window overlaps validation; embargo ~ label horizon after each test block.
- **CPCV** (combinatorial purged CV): many train/test combinations -> a distribution of backtest paths, not one lucky path.
- **Final holdout**: last ~6-12 months, touched **once per strategy family**, then burned. New holdouts come from paper trading.
- Cross-sectional models: split by time, never by stock only.

## 2. Overfitting defences
- Log every trial in the registry (count N for Deflated Sharpe).
- Deflated Sharpe Ratio / Probability of Backtest Overfitting; White's Reality Check or SPA for the best-of-many problem.
- Hyperparameter search inside the training window only (nested CV).
- Prefer few features, monotone constraints, shallow trees, early stopping, seed ensembles.
- Feature importance stability across folds; drop unstable features.
- Sample weights by label uniqueness (overlapping triple-barrier labels).
- Sanity tests: shuffled-label model must score ~0; random-signal strategy must lose ~costs.

## 3. Metrics (model level vs strategy level)
- Model: rank-IC, IC-IR, calibration (for meta-labels), log-loss, precision at top-k. **Not** raw accuracy.
- Strategy: net Sharpe/Sortino/Calmar, max drawdown, turnover, hit rate, profit factor, tail ratio, PSR/DSR, cost/gross ratio, capacity.

## 4. Compute workflow
1. Develop & debug locally on small slices (Polars/DuckDB, LightGBM).
2. Heavy jobs (transformer, Kronos fine-tune, big sweeps) on Colab TPU v5e-1 / GPU; checkpoint to Google Drive every N steps because sessions can be cut or idle-disconnected. Colab facts (official FAQ): limits are not published and vary; free max 12h; 24h continuous run is tied to Pro+ with enough compute units; backend terminates when compute units run out. Third-party estimate: ~33h of TPU v5e-1 per 100 units (unofficial). Check actual burn rate in the Colab UI.
3. Code lives in repo; notebooks are thin wrappers calling `src/` modules with a config file.
4. Artifacts + metrics logged to registry (MLflow local or a simple SQLite/CSV + Parquet store).
5. Seeds fixed; environment pinned (`requirements.lock`, Docker image).
- Colab cost-saver: don't use TPU for GBDT; run sweeps with Optuna on local CPU overnight.

## 5. Retraining & drift
- Scheduled retrain (weekly/monthly) with champion/challenger comparison.
- Drift monitors: feature distributions (PSI), prediction distribution, rolling IC, rolling live-vs-backtest slippage.
- Auto-demote model to "paper only" if rolling IC/PnL breaches thresholds.

## 6. Step-by-step training recipe (per experiment)
1. Define hypothesis + family ID + labels + features (pre-registered in experiment card).
2. Build dataset snapshot (hash).
3. Baselines: naive, linear, previous champion.
4. Train candidate under purged walk-forward; tune within folds.
5. Evaluate model metrics -> strategy backtest with full cost model.
6. Run DSR/PBO with trial count.
7. Stress tests (doc 07).
8. Write result + decision in the experiment log (keep failures too).
