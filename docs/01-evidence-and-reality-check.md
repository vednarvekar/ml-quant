# 01 - Evidence & reality check

## 1. Why the old CNN failed (from `MODEL_TRAINING_LOG.md`)
- 3-class (HOLD/BUY/SELL) val acc ~0.36-0.41, majority baseline ~0.44 -> **no signal learned**.
- Train macro-F1 0.53 vs val 0.37 -> memorisation/overfit.
- Label issues: asymmetric barriers (1.5x up / 1.0x down) -> SELL-heavy; labels from one fixed horizon.
- Single stock-by-stock chronological split -> regime mismatch; no purging/embargo (overlapping labels leak).
- Inputs: only past OHLCV of the same stock. That's the most picked-over information set in existence.
- Metric: accuracy/F1 never tells whether it makes money after costs.
- Conclusion: **problem framing, labels, validation and features - not model size - were the bottleneck.**

## 2. What literature says (summarised, treat as indicative)
- Intraday index returns showed predictability with regularised linear + tree ensembles, with net Sharpe ~0.98 after costs in one study (J. Financial Econometrics 2023) **[verify details]**; a 2025 Management Science paper reports similar ETF results **[verify]**.
- Other evidence (frontier markets) says intraday prices are close to weak-form efficient. So: edges exist but are thin, market-specific, and decay.
- General time-series foundation models (TimesFM, Chronos) reported **negative** zero-shot R2 on financial returns in one secondary-cited study **[verify]**. Kronos (finance-specific, Tsinghua, 2025) claims big gains on its own benchmarks - not independent trading results.
- WorldQuant "101 Formulaic Alphas": production alphas are many, weak, low-correlation (avg pairwise corr ~16%), short holding (0.6-6.4 days avg). Edge = diversification of many weak signals.
- Lopez de Prado toolkit (Advances in Financial ML): triple-barrier labels, meta-labeling, purged K-fold + embargo, CPCV, Deflated Sharpe Ratio. These exist precisely because naive ML backtests overstate performance.
- RL (FinRL etc.): noisy/delayed rewards, non-stationarity, backtest overfitting, profit-only evaluation. Good for execution/sizing research later, risky as a primary alpha source.
- LLM/FinBERT sentiment for Indian news: promising but evidence is thin (tiny samples, no cost-aware backtests found).

## 3. What this implies for us
| Insight | Consequence |
|---|---|
| Signal-to-noise is tiny | Prefer low-capacity models, strong regularisation, lots of data, many instruments |
| Costs dominate | Lower turnover, longer holding (hours-days), liquid names only |
| Alpha decays | Continuous retraining + monitoring, many uncorrelated strategies |
| Overfitting is the main enemy | CPCV, DSR, trial logging, holdout we touch once |
| Direction prediction is the hardest framing | Try ranking (cross-sectional), volatility, regime, meta-labelling instead |
| Retail speed is slow | Compete on horizon & breadth, not latency |

## 4. Priors we hold (to be tested, not believed)
- Tree models (LightGBM) >= small deep nets on tabular market features.
- Cross-sectional daily ranking has more robust signal than single-stock minute direction.
- Volatility and regime are far more predictable than direction -> use for sizing/filtering.
- Meta-labelling on a simple primary strategy beats end-to-end direction nets.
- Foundation models: zero-shot likely weak; fine-tuned / embedding use is worth one experiment.
