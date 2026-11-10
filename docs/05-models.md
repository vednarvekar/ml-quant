# 05 - Model zoo

Principle: **small, specialised, regularised.** Each model has one job and a baseline it must beat. Bigger models only after smaller ones show signal.

## 1. Models by job
| Job | Candidates (simple -> complex) | Output | Used by |
|---|---|---|---|
| **Return/rank predictor** | Ridge/ElasticNet -> LightGBM/XGBoost/CatBoost -> MLP (1-5M) -> TCN/GRU -> PatchTST-style transformer (5-20M) | Expected return or rank score | S1, S3 |
| **Meta-labeller** (take/skip + confidence) | Logistic -> LightGBM classifier -> small MLP | P(primary signal wins) | S3 |
| **Volatility forecaster** | EWMA -> HAR-RV -> GARCH/EGARCH -> LightGBM on realised-vol features | Next-period vol | S5 sizing, stops |
| **Regime detector** | Rules (trend/vol thresholds) -> HMM / GMM -> GBM classifier | Regime probs | S4 |
| **News/event sentiment** | Lexicon -> FinBERT-India -> fine-tuned small LLM (3B, quantised) | Sentiment, event type | S6 |
| **Foundation model** | Kronos (VERIFIED: MIT licence; mini 4.1M ctx 2048, small 24.7M ctx 512, base 102.3M ctx 512; large 499M NOT public; PyTorch-implied, CUDA/torchrun fine-tune scripts built around Qlib daily data; no TPU/JAX support documented; authors say demo pipeline is "not production-ready" and ignores costs), Chronos-2/TimesFM as baselines | Forecast / embeddings | S8 |
| **Anomaly / data-quality** | Z-score -> isolation forest -> autoencoder | Flag bad data/market stress | Risk layer |
| **Execution / sizing (late)** | Kelly-fraction + vol targeting rules -> RL (PPO) | Order size/timing | S10 |
| **Ensemble/stacker** | Equal weight -> rank average -> ridge stacking -> hierarchical risk parity on strategy returns | Final score / strategy weights | Portfolio |

## 2. How the pieces connect (inference flow)
```
market data -> feature pipeline -> [rank model] [vol model] [regime model] [sentiment model]
                                      |             |            |               |
                                      +----> signal combiner (S1..Sn) <----------+
                                                  |
                                       meta-labeller (take / skip, size)
                                                  |
                                   portfolio + risk layer (deterministic rules)
                                                  |
                                  approval UI (semi-auto) -> order router -> Upstox
```

## 3. Training details per family
- **GBDT**: Optuna search over shallow trees (depth 3-6), strong L2, min_child_samples large, early stopping on purged validation, feature subsampling. Train per-horizon. CPU, minutes.
- **Linear**: baseline + sanity check for GBDT; rank-IC as metric.
- **Small NNs**: windows of normalised features (not raw price), dropout, weight decay, adaptive pooling (old repo's own to-do), ensembling of seeds. CPU/GPU local; Colab TPU only if JAX/Flax ports are worth it.
- **Kronos**: zero-shot eval first; then light fine-tune on Indian candles (Colab TPU/GPU; check Kronos code is PyTorch -> TPU needs torch_xla or port **[verify]**).
- **LLM sentiment**: label via market-adjusted returns (as in Indian-finance LLM research) not human tone; 4-bit quantised for local inference.
- **HMM/GARCH**: statsmodels/arch; fit rolling.

## 4. Compute fit
**Local machine (known): i5-1135G7, 4 cores/8 threads, 16 GB RAM, Iris Xe, no CUDA.** Implications: GBDT/linear/HMM/GARCH comfortable (limit datasets to a few GB in RAM; use Polars lazy/DuckDB); NNs limited to ~1-5M params, small batches; Kronos-mini/small inference OK on CPU, Kronos-base slow; no 100M-param training locally -> use Colab (GPU runtime for PyTorch models, TPU only for JAX). Live inference of our small models on CPU is easily fast enough.

| Model | Local PC | Colab TPU v5e-1 |
|---|---|---|
| GBDT, linear, HMM, GARCH | Yes | Not useful (TPU doesn't accelerate GBDT) |
| MLP/TCN/GRU <20M | Yes (CPU ok, GPU better) | Optional |
| Transformer 20-100M | Fine-tune/infer yes if GPU; slow on CPU | Good fit (JAX) |
| Kronos base (102M) | Inference ok (GPU preferred) | **Not TPU-ready** (CUDA/PyTorch only documented) -> use Colab GPU runtime or local GPU for Kronos; TPU only for our own JAX models |
| 3B LLM | Quantised inference only if enough RAM/VRAM | Fine-tune is heavy; likely use API or skip |
Local GPU specs unknown -> open question (doc 13).

## 5. Model acceptance rules
A model enters the portfolio only if, on purged walk-forward + untouched holdout:
- rank-IC (or equivalent) significantly > 0 and stable across years/regimes
- beats the previous simpler model net of costs
- Deflated Sharpe > threshold given number of trials
- Passes stress tests (doc 07) and has bounded turnover
- Reproducible from config + seed, stored with data/feature hashes

## 6. Model registry
Each model version: id, config, data hash, feature list, train window, metrics, artifacts (ONNX/pickle), status (research/paper/live/retired).
