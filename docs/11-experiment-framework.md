# 11 - Experiment framework (many approaches, fair comparison)

## 1. Rules
- Every idea = an **experiment card** (YAML/MD) written *before* running.
- Every run goes through the **same harness** (doc 06/07): same data snapshot, same costs, same splits, same metrics.
- All trials logged, including failures. Trial count feeds Deflated Sharpe.
- Champion/challenger: a new approach must beat the current champion net of costs to replace it.
- Weekly review: leaderboard, correlation matrix of strategy returns, decide what to kill/keep/combine.

## 2. Experiment card template
```
id: E-S1-003
family: S1
hypothesis: <one sentence>
universe / period / horizon:
features (version): 
labels (version):
model + hyperparameter space:
split scheme + embargo:
baselines to beat:
success criteria (pre-registered):
compute: local | colab-tpu
status: planned | running | done | killed
result summary + link to artifacts:
decision:
```

## 3. Registry fields per run
run id, git commit, config hash, data snapshot hash, seed, metrics (model + strategy), trial counter, artifacts path, wall time, compute used.

## 4. Starter experiment list
| ID | Question |
|---|---|
| E-B1 | Do buy&hold / equal-weight / 12-1 momentum set a sane benchmark on Nifty 200? |
| E-B2 | Old 1D-CNN on 5M data under new harness: still chance? |
| E-S1-1..n | Ridge vs LightGBM vs MLP on Alpha158-like features, 1/5-day rank targets |
| E-S1-L | Label comparison: raw return vs rank vs vol-normalised |
| E-S2-1 | Trend + vol-target on index futures-like series |
| E-S3-1 | ORB / VWAP-reversion primary + LightGBM meta-label, triple-barrier |
| E-S3-2 | Same with small TCN; does deep beat GBM? |
| E-S4-1 | HMM regime gating of S1/S3: does it improve Sharpe/DD? |
| E-S5-1 | HAR vs GARCH vs GBM for vol; effect on position sizing |
| E-S6-1 | FinBERT-India sentiment as feature; incremental IC |
| E-S8-1 | Kronos zero-shot vs fine-tuned vs embeddings as features |
| E-S9-1 | Options-derived features incremental value |
| E-P1 | Portfolio combiners: equal vs rank-avg vs ridge stack vs HRP |
| E-X1 | Cost sensitivity and capacity curve for each candidate |

## 5. Tooling
Start minimal: config files + Python runner + SQLite/Parquet registry + markdown reports. Add MLflow/W&B only if it earns its keep. Optional later: RD-Agent to generate factor ideas (still passes through the same harness).
