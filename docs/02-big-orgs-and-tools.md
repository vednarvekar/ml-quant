# 02 - Big orgs & tools: what they do, what we copy

Note: firm internals are mostly private. Statements about firms below are public knowledge/inference **[verify]**; tool facts come from docs/search results.

## A. Firms (approach -> can we copy?)
| Org | Public approach | Copy? |
|---|---|---|
| **Renaissance (Medallion)** | Huge data cleaning, many short-horizon weak signals, strict cost control, heavy stat rigor, small team of scientists | Copy the *process*: obsessive data quality, many weak signals, cost-awareness. Not the speed/data. |
| **WorldQuant** | "Alpha factory": thousands of formulaic alphas, low correlation, combined; published *101 Formulaic Alphas* (arXiv 1601.00991) | **Yes.** Implement a subset as features/signals; combine via ML. Daily horizon, doable for us. |
| **Two Sigma / D.E. Shaw** | Data-driven ML at scale, alt data, rigorous research platforms | Copy research hygiene + experiment tracking. Alt data limited for us. |
| **Jane Street / Citadel Securities / HFTs** | Market making, latency, inventory risk | **No.** Needs colocation, exchange membership. |
| **AQR** | Systematic factors: value, momentum, quality, low-vol; diversified, transparent | **Yes.** Factor/momentum on Nifty 200 at daily-weekly horizon is our most realistic baseline. |
| **Man AHL / trend followers** | Diversified trend-following, vol targeting | Copy vol-targeting + trend overlay on index/stock futures. |
| **Microsoft Qlib team** | Open pipeline: data -> Alpha158/360 features -> model zoo -> backtest; RD-Agent for LLM-driven factor R&D | **Yes**, borrow feature sets, evaluation (IC/RankIC), config-driven experiments. |

## B. Open-source tools
| Tool | Role | Verdict |
|---|---|---|
| **Qlib (Microsoft)** | Research platform, 40+ models, Alpha158, cross-sectional focus. Research only, not live trading. | Use as reference & maybe for baseline benchmark; India data adapter needed. |
| **RD-Agent** | LLM agents propose/test factors; Linux+Docker | Later experiment (Phase 5+); run on Colab/WSL. |
| **vectorbt** | Fast vectorised backtests (pandas/numba) | Good for rapid idea screening. |
| **backtesting.py** | Simple single-asset backtests | Teaching only. |
| **NautilusTrader** | Rust core + Python, event-driven, same code backtest/live; LGPL-3.0; high MSRV **[verify]** | Study its architecture closely; maybe use for backtest/live parity or copy design. No Upstox adapter. |
| **barter-rs** | Rust trading framework, MIT, young/changing API | Reference for Rust engine; pin version if used. |
| **FinRL** | RL envs for trading | Research only; RL pitfalls in doc 01. |
| **Kronos** | Finance foundation model for candlesticks (tokenizer + autoregressive), MIT, mini/small/base public (4.1M/24.7M/102.3M), large not public; CUDA only | Experiment E-F1 (doc 03). |
| **Chronos-2 / TimesFM** | General TSFMs | Baseline only; expected weak on returns. |
| **FinBERT-India-v1 / Llama-3.2-3B finance FT** | Indian news sentiment | Experiment E-N1; evaluate cost-aware. |
| **Lopez de Prado methods** (mlfinlab-style, reimplement) | Triple barrier, meta-label, purged CV, DSR | Core of our validation. Reimplement ourselves for correctness. |
| **LightGBM / XGBoost / CatBoost** | Tabular SOTA | Primary model family. |
| **Polars + DuckDB + Parquet** | Fast local data stack | Data layer. |
| **ONNX Runtime (`ort` crate) / candle / burn** | Rust inference | Phase 6 option; **[verify]** maturity. |

## C. Reality of "copying" big orgs
- We cannot copy: speed, data volume, capital, execution infrastructure.
- We can copy: validation rigor, alpha diversification, factor models, vol targeting, meta-labelling, risk limits, experiment discipline.
- Our structural edges: small capital (no market impact), flexibility, can trade less-crowded mid-caps within liquidity limits, can be patient.
- Our structural disadvantages: retail costs/STT, slow API, limited data history (Upstox minute data from Jan 2022 **[verify]**), no shorting in delivery cash market.
