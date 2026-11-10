# 04 - Data plan

## 1. Sources
| Data | Source | Notes |
|---|---|---|
| OHLCV daily | Upstox Historical Candle V3 (daily back to 2000) **[verify]**, NSE bhavcopy (free, official EOD) | Use bhavcopy as cross-check for Upstox |
| OHLCV intraday (1m-1h) | Upstox V3 (VERIFIED: minute/hour data from Jan 2022; fetch 1 month per call for 1-15m, 1 quarter otherwise; 2000 calls/30 min cap); existing repo has 1D/1H/15M/3M/5M for 14 symbols | ~4.8y max of minutes -> limited regimes |
| Live ticks/quotes | Upstox V3 market data websocket (VERIFIED: protobuf; ltpc/full 5-level/option_greeks modes; snapshot then live updates) | 2 connections; full mode 2000 keys/connection (1500 combined) - fits Nifty 200 |
| Option chain, OI | Upstox option chain API **[verify]** | For S9 |
| Index constituents (point-in-time) | NSE index rebalance circulars | Avoid survivorship bias |
| Corporate actions (splits, bonus, dividends) | NSE/BSE announcements | Needed to adjust prices |
| Results/announcement calendar | BSE/NSE corporate filings | For S6 |
| India VIX, Nifty, sector indices | NSE | Regime features |
| FII/DII flows | NSE daily | Macro flow feature |
| Global cues | S&P/Nasdaq futures, GIFT Nifty, USDINR, Brent | Overnight gap predictor; source TBD (free APIs/yfinance as fallback, unreliable) |
| News | RSS (ET, Moneycontrol, etc.) + BSE announcements | Check licences/ToS **[verify]** |

## 2. Storage & layout
- Raw -> `data/raw/<source>/<timeframe>/<symbol>.parquet` (convert existing JSON to Parquet, JSON is 4-22MB each and slow).
- Clean/adjusted -> `data/clean/`; features -> `data/features/<version>/`; labels -> `data/labels/<version>/`.
- Query via DuckDB/Polars; version datasets with DVC or content hashes; **raw data immutable**.
- Large files stay out of git (current repo commits ~600MB+ JSON: move to Git LFS/DVC or external storage).

## 3. Cleaning checklist
- Timezone (IST), trading calendar + holidays, special sessions (Muhurat).
- Adjust for splits/bonus/dividends; keep both raw and adjusted close.
- Zero-volume indices: separate feature handling (old repo issue).
- Missing/duplicate candles, bad ticks, zero/negative prices, gaps, circuit-hit days.
- Survivorship: include delisted/removed names where possible.
- Look-ahead guard: every feature stamped with availability time; features at bar t use data <= t only; orders fill at t+1 open or later.

## 4. Feature families
1. Returns multi-horizon (1, 5, 15, 60 min, 1, 5, 20, 60 d), log, volatility-scaled
2. Volatility: realised, Parkinson, Garman-Klass, ATR, vol-of-vol, overnight vs intraday split
3. Volume/liquidity: relative volume, dollar volume, Amihud, VWAP deviation
4. Microstructure-lite: gap, opening range, time-of-day, day-of-week, minutes to expiry
5. Technical (kept minimal): RSI, MACD, Bollinger %b, distance to MA/high/low
6. Cross-sectional: rank/z-score within universe and sector, beta to Nifty, residual return
7. Market/regime: Nifty trend, VIX level/change, breadth (adv/dec), sector rotation
8. Alpha101 / Qlib Alpha158 subset
9. Flows/macro: FII/DII, USDINR, crude, global index overnight return
10. Options: PCR, OI change, IV rank, skew (S9)
11. Event/news: days-to-earnings, sentiment score/volume (S6)
12. Model-derived: Kronos embeddings (S8), regime probabilities (S4), predicted vol (S5)

## 5. Labels (all tried, compared)
- L1: forward return over horizon h (regression), vol-normalised
- L2: cross-sectional rank / quantile of forward return
- L3: **triple barrier** (symmetric, vol-scaled barriers, vertical time barrier) + sample weights for overlap/uniqueness
- L4: meta-label (did primary signal win after costs?)
- L5: forward realised volatility
- L6: regime label (from HMM / rules)
Rule: labels use only future data after decision time; purge overlapping label windows in CV.

## 6. Data quality gate (Phase 1 exit)
- Reconcile Upstox vs bhavcopy daily close within tolerance for all symbols.
- Automated report: missing bars, outliers, adjustment checks, per-symbol coverage.
- Re-runnable pipeline: raw -> clean -> features reproducible from a single command.

## 7. Size estimate
Nifty 200 x 5y x 1m ~ 200 * 1250d * 375 bars ~ 94M rows (Parquet ~ few GB). Fine locally. Daily is trivial.
