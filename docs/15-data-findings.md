# 15 - Data findings (Phase 1, updated 2026-11-10)

Source: Upstox V3, Nifty 100 (list from NSE `ind_nifty100list.csv`). Code: `src/mlq/{upstox,quality,clean,nse_bhav,adjust,adjust_5m}.py`. Raw in `data/raw/`, cleaned in `data/clean/` (git-ignored). 8 unit tests pass.

**Scope (user decision 2026-11-10): Indian market (NSE) only.** We trade only NSE stocks. Global data (USDINR, Brent, Nasdaq, US VIX) is used purely as *input signals* for Indian stocks, never traded.

## Coverage
- Daily: 100 symbols, 441,023 rows, 2005-01-03 -> 2026-11-08 (newer listings start later, e.g. VAML Jun 2026, TMCV Nov 2025, HYUNDAI Oct 2024).
- 5-minute: 100 symbols, 8.61M rows, 2022-01-03 -> 2026-11-08. 75 bars on a normal day.
- NSE bhavcopy (unadjusted EOD, raw): 5,401 trading days 2005-2026 for ALL EQ symbols (old `cm*bhav.csv.zip` format until mid-2024, `sec_bhavdata_full_*.csv` after). 3 weekend special-session days have no file.
- Current constituents only -> **survivorship bias**. Must add point-in-time membership before any universe-level backtest.

## Issues found and handled
| Issue | Detail | Action |
|---|---|---|
| Daily duplicates | 75 symbol-dates stamped both 00:00 and 09:15 | Fixed in `clean_daily` |
| Bad OHLC | 10 daily rows (MAZDOCK 2017-18 zeros, VEDL Jan 2005), 12 five-minute bars | Dropped |
| Special sessions | Muhurat 2022-10-24, 2023-11-12, 2024-11-01, 2025-10-21, 2026-11-08; Saturday sessions 2024-03-02, 2024-05-18 | `special_session` flag; excluded from training by default |
| Zero-volume | 21 daily, 13,744 five-minute bars | Keep, flag |
| Fetch reliability | Home network drops -> DNS/timeouts | Retry with backoff, stop after 3 failures, resumable |

## Price adjustment: what we learned (and one mistake)
1. **Upstox history is adjusted** for corporate actions (and apparently dividends; e.g. IOC's factor drifts up ~12% over 2022-23 in steps). Recent dates equal NSE raw (RELIANCE 2025-11-07: 1478.00).
2. **Mistake corrected 2026-11-10:** I first assumed NSE bhavcopy `PREV_CLOSE` is adjusted on ex-dates and built an adjusted series from it. **It is not** (IOC 1:2 bonus on 2022-06-30: prev_close 109.8 vs close 74.25 -> fake -32%). That series and the "1,230 events" were wrong and were removed. Earlier claims that "Upstox does not adjust dividends" and that IOC/BPCL/etc. were bad Upstox data were also wrong: those mismatches came from my broken chain.
3. **Check that settled it:** on 134 days where NSE raw moved >30% (splits/bonus: Bajaj Finance -89.9%, HDFC Bank -50%, Kotak -80%, Shriram -81%, Nestle -51% ...), Upstox daily shows a normal return (e.g. +0.5%, -0.9%). So Upstox daily handles splits/bonus correctly.
4. **Decision:** research basis = **Upstox daily adjusted** (`data/clean/adj_daily`), with NSE raw (`raw_close`, `raw_prev_close`) alongside for cross-checks and lot/tick/price-band logic. Flags: `adj_step` (|adjusted ret - raw ret| > 0.3%: ex-date step, 10,306 rows), `review` (|ret| > 25%: 71 rows), `manual_event`.
5. **5m basis:** Upstox 5m and daily use slightly different factors (IOC 2022: 5m ~0.9x daily). `adj_5m` rescales each 5m bar by a per-segment median of (last 5m close / daily close), breaking at `adj_step` dates and >1% jumps. Result: days with >2% gap between last 5m close and daily close = 0.017% (was 0.18% on the broken basis); >0.5% = 4.4% on average (last candle != official close, so some noise is expected). Validation table: `data/clean/adj_5m_validation.csv`.

## Problem-stock review results
| Stock | Finding | Status |
|---|---|---|
| IOC, BPCL, GAIL, BANKBARODA | Earlier mismatch caused by my broken NSE chain (bonus/split days); under Upstox basis IOC >2% gaps = 0 | Resolved |
| BAJFINANCE, HDFCBANK, KOTAKBANK, SHRIRAMFIN, NESTLEIND, PIDILITIND, MAZDOCK, HDFCAMC, BSE, TRENT, MOTHERSON, ADANIPOWER (2024-25 splits/bonus) | Raw NSE shows -34% to -90%; Upstox adjusts correctly | Resolved |
| **SIEMENS 2025-04-07 (-43%), TMPV 2025-10-14 (-40%), VEDL 2026-04-30 (-65%)** | Upstox shows the full drop => unadjusted, look like **demergers** (inferred from price pattern, not verified) | `manual_event` flag; mask these days (return not representative) |
| 71 days with |ret| > 25% | **Classified by `mlq.review`** (rules in file, output `data/clean/review_classified.csv`): tick_quantization 40 (adjusted price < Rs 20, e.g. BAJFINANCE 2005-09, VEDL 2005: +-33/50/100% are rounding artifacts), sector_group 10 (PSU banks 2017-10-25, Adani 2023-02-01), market_wide 9 (2008, 2009-05-18 election, 2020-03), news_vol_spike 6 (e.g. ADANIENT 2023-02-02, IDEA 2017), demerger_unadjusted 3, suspected_unadjusted 3 (CGPOWER 2015-01-01 +191%, CGPOWER 2016-03-15 -72%, ADANIENT 2015-06-03 -83%: likely unadjusted corporate actions, unverified) | `exclude_ret` = True on 46 stock-days; training must drop them |
| CANBK (25.6% of days >0.5%, 1.3% >2%), UNIONBANK (22%), IDEA, POWERINDIA (many short segments) | Likely closing-auction noise / low-price tick rounding; CANBK's 1.3% >2% unexplained | Open: inspect CANBK 2024 split period |
| 11,137 daily rows without NSE raw match | Likely pre-ISIN-era symbol renames not linked, plus weekend sessions | Open: improve symbol linking for pre-2011 |

## Expanded universe + one-command rebuild (2026-11-10)
- Downloaded the 155 extra point-in-time names -> **255 symbols**, 982,886 daily rows, 21.3M 5m rows. No download failures.
- `uv run python -m mlq.rebuild` rebuilds clean -> adjust -> adjust_5m -> review -> universe -> calendar (offline, ~1 min). Result: 5,402 trading days; 149 days with |ret|>25% classified (64 stock-days excluded from training); 5m vs daily gap >2% on only 0.007% of days.
- **New open issues:** (a) 28,413 daily rows (2.9%) have no NSE raw match (mostly pre-ISIN renames; was 2.5%). (b) ~~stray early rows~~ fixed: `drop_stray_early` in clean.py removes short leading segments before a >1-year gap (SBICARD had 20 rows from 2015, listed 2020). 8 symbols still have missing days; OLECTRA (987), FORCEMOT (76), AWL (71) look like real illiquidity/suspension gaps, the rest 1-2 days. (c) `ret_class` "unclear": 2 rows.

## Remaining caveats
- Upstox adjustment methodology is undocumented to us; dividend inclusion is inferred. Prices are total-return-like, so use returns, not price levels, as features.
- Demerger list and renames (ETERNAL<-ZOMATO, LTM<-LTIM, TMPV<-TATAMOTORS) come from my own knowledge and price patterns; spot-check against NSE announcements.
- ~104 five-minute rows per symbol dropped (weekend special sessions with no daily anchor).

## Still to do in Phase 1
1. ~~Classify the 71 `review` days~~ done. Open: check CANBK 2024; improve pre-2011 symbol linking; verify the 3 suspected + 3 demerger cases against NSE announcements.
2. **Survivorship - done (v1)** (`mlq.universe` -> `data/clean/universe.parquet`): every month, all NSE EQ stocks (incl. later-delisted) ranked by 60-day median traded value (close x volume) from bhavcopy; top 200 kept. ETFs/fund units removed (ISIN starts `INF`). Since 2022: 261 companies were ever top-100; 168 of them are not in today's Nifty 100 (155 still listed -> being downloaded by `mlq.upstox.bulk_universe`; 13 delisted/renamed -> no Upstox history, NSE raw only). Since 2005: 538 ever top-100, 176 with no Upstox history. Caveats: liquidity rank != Nifty's market-cap rank (latest month overlap 64/100; hot speculative names enter); monthly churn is high - consider longer window/hysteresis.
2b. **Market context - done**: `mlq.upstox.indices` -> 29 indices daily (mostly 2005+, India VIX 2009+) and 28 at 5m (2022+): Nifty 50/Next 50/100/200/500, Midcap 100, Smallcap 100, India VIX, 20 sector indices, Nifty GS 10Yr bond index. Upstox quirk: 2 rows per date (close-only midnight row + real-range 09:15 row) and bond index has weekend rows -> `clean_index_daily` keeps the real-range row on equity trading days (`data/clean/idx/`). Many sector indices are close-only in early years (flat OHLC on 1,500-4,000 days) -> use index close-to-close returns, not index high/low, for history.
2c. **Global + positioning**: `mlq.context` -> FRED (US Fed public data): USDINR 1973+, Brent 1987+, Nasdaq 1971+, US VIX 1990+, US 10y 1962+, S&P 500 only last 10y (FRED licence). US data dated D may only be used for Indian day D+1 (US closes after India). NSE participant-wise open interest (Client/DII/FII/Pro long/short in index & stock futures/options) daily 2012+ (public archive; done: 3,668 days, 2 days no file, 1 odd-format file 2014-02-24 saved in `_unparsed/`). FII/DII *cash* flows: NSE API blocks scripts -> not available; participant OI is our positioning proxy.
3. **Trading calendar - done** (`mlq.calendar_check`): 5,401 trading days 2005-01-03 -> 2026-11-08 (~246-249/yr; includes the 5 Muhurat/Saturday special sessions, which are flagged). Only 1 of 100 symbols has missing days: VEDL 2005-01-12/13 (bad zero rows we dropped). Report: `data/clean/missing_days.csv`.
4. Context data: India VIX, Nifty/sector indices, FII/DII, USDINR.
5. Single rebuild command (raw -> clean -> adjust -> features).

## Gate check (Phase 1 exit)
- [x] Reproducible download + clean + adjust commands
- [x] Upstox vs NSE raw reconciliation (returns match on non-event days; events explained or flagged)
- [x] Adjustment basis decision (Upstox adjusted, NSE raw alongside)
- [x] One command rebuilds all derived datasets (`python -m mlq.rebuild`)
- [x] Coverage with survivorship handling (255 symbols, point-in-time `universe.parquet`)
- [x] Context data + calendar (indices, VIX, FRED, participant OI, 5,402-day calendar)
- Open, non-blocking: verify demergers/suspected cases, CANBK, pre-2011 symbol links, FII/DII cash flows
