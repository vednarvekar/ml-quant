# 10 - Broker, regulation, tax

Verification status as of 2026-11-09. **VERIFIED** = read on Upstox/NSE primary pages. **[verify]** = still unconfirmed.

## 1. Upstox API (VERIFIED from Upstox docs)
**Historical Candle V3** (`GET /v3/historical-candle/:instrument_key/:unit/:interval/:to_date/:from_date`)
| Unit | Intervals | History from | Max range per request |
|---|---|---|---|
| minutes | 1-300 | Jan 2022 | 1 month (intervals 1-15), 1 quarter (>15) |
| hours | 1-5 | Jan 2022 | 1 quarter |
| days | 1 | Jan 2000 | 1 decade |
| weeks / months | 1 | Jan 2000 | no limit |
=> ~4.8 years of intraday max. Must page requests by month for 1-15m data.

**Rate limits** (per API, per user): order place/modify/cancel/GTT combined: **10 req/s, 500/min, 2000 per 30 min** for non-registered algos; 50 req/s for SEBI-registered algos. Other standard APIs (incl. historical candles, positions, funds): 50 req/s, 500/min, 2000/30 min. Exceeding can suspend access. => 2000 calls/30 min cap matters for bulk historical download (throttle).

**Market data websocket V3** (protobuf, binary): 2 connections/user (5 with Upstox Plus). Per connection: LTPC 5000 keys (2000 combined), Option Greeks 3000 (2000), Full 2000 (1500). Modes: ltpc, option_greeks, full (5-level depth), full_d30 (Plus). Snapshot first, then live updates; not documented as tick-by-tick. => Enough for Nifty 200 in `full` mode.

**Tokens**: access_token expires **3:30 AM next day** regardless of issue time -> daily re-login. `extended_token` is read-only. TOTP login endpoint exists (revokes previous token). Analytics token: long-lived, read-only market data; trading APIs need static IP.

**Sandbox**: exists; 30-day token; one sandbox app per user; **only order APIs** (place/modify/cancel, v2/v3, multi-order) are sandboxed; cannot be used for live. => Paper-trading fills must come from our own simulator + live data feed; sandbox only tests order plumbing.

## 2. Regulation (SEBI retail algo framework)
Primary-document status: NSE circular **NSE/INVG/67858** (May 2025, implementation standards) found; SEBI circular SEBI/HO/MIRSD/MIRSD-PoD/P/CIR/2025/0000013 text **not retrieved** (still [verify]); later NSE circulars INVG/72657, INVG/73992 (30 Apr 2026) not opened.

Verified from Upstox (community guide, rate-limit page, "regulatory changes now live" post):
- Framework effective Oct 2025 with hard stop **1 Apr 2026**; changes are live on Upstox.
- **Direct API users <= 10 orders/sec**: no exchange strategy registration; Upstox tags orders with generic Algo ID 99999; **static IP mandatory** and whitelisted in Upstox "My Apps"; OAuth + 2FA each session; Upstox rejects orders above 10 OPS.
- **> 10 OPS**: exchange registration (strategy writeup, RMS writeup, auditor certificate), unique Algo ID.
- Static IP can be changed **once a week** (NSE circular too). Order place/modify/cancel need the IP; analytics (read-only) token does not.
- Third-party algo providers must be NSE-empanelled; any non-TWAP/VWAP strategy offered to others = "black box" needing Research Analyst registration. **Selling/sharing our strategy = out of scope.**
- API trading on MCX temporarily disabled at Upstox.
- Family/third-party use of one API app is not allowed (own account only) **[verify wording]**.

Implication: our own personal bot at <= 10 OPS is allowed, needs a **static IP** (home ISP static IP or a India-region VPS). Our strategy will never approach 10 OPS.
Still to do: read SEBI circular + NSE/INVG/67858 + NSE/INVG/73992 directly; confirm Upstox "May 2026" algo-app requirements (e.g. X-Algo-Name header mentioned by a vendor blog).

## 3. Product constraints [verify with Upstox]
MIS auto square-off near close, CNC cannot short, F&O lots/margin/expiry/ban list.

## 4. Costs (see doc 07). STT: delivery 0.1% both sides; intraday 0.025% sell; futures sell 0.05% (from 1 Apr 2026); options sell 0.15% of premium (multiple sources agree; one broker table was stale). Upstox brokerage/other charges: read Upstox pricing page / first contract note.

## 5. Tax (not verified; ask a CA)
Delivery: capital gains (STCG 20%, LTCG 12.5% above Rs 1.25L - from memory). Intraday = speculative business income; F&O = non-speculative business income (ITR-3, loss carry-forward, audit thresholds). Keep full trade logs.

## 6. Data/licence
Upstox data for personal use only; do not redistribute. News scraping: check ToS.
