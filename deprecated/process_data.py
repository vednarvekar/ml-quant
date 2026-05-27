"""
prepare_data.py
---------------
Builds per-stock .npy sample files ready to feed into the CNN.

What this adds over the old version
────────────────────────────────────
1.  Candlestick pattern features   (body ratio, wicks, gap, engulf, etc.)
2.  Volume / momentum features     (volume surge, RVOL, OI change)
3.  Support & resistance distances (pivot highs/lows from daily history)
4.  Trend context features         (EMA slope, ATR regime, candle position)
5.  Triple-barrier labeling        (ATR-scaled, asymmetric, time-capped)
6.  Anchor-relative normalisation  (pct change from window open, not min-max)
7.  Per-timeframe feature matrix   shape (N, WINDOW, N_FEATURES) per TF
8.  Extra scalar feature vector    shape (N, N_SCALARS) — S/R + regime

Output per stock  (in data/processed/<stock>/)
───────────────────────────────────────────────
  X_3min.npy        float32  (N, 60, F)
  X_5min.npy        float32  (N, 60, F)
  X_15min.npy       float32  (N, 60, F)
  X_1hr.npy         float32  (N, 60, F)
  X_1d.npy          float32  (N, 60, F)
  X_scalar.npy      float32  (N, S)      — S/R distances, regime, spread
  y_labels.npy      int64    (N,)        — 0 hold, 1 buy, 2 sell
  y_returns.npy     float32  (N,)        — raw forward return (for analysis)
  anchor_timestamps.npy  int64  (N,)
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

# ── paths ────────────────────────────────────────────────────────────────────
BASE_DIR     = Path(__file__).resolve().parent.parent
RAW_DIR      = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

# ── window sizes (candles kept per timeframe) ─────────────────────────────────
WINDOW = 60          # same for all TFs — change here to affect everything

# ── labeling ─────────────────────────────────────────────────────────────────
HORIZON_5M   = 10    # how many 5-min candles ahead to watch
ATR_PERIOD   = 14    # ATR lookback for barrier scaling
UPPER_MULT   = 1.5   # profit barrier = 1.5 × ATR  (positive asymmetry)
LOWER_MULT   = 1.0   # stop barrier   = 1.0 × ATR

# ── feature columns in raw files ─────────────────────────────────────────────
RAW_COLS = ["open", "high", "low", "close", "volume", "oi"]

# ── how many engineered features we produce per candle ───────────────────────
#   open, high, low, close, volume, oi          → 6  (anchor-relative pct)
#   body_ratio, upper_wick, lower_wick          → 3
#   gap_pct                                     → 1
#   volume_surge (vs 20-bar rolling mean)       → 1
#   oi_change_pct                               → 1
#   engulf_bull, engulf_bear                    → 2
#   doji_flag                                   → 1
#   total = 15
N_FEATURES = 15

# ── scalar features (one vector per sample, not per candle) ──────────────────
#   dist_to_resistance, dist_to_support         → 2
#   atr_pct (volatility regime)                 → 1
#   trend_slope_1h (EMA20 slope on 1h)          → 1
#   trend_slope_1d (EMA20 slope on 1d)          → 1
#   rvol_5m (relative volume vs 20-day same-hour mean) → 1
#   total = 6
N_SCALARS = 6


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--stocks", nargs="+", default=["all"])
    return p.parse_args()


def list_stocks():
    stocks = set()
    for tf in ["3M", "5M", "15M", "1H", "1D"]:
        for path in (RAW_DIR / tf).glob("*_ohlcv.json"):
            stocks.add(path.stem.replace("_ohlcv", ""))
    return sorted(stocks)


def get_stocks_to_process(requested):
    all_stocks = list_stocks()
    if requested == ["all"]:
        return all_stocks
    for s in requested:
        if s not in all_stocks:
            raise ValueError(f"Unknown stock: {s}. Available: {all_stocks}")
    return requested


# ─────────────────────────────────────────────────────────────────────────────
# I/O helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_frame(path: Path) -> pd.DataFrame:
    with open(path) as f:
        rows = json.load(f)
    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError(f"Empty file: {path}")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = (df.drop_duplicates("timestamp")
            .sort_values("timestamp")
            .set_index("timestamp"))
    # ensure all columns exist; fill missing oi with 0
    for col in RAW_COLS:
        if col not in df.columns:
            df[col] = 0.0
    return df[RAW_COLS].astype(np.float64)


# ─────────────────────────────────────────────────────────────────────────────
# ATR (used for labeling)
# ─────────────────────────────────────────────────────────────────────────────

def compute_atr_series(df: pd.DataFrame, period: int = 14) -> pd.Series:
    h, l, c = df["high"], df["low"], df["close"]
    prev_c = c.shift(1)
    tr = pd.concat([h - l,
                    (h - prev_c).abs(),
                    (l - prev_c).abs()], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


# ─────────────────────────────────────────────────────────────────────────────
# Triple-barrier labeling
# ─────────────────────────────────────────────────────────────────────────────

def triple_barrier_label(prices: np.ndarray,
                         idx: int,
                         atr_val: float,
                         horizon: int,
                         upper_mult: float,
                         lower_mult: float):
    """
    Returns (label, forward_return)
      label: 1=buy, 2=sell, 0=hold/timeout
    """
    p0 = prices[idx]
    if p0 <= 0 or np.isnan(atr_val) or atr_val <= 0:
        return 0, 0.0

    upper = p0 + upper_mult * atr_val
    lower = p0 - lower_mult * atr_val

    for j in range(1, horizon + 1):
        fi = idx + j
        if fi >= len(prices):
            break
        pf = prices[fi]
        if pf >= upper:
            return 1, float((pf - p0) / p0)
        if pf <= lower:
            return 2, float((pf - p0) / p0)

    # time barrier — still record the actual return for analysis
    last_i = min(idx + horizon, len(prices) - 1)
    raw_ret = float((prices[last_i] - p0) / p0)
    return 0, raw_ret


# ─────────────────────────────────────────────────────────────────────────────
# Per-candle feature engineering  →  shape (T, N_FEATURES)
# ─────────────────────────────────────────────────────────────────────────────

def build_features(df: pd.DataFrame) -> np.ndarray:
    """
    Takes a raw OHLCV+OI dataframe and returns a feature matrix (T, N_FEATURES).
    All values are computed for EVERY row so we can slice windows later.
    """
    o = df["open"].to_numpy(dtype=float)
    h = df["high"].to_numpy(dtype=float)
    l = df["low"].to_numpy(dtype=float)
    c = df["close"].to_numpy(dtype=float)
    v = df["volume"].to_numpy(dtype=float)
    oi = df["oi"].to_numpy(dtype=float)
    T = len(df)

    eps = 1e-9

    # ── price range (used many times) ────────────────────────────────────────
    hl_range = h - l
    hl_range = np.where(hl_range < eps, eps, hl_range)

    # ── body ─────────────────────────────────────────────────────────────────
    body      = c - o                               # signed body
    body_ratio = body / hl_range                   # [-1, 1]

    # ── wicks ────────────────────────────────────────────────────────────────
    upper_wick = (h - np.maximum(o, c)) / hl_range  # [0, 1]
    lower_wick = (np.minimum(o, c) - l) / hl_range  # [0, 1]

    # ── gap from previous close ───────────────────────────────────────────────
    prev_c   = np.roll(c, 1);  prev_c[0] = c[0]
    gap_pct  = (o - prev_c) / (prev_c + eps)

    # ── volume surge (vs rolling 20-bar mean) ────────────────────────────────
    v_mean20 = pd.Series(v).rolling(20, min_periods=1).mean().to_numpy()
    vol_surge = v / (v_mean20 + eps)
    vol_surge = np.clip(vol_surge, 0, 10)           # cap outliers

    # ── OI change pct ─────────────────────────────────────────────────────────
    prev_oi   = np.roll(oi, 1);  prev_oi[0] = oi[0]
    oi_chg    = (oi - prev_oi) / (np.abs(prev_oi) + eps)
    oi_chg    = np.clip(oi_chg, -1, 1)

    # ── bullish engulfing flag ────────────────────────────────────────────────
    prev_o = np.roll(o, 1); prev_o[0] = o[0]
    prev_body = np.roll(body, 1); prev_body[0] = body[0]
    engulf_bull = ((body > 0) &
                   (prev_body < 0) &
                   (o < prev_c) &
                   (c > prev_o)).astype(np.float32)

    # ── bearish engulfing flag ────────────────────────────────────────────────
    engulf_bear = ((body < 0) &
                   (prev_body > 0) &
                   (o > prev_c) &
                   (c < prev_o)).astype(np.float32)

    # ── doji flag (body < 10% of range) ─────────────────────────────────────
    doji_flag = (np.abs(body_ratio) < 0.1).astype(np.float32)

    # ── assemble ─────────────────────────────────────────────────────────────
    # Columns 0-5: raw OHLCV+OI (will be anchor-normalised per window)
    # Columns 6-14: pattern/volume features (already dimensionless ratios)
    feat = np.stack([
        o, h, l, c, v, oi,          # 0-5  (raw, will normalise later)
        body_ratio,                  # 6
        upper_wick,                  # 7
        lower_wick,                  # 8
        gap_pct,                     # 9
        vol_surge,                   # 10
        oi_chg,                      # 11
        engulf_bull,                 # 12
        engulf_bear,                 # 13
        doji_flag,                   # 14
    ], axis=1).astype(np.float64)    # (T, 15)

    return feat


# ─────────────────────────────────────────────────────────────────────────────
# Window normalisation
# ─────────────────────────────────────────────────────────────────────────────

def normalise_window(window: np.ndarray) -> np.ndarray:
    """
    window: (WINDOW, N_FEATURES)

    Columns 0-5  (OHLCV + OI) → anchor-relative pct change from first-bar close.
                                  Preserves move magnitude AND direction.
    Columns 6-14 (ratios/flags) → already [−1,1] or [0,1] — just clip-scale.

    Returns float32 (WINDOW, N_FEATURES).
    """
    w = window.copy()

    # ── price / volume columns: pct change from anchor ───────────────────────
    anchor_close  = w[0, 3] + 1e-9          # first candle's close
    anchor_volume = w[0, 4] + 1e-9
    anchor_oi     = abs(w[0, 5]) + 1e-9

    w[:, 0] = (w[:, 0] - anchor_close)  / anchor_close   # open
    w[:, 1] = (w[:, 1] - anchor_close)  / anchor_close   # high
    w[:, 2] = (w[:, 2] - anchor_close)  / anchor_close   # low
    w[:, 3] = (w[:, 3] - anchor_close)  / anchor_close   # close
    w[:, 4] = (w[:, 4] - anchor_volume) / anchor_volume  # volume
    w[:, 5] = (w[:, 5] - anchor_oi)     / anchor_oi      # oi

    # ── clip pct-change columns to ±50 % (handles splits / bad ticks) ───────
    w[:, :6] = np.clip(w[:, :6], -0.5, 0.5)

    # ── ratio/flag columns: already in reasonable range, just clip ───────────
    w[:, 6]  = np.clip(w[:, 6],  -1,   1)   # body_ratio
    w[:, 7]  = np.clip(w[:, 7],   0,   1)   # upper_wick
    w[:, 8]  = np.clip(w[:, 8],   0,   1)   # lower_wick
    w[:, 9]  = np.clip(w[:, 9], -0.1, 0.1)  # gap_pct
    w[:, 10] = np.clip(w[:, 10],  0,  10)   # vol_surge (already clipped, belt+braces)
    w[:, 11] = np.clip(w[:, 11], -1,   1)   # oi_chg
    # cols 12-14 are binary flags, no change needed

    return w.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Scalar (per-sample) feature builder
# ─────────────────────────────────────────────────────────────────────────────

def build_scalar_features(df_1d: pd.DataFrame,
                           df_1h: pd.DataFrame,
                           df_5m: pd.DataFrame,
                           anchor_time,
                           atr_val: float) -> np.ndarray:
    """
    Returns 1-D array of N_SCALARS floats.
    All values are dimensionless (ratios / slopes).
    """
    # ── current price ─────────────────────────────────────────────────────────
    cur = float(df_5m.loc[:anchor_time]["close"].iloc[-1])
    eps = 1e-9

    # ── support & resistance from daily pivots (last 120 days) ───────────────
    hist_1d = df_1d.loc[:anchor_time].tail(120)
    if len(hist_1d) >= 5:
        highs = hist_1d["high"].to_numpy(dtype=np.float64)
        lows = hist_1d["low"].to_numpy(dtype=np.float64)
        res_candidates = highs[highs > cur]
        sup_candidates = lows[lows < cur]
        dist_res = float((res_candidates.min() - cur) / cur) if len(res_candidates) else 0.05
        dist_sup = float((cur - sup_candidates.max()) / cur) if len(sup_candidates) else 0.05
    else:
        dist_res, dist_sup = 0.05, 0.05

    dist_res = float(np.clip(dist_res, 0, 0.2))
    dist_sup = float(np.clip(dist_sup, 0, 0.2))

    # ── ATR as % of price (volatility regime) ────────────────────────────────
    atr_pct = float(np.clip(atr_val / (cur + eps), 0, 0.05))

    # ── EMA-20 slope on 1-hour (trend context) ────────────────────────────────
    hist_1h = df_1h.loc[:anchor_time].tail(40)
    if len(hist_1h) >= 22:
        ema = hist_1h["close"].ewm(span=20, adjust=False).mean().to_numpy(dtype=np.float64)
        # slope = (last EMA − EMA 5 bars ago) / price, normalised
        slope_1h = float(np.clip((ema[-1] - ema[-6]) / (cur + eps), -0.05, 0.05))
    else:
        slope_1h = 0.0

    # ── EMA-20 slope on daily ─────────────────────────────────────────────────
    if len(hist_1d) >= 22:
        ema_d = hist_1d["close"].ewm(span=20, adjust=False).mean().to_numpy(dtype=np.float64)
        slope_1d = float(np.clip((ema_d[-1] - ema_d[-6]) / (cur + eps), -0.1, 0.1))
    else:
        slope_1d = 0.0

    # ── relative volume (5-min bar vs same-hour average over last 20 days) ───
    hist_5m = df_5m.loc[:anchor_time].tail(1)
    last_vol = float(hist_5m["volume"].iloc[-1]) if len(hist_5m) else 0.0
    # rolling mean over the last 100 5-min bars
    vol_history = float(df_5m.loc[:anchor_time].tail(100)["volume"].mean())
    rvol = float(np.clip(last_vol / (vol_history + eps), 0, 10))

    return np.array([dist_res, dist_sup, atr_pct, slope_1h, slope_1d, rvol],
                    dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Main per-stock processing
# ─────────────────────────────────────────────────────────────────────────────

def process_stock(stock: str):
    print(f"\n{'─'*60}")
    print(f"  Processing {stock}")
    print(f"{'─'*60}")

    # ── load raw frames ───────────────────────────────────────────────────────
    paths = {
        "3min":  RAW_DIR / "3M"  / f"{stock}_ohlcv.json",
        "5min":  RAW_DIR / "5M"  / f"{stock}_ohlcv.json",
        "15min": RAW_DIR / "15M" / f"{stock}_ohlcv.json",
        "1hr":   RAW_DIR / "1H"  / f"{stock}_ohlcv.json",
        "1d":    RAW_DIR / "1D"  / f"{stock}_ohlcv.json",
    }
    for k, p in paths.items():
        if not p.exists():
            raise FileNotFoundError(f"Missing {k} file for {stock}: {p}")

    df_3m  = load_frame(paths["3min"])
    df_5m  = load_frame(paths["5min"])
    df_15m = load_frame(paths["15min"])
    df_1h  = load_frame(paths["1hr"])
    df_1d  = load_frame(paths["1d"])

    # ── build full feature matrices (one row per candle) ─────────────────────
    print(f"  Building feature matrices...", flush=True)
    feat_3m  = build_features(df_3m)
    feat_5m  = build_features(df_5m)
    feat_15m = build_features(df_15m)
    feat_1h  = build_features(df_1h)
    feat_1d  = build_features(df_1d)

    # ── ATR on 5-min for labeling ─────────────────────────────────────────────
    atr_5m = compute_atr_series(df_5m, ATR_PERIOD).values
    closes_5m = df_5m["close"].to_numpy()

    # ── iterate over 5-min anchor bars ────────────────────────────────────────
    # Start late enough that we have 60 daily candles (≈60 trading days ≈ 3 months)
    start_index = max(WINDOW, 720)   # 720 5-min bars ≈ 1 trading week

    x3_list, x5_list, x15_list, xh_list, xd_list = [], [], [], [], []
    xs_list = []
    y_list, yret_list, ts_list = [], [], []

    for i in range(start_index, len(df_5m) - HORIZON_5M):
        anchor_time = df_5m.index[i]

        # ── slice windows (by iloc-equivalent into pre-built matrices) ────────
        # For non-5m frames we use .loc to get all bars up to anchor_time
        # then take the tail — this is safe across all TFs

        # helper: get positional slice for a frame aligned to anchor_time
        def tail_feat(df, feat_mat, n):
            pos = df.index.searchsorted(anchor_time, side="right")  # exclusive
            start = pos - n
            if start < 0 or pos > len(feat_mat):
                return None
            return feat_mat[start:pos]

        w3m  = tail_feat(df_3m,  feat_3m,  WINDOW)
        w5m  = tail_feat(df_5m,  feat_5m,  WINDOW)
        w15m = tail_feat(df_15m, feat_15m, WINDOW)
        w1h  = tail_feat(df_1h,  feat_1h,  WINDOW)
        w1d  = tail_feat(df_1d,  feat_1d,  WINDOW)

        if any(w is None or len(w) < WINDOW for w in [w3m, w5m, w15m, w1h, w1d]):
            continue

        assert w3m is not None
        assert w5m is not None
        assert w15m is not None
        assert w1h is not None
        assert w1d is not None

        # ── label via triple barrier ──────────────────────────────────────────
        atr_val = float(atr_5m[i]) if i < len(atr_5m) and not np.isnan(atr_5m[i]) else 0.0
        label, fwd_ret = triple_barrier_label(
            closes_5m, i, atr_val,
            HORIZON_5M, UPPER_MULT, LOWER_MULT
        )

        # ── scalar features ───────────────────────────────────────────────────
        scalar = build_scalar_features(df_1d, df_1h, df_5m, anchor_time, atr_val)

        # ── normalise windows ─────────────────────────────────────────────────
        x3_list.append(normalise_window(w3m))
        x5_list.append(normalise_window(w5m))
        x15_list.append(normalise_window(w15m))
        xh_list.append(normalise_window(w1h))
        xd_list.append(normalise_window(w1d))
        xs_list.append(scalar)
        y_list.append(label)
        yret_list.append(fwd_ret)
        ts_list.append(anchor_time.value)

        if len(y_list) % 10_000 == 0:
            print(f"  {stock}: {len(y_list):,} samples...", flush=True)

    if len(y_list) == 0:
        print(f"  WARNING: no samples generated for {stock}. Skipping.")
        return

    # ── save ─────────────────────────────────────────────────────────────────
    out = PROCESSED_DIR / stock
    out.mkdir(parents=True, exist_ok=True)

    np.save(out / "X_3min.npy",  np.asarray(x3_list,   dtype=np.float32))
    np.save(out / "X_5min.npy",  np.asarray(x5_list,   dtype=np.float32))
    np.save(out / "X_15min.npy", np.asarray(x15_list,  dtype=np.float32))
    np.save(out / "X_1hr.npy",   np.asarray(xh_list,   dtype=np.float32))
    np.save(out / "X_1d.npy",    np.asarray(xd_list,   dtype=np.float32))
    np.save(out / "X_scalar.npy",np.asarray(xs_list,   dtype=np.float32))
    np.save(out / "y_labels.npy",np.asarray(y_list,    dtype=np.int64))
    np.save(out / "y_returns.npy",np.asarray(yret_list, dtype=np.float32))
    np.save(out / "anchor_timestamps.npy", np.asarray(ts_list, dtype=np.int64))

    y_arr = np.asarray(y_list)
    counts = {lbl: int((y_arr == lbl).sum()) for lbl in [0, 1, 2]}
    total  = len(y_arr)
    print(f"  ✓ {stock}: {total:,} samples")
    print(f"    HOLD {counts[0]:,} ({100*counts[0]//total}%)  "
          f"BUY {counts[1]:,} ({100*counts[1]//total}%)  "
          f"SELL {counts[2]:,} ({100*counts[2]//total}%)")


# ─────────────────────────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    stocks = get_stocks_to_process(args.stocks)
    print(f"Stocks to process: {stocks}")
    for stock in stocks:
        process_stock(stock)
    print("\n✓ prepare_data done.")


if __name__ == "__main__":
    main()
