import numpy as np
import pandas as pd
from pathlib import Path
import json
from collections import Counter

# ─── DIRECTORY CONFIGURATION ─────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / 'data' / 'raw' / '5M'
PROCESSED_DIR = BASE_DIR / 'data' / 'processed' / '5M'
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# ─── HYPERPARAMETERS ─────────────────────────────────────────────────────────
WINDOW     = 60     # Lookback window size (number of 5-minute candles)
HORIZON    = 30     # Forward-looking horizon: 30 bars x 5min = 2.5 hours. Longer horizon = cleaner labels.
ATR_PERIOD = 14     # Period used for Average True Range calculation

# [CHANGED] Switched from barrier-touch labeling to trend-following labeling.
# Old barrier method: checked which barrier (upper/lower) price touched first within HORIZON.
# Problem: with asymmetric or small ATR, SELL labels got hit more often purely by chance.
# New method: look at where price actually IS after HORIZON bars, then compare the move
# to local ATR volatility. This produces much cleaner, less noisy labels.
# LABEL_MULT controls the threshold: move must be > 1x ATR to count as BUY/SELL.
# Moves smaller than that are HOLD (noise, not a real trend).
LABEL_MULT = 1.0

VAL_SPLIT  = 0.1
TEST_SPLIT = 0.1

# [CHANGED] N_FEATURES updated from 12 to 18.
# Added 6 trend/momentum features so the CNN has sequential structure to convolve over.
# Without these, all 12 original features were per-bar statistics — the CNN had nothing
# to detect trends or momentum shifts across the 60-bar window.
N_FEATURES = 18


# ─── FEATURE EXTRACTION ENGINE ───────────────────────────────────────────────
def extract_features(df: pd.DataFrame) -> np.ndarray:
    """Transforms raw OHLCV bars into 18 normalized, stationary ML features."""
    o = df['open'].to_numpy(float)
    h = df['high'].to_numpy(float)
    l = df['low'].to_numpy(float)
    c = df['close'].to_numpy(float)
    v = df['volume'].to_numpy(float)
    eps = 1e-9

    # ── ORIGINAL 12 FEATURES (unchanged) ─────────────────────────────────────

    # 1. Differential Returns
    prev_c, prev_v = np.roll(c, 1), np.roll(v, 1)
    prev_c[0], prev_v[0] = c[0], v[0]

    close_ret = (c - prev_c) / (prev_c + eps)
    open_ret  = (o - prev_c) / (prev_c + eps)
    high_ret  = (h - prev_c) / (prev_c + eps)
    low_ret   = (l - prev_c) / (prev_c + eps)
    vol_ret   = np.clip((v - prev_v) / (prev_v + eps), -10, 10)

    # 2. Candlestick Structure
    hl         = np.where((h - l) < eps, eps, h - l)
    body       = c - o
    body_ratio = body / hl
    upper_wick = (h - np.maximum(o, c)) / hl
    lower_wick = (np.minimum(o, c) - l) / hl

    # 3. Structural Patterns
    prev_body, prev_o = np.roll(body, 1), np.roll(o, 1)
    prev_body[0], prev_o[0] = body[0], o[0]

    engulf_bull = ((body > 0) & (prev_body < 0) & (o < prev_c) & (c > prev_o)).astype(float)
    engulf_bear = ((body < 0) & (prev_body > 0) & (o > prev_c) & (c < prev_o)).astype(float)
    doji        = (np.abs(body_ratio) < 0.1).astype(float)

    # 4. Volume Dynamics
    v_mean20  = pd.Series(v).rolling(20, min_periods=1).mean().to_numpy()
    vol_surge = np.clip(v / (v_mean20 + eps), 0, 10)

    # ── NEW 6 TREND/MOMENTUM FEATURES ────────────────────────────────────────
    # These give the CNN actual sequential structure to detect trends, breakouts,
    # and range position across the full 60-bar window. Without these the convolutions
    # were operating on per-bar noise with no multi-bar context.

    # Multi-timeframe momentum: where is price relative to N bars ago?
    # Clipped to [-0.1, 0.1] to remove extreme outlier spikes from data errors.
    c_5  = np.roll(c, 5);  c_5[:5]   = c[0]
    c_10 = np.roll(c, 10); c_10[:10] = c[0]
    c_20 = np.roll(c, 20); c_20[:20] = c[0]
    ret_5  = np.clip((c - c_5)  / (c_5  + eps), -0.1, 0.1)  # 25min momentum
    ret_10 = np.clip((c - c_10) / (c_10 + eps), -0.1, 0.1)  # 50min momentum
    ret_20 = np.clip((c - c_20) / (c_20 + eps), -0.1, 0.1)  # 100min momentum

    # Price position within recent 20-bar range (0 = at the low, 1 = at the high).
    # Tells the CNN if price is near a local top or bottom — critical for mean-reversion
    # and breakout detection.
    roll_high  = pd.Series(h).rolling(20, min_periods=1).max().to_numpy()
    roll_low   = pd.Series(l).rolling(20, min_periods=1).min().to_numpy()
    trend_pos  = (c - roll_low) / (roll_high - roll_low + eps)
    trend_pos  = np.clip(trend_pos, 0.0, 1.0)

    # Volume trend: is volume accelerating or decelerating recently?
    # Ratio of 5-bar avg volume to 20-bar avg volume, clipped to remove spikes.
    vol_ma5   = pd.Series(v).rolling(5,  min_periods=1).mean().to_numpy()
    vol_ma20  = pd.Series(v).rolling(20, min_periods=1).mean().to_numpy()
    vol_trend = np.clip((vol_ma5 - vol_ma20) / (vol_ma20 + eps), -5.0, 5.0)

    # ATR % of price: normalised local volatility. Tells the model whether the market
    # is calm or volatile right now — important context for interpreting price moves.
    h_s, l_s, c_s = df['high'], df['low'], df['close']
    tr    = pd.concat([h_s - l_s, (h_s - c_s.shift(1)).abs(), (l_s - c_s.shift(1)).abs()], axis=1).max(axis=1)
    atr14 = tr.ewm(span=ATR_PERIOD, adjust=False).mean().to_numpy()
    atr_pct = np.clip(atr14 / (c + eps), 0.0, 0.05)  # cap at 5% to remove insane outliers

    return np.stack([
        # Original 12
        close_ret, open_ret, high_ret, low_ret, vol_ret,
        body_ratio, upper_wick, lower_wick,
        engulf_bull, engulf_bear, doji, vol_surge,
        # New 6
        ret_5, ret_10, ret_20,
        trend_pos, vol_trend, atr_pct,
    ], axis=1).astype(np.float32)


def compute_atr(df):
    """Calculates wilder-style exponential rolling ATR."""
    h, l, c = df['high'], df['low'], df['close']
    prev_c  = c.shift(1)
    tr = pd.concat([h - l, (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    return tr.ewm(span=ATR_PERIOD, adjust=False).mean().to_numpy()


# [CHANGED] Replaced barrier-touch labeling with trend-following labeling.
# Old: scan forward bar by bar, label whichever barrier (upper/lower) price touches first.
# Problem with old: asymmetric hit rates even with equal multipliers because small ATR
# on 5min bars meant barriers were touched almost at random, producing very noisy labels.
# New: look at the single closing price exactly HORIZON bars ahead.
# If it moved up by more than LABEL_MULT * ATR → BUY.
# If it moved down by more than LABEL_MULT * ATR → SELL.
# Otherwise → HOLD (the move was smaller than normal volatility, i.e. noise).
# This gives the model a clear directional target based on actual future price,
# not a race to see which randomly-placed barrier gets hit first.
def create_labels(closes, atr_arr):
    """Trend-following labels: compare future close to current ATR threshold."""
    labels = np.zeros(len(closes), dtype=np.int64)

    for i in range(len(closes) - HORIZON):
        p0  = closes[i]
        atr = atr_arr[i]

        if p0 <= 0 or atr <= 0 or np.isnan(atr):
            continue

        future_return = (closes[i + HORIZON] - p0) / (p0 + 1e-9)
        atr_pct       = atr / (p0 + 1e-9)
        threshold     = LABEL_MULT * atr_pct

        if future_return > threshold:
            labels[i] = 1   # BUY
        elif future_return < -threshold:
            labels[i] = 2   # SELL
        # else: stays 0 (HOLD)

    return labels


# ─── CORE PROCESSING ENGINE ──────────────────────────────────────────────────
def process_all_stocks(DATA_DIR: Path, PROCESSED_DIR: Path):
    files = sorted(DATA_DIR.glob('*_ohlcv.json'))
    print(f"Found {len(files)} stocks. Slicing sequence boundaries...\n")

    total_train, total_val, total_test = 0, 0, 0
    stock_metadata = []

    # ── PASS 1: Calculate split sizes ────────────────────────────────────────
    for file in files:
        with open(file, 'r') as f:
            data = json.load(f)
        df        = pd.DataFrame(data).drop(columns=['timestamp'], errors='ignore').astype(float)
        n_windows = len(range(WINDOW, len(df) - HORIZON))

        if n_windows > 0:
            n_test  = int(n_windows * TEST_SPLIT)
            n_val   = int(n_windows * VAL_SPLIT)
            n_train = n_windows - n_val - n_test

            total_train += n_train
            total_val   += n_val
            total_test  += n_test
            stock_metadata.append({'file': file, 'n_train': n_train, 'n_val': n_val, 'n_test': n_test})
        del data, df

    print(f"Global Allocation Sizes — Train: {total_train:,} | Val: {total_val:,} | Test: {total_test:,}")

    with open(PROCESSED_DIR / 'shapes.json', 'w') as sf:
        json.dump({
            'train_shape': [total_train, WINDOW, N_FEATURES],
            'val_shape':   [total_val,   WINDOW, N_FEATURES],
            'test_shape':  [total_test,  WINDOW, N_FEATURES],
        }, sf)

    X_train_mm = np.memmap(PROCESSED_DIR / 'X_train.npy', dtype=np.float32, mode='w+', shape=(total_train, WINDOW, N_FEATURES))
    y_train_mm = np.memmap(PROCESSED_DIR / 'y_train.npy', dtype=np.int64,   mode='w+', shape=(total_train,))
    X_val_mm   = np.memmap(PROCESSED_DIR / 'X_val.npy',   dtype=np.float32, mode='w+', shape=(total_val,   WINDOW, N_FEATURES))
    y_val_mm   = np.memmap(PROCESSED_DIR / 'y_val.npy',   dtype=np.int64,   mode='w+', shape=(total_val,))
    X_test_mm  = np.memmap(PROCESSED_DIR / 'X_test.npy',  dtype=np.float32, mode='w+', shape=(total_test,  WINDOW, N_FEATURES))
    y_test_mm  = np.memmap(PROCESSED_DIR / 'y_test.npy',  dtype=np.int64,   mode='w+', shape=(total_test,))

    # ── PASS 2: Build windows and write to disk ───────────────────────────────
    c_train, c_val, c_test = 0, 0, 0
    global_label_counts    = Counter()

    for meta in stock_metadata:
        stock = meta['file'].stem.replace('_ohlcv', '').upper()
        with open(meta['file'], 'r') as f:
            data = json.load(f)

        df       = pd.DataFrame(data).drop(columns=['timestamp'], errors='ignore').astype(float)
        features = extract_features(df)
        labels   = create_labels(df['close'].to_numpy(float), compute_atr(df))

        stock_windows, stock_labels = [], []
        for i in range(WINDOW, len(features) - HORIZON):
            stock_windows.append(features[i - WINDOW : i])
            # [FIXED] Was labels[i-1] — that used the label of the LAST bar inside the window,
            # meaning the forward horizon partially overlapped with the window itself (data leak).
            # Now labels[i]: the label computed AT the end of the window, looking forward from there.
            stock_labels.append(labels[i])

        s_windows = np.asarray(stock_windows, dtype=np.float32)
        s_labels  = np.asarray(stock_labels,  dtype=np.int64)
        nt, nv, nte = meta['n_train'], meta['n_val'], meta['n_test']

        if nt > 0:
            X_train_mm[c_train : c_train + nt] = s_windows[:nt]
            y_train_mm[c_train : c_train + nt] = s_labels[:nt]
            c_train += nt
        if nv > 0:
            X_val_mm[c_val : c_val + nv] = s_windows[nt : nt + nv]
            y_val_mm[c_val : c_val + nv] = s_labels[nt : nt + nv]
            c_val += nv
        if nte > 0:
            X_test_mm[c_test : c_test + nte] = s_windows[nt + nv :]
            y_test_mm[c_test : c_test + nte] = s_labels[nt + nv :]
            c_test += nte

        global_label_counts.update(s_labels.tolist())
        total = len(stock_labels)
        dist  = {k: f"{v/total*100:.1f}%" for k, v in sorted(global_label_counts.items())}
        print(f"  Processed {stock} | windows: {total} | running label dist: {dist}")
        del data, df, features, labels, stock_windows, stock_labels

    del X_train_mm, y_train_mm, X_val_mm, y_val_mm, X_test_mm, y_test_mm

    total_labels = sum(global_label_counts.values())
    print(f"\nFinal label distribution:")
    print(f"  HOLD: {global_label_counts[0]/total_labels*100:.1f}%")
    print(f"  BUY:  {global_label_counts[1]/total_labels*100:.1f}%")
    print(f"  SELL: {global_label_counts[2]/total_labels*100:.1f}%")
    print(f"\n✅ Done. Dataset saved to {PROCESSED_DIR}")


if __name__ == '__main__':
    process_all_stocks(DATA_DIR, PROCESSED_DIR)
