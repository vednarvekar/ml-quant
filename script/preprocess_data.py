import numpy as np
import pandas as pd
from pathlib import Path
import json
from collections import Counter

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / 'data' / 'raw' / '5M'
PROCESSED_DIR = BASE_DIR / 'data' / 'processed' / '5M'
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

WINDOW  = 60
HORIZON = 10
ATR_PERIOD = 14
UPPER_MULT = 1.5
LOWER_MULT = 1.0

def list_stocks():
    stocks = set()
    for file in DATA_DIR.glob('*_ohlcv.json'):
        file_name = file.stem.replace('_ohlcv', '').upper()
        stocks.add(file_name)
    print(*sorted(stocks), sep=', ')


#------------- FEATURE EXTRACTION -------------
def extract_features(df: pd.DataFrame) -> np.ndarray:
    """
    Input : df with columns open, high, low, close, volume, oi
    Output: np.array shape (T, 12) — one row per candle
    """
    o = df['open'].to_numpy(float)
    h = df['high'].to_numpy(float)
    l = df['low'].to_numpy(float)
    c = df['close'].to_numpy(float)
    v = df['volume'].to_numpy(float)
    eps = 1e-9

    # --- Differential Encoding ---
    prev_c = np.roll(c, 1); prev_c[0] = c[0]
    prev_v = np.roll(v, 1); prev_v[0] = v[0]

    close_ret = (c - prev_c) / (prev_c + eps)  # Col 1
    open_ret = (o - prev_c) / (prev_c + eps)   # Col 2
    high_ret = (h - prev_c) / (prev_c + eps)   # Col 3
    low_ret  = (l - prev_c) / (prev_c + eps)   # Col 4
    vol_ret = (v - prev_v) / (prev_v + eps)    # Col 5

    # --- Candlestick Features ---
    hl = np.where((h - l) < eps, eps, h - l)
    body = c - o
    body_ratio = body / hl                      # Col 6           
    upper_wick = (h - np.maximum(o, c)) / hl    # Col 7
    lower_wick = (np.minimum(o, c) - l) / hl    # Col 8

    # --- Candlesticks ---
    prev_body = np.roll(body, 1); prev_body[0] = body[0]
    prev_o = np.roll(o, 1); prev_o[0] = o[0]

    engulf_bull = ((body > 0) & (prev_body < 0) & (o < prev_c) & (c > prev_o)).astype(float)  # Col 9
    engulf_bear = ((body < 0) & (prev_body > 0) & (o > prev_c) & (c < prev_o)).astype(float)  # Col 10
    doji = (np.abs(body_ratio) < 0.1).astype(float)                                            # Col 11

    # --- Volume --- 
    v_mean20 = pd.Series(v).rolling(20, min_periods=1).mean().to_numpy()
    vol_surge = np.clip(v / (v_mean20 + eps), 0, 10)                                          # Col 12

    return np.stack([
        close_ret, open_ret, high_ret, low_ret, vol_ret,
        body_ratio, upper_wick, lower_wick,
        engulf_bull, engulf_bear, doji, vol_surge,
    ], axis=1).astype(np.float32)


# ------------ Average True Range (ATR) -------------
def compute_atr(df):
    h, l, c  = df['high'], df['low'], df['close']
    prev_c   = c.shift(1)
    tr = pd.concat([h-l, (h-prev_c).abs(), (l-prev_c).abs()], axis=1).max(axis=1)
    return tr.ewm(span=ATR_PERIOD, adjust=False).mean().to_numpy()


# ------------- Creating Labels -------------
def create_labels(closes, atr_arr):
    labels, returns = [], []
    for i in range(len(closes)):
        p0 = closes[i]
        atr = atr_arr[i]
        if p0 <= 0 or atr <= 0 or np.isnan(atr):
            labels.append(0); returns.append(0.0); continue
        upper = p0 + UPPER_MULT * atr
        lower = p0 - LOWER_MULT * atr
        label, ret = 0, 0.0

        for j in range(1, HORIZON + 1):
            fi = i + j
            if fi >= len(closes): break
            if closes[fi] >= upper: label, ret = 1, (closes[fi]-p0)/p0; break
            if closes[fi] <= lower: label, ret = 2, (closes[fi]-p0)/p0; break
        else:
            last = min(i + HORIZON, len(closes)-1)
            ret  = (closes[last] - p0) / p0
        labels.append(label);  returns.append(ret)

    return np.array(labels, dtype=np.int64), np.array(returns, dtype=np.float32)


# ------------- MAIN -------------
def process_all_stocks(DATA_DIR: Path, PROCESSED_DIR: Path):
    files = sorted(DATA_DIR.glob('*_ohlcv.json'))
    print(f"Found {len(files)} stocks to process. Running memory-safe execution...\n")

    total_windows_saved = 0
    global_label_counts = Counter()

    for file in files:
        stock = file.stem.replace('_ohlcv', '').upper()
        print(f"Processing stock {stock}...")

        # 1. Load data for single file
        with open(file, 'r') as f:
            data = json.load(f)

        df = pd.DataFrame(data)
        df = df.drop(columns=['timestamp'], errors='ignore')
        df = df.astype(float)

        # 2. Extract calculations
        features = extract_features(df)
        closes = df['close'].to_numpy(float)
        atr_arr = compute_atr(df)
        labels, returns = create_labels(closes, atr_arr)

        # Local containers for this stock only
        X_stock, y_stock, r_stock = [], [], []

        # 3. Slice windows
        for i in range(WINDOW, len(features) - HORIZON):
            window = features[i-WINDOW : i]
            X_stock.append(window)
            y_stock.append(labels[i-1])
            r_stock.append(returns[i-1])

        # Skip files that don't yield windows
        if not X_stock:
            print(f"  Skipped {stock} — insufficient historical records.")
            continue

        # 4. Cast to NumPy structures immediately
        X = np.asarray(X_stock, dtype=np.float32)
        y = np.asarray(y_stock, dtype=np.int64)
        r = np.asarray(r_stock, dtype=np.float32)

        # 5. Flush directly to disk (clears RAM immediately on loop cycle)
        np.save(PROCESSED_DIR / f'{stock}_X.npy', X)
        np.save(PROCESSED_DIR / f'{stock}_y.npy', y)
        np.save(PROCESSED_DIR / f'{stock}_r.npy', r)

        # Metrics Tracking
        num_windows = len(y)
        total_windows_saved += num_windows
        global_label_counts.update(y.tolist())

        print(f"  Done — Generated {num_windows:,} windows ({X.nbytes/1e6:.1f} MB saved to disk)")

        # Explicitly clear variables from scope to free RAM space
        del data, df, features, closes, atr_arr, labels, returns, X_stock, y_stock, r_stock, X, y, r

    # Summary report
    print(f"\n All stocks processed safely without memory spikes!")
    print(f"Total windows stored across dataset: {total_windows_saved:,}")
    print(f"Global label balance — HOLD: {global_label_counts[0]/total_windows_saved:.1%} | "
          f"BUY: {global_label_counts[1]/total_windows_saved:.1%} | "
          f"SELL: {global_label_counts[2]/total_windows_saved:.1%}")


process_all_stocks(DATA_DIR, PROCESSED_DIR)