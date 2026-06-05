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
WINDOW  = 60        # Lookback window size (number of 5-minute candles)
HORIZON = 10        # Forward-looking horizon to check for targets
ATR_PERIOD = 14     # Period used for Average True Range calculation
UPPER_MULT = 1.5    # Multiplier for upper barrier (BUY label target)
LOWER_MULT = 1.0    # Multiplier for lower barrier (SELL label target)

# Chronological data allocation ratios
VAL_SPLIT  = 0.1    # 10% of each stock's timeline goes to validation data
TEST_SPLIT = 0.1    # 10% of each stock's timeline goes to test data


# ─── FEATURE EXTRACTION ENGINE ───────────────────────────────────────────────
def extract_features(df: pd.DataFrame) -> np.ndarray:
    """Transforms raw OHLCV bars into 12 normalized, stationary ML features."""
    o = df['open'].to_numpy(float)
    h = df['high'].to_numpy(float)
    l = df['low'].to_numpy(float)
    c = df['close'].to_numpy(float)
    v = df['volume'].to_numpy(float)
    eps = 1e-9  # Small constant to prevent division-by-zero errors

    # 1. Differential Returns (Makes data stationary over time)
    prev_c, prev_v = np.roll(c, 1), np.roll(v, 1)
    prev_c[0], prev_v[0] = c[0], v[0]  # Avoid boundary artifacts from roll

    close_ret = (c - prev_c) / (prev_c + eps)
    open_ret = (o - prev_c) / (prev_c + eps)
    high_ret = (h - prev_c) / (prev_c + eps)
    low_ret  = (l - prev_c) / (prev_c + eps)
    vol_ret = (v - prev_v) / (prev_v + eps)
    vol_ret = np.clip(vol_ret, -10, 10)  

    # 2. Candlestick Structure Properties
    hl = np.where((h - l) < eps, eps, h - l)
    body = c - o
    body_ratio = body / hl                             
    upper_wick = (h - np.maximum(o, c)) / hl    
    lower_wick = (np.minimum(o, c) - l) / hl    

    # 3. Structural Patterns (Binary flags mapped to float values)
    prev_body, prev_o = np.roll(body, 1), np.roll(o, 1)
    prev_body[0], prev_o[0] = body[0], o[0]

    engulf_bull = ((body > 0) & (prev_body < 0) & (o < prev_c) & (c > prev_o)).astype(float)
    engulf_bear = ((body < 0) & (prev_body > 0) & (o > prev_c) & (c < prev_o)).astype(float)
    doji = (np.abs(body_ratio) < 0.1).astype(float)

    # 4. Volume Dynamics
    v_mean20 = pd.Series(v).rolling(20, min_periods=1).mean().to_numpy()
    vol_surge = np.clip(v / (v_mean20 + eps), 0, 10)                                          

    # Combine all processed features into an output shape of (Total Candles, 12)
    return np.stack([
        close_ret, open_ret, high_ret, low_ret, vol_ret,
        body_ratio, upper_wick, lower_wick,
        engulf_bull, engulf_bear, doji, vol_surge,
    ], axis=1).astype(np.float32)


def compute_atr(df):
    """Calculates wilder-style exponential rolling market volatility."""
    h, l, c  = df['high'], df['low'], df['close']
    prev_c   = c.shift(1)
    tr = pd.concat([h-l, (h-prev_c).abs(), (l-prev_c).abs()], axis=1).max(axis=1)
    return tr.ewm(span=ATR_PERIOD, adjust=False).mean().to_numpy()


def create_labels(closes, atr_arr):
    """Executes rolling horizontal-barrier checks based on local volatility."""
    labels = []
    for i in range(len(closes)):
        p0, atr = closes[i], atr_arr[i]
        if p0 <= 0 or atr <= 0 or np.isnan(atr):
            labels.append(0); continue
        
        # Calculate localized price bounds based on current ATR
        upper = p0 + UPPER_MULT * atr
        lower = p0 - LOWER_MULT * atr
        label = 0  # Default label is 0 (HOLD)

        # Look forward up to HORIZON steps to see which barrier breaks first
        for j in range(1, HORIZON + 1):
            fi = i + j
            if fi >= len(closes): 
                break
            if closes[fi] >= upper: 
                label = 1; break  # BUY
            if closes[fi] <= lower: 
                label = 2; break  # SELL
        labels.append(label)
    return np.array(labels, dtype=np.int64)


# ─── CORE PROCESSING ENGINE ──────────────────────────────────────────────────
def process_all_stocks(DATA_DIR: Path, PROCESSED_DIR: Path):
    files = sorted(DATA_DIR.glob('*_ohlcv.json'))
    print(f"Found {len(files)} stocks. Slicing sequence boundaries...\n")

    # ──── PASS 1: Calculate precise split matrices sizes per stock ────
    total_train, total_val, total_test = 0, 0, 0
    stock_metadata = []

    for file in files:
        with open(file, 'r') as f:
            data = json.load(f)
        df = pd.DataFrame(data).drop(columns=['timestamp'], errors='ignore').astype(float)
        n_windows = len(range(WINDOW, len(df) - HORIZON))
        
        if n_windows > 0:
            # Chronologically calculate allocations specifically for this stock
            n_test = int(n_windows * TEST_SPLIT)
            n_val  = int(n_windows * VAL_SPLIT)
            n_train = n_windows - n_val - n_test
            
            total_train += n_train
            total_val   += n_val
            total_test  += n_test
            
            stock_metadata.append({'file': file, 'n_train': n_train, 'n_val': n_val, 'n_test': n_test})
        del data, df

    print(f"Global Allocation Sizes — Train: {total_train:,} | Val: {total_val:,} | Test: {total_test:,}")
    N_FEATURES = 12

    # Save array geometry data so training script can read raw binary matrices back
    with open(PROCESSED_DIR / 'shapes.json', 'w') as sf:
        json.dump({
            'train_shape': [total_train, WINDOW, N_FEATURES],
            'val_shape': [total_val, WINDOW, N_FEATURES],
            'test_shape': [total_test, WINDOW, N_FEATURES]
        }, sf)

    # Pre-allocate binary file blocks directly on your storage disk (zero RAM footprint)
    X_train_mm = np.memmap(PROCESSED_DIR / 'X_train.npy', dtype=np.float32, mode='w+', shape=(total_train, WINDOW, N_FEATURES))
    y_train_mm = np.memmap(PROCESSED_DIR / 'y_train.npy', dtype=np.int64,   mode='w+', shape=(total_train,))
    X_val_mm   = np.memmap(PROCESSED_DIR / 'X_val.npy',   dtype=np.float32, mode='w+', shape=(total_val, WINDOW, N_FEATURES))
    y_val_mm   = np.memmap(PROCESSED_DIR / 'y_val.npy',   dtype=np.int64,   mode='w+', shape=(total_val,))
    X_test_mm  = np.memmap(PROCESSED_DIR / 'X_test.npy',  dtype=np.float32, mode='w+', shape=(total_test, WINDOW, N_FEATURES))
    y_test_mm  = np.memmap(PROCESSED_DIR / 'y_test.npy',  dtype=np.int64,   mode='w+', shape=(total_test,))

    # ──── PASS 2: Compute and write cross-sectional slices to disk ────
    c_train, c_val, c_test = 0, 0, 0
    global_label_counts = Counter()

    for meta in stock_metadata:
        stock = meta['file'].stem.replace('_ohlcv', '').upper()
        with open(meta['file'], 'r') as f:
            data = json.load(f)

        df = pd.DataFrame(data).drop(columns=['timestamp'], errors='ignore').astype(float)
        features = extract_features(df)
        labels = create_labels(df['close'].to_numpy(float), compute_atr(df))

        # Build rolling windows sequences
        stock_windows, stock_labels = [], []
        for i in range(WINDOW, len(features) - HORIZON):
            stock_windows.append(features[i-WINDOW : i])
            stock_labels.append(labels[i-1])

        s_windows = np.asarray(stock_windows, dtype=np.float32)
        s_labels  = np.asarray(stock_labels, dtype=np.int64)
        nt, nv, nte = meta['n_train'], meta['n_val'], meta['n_test']

        # Map subsets into distinct structural files sequentially
        if nt > 0:
            X_train_mm[c_train : c_train+nt] = s_windows[:nt]
            y_train_mm[c_train : c_train+nt] = s_labels[:nt]
            c_train += nt
        if nv > 0:
            X_val_mm[c_val : c_val+nv] = s_windows[nt : nt+nv]
            y_val_mm[c_val : c_val+nv] = s_labels[nt : nt+nv]
            c_val += nv
        if nte > 0:
            X_test_mm[c_test : c_test+nte] = s_windows[nt+nv :]
            y_test_mm[c_test : c_test+nte] = s_labels[nt+nv :]
            c_test += nte

        global_label_counts.update(stock_labels)
        print(f"  Processed {stock} | Safe chunks mapped onto disk arrays.")
        del data, df, features, labels, stock_windows, stock_labels

    # Close and flush binary file system streams from RAM
    del X_train_mm, y_train_mm, X_val_mm, y_val_mm, X_test_mm, y_test_mm
    print(f"\n✅ Done. Preallocated dataset segments saved to {PROCESSED_DIR}")

if __name__ == '__main__':
    process_all_stocks(DATA_DIR, PROCESSED_DIR)