import numpy as np
import pandas as pd
from pathlib import Path
import json

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
    oi = df['oi'].to_numpy(float)
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

    engulf_bull = (         # Col 8
        (body > 0) &
        (prev_body < 0) &
        (o < prev_c) &
        (c > prev_o)
        ).astype(float)
    
    engulf_bear = (         # Col 9
        (body < 0) & 
        (prev_body > 0) & 
        (o > prev_c) & 
        (c < prev_o)
        ).astype(float)   
    
    doji = (np.abs(body_ratio) < 0.1).astype(float)             # Col 10

    # --- Volume & Open Interest--- 
    v_mean20 = pd.Series(v).rolling(20, min_periods=1).mean().to_numpy()
    oi_mean20 = pd.Series(oi).rolling(20, min_periods=1).mean().to_numpy()
    vol_surge = np.clip(v / (v_mean20 + eps), 0, 10)            # Col 11
    oi_surge  = np.clip(oi / (oi_mean20 + eps), 0, 10)          # Col 12

    return np.stack([
        close_ret, open_ret, high_ret, low_ret, vol_ret,
        body_ratio, upper_wick, lower_wick,
        engulf_bull, engulf_bear, doji, vol_surge, oi_surge
    ], axis=1).astype(np.float32)                               # (T, 13)


# ------------ Average True Range (ATR) -------------
def compute_atr(df):
    h, l, c  = df['high'], df['low'], df['close']
    prev_c   = c.shift(1)
    tr = pd.concat([h-l, (h-prev_c).abs(), (l-prev_c).abs()], axis=1).max(axis=1)
    return tr.ewm(span=ATR_PERIOD, adjust=False).mean().to_numpy()


# ------------- Creating Lables -------------
def create_lables(closes, atr_arr):
    lables, returns = [], []
    for i in range(len(closes)):
        p0 = closes[i]
        atr = atr_arr[i]
        if p0 <= 0 or atr

def load_data(path: Path):
    for file in DATA_DIR.glob('*_ohlcv.json'):
        with open(file, 'r') as f:
            data = json.load(f)

        df = pd.DataFrame(data)
