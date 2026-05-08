# prepare_data.py

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

# We predict the move 10 future 5-minute candles ahead.
HORIZON_5M = 10

# Each sample keeps 60 candles from each timeframe.
WINDOW_1M = 60
WINDOW_5M = 60
WINDOW_1H = 60
WINDOW_15M = 60
WINDOW_1D = 60

# Current labeling rule.
BUY_THRESHOLD = 0.007
SELL_THRESHOLD = -0.007

COLUMNS = ["open", "high", "low", "close", "volume", "oi"]


def parse_args():
    parser = argparse.ArgumentParser(description="Build processed CNN datasets from raw OHLCV files.")
    parser.add_argument("--stocks", nargs="+", default=["all"])
    return parser.parse_args()


def list_stocks():
    stocks = set()
    for timeframe in ["1min", "5min", "15min", "1hr", "1d"]:
        for path in (RAW_DIR / timeframe).glob("*_ohlcv.json"):
            stocks.add(path.name.replace("_ohlcv.json", ""))
    return sorted(stocks)


def get_stocks_to_process(requested):
    all_stocks = list_stocks()
    if requested == ["all"]:
        return all_stocks

    for stock in requested:
        if stock not in all_stocks:
            raise ValueError(f"Unknown stock: {stock}. Available: {all_stocks}")
    return requested


def load_frame(path):
    with open(path, "r") as f:
        rows = json.load(f)

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError(f"No data found in {path}")

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.drop_duplicates("timestamp").sort_values("timestamp").set_index("timestamp")
    return df[COLUMNS]


def normalize_window(values):
    # Local min-max scaling keeps the CNN focused on pattern shape, not price level.
    return (values - values.min(axis=0)) / (values.max(axis=0) - values.min(axis=0) + 1e-7)


def make_label(price_now, price_future):
    move = (price_future - price_now) / price_now
    if move > BUY_THRESHOLD:
        return 1
    if move < SELL_THRESHOLD:
        return 2
    return 0


def process_stock(stock):
    print(f"\nProcessing {stock}...")

    path_1m = RAW_DIR / "1min" / f"{stock}_ohlcv.json"
    path_5m = RAW_DIR / "5min" / f"{stock}_ohlcv.json"
    path_15m = RAW_DIR / "15min" / f"{stock}_ohlcv.json"
    path_1h = RAW_DIR / "1hr" / f"{stock}_ohlcv.json"
    path_1d = RAW_DIR / "1d" / f"{stock}_ohlcv.json"

    if not path_1m.exists() or not path_5m.exists() or not path_15m.exists() or not path_1h.exists() or not path_1d.exists():
        raise FileNotFoundError(f"Missing one or more raw files for {stock}")

    df_1m = load_frame(path_1m)
    df_5m = load_frame(path_5m)
    df_15m = load_frame(path_15m)
    df_1h = load_frame(path_1h)
    df_1d = load_frame(path_1d)

    x1_list = []
    x5_list = []
    x15_list = []
    xh_list = []
    xd_list = []
    y_list = []
    timestamp_list = []

    # Start late enough that 60 hourly candles already exist.
    start_index = 720

    for i in range(start_index, len(df_5m) - HORIZON_5M):
        anchor_time = df_5m.index[i]

        # All three windows end at the same 5-minute anchor time.
        window_1m = df_1m.loc[:anchor_time].tail(WINDOW_1M).values
        window_15m = df_15m.loc[:anchor_time].tail(WINDOW_15M).values
        window_5m = df_5m.loc[:anchor_time].tail(WINDOW_5M).values
        window_1h = df_1h.loc[:anchor_time].tail(WINDOW_1H).values
        window_1d = df_1d.loc[:anchor_time].tail(WINDOW_1D).values

        # Skip samples that do not have full history in all timeframes.
        if (
            len(window_1m) < WINDOW_1M or 
            len(window_5m) < WINDOW_5M or 
            len(window_15m) < WINDOW_15M or 
            len(window_1h) < WINDOW_1H or 
            len(window_1d) < WINDOW_1D
        ):
            continue

        price_now = float(df_5m.iloc[i]["close"])
        price_future = float(df_5m.iloc[i + HORIZON_5M]["close"])
        label = make_label(price_now, price_future)

        x1_list.append(normalize_window(window_1m))
        x5_list.append(normalize_window(window_5m))
        x15_list.append(normalize_window(window_15m))
        xh_list.append(normalize_window(window_1h))
        xd_list.append(normalize_window(window_1d))
        y_list.append(label)
        timestamp_list.append(anchor_time.value)

        if len(y_list) % 10000 == 0:
            print(f"{stock}: {len(y_list)} samples")

    x1 = np.asarray(x1_list, dtype=np.float32)
    x5 = np.asarray(x5_list, dtype=np.float32)
    x15 = np.asarray(x15_list, dtype=np.float32)
    xh = np.asarray(xh_list, dtype=np.float32)
    xd = np.asarray(xd_list, dtype=np.float32)
    y = np.asarray(y_list, dtype=np.int64)
    timestamps = np.asarray(timestamp_list, dtype=np.int64)

    output_dir = PROCESSED_DIR / stock
    output_dir.mkdir(parents=True, exist_ok=True)

    np.save(output_dir / "X_1min.npy", x1)
    np.save(output_dir / "X_5min.npy", x5)
    np.save(output_dir / "X_15min.npy", x15)
    np.save(output_dir / "X_1hr.npy", xh)
    np.save(output_dir / "X_1d.npy", xd)
    np.save(output_dir / "y_labels.npy", y)
    np.save(output_dir / "anchor_timestamps.npy", timestamps)

    counts = {label: int((y == label).sum()) for label in [0, 1, 2]}
    print(f"Saved {stock} -> samples={len(y)} labels={counts}")


def main():
    args = parse_args()
    stocks = get_stocks_to_process(args.stocks)

    for stock in stocks:
        process_stock(stock)

    print("\nDone.")


if __name__ == "__main__":
    main()
