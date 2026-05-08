# merge_data.py

from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
MASTER_DIR = BASE_DIR / "data" / "training_data"
MASTER_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15


def list_stocks():
    stocks = []
    for path in sorted(PROCESSED_DIR.iterdir()):
        if not path.is_dir():
            continue

        needed = [
            path / "X_1min.npy",
            path / "X_5min.npy",
            path / "X_15min.npy",
            path / "X_1hr.npy",
            path / "X_1d.npy",
            path / "y_labels.npy",
            path / "anchor_timestamps.npy",
        ]
        if all(file.exists() for file in needed):
            stocks.append(path.name)

    return stocks


def save_split(prefix, x1_parts, x5_parts, x15_parts, xh_parts, xd_parts, y_parts, time_parts):
    # We keep time order inside each stock split. There is no random shuffle here.
    x1 = np.concatenate(x1_parts, axis=0).astype(np.float32)
    x5 = np.concatenate(x5_parts, axis=0).astype(np.float32)
    x15 = np.concatenate(x15_parts, axis=0).astype(np.float32)
    xh = np.concatenate(xh_parts, axis=0).astype(np.float32)
    xd = np.concatenate(xd_parts, axis=0).astype(np.float32)
    y = np.concatenate(y_parts, axis=0).astype(np.int64)
    timestamps = np.concatenate(time_parts, axis=0).astype(np.int64)

    if prefix == "TRAIN":
        print(f"Shuffling {prefix} set for better learning...")
        indices = np.random.permutation(len(y))
        x1, x5, x15, xh, xd, y, timestamps = x1[indices], x5[indices], x15[indices], xh[indices], xd[indices], y[indices], timestamps[indices]

    np.save(MASTER_DIR / f"{prefix}_X1.npy", x1)
    np.save(MASTER_DIR / f"{prefix}_X5.npy", x5)
    np.save(MASTER_DIR / f"{prefix}_X15.npy", x15)
    np.save(MASTER_DIR / f"{prefix}_XH.npy", xh)
    np.save(MASTER_DIR / f"{prefix}_XD.npy", xd)
    np.save(MASTER_DIR / f"{prefix}_y.npy", y)
    np.save(MASTER_DIR / f"{prefix}_timestamps.npy", timestamps)

    counts = {label: int((y == label).sum()) for label in [0, 1, 2]}
    print(f"{prefix}: samples={len(y)} labels={counts}")


def main():
    stocks = list_stocks()
    if not stocks:
        raise FileNotFoundError("No processed stock data found. Run prepare_data.py first.")

    print(f"Stocks to merge: {stocks}")

    train_x1, train_x5, train_x15, train_xh, train_xd, train_y, train_t = [], [], [], [], [], [], []
    val_x1, val_x5, val_x15, val_xh, val_xd, val_y, val_t = [], [], [], [], [], [], []
    test_x1, test_x5, test_x15, test_xh, test_xd, test_y, test_t = [], [], [], [], [], [], []

    for stock in stocks:
        print(f"Merging {stock}...")
        stock_dir = PROCESSED_DIR / stock

        x1 = np.load(stock_dir / "X_1min.npy")
        x5 = np.load(stock_dir / "X_5min.npy")
        x15 = np.load(stock_dir / "X_15min.npy")
        xh = np.load(stock_dir / "X_1hr.npy")
        xd = np.load(stock_dir / "X_1d.npy")
        y = np.load(stock_dir / "y_labels.npy")
        timestamps = np.load(stock_dir / "anchor_timestamps.npy")

        # Time-safe split: early data -> train, middle -> val, latest -> test.
        train_end = int(len(y) * TRAIN_RATIO)
        val_end = train_end + int(len(y) * VAL_RATIO)

        train_x1.append(x1[:train_end])
        train_x5.append(x5[:train_end])
        train_x15.append(x15[:train_end])
        train_xh.append(xh[:train_end])
        train_xd.append(xd[:train_end])
        train_y.append(y[:train_end])
        train_t.append(timestamps[:train_end])

        val_x1.append(x1[train_end:val_end])
        val_x5.append(x5[train_end:val_end])
        val_x15.append(x15[train_end:val_end])
        val_xh.append(xh[train_end:val_end])
        val_xd.append(xd[train_end:val_end])
        val_y.append(y[train_end:val_end])
        val_t.append(timestamps[train_end:val_end])

        test_x1.append(x1[val_end:])
        test_x5.append(x5[val_end:])
        test_x15.append(x15[val_end:])
        test_xh.append(xh[val_end:])
        test_xd.append(xd[val_end:])
        test_y.append(y[val_end:])
        test_t.append(timestamps[val_end:])

    save_split("TRAIN", train_x1, train_x5, train_x15, train_xh, train_xd, train_y, train_t)
    save_split("VAL", val_x1, val_x5, val_x15, val_xh, val_xd, val_y, val_t)
    save_split("TEST", test_x1, test_x5, test_x15, test_xh, test_xd, test_y, test_t)

    print("Done.")


if __name__ == "__main__":
    main()
