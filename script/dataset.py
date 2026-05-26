import numpy as np
from pathlib import Path
import gc

# ── paths ────────────────────────────────────────────────────────────────────
BASE_DIR      = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
MASTER_OUT_DIR = PROCESSED_DIR / "master_splits"

def merge_and_split_dataset(train_ratio=0.7, val_ratio=0.15):
    print("Phase 1: Calculating total dataset shapes across all stocks...")
    
    total_samples = 0
    stock_dirs = []
    
    for stock_dir in PROCESSED_DIR.iterdir():
        if stock_dir.is_dir() and stock_dir.name != "master_splits":
            labels_path = stock_dir / "y_labels.npy"
            if labels_path.exists():
                y_meta = np.load(labels_path, mmap_mode='r')
                total_samples += y_meta.shape[0]
                stock_dirs.append((stock_dir, y_meta.shape[0]))

    if total_samples == 0:
        print("Error: No processed stock data found.")
        return

    print(f"Total combined sample size across all stocks: {total_samples:,}")
    MASTER_OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Phase 2: Initialize Consolidated Global Arrays on Disk ───────────────
    print("\nPhase 2: Initializing continuous temporary master files on disk...")
    
    # Allocating ONE massive continuous file per block instead of splitting early
    temp_masters = {
        "X_3min": np.memmap(MASTER_OUT_DIR / "total_X_3min.tmp", dtype='float32', mode='w+', shape=(total_samples, 60, 15)),
        "X_5min": np.memmap(MASTER_OUT_DIR / "total_X_5min.tmp", dtype='float32', mode='w+', shape=(total_samples, 60, 15)),
        "X_1hr":  np.memmap(MASTER_OUT_DIR / "total_X_1hr.tmp",  dtype='float32', mode='w+', shape=(total_samples, 60, 15)),
        "y_labels": np.memmap(MASTER_OUT_DIR / "total_y_labels.tmp", dtype='int64', mode='w+', shape=(total_samples,))
    }

    # ── Phase 3: Straight Monolithic Appending ───────────────────────────────
    print("\nPhase 3: Streaming stock data seamlessly to global matrix...")
    
    global_ptr = 0
    for stock_dir, n_samples in stock_dirs:
        print(f"-> Appending arrays from: {stock_dir.name} ({n_samples:,} samples)")
        
        x3 = np.load(stock_dir / "X_3min.npy")
        x5 = np.load(stock_dir / "X_5min.npy")
        xh = np.load(stock_dir / "X_1hr.npy")
        y  = np.load(stock_dir / "y_labels.npy")

        # Drop the entire stock into the current global tracking window
        temp_masters["X_3min"][global_ptr : global_ptr + n_samples] = x3
        temp_masters["X_5min"][global_ptr : global_ptr + n_samples] = x5
        temp_masters["X_1hr"][global_ptr : global_ptr + n_samples]  = xh
        temp_masters["y_labels"][global_ptr : global_ptr + n_samples] = y

        global_ptr += n_samples

        # Flush data changes out and clear memory references instantly
        for key in temp_masters:
            temp_masters[key].flush()
            
        del x3, x5, xh, y
        gc.collect()

    # ── Phase 4: Clean Slice & Partition ─────────────────────────────────────
    print("\nPhase 4: Slicing continuous matrix into Train/Val/Test files...")
    
    train_size = int(total_samples * train_ratio)
    val_size = int(total_samples * val_ratio)
    
    splits = {
        "TRAIN": slice(0, train_size),
        "VAL": slice(train_size, train_size + val_size),
        "TEST": slice(train_size + val_size, None)
    }

    for prefix, slc in splits.items():
        print(f"   Generating clean {prefix} split (.npy files)...")
        # Load precise structural slices from the unified memmap block and save directly
        np.save(MASTER_OUT_DIR / f"{prefix}_X_3min.npy", temp_masters["X_3min"][slc])
        np.save(MASTER_OUT_DIR / f"{prefix}_X_5min.npy", temp_masters["X_5min"][slc])
        np.save(MASTER_OUT_DIR / f"{prefix}_X_1hr.npy",  temp_masters["X_1hr"][slc])
        np.save(MASTER_OUT_DIR / f"{prefix}_y_labels.npy", temp_masters["y_labels"][slc])

    # Unlink active memory references to allow raw file access
    for key in list(temp_masters.keys()):
        del temp_masters[key]
    gc.collect()

    # Clean up the large temp files from your hard drive
    print("\nCleaning up temporary storage files...")
    for name in ["X_3min", "X_5min", "X_1hr", "y_labels"]:
        tmp_file = MASTER_OUT_DIR / f"total_{name}.tmp"
        if tmp_file.exists():
            tmp_file.unlink()

    print("\n✓ Master datasets split and compiled successfully without alignment errors!")

if __name__ == "__main__":
    merge_and_split_dataset()