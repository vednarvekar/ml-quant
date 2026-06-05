# import numpy as np
# from pathlib import Path

# PROCESSED_DIR = Path('data/processed/5M')

# X_train = np.memmap(PROCESSED_DIR / 'X_train.npy', dtype=np.float32, mode='r')
# X_val   = np.memmap(PROCESSED_DIR / 'X_val.npy',   dtype=np.float32, mode='r')

# # Check feature means — if train and val are from same distribution these should be similar
# print("Train feature means:", X_train.reshape(-1, 12).mean(axis=0).round(4))
# print("Val feature means:  ", X_val.reshape(-1, 12).mean(axis=0).round(4))

# # Check feature std
# print("Train feature stds:", X_train.reshape(-1, 12).std(axis=0).round(4))
# print("Val feature stds:  ", X_val.reshape(-1, 12).std(axis=0).round(4))

import numpy as np
import json
from pathlib import Path

DATA_DIR = Path('data/raw/5M')

for file in sorted(DATA_DIR.glob('*_ohlcv.json')):
    with open(file) as f:
        data = json.load(f)
    
    v = np.array([d['volume'] for d in data], dtype=float)
    prev_v = np.roll(v, 1)
    prev_v[0] = v[0]
    vol_ret = (v - prev_v) / (prev_v + 1e-9)
    
    # 1. Grab the true absolute maximum value BEFORE clipping it down
    true_max = np.abs(vol_ret).max()
    
    # 2. Now clip the matrix so your downstream data processing stays stable
    vol_ret = np.clip(vol_ret, -10, 10)         
    
    # 3. Check the true_max instead of the clipped matrix
    if true_max > 1000:
        stock = file.stem.replace('_ohlcv', '').upper()
        print(f"{stock}: max vol_ret = {true_max:.0f}, mean = {vol_ret.mean():.2f}")