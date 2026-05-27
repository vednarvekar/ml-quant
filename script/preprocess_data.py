import numpy as np
import pandas as pd
from pathlib import Path
import json

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / 'data' / 'raw' / '5M'

def list_stocks():
    stocks = set()
    for file in DATA_DIR.glob('*_ohlcv.json'):
        stocks.add(file.stem.replace('_ohlcv', ''))
    return sorted(stocks)

def load_data(path: Path):
    with open(path) as f:
        rows = json.load(f)

    df = pd.DataFrame(rows)