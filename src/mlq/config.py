from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RAW = DATA / "raw" / "upstox"
SECRETS = ROOT / ".secrets"
TOKEN_FILE = SECRETS / "upstox_token.json"
INSTRUMENTS_FILE = DATA / "instruments_nse.parquet"
