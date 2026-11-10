"""Experiment registry: every strategy variant we try is logged, so Deflated Sharpe knows the
true number of trials (hiding failed tries is the #1 way backtests lie).

Append-only JSONL at data/experiments.jsonl. One line per trial.
"""
import json
import time
from pathlib import Path

import numpy as np

from mlq.config import DATA
from mlq.validation import deflated_sharpe, sharpe

FILE = DATA / "experiments.jsonl"


def log_trial(name: str, family: str, params: dict, returns: np.ndarray, extra: dict | None = None,
              file: Path = FILE) -> dict:
    """returns = per-period (daily) net strategy returns of this trial."""
    r = np.asarray(returns, float)
    rec = {"ts": time.strftime("%Y-%m-%d %H:%M:%S"), "name": name, "family": family,
           "params": params, "n": int(len(r)), "sr_period": sharpe(r) if len(r) > 2 else 0.0,
           "sr_annual": sharpe(r) * np.sqrt(252) if len(r) > 2 else 0.0,
           "mean": float(r.mean()) if len(r) else 0.0, **(extra or {})}
    file.parent.mkdir(parents=True, exist_ok=True)
    with open(file, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")
    return rec


def load(file: Path = FILE) -> list[dict]:
    if not file.exists():
        return []
    return [json.loads(x) for x in file.read_text(encoding="utf-8").splitlines() if x.strip()]


def n_trials(family: str | None = None, file: Path = FILE) -> int:
    t = load(file)
    return len([x for x in t if family is None or x["family"] == family])


def dsr(returns: np.ndarray, family: str | None = None, file: Path = FILE) -> float:
    """Deflated Sharpe of `returns` against ALL logged trials (or one family)."""
    srs = np.array([x["sr_period"] for x in load(file) if family is None or x["family"] == family])
    if len(srs) < 2:
        return float("nan")
    return deflated_sharpe(np.asarray(returns, float), srs)
