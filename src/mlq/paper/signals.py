"""Signal generation for paper trading: same features, model and selection rules as the backtest.

Selection = top-k by predicted relative 5-day return among F&O names in the point-in-time universe,
with hysteresis (a current holding is kept while its rank is within `keep`). Equal weights.
"""
import json
from pathlib import Path

import numpy as np
from lightgbm import Booster

from mlq.config import DATA
from mlq.ml_swing import HOLD, build, make_model

MODEL_DIR = DATA / "paper"


def select_names(score: dict, held: set, k: int, keep: int) -> list:
    """score: sym -> predicted value (NaN ignored). Returns <= k symbols (hysteresis)."""
    ranked = [s for s, v in sorted(score.items(), key=lambda kv: -kv[1]) if np.isfinite(v)]
    if len(ranked) < k:
        return []
    rank = {s: i for i, s in enumerate(ranked)}
    chosen = sorted((s for s in held if rank.get(s, 10**9) < keep), key=lambda s: rank[s])[:k]
    chosen += [s for s in ranked if s not in chosen][:k - len(chosen)]
    return chosen


def weights_for(names: list) -> dict:
    return {s: 1.0 / len(names) for s in names} if names else {}


def features(ctx, p):
    """Feature cube X (T,N,F), label yd, universe mask M for the full history (+ latest day)."""
    f, yd, M, _ = build(ctx, p)
    names = list(f)
    return np.stack([f[k] for k in names], axis=-1), yd, M, names


def fit_from(X, yd, M, embargo: int = 5):
    """Fit on every day whose label (open[t+1] -> open[t+1+HOLD]) is already known, minus embargo.
    Returns (model, index of the last training day)."""
    last = X.shape[0] - 2 - HOLD - embargo
    m = M[:last + 1] & ~np.isnan(yd[:last + 1])
    Xtr, ytr = X[:last + 1][m], yd[:last + 1][m]
    ok = ~np.isnan(Xtr).all(axis=1)
    mdl = make_model()
    mdl.fit(Xtr[ok], ytr[ok])
    return mdl, last


def save_model(mdl, names, trained_through, tag: str) -> Path:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    f = MODEL_DIR / f"model_{tag}.txt"
    mdl.booster_.save_model(str(f))
    (MODEL_DIR / f"model_{tag}.json").write_text(
        json.dumps({"features": names, "trained_through": str(trained_through), "hold": HOLD}))
    return f


def load_model(tag: str):
    f = MODEL_DIR / f"model_{tag}.txt"
    meta = json.loads((MODEL_DIR / f"model_{tag}.json").read_text())
    return Booster(model_file=str(f)), meta


def latest_model_tag() -> str | None:
    fs = sorted(MODEL_DIR.glob("model_*.json"))
    return fs[-1].stem.replace("model_", "") if fs else None


def score_day(booster, X_t: np.ndarray, M_t: np.ndarray, fno: np.ndarray, syms: list) -> dict:
    """Predict for every tradable (universe & F&O) symbol with at least one feature."""
    sel = M_t & fno & ~np.isnan(X_t).all(axis=1)
    if not sel.any():
        return {}
    pred = booster.predict(X_t[sel])
    return dict(zip([s for s, k in zip(syms, sel) if k], pred.tolist()))
