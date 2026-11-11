"""The fixed measuring stick (audit step 1). Every strategy is reported the same way:

- averaged over ALL rebalance phases (same rule started on different days), mean / min / max Sharpe
- against fixed baselines built with the SAME construction and engine:
    * model-universe equal-weight (no model)
    * random scores (several seeds)  <- what plain beta + costs give you for free
    * 5-day reversal rule            <- the simple thing the ML mostly learned
    * index buy&hold: Nifty 50, Midcap 100 (daily, no costs)
- a factor regression of daily returns on Nifty 50, (Midcap-Nifty) and (Smallcap-Midcap):
  annual alpha and its t-stat. Alpha t < 2 means "not distinguishable from market/size exposure".

uv run python -m mlq.evaluate
"""
from dataclasses import dataclass

import numpy as np
import polars as pl

from mlq.config import DATA

CLEAN_START = "2022-01-03"


def idx_returns(dates: list) -> dict:
    out = {}
    for key, f in (("n50", "Nifty_50"), ("mid", "NIFTY_MIDCAP_100"), ("small", "NIFTY_SMLCAP_100"),
                   ("n500", "Nifty_500")):
        p = DATA / "clean" / "idx" / "1d" / f"{f}.parquet"
        if not p.exists():
            continue
        d = pl.read_parquet(p).select(["date", "close"])
        m = dict(zip(d["date"].to_list(), d["close"].to_list()))
        c = np.array([m.get(x, np.nan) for x in dates], float)
        r = np.full_like(c, np.nan)
        r[1:] = c[1:] / c[:-1] - 1
        out[key] = r
    return out


def stats(r: np.ndarray) -> dict:
    r = r[np.isfinite(r)]
    sd = r.std(ddof=1)
    eq = np.cumprod(1 + r)
    yrs = len(r) / 252
    return {"cagr": eq[-1] ** (1 / yrs) - 1, "sharpe": float(r.mean() / sd * np.sqrt(252)) if sd > 0 else 0.0,
            "max_dd": float((eq / np.maximum.accumulate(eq) - 1).min())}


def factor_alpha(r: np.ndarray, fac: dict) -> tuple[float, float, float]:
    """OLS of daily strategy returns on [n50, mid-n50, small-mid]. Returns (alpha per year, t, R^2)."""
    ok = np.isfinite(r) & np.isfinite(fac["n50"]) & np.isfinite(fac["mid"]) & np.isfinite(fac["small"])
    y = r[ok]
    X = np.column_stack([np.ones(ok.sum()), fac["n50"][ok], (fac["mid"] - fac["n50"])[ok],
                         (fac["small"] - fac["mid"])[ok]])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ beta
    s2 = res @ res / (len(y) - X.shape[1])
    cov = s2 * np.linalg.inv(X.T @ X)
    r2 = 1 - (res @ res) / ((y - y.mean()) @ (y - y.mean()))
    return float(beta[0] * 252), float(beta[0] / np.sqrt(cov[0, 0])), float(r2)


@dataclass
class Row:
    name: str
    sharpes: list
    cagrs: list
    dds: list
    alpha: list
    alpha_t: list

    def summary(self) -> dict:
        return {"strategy": self.name, "sharpe_mean": round(float(np.mean(self.sharpes)), 2),
                "sharpe_min": round(float(np.min(self.sharpes)), 2),
                "sharpe_max": round(float(np.max(self.sharpes)), 2),
                "cagr_mean": round(float(np.mean(self.cagrs)), 3),
                "maxdd_worst": round(float(np.min(self.dds)), 3),
                "alpha_yr": round(float(np.mean(self.alpha)), 3),
                "alpha_t_mean": round(float(np.mean(self.alpha_t)), 2),
                "alpha_t_max": round(float(np.max(self.alpha_t)), 2)}


def run_phases(name: str, runner, phases: list[int], fac: dict) -> Row:
    """runner(phase) -> daily return array aligned to the evaluation window."""
    row = Row(name, [], [], [], [], [])
    for p in phases:
        r = runner(p)
        s = stats(r)
        a, t, _ = factor_alpha(r, fac)
        row.sharpes.append(s["sharpe"])
        row.cagrs.append(s["cagr"])
        row.dds.append(s["max_dd"])
        row.alpha.append(a)
        row.alpha_t.append(t)
    return row


def index_rows(dates: list, i0: int) -> list[dict]:
    rows = []
    fac = idx_returns(dates)
    for key, nm in (("n50", "Nifty 50 (price index)"), ("mid", "Midcap 100 index"),
                    ("small", "Smallcap 100 index"), ("n500", "Nifty 500 index")):
        if key in fac:
            s = stats(fac[key][i0:])
            rows.append({"strategy": nm + " buy&hold", "sharpe_mean": round(s["sharpe"], 2),
                         "cagr_mean": round(s["cagr"], 3), "maxdd_worst": round(s["max_dd"], 3)})
    return rows
