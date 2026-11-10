"""Daily-bar backtest engine (swing). Honest by construction:

- A target weight row dated t may only use information up to the CLOSE of day t.
- It is executed at the OPEN of day t+1 (never at the close that produced it).
- Every fill pays slippage (bps against you) and the full India fee model (mlq.costs).
- Positions are marked to market at each close. Delivery kind => long only.

Weights are fractions of current equity (sum of |w| <= max_gross). Rebalance happens only when the
target share count changes by >= min_trade_value rupees (avoids tiny churn).

Intraday strategies need a separate 5-minute engine (later); this one is for daily/swing.
"""
from dataclasses import dataclass

import numpy as np
import polars as pl

from mlq.costs import DEFAULT, Rates, leg_cost


@dataclass
class Result:
    equity: pl.DataFrame      # date, equity
    trades: int
    fees: float
    slippage: float
    turnover: float           # total traded value / average equity / years

    def metrics(self) -> dict:
        e = self.equity["equity"].to_numpy()
        r = e[1:] / e[:-1] - 1
        yrs = len(r) / 252
        sd = r.std(ddof=1) if len(r) > 2 else np.nan
        peak = np.maximum.accumulate(e)
        return {"total_return": e[-1] / e[0] - 1,
                "cagr": (e[-1] / e[0]) ** (1 / yrs) - 1 if yrs > 0 else np.nan,
                "sharpe": (r.mean() / sd * np.sqrt(252)) if sd and sd > 0 else 0.0,
                "max_dd": float((e / peak - 1).min()),
                "trades": self.trades, "fees": self.fees, "slippage": self.slippage,
                "turnover_per_year": self.turnover}


def panel(prices: pl.DataFrame, col: str) -> tuple[list, list, np.ndarray]:
    """prices (sym,date,col...) -> dates, syms, T x N matrix (NaN where missing)."""
    w = prices.pivot(on="sym", index="date", values=col).sort("date")
    dates = w["date"].to_list()
    syms = [c for c in w.columns if c != "date"]
    return dates, syms, w.select(syms).to_numpy().astype(float)


def simulate(open_: np.ndarray, close: np.ndarray, weights: np.ndarray, dates: list,
             capital: float = 1_000_000.0, kind: str = "delivery", slippage_bps: float = 5.0,
             min_trade_value: float = 2_000.0, max_gross: float = 1.0,
             rates: Rates = DEFAULT) -> Result:
    T, N = close.shape
    if weights.shape != (T, N):
        raise ValueError("weights must be T x N aligned with prices")
    w = np.nan_to_num(weights, nan=0.0)
    if kind == "delivery" and (w < -1e-12).any():
        raise ValueError("delivery cannot short; use kind='futures' or intraday")
    gross = np.abs(w).sum(axis=1)
    if (gross > max_gross + 1e-9).any():
        raise ValueError(f"gross weight exceeds {max_gross}")
    s = slippage_bps / 1e4
    cash, shares = capital, np.zeros(N)
    last = np.full(N, np.nan)
    eq = np.empty(T)
    eq[0] = capital
    n_tr, fees, slip, traded = 0, 0.0, 0.0, 0.0
    for t in range(T - 1):
        o, c = open_[t + 1], close[t + 1]
        ref = np.where(np.isnan(last), 0.0, last)                 # prior close for mark
        px_open = np.where(np.isnan(o), ref, o)
        equity_open = cash + (shares * np.where(np.isnan(px_open), 0.0, px_open)).sum()
        tradable = ~np.isnan(o) & (o > 0)
        target = np.zeros(N)
        target[tradable] = np.floor(w[t, tradable] * equity_open / o[tradable])
        keep = ~tradable
        target[keep] = shares[keep]                               # can't trade: hold
        delta = target - shares
        small = np.abs(delta) * np.where(tradable, o, 0.0) < min_trade_value
        small &= ~((target == 0) & (shares != 0) & tradable)       # always allow full exit
        delta[small] = 0.0
        for j in np.nonzero(delta)[0]:
            qty = delta[j]
            side = "buy" if qty > 0 else "sell"
            px = o[j] * (1 + s) if qty > 0 else o[j] * (1 - s)
            val = abs(qty) * px
            fee = leg_cost(val, side, kind, rates)
            cash -= qty * px + fee
            slip += abs(qty) * o[j] * s
            fees += fee
            traded += val
            n_tr += 1
        shares = shares + delta
        mk = np.where(np.isnan(c), np.where(np.isnan(last), 0.0, last), c)
        last = np.where(np.isnan(c), last, c)
        eq[t + 1] = cash + (shares * mk).sum()
    yrs = max((T - 1) / 252, 1e-9)
    return Result(pl.DataFrame({"date": dates, "equity": eq}), n_tr, fees, slip,
                  traded / eq.mean() / yrs)
