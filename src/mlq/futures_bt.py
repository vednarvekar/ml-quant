"""Lot-aware stock-futures backtest with real capital, margin and monthly roll cost.

Differences from the simple engine: positions are whole LOTS (today's lot sizes), P&L is the futures
mark-to-market on lots*lot_size*price move, equity = capital + cumulative P&L - costs, notional is
capped by exposure (x equity) and by margin (MARGIN_PCT of notional must fit in equity).
Signals decided at close t trade at open t+1 with slippage; on the first trading day of each month
all held notional pays a full round trip (roll to next expiry).

Approximations: today's lot sizes for all history, adjusted prices for sizing, flat margin %,
no futures basis/carry (assumes cash earns what carry costs).
"""
from dataclasses import dataclass

import numpy as np
import polars as pl

from mlq.costs import DEFAULT, Rates, leg_cost
from mlq.futures import MARGIN_PCT


@dataclass
class FutResult:
    equity: np.ndarray
    fees: float
    slippage: float
    avg_positions: float
    avg_exposure: float
    trades: int

    def metrics(self) -> dict:
        e = self.equity
        r = e[1:] / e[:-1] - 1
        yrs = len(r) / 252
        sd = r.std(ddof=1)
        return {"cagr": (e[-1] / e[0]) ** (1 / yrs) - 1, "sharpe": float(r.mean() / sd * np.sqrt(252)) if sd > 0 else 0.0,
                "max_dd": float((e / np.maximum.accumulate(e) - 1).min()), "fees": self.fees,
                "slippage": self.slippage, "avg_positions": self.avg_positions,
                "avg_exposure": self.avg_exposure, "trades": self.trades}


def simulate_futures(open_, close, W, dates, lot, capital=2_000_000.0, exposure=1.0,
                     margin_pct=MARGIN_PCT, slippage_bps=5.0, roll=True,
                     rates: Rates = DEFAULT) -> FutResult:
    """W: T x N target weights (fractions of gross exposure; sum |w| <= 1, long and/or short)."""
    T, N = close.shape
    s = slippage_bps / 1e4
    can_trade = ~np.isnan(lot)              # only F&O names have a lot size
    lot = np.nan_to_num(lot, nan=0.0)       # keep arithmetic NaN-free (0 lot => 0 exposure)
    lots = np.zeros(N)                      # signed lots held
    eq = np.empty(T)
    eq[0] = capital
    equity = capital
    last = np.full(N, np.nan)
    fees = slip = 0.0
    trades = 0
    npos, expo = [], []
    for t in range(T - 1):
        o, c = open_[t + 1], close[t + 1]
        tradable = ~np.isnan(o) & (o > 0) & can_trade
        prev = np.where(np.isnan(last), o, last)
        held = lots != 0
        # gap P&L from last close to today's open on positions carried over
        gap = np.where(held & ~np.isnan(o) & ~np.isnan(prev), lots * lot * (o - prev), 0.0)
        equity += np.nansum(gap)
        # monthly roll: pay a round trip on carried notional
        if roll and dates[t + 1].month != dates[t].month and held.any():
            nv = np.abs(lots) * lot * np.where(tradable, o, 0.0)
            for j in np.nonzero(held & tradable)[0]:
                cst = leg_cost(nv[j], "buy", "futures", rates) + leg_cost(nv[j], "sell", "futures", rates)
                equity -= cst + 2 * nv[j] * s
                fees += cst
                slip += 2 * nv[j] * s
        # target lots
        w = np.nan_to_num(W[t], nan=0.0)
        tgt = lots.copy()
        notional_budget = max(equity, 0.0) * exposure
        for j in np.nonzero(tradable)[0]:
            want = w[j] * notional_budget
            n = np.sign(want) * np.floor(abs(want) / (lot[j] * o[j]) + 0.5)
            tgt[j] = n
        mgn = (np.abs(tgt) * lot * np.where(tradable, o, np.where(np.isnan(prev), 0, prev))).sum() * margin_pct
        if mgn > equity > 0:                                    # margin cap: scale down proportionally
            tgt = np.sign(tgt) * np.floor(np.abs(tgt) * equity / mgn)
        delta = np.where(tradable, tgt - lots, 0.0)
        for j in np.nonzero(delta)[0]:
            nv = abs(delta[j]) * lot[j] * o[j]
            side = "buy" if delta[j] > 0 else "sell"
            cst = leg_cost(nv, side, "futures", rates)
            equity -= cst + nv * s
            fees += cst
            slip += nv * s
            trades += 1
        lots = lots + delta
        # intraday part of today's move from open to close
        day = np.where((lots != 0) & ~np.isnan(c) & ~np.isnan(o), lots * lot * (c - o), 0.0)
        equity += np.nansum(day)
        last = np.where(np.isnan(c), last, c)
        eq[t + 1] = equity
        npos.append(int((lots != 0).sum()))
        expo.append(float((np.abs(lots) * lot * np.where(np.isnan(c), 0, c)).sum() / max(equity, 1)))
    return FutResult(eq, fees, slip, float(np.mean(npos)), float(np.mean(expo)), trades)
