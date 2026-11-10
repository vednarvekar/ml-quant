"""5-minute intraday engine (MIS: squared off same day, long and short, India costs).

Arrays are shaped (D days, B=75 bars, N symbols). Signal arrays P (+1/0/-1) and S (priority score)
are decided at the CLOSE of bar b and acted on at the OPEN of bar b+1 (never the same bar).
Rules: flat at the start and end of every day; no signal from bar SQ-1 on, so everything is
closed at the open of bar SQ (15:15); at most K concurrent positions (highest |score| wins free
slots); each position is a fixed notional V = capital * leverage / K; every leg pays slippage and
the intraday fee model from mlq.costs.
"""
from dataclasses import dataclass

import numpy as np
import polars as pl

from mlq.config import DATA
from mlq.costs import DEFAULT, Rates, leg_cost

B = 75
SQ = 72          # bar index of 15:15; flat from this bar's open


@dataclass
class IntradayResult:
    daily: np.ndarray        # net rupee P&L per day
    gross: float
    fees: float
    slippage: float
    trades: int              # round trips (entries)
    capital: float
    notional: float          # rupees per position

    def returns(self) -> np.ndarray:
        return self.daily / self.capital

    def metrics(self) -> dict:
        r = self.returns()
        sd = r.std(ddof=1)
        eq = np.cumprod(1 + r)
        return {"net_pnl": float(self.daily.sum()), "gross_pnl": self.gross, "fees": self.fees,
                "slippage": self.slippage, "trades": self.trades,
                "net_bps_per_trade": (self.daily.sum() / max(self.trades, 1)) / (self.notional / 1e4),
                "sharpe": float(r.mean() / sd * np.sqrt(252)) if sd > 0 else 0.0,
                "max_dd": float((eq / np.maximum.accumulate(eq) - 1).min()),
                "ret_per_year": float(r.mean() * 252)}


def simulate(O, P, S=None, capital: float = 200_000.0, leverage: float = 5.0, K: int = 5,
             slippage_bps: float = 5.0, rates: Rates = DEFAULT, sq: int = SQ) -> IntradayResult:
    D, Bn, N = O.shape
    V = capital * leverage / K
    s = slippage_bps / 1e4
    fb, fs = leg_cost(V, "buy", "intraday", rates), leg_cost(V, "sell", "intraday", rates)
    S = np.abs(P).astype(float) if S is None else np.abs(S)
    daily = np.zeros(D)
    gross = fees = slip = 0.0
    trades = 0
    for d in range(D):
        Q = np.zeros(N)
        o = O[d]
        for b in range(1, Bn):
            want = np.nan_to_num(P[d, b - 1], nan=0.0)
            if b >= sq:
                want = np.zeros(N)
            ok = ~np.isnan(o[b]) & (o[b] > 0)
            ex = (Q != 0) & (want != Q) & ok                     # close: signal gone or flipped
            f = (ex & (Q < 0)).sum() * fb + (ex & (Q > 0)).sum() * fs
            sl = ex.sum() * V * s
            Q[ex] = 0.0
            if b < sq:
                free = K - int((Q != 0).sum())
                cand = np.nonzero((want != 0) & (Q == 0) & ok)[0]
                if free > 0 and cand.size:
                    cand = cand[np.argsort(-S[d, b - 1, cand])][:free]
                    Q[cand] = want[cand]
                    f += (want[cand] > 0).sum() * fb + (want[cand] < 0).sum() * fs
                    sl += cand.size * V * s
                    trades += cand.size
            fees += f
            slip += sl
            daily[d] -= f + sl
            if b < Bn - 1:
                pos = np.nonzero(Q != 0)[0]
                if pos.size:
                    ratio = np.nan_to_num(O[d, b + 1, pos] / o[b, pos] - 1.0, nan=0.0)
                    g = float((Q[pos] * ratio).sum() * V)
                    daily[d] += g
                    gross += g
    return IntradayResult(daily, gross, fees, slip, trades, capital, V)


def load_panel(syms: list[str], start: str = "2022-01-03"):
    """Load adj_5m for syms -> dates, O,H,L,C,Vol arrays (D,75,N) float32, NaN where missing.
    Special sessions are dropped."""
    frames = []
    for s in syms:
        f = DATA / "clean" / "adj_5m" / f"{s}.parquet"
        frames.append(pl.read_parquet(f).filter(~pl.col("special_session")
                      & (pl.col("date") >= pl.lit(start).str.to_date())))
    allf = pl.concat(frames)
    dates = sorted(allf["date"].unique().to_list())
    di = {d: i for i, d in enumerate(dates)}
    shp = (len(dates), B, len(syms))
    arrs = {k: np.full(shp, np.nan, np.float32) for k in ("o", "h", "l", "c", "v")}
    for j, s in enumerate(syms):
        g = frames[j]
        g = g.with_columns(((pl.col("ts").dt.hour().cast(pl.Int32) * 60 + pl.col("ts").dt.minute().cast(pl.Int32) - 555) // 5)
                           .alias("bar")).filter((pl.col("bar") >= 0) & (pl.col("bar") < B))
        d_i = np.array([di[x] for x in g["date"].to_list()])
        b_i = g["bar"].to_numpy().astype(int)
        for k, col in (("o", "open"), ("h", "high"), ("l", "low"), ("c", "close"), ("v", "volume")):
            arrs[k][d_i, b_i, j] = g[col].to_numpy().astype(np.float32)
    return dates, arrs
