"""Paper broker for stock futures. Same accounting as mlq.futures_bt, one day at a time, with state
persisted to disk so the strategy can run day after day. It never touches a real broker.

Order of events on each trading day d (identical to the backtest):
  1. gap P&L on carried positions (prev close -> today's open)
  2. monthly roll cost on carried notional (first trading day of a month)
  3. execute the pending decision (made at the previous close) at today's OPEN, slippage + fees
  4. intraday P&L (open -> close) on the positions held after trading
"""
import json
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import numpy as np

from mlq.costs import DEFAULT, Rates, leg_cost
from mlq.futures import MARGIN_BUFFER, MARGIN_PCT


@dataclass
class PaperConfig:
    capital: float = 2_000_000.0
    k: int = 10                  # names held
    keep: int = 20               # hysteresis: keep a holding while its rank is within this
    exposure: float = 1.0        # notional / equity
    rebalance_days: int = 5      # trading days between decisions
    margin_pct: float = MARGIN_PCT
    slippage_bps: float = 5.0
    universe: int = 150          # point-in-time liquidity universe size
    max_stale_days: int = 4      # no new orders if data is older than this (calendar days)
    retrain_days: int = 126      # retrain the model at least this often (trading days)
    mode: str = "long"           # "long" = long-only top-k; "ls" = long top-k / short bottom-k


@dataclass
class PaperBroker:
    path: Path
    cfg: PaperConfig
    equity: float = 0.0
    lots: dict = field(default_factory=dict)          # sym -> signed lots
    pending: dict | None = None                       # {"decided": "YYYY-MM-DD", "weights": {sym: w}}
    last_date: str | None = None                      # last trading day fully processed
    days_since: int = 10**6                           # trading days since last decision
    start_equity: float = 0.0
    start_date: str | None = None
    target: list = field(default_factory=list)       # names chosen at the last decision (for hysteresis)
    last_roll: str | None = None                      # expiry date already rolled (avoid double roll)
    target_short: list = field(default_factory=list)  # short names chosen at the last decision (ls mode)
    rates: Rates = DEFAULT

    @classmethod
    def load(cls, path: Path, cfg: PaperConfig) -> "PaperBroker":
        path = Path(path)
        if path.exists():
            s = json.loads(path.read_text(encoding="utf-8"))
            return cls(path=path, cfg=cfg, equity=s["equity"], lots=s["lots"], pending=s["pending"],
                       last_date=s["last_date"], days_since=s["days_since"],
                       start_equity=s["start_equity"], start_date=s["start_date"],
                       target=s.get("target", []), last_roll=s.get("last_roll"),
                       target_short=s.get("target_short", []))
        return cls(path=path, cfg=cfg, equity=cfg.capital, start_equity=cfg.capital)

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        s = {"equity": self.equity, "lots": self.lots, "pending": self.pending,
             "last_date": self.last_date, "days_since": self.days_since,
             "start_equity": self.start_equity, "start_date": self.start_date,
             "target": self.target, "last_roll": self.last_roll,
             "target_short": self.target_short}
        self.path.write_text(json.dumps(s, indent=1), encoding="utf-8")

    # ------------------------------------------------------------------
    def process_day(self, d: date, o: dict, c: dict, prev_c: dict, lot: dict, new_month: bool) -> dict:
        """o/c/prev_c: sym -> price (NaN/missing = no data). lot: sym -> lot size.
        Returns a summary dict with trades made today."""
        cfg, s = self.cfg, self.cfg.slippage_bps / 1e4
        ok = lambda x: x is not None and np.isfinite(x) and x > 0                      # noqa: E731
        trades, fees, slip = [], 0.0, 0.0
        eq = self.equity
        for sym, n in self.lots.items():                       # 1. gap P&L
            if n and ok(o.get(sym)) and ok(prev_c.get(sym)):
                eq += n * lot[sym] * (o[sym] - prev_c[sym])
        if new_month:                                          # 2. monthly roll round trip
            for sym, n in self.lots.items():
                if n and ok(o.get(sym)):
                    nv = abs(n) * lot[sym] * o[sym]
                    cst = leg_cost(nv, "buy", "futures", self.rates) + leg_cost(nv, "sell", "futures", self.rates)
                    eq -= cst + 2 * nv * s
                    fees += cst
                    slip += 2 * nv * s
        if self.pending is not None:                           # 3. execute at the open
            budget = max(eq, 0.0) * cfg.exposure
            tgt = dict(self.lots)
            w = self.pending["weights"]
            for sym in set(w) | {x for x, n in self.lots.items() if n}:
                if not (ok(o.get(sym)) and sym in lot):
                    continue                                   # cannot trade today: leave as is
                want = w.get(sym, 0.0) * budget
                tgt[sym] = float(np.sign(want) * np.floor(abs(want) / (lot[sym] * o[sym]) + 0.5))
            mgn = sum(abs(n) * lot[x] * o[x] for x, n in tgt.items() if n and ok(o.get(x))) * cfg.margin_pct
            cap = eq * MARGIN_BUFFER
            if mgn > cap > 0:                                  # margin cap: scale down
                tgt = {x: float(np.sign(n) * np.floor(abs(n) * cap / mgn)) for x, n in tgt.items()}
            for sym, n_new in sorted(tgt.items()):
                delta = n_new - self.lots.get(sym, 0)
                if delta == 0 or not ok(o.get(sym)):
                    continue
                nv = abs(delta) * lot[sym] * o[sym]
                side = "buy" if delta > 0 else "sell"
                cst = leg_cost(nv, side, "futures", self.rates)
                eq -= cst + nv * s
                fees += cst
                slip += nv * s
                px = o[sym] * (1 + s) if delta > 0 else o[sym] * (1 - s)
                trades.append({"date": str(d), "sym": sym, "side": side, "lots": abs(delta),
                               "lot_size": lot[sym], "price": round(px, 2), "notional": round(nv),
                               "fee": round(cst, 2), "slippage": round(nv * s, 2)})
            self.lots = {x: n for x, n in tgt.items() if n}
            self.pending = None
        for sym, n in self.lots.items():                       # 4. intraday P&L
            if n and ok(o.get(sym)) and ok(c.get(sym)):
                eq += n * lot[sym] * (c[sym] - o[sym])
        self.equity = eq
        self.last_date = str(d)
        return {"date": str(d), "equity": eq, "fees": fees, "slippage": slip, "trades": trades}
