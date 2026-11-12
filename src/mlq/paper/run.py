"""Paper-trading runner. Run once per day after the market closes (data for today must be in).

uv run python -m mlq.paper.run --update          # refresh data, process new days, decide, report
uv run python -m mlq.paper.run                   # same, using data already on disk
uv run python -m mlq.paper.run --replay 2022-01-03   # self-test: replay history through the live code

Safety: this module has no connection to any broker order API. A file `data/paper/KILL` stops all
new decisions. Stale data (older than max_stale_days) also blocks new decisions.
"""
import argparse
import json
from datetime import date, datetime
from pathlib import Path

import numpy as np
import polars as pl

from mlq.backtest import panel
from mlq.baselines import load_prices
from mlq.config import DATA
from mlq.futures import lot_sizes
from mlq.paper import signals
from mlq.paper.broker import PaperBroker, PaperConfig
from mlq.swing import make_ctx

HOME = DATA / "paper"


def world():
    p = load_prices()
    ctx = make_ctx()
    _, _, O = panel(p, "open")
    lots = lot_sizes()
    return p, ctx, O, {s: lots[s] for s in ctx.syms if s in lots}


def day_prices(ctx, O, i, lot):
    fut = [(j, s) for j, s in enumerate(ctx.syms) if s in lot]
    o = {s: float(O[i, j]) for j, s in fut}
    c = {s: float(ctx.close[i, j]) for j, s in fut}
    pc = {s: float(ctx.close[i - 1, j]) for j, s in fut}
    return o, c, pc


def append_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    df = pl.DataFrame(rows)
    if path.exists():
        df = pl.concat([pl.read_csv(path, infer_schema_length=0).cast(pl.String), df.cast(pl.String)])
    df.write_csv(path)


def decide(b: PaperBroker, ctx, i, scores: dict, alerts: list, home: Path, blocked: str | None):
    """Make the end-of-day decision at index i; stores b.pending (executes at next open)."""
    if b.days_since < b.cfg.rebalance_days:
        return
    if blocked:
        alerts.append(f"no decision: {blocked}")
        return
    if b.cfg.mode == "ls":
        longs, shorts = signals.select_ls(scores, set(b.target), set(b.target_short), b.cfg.k, b.cfg.keep)
        if not longs:
            alerts.append("no decision: not enough scored names")
            return
        b.pending = {"decided": str(ctx.dates[i]), "weights": signals.weights_ls(longs, shorts)}
        b.target, b.target_short, names = list(longs), list(shorts), list(longs) + list(shorts)
    else:
        names = signals.select_names(scores, set(b.target), b.cfg.k, b.cfg.keep)
        if not names:
            alerts.append("no decision: not enough scored names")
            return
        b.pending = {"decided": str(ctx.dates[i]), "weights": signals.weights_for(names)}
        b.target = list(names)
    b.days_since = 0
    with open(home / "decisions.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps({"date": str(ctx.dates[i]), "names": names, "mode": b.cfg.mode,
                            "scores": {s: round(scores[s], 5) for s in names}}) + "\n")


def equity_future_expiries() -> list:
    """Monthly stock-futures expiry dates from the Upstox instrument master (current + next series)."""
    from datetime import datetime as _dt

    from mlq.config import INSTRUMENTS_FILE
    d = pl.read_parquet(INSTRUMENTS_FILE).filter((pl.col("segment") == "NSE_FO")
                                                 & (pl.col("instrument_type") == "FUT")
                                                 & (pl.col("underlying_type") == "EQUITY"))
    return sorted({_dt.fromtimestamp(x / 1000).date() for x in d["expiry"].drop_nulls().unique().to_list()})


def roll_due(b, d, expiries, roll_days: int = 2) -> bool:
    """Stock futures are PHYSICALLY settled and expire monthly: roll `roll_days` trading days before
    expiry (weekday count; holidays make it at most a day early) and only once per expiry."""
    nxt = next((e for e in expiries if e >= d), None)
    if nxt is None or b.last_roll == str(nxt):
        return False
    if np.busday_count(d, nxt) <= roll_days:
        b.last_roll = str(nxt)
        return True
    return False


def advance(b, ctx, O, lot, i, home, alerts, trades_out, expiries=None):
    if expiries is None:                                  # replay/self-test: month-based (matches backtest)
        new_month = ctx.dates[i].month != ctx.dates[i - 1].month
    else:
        new_month = roll_due(b, ctx.dates[i], expiries)
    o, c, pc = day_prices(ctx, O, i, lot)
    r = b.process_day(ctx.dates[i], o, c, pc, lot, new_month)
    b.days_since += 1
    trades_out.extend(r["trades"])
    return r


def report(b, ctx, home, alerts, last_trades):
    i = len(ctx.dates) - 1
    idx0 = next((k for k, d in enumerate(ctx.dates) if str(d) >= (b.start_date or "9999")), i)
    nifty = ctx.idx_close[i] / ctx.idx_close[idx0] - 1 if np.isfinite(ctx.idx_close[idx0]) else float("nan")
    lines = [f"# Paper trading report - data through {ctx.dates[i]}", "",
             f"- Equity: Rs {b.equity:,.0f} (start Rs {b.start_equity:,.0f}, since {b.start_date}) = "
             f"{b.equity / b.start_equity - 1:+.2%}; Nifty 50 over same period {nifty:+.2%}",
             f"- Last processed day: {b.last_date}; days since last decision: {b.days_since}",
             f"- Config: {b.cfg}", ""]
    if alerts:
        lines += ["## ALERTS"] + [f"- {a}" for a in alerts] + [""]
    lines += ["## Open positions (lots)"]
    for s, n in sorted(b.lots.items()):
        lines.append(f"- {s}: {n:+.0f} lots")
    if b.pending:
        lines += ["", f"## Pending (decided {b.pending['decided']}, executes at next open)"]
        lines += [f"- {s}: weight {w:.2f}" for s, w in b.pending["weights"].items()]
    if last_trades:
        lines += ["", "## Trades today"] + [f"- {t['date']} {t['side']} {t['lots']:.0f} x {t['sym']} @ {t['price']}"
                                           for t in last_trades]
    (home / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


ACCOUNTS = [  # name, config. Sized so whole futures lots can actually hold the book (see docs/19)
    ("ls_5cr", PaperConfig(capital=50_000_000, k=10, keep=20, mode="ls")),
    ("long_1cr", PaperConfig(capital=10_000_000, k=20, keep=40, mode="long")),
]


def run_live(accounts=None):
    accounts = accounts or ACCOUNTS
    p, ctx, O, lot = world()
    D = ctx.dates[-1]
    cfg0 = accounts[0][1]
    # ---- model (shared by all accounts): retrain if none, old, or the feature list changed
    X, yd, M, names = signals.features(ctx, p)
    tag = signals.latest_model_tag()
    meta = json.loads((signals.MODEL_DIR / f"model_{tag}.json").read_text()) if tag else None
    age = (len(ctx.dates) - 1 - next(k for k, d in enumerate(ctx.dates) if str(d) >= meta["trained_through"])
           ) if meta else 10**9
    if age > cfg0.retrain_days or (meta and meta["features"] != names):
        mdl, last = signals.fit_from(X, yd, M)
        tag = str(D)
        signals.save_model(mdl, names, ctx.dates[last], tag)
        print(f"retrained model, trained through {ctx.dates[last]}")
    booster, meta = signals.load_model(tag)
    i = len(ctx.dates) - 1
    fno = np.array([s in lot for s in ctx.syms])
    scores = signals.score_day(booster, X[i], M[i], fno, ctx.syms)
    expiries = equity_future_expiries()
    out = {}
    for name, cfg in accounts:
        home = HOME / name
        home.mkdir(parents=True, exist_ok=True)
        b = PaperBroker.load(home / "state.json", cfg)
        alerts, trades = [], []
        stale = (date.today() - D).days > cfg.max_stale_days
        blocked = ("data is stale (latest " + str(D) + ")" if stale else
                   "KILL file present" if ((home / "KILL").exists() or (HOME / "KILL").exists()) else None)
        if meta["features"] != names:
            alerts.append("feature list changed since the model was trained: retrain required")
            blocked = blocked or "feature mismatch"
        if b.last_date is None:                               # first ever run: decide at latest close
            b.start_date, b.last_date, b.days_since = str(D), str(D), cfg.rebalance_days
            start_i = len(ctx.dates)
        else:
            start_i = next((k for k, d in enumerate(ctx.dates) if str(d) > b.last_date), len(ctx.dates))
        for k in range(start_i, len(ctx.dates)):
            r = advance(b, ctx, O, lot, k, home, alerts, trades, expiries)
            append_csv(home / "equity.csv", [{"date": str(ctx.dates[k]), "equity": round(r["equity"], 2),
                                              "nifty": round(float(ctx.idx_close[k]), 2),
                                              "positions": sum(1 for n in b.lots.values() if n)}])
        append_csv(home / "trades.csv", trades)
        held_now = [s for s, n in b.lots.items() if n]
        for j, s in enumerate(ctx.syms):                       # corporate-action guard on holdings
            if s in held_now and np.isfinite(ctx.close[i, j]) and np.isfinite(ctx.close[i - 1, j]):
                if abs(ctx.close[i, j] / ctx.close[i - 1, j] - 1) > 0.15:
                    alerts.append(f"{s} moved >15% today: check for split/bonus/demerger before trusting P&L")
        decide(b, ctx, i, scores, alerts, home, blocked)
        b.save()
        print(f"\n######## account {name} ({cfg.mode}, Rs {cfg.capital / 1e7:.1f} crore, k={cfg.k}) ########")
        report(b, ctx, home, alerts, [t for t in trades if t["date"] == str(D)])
        out[name] = b
    return out


def run_replay(start: str, cfg: PaperConfig):
    """Self-test: push history through the live broker/selection code using the saved walk-forward
    predictions as the scorer, and compare with the backtest engine."""
    import shutil

    from mlq.futures_bt import simulate_futures
    from mlq.swing_portfolio import build_weights

    home = DATA / "paper_replay"
    shutil.rmtree(home, ignore_errors=True)
    home.mkdir(parents=True)
    p, ctx, O, lot = world()
    pred = np.load(DATA / "clean" / "ml_swing_pred_h5.npy")
    i0 = next(i for i, d in enumerate(ctx.dates) if str(d) >= start)
    b = PaperBroker.load(home / "state.json", cfg)
    b.days_since = cfg.rebalance_days
    alerts, trades = [], []
    fno = np.array([s in lot for s in ctx.syms])

    def scores(i):
        return {s: pred[i, j] for j, s in enumerate(ctx.syms) if fno[j] and np.isfinite(pred[i, j])}

    decide(b, ctx, i0 - 1, scores(i0 - 1), alerts, home, None)     # same start as the backtest
    for i in range(i0, len(ctx.dates)):
        advance(b, ctx, O, lot, i, home, alerts, trades)
        decide(b, ctx, i, scores(i), alerts, home, None)
    # backtest reference with the same predictions and parameters
    lot_arr = np.array([lot.get(s, np.nan) for s in ctx.syms])
    W = build_weights(np.where(fno[None, :], pred, np.nan), ctx, i0, cfg.rebalance_days, "hyst",
                      k=cfg.k, keep=cfg.keep)
    sl = slice(i0 - 1, len(ctx.dates))
    r = simulate_futures(O[sl], ctx.close[sl], W[sl], ctx.dates[sl], lot_arr, capital=cfg.capital,
                         exposure=cfg.exposure, slippage_bps=cfg.slippage_bps)
    print(f"replay from {ctx.dates[i0]}: live-code equity Rs {b.equity:,.0f} | backtest engine Rs {r.equity[-1]:,.0f} "
          f"| diff {b.equity / r.equity[-1] - 1:+.2%} | trades {len(trades)}")
    return b.equity, r.equity[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--update", action="store_true")
    ap.add_argument("--replay", metavar="YYYY-MM-DD")
    a = ap.parse_args()
    if a.replay:
        run_replay(a.replay, PaperConfig())
        return
    if a.update:
        from mlq.paper import update
        print(update.refresh_data(), flush=True)
        update.rebuild_offline()
    run_live()


if __name__ == "__main__":
    main()
