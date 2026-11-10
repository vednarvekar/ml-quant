"""How much capital does the ML futures strategy need? Lot-aware runs across account sizes.

uv run python -m mlq.futures_run
"""
import numpy as np
import polars as pl

from mlq import registry
from mlq.backtest import panel
from mlq.baselines import load_prices
from mlq.config import DATA
from mlq.futures import lot_sizes
from mlq.futures_bt import simulate_futures
from mlq.swing import make_ctx
from mlq.swing_portfolio import CLEAN_START, build_weights

CAPITALS = [5, 10, 20, 50, 100]          # lakh rupees
STRATS = [("weekly hyst k20", 5, "hyst", dict(k=20, keep=40)),
          ("weekly hyst k10", 5, "hyst", dict(k=10, keep=20)),
          ("weekly top5", 5, "topk", dict(k=5)),
          ("20-day top20", 20, "topk", dict(k=20))]


def main():
    p = load_prices()
    ctx = make_ctx()
    _, _, O = panel(p, "open")
    lots = lot_sizes()
    lot = np.array([lots.get(s, np.nan) for s in ctx.syms], float)
    pred = np.load(DATA / "clean" / "ml_swing_pred_h5.npy")
    pred = np.where(ctx.fno[None, :], pred, np.nan)               # only F&O names can be traded
    i0 = next(i for i, d in enumerate(ctx.dates) if d.isoformat() >= CLEAN_START)
    rows = []
    for name, R, kind, kw in STRATS:
        W = build_weights(pred, ctx, i0, R, kind, **kw)
        for cap in CAPITALS:
            for expo in (1.0, 2.0):
                sl = slice(i0 - 1, len(ctx.dates))
                r = simulate_futures(O[sl], ctx.close[sl], W[sl], ctx.dates[sl], lot,
                                     capital=cap * 1e5, exposure=expo)
                m = r.metrics()
                eq = r.equity
                registry.log_trial(f"fut-lot {name} {cap}L x{expo}", "futures-lot",
                                   {"cap_lakh": cap, "expo": expo, "strat": name}, eq[1:] / eq[:-1] - 1)
                rows.append({"strategy": name, "capital_L": cap, "expo": expo, "cagr": round(m["cagr"], 3),
                             "sharpe": round(m["sharpe"], 2), "max_dd": round(m["max_dd"], 3),
                             "avg_pos": round(m["avg_positions"], 1), "avg_expo": round(m["avg_exposure"], 2),
                             "fees_L": round(m["fees"] / 1e5, 1)})
        print(name, "done", flush=True)
    out = pl.DataFrame(rows)
    out.write_csv(DATA / "clean" / "futures_lot_results.csv")
    with pl.Config(tbl_rows=80, tbl_width_chars=150):
        print(out)


if __name__ == "__main__":
    main()
