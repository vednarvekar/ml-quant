"""Turn ML predictions into low-turnover portfolios (the cost problem).

Takes saved walk-forward predictions (`ml_swing_pred_h{H}.npy`) and tries several portfolio
constructions that trade less: slower rebalancing, rank hysteresis, score smoothing, partial
rebalancing, trend gating. Delivery, long-only, 5 bps slippage (conservative). All logged.

uv run python -m mlq.swing_portfolio
"""
import numpy as np
import polars as pl

from mlq import registry
from mlq.backtest import panel, simulate
from mlq.baselines import load_prices
from mlq.config import DATA
from mlq.swing import make_ctx, trend_on

CLEAN_START = "2022-01-03"


def ranks(score):
    idx = [j for j in np.argsort(np.where(np.isnan(score), -np.inf, score))[::-1] if not np.isnan(score[j])]
    return idx, {j: r for r, j in enumerate(idx)}


def build_weights(pred, ctx, first, R, kind, k=20, keep=40, alpha=0.5, frac=0.35, gate=False):
    T, N = pred.shape
    W = np.zeros((T, N))
    cur, ema = np.zeros(N), np.full(N, np.nan)
    since = 10**9
    for t in range(max(first - 1, 0), T):
        since += 1
        sc = pred[t]
        if not np.all(np.isnan(sc)):
            if kind == "ema":                      # smooth every day so noise averages out
                ema = np.where(np.isnan(ema), sc, np.where(np.isnan(sc), ema, alpha * sc + (1 - alpha) * ema))
            if since >= R:
                since = 0
                s = ema if kind == "ema" else sc
                idx, rk = ranks(s)
                if len(idx) >= k:
                    chosen = idx[:k]
                    if kind == "hyst":
                        held = [j for j in np.nonzero(cur > 0)[0] if rk.get(j, 10**9) < keep]
                        chosen = sorted(held, key=lambda j: rk[j])[:k]
                        chosen += [j for j in idx if j not in chosen][:k - len(chosen)]
                    tgt = np.zeros(N)
                    tgt[chosen] = 1.0 / k
                    cur = cur + frac * (tgt - cur) if kind == "partial" else tgt
        W[t] = cur * ((1.0 if trend_on(ctx, t) else 0.5) if gate else 1.0)
    return W


def metrics(ctx, O, W, lo, slip=5.0):
    sl = slice(max(lo - 1, 0), len(ctx.dates))
    r = simulate(O[sl], ctx.close[sl], W[sl], ctx.dates[sl], kind="delivery", slippage_bps=slip)
    e = r.equity["equity"].to_numpy()
    return r.metrics(), e[1:] / e[:-1] - 1


def main():
    p = load_prices()
    ctx = make_ctx()
    _, _, O = panel(p, "open")
    i_clean = next(i for i, d in enumerate(ctx.dates) if d.isoformat() >= CLEAN_START)
    specs = []
    for h in (5, 10, 20):
        for R in (5, 10, 20):
            specs.append((h, R, "topk", {}))
    for h, R in ((5, 5), (10, 10), (20, 20), (10, 5)):
        specs += [(h, R, "hyst", {"keep": 40}), (h, R, "ema", {"alpha": 0.4}),
                  (h, R, "partial", {"frac": 0.35}), (h, R, "topk", {"gate": True}),
                  (h, R, "hyst", {"keep": 40, "gate": True})]
    rows = []
    for h, R, kind, kw in specs:
        f = DATA / "clean" / f"ml_swing_pred_h{h}.npy"
        if not f.exists():
            continue
        pred = np.load(f)
        first = int(np.nonzero(~np.all(np.isnan(pred), axis=1))[0][0])
        W = build_weights(pred, ctx, first, R, kind, **kw)
        for tag, lo in (("all OOS", first), ("2022+ clean", i_clean)):
            m, d = metrics(ctx, O, W, lo)
            name = f"h{h} R{R} {kind} {kw}"
            registry.log_trial(f"{name} [{tag}]", "swing-ml", {"h": h, "R": R, "kind": kind, **kw}, d)
            rows.append({"period": tag, "h": h, "R": R, "kind": kind, "opts": str(kw),
                         "cagr": round(m["cagr"], 3), "sharpe": round(m["sharpe"], 2),
                         "max_dd": round(m["max_dd"], 3), "turn/yr": round(m["turnover_per_year"], 1),
                         "fees_k": round(m["fees"] / 1e3), "dsr": round(registry.dsr(d, "swing-ml"), 2)})
        print(rows[-1], flush=True)
    out = pl.DataFrame(rows).sort(["period", "sharpe"], descending=[False, True])
    out.write_csv(DATA / "clean" / "swing_portfolio_results.csv")
    with pl.Config(tbl_rows=60, tbl_width_chars=170):
        print(out)


if __name__ == "__main__":
    main()
