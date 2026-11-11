"""Futures evaluation with the biases removed (audit step 2) and a Nifty-hedged variant (step 4).

- point-in-time F&O membership (a stock can only be picked on dates it had futures)
- historical lot sizes (new format given; old format inferred from open interest)
- ex-dividend prices (futures holders do not receive dividends)
- roll 2 trading days before each monthly expiry (stock futures are physically settled)
- phase-averaged, with random-score and reversal baselines and factor-alpha t-stats

uv run python -m mlq.compare_v2 [pred_file ...]
"""
import sys

import numpy as np
import polars as pl

from mlq import fo_features as ff
from mlq.config import DATA
from mlq.evaluate import CLEAN_START, idx_returns, index_rows, run_phases
from mlq.futures_bt import simulate_futures
from mlq.ml_v2 import idx_open_close, rolling_beta
from mlq.swing import make_ctx
from mlq.swing_portfolio import build_weights


def hedge_column(W, beta, rebal_flag):
    """Short Nifty notional = portfolio beta at the last rebalance, carried until the next one."""
    T = W.shape[0]
    col = np.zeros(T)
    cur = 0.0
    for t in range(T):
        if rebal_flag[t]:
            b = np.where(np.isfinite(beta[t]), beta[t], 1.0)
            cur = -float(np.clip((W[t] * b).sum(), 0, 1.5))
        col[t] = cur if W[t].sum() > 0 else 0.0
    return col


def main(pred_files: list[str]):
    ctx = make_ctx()
    T, N = ctx.close.shape
    fo = ff.build(ctx)
    ox, cx, _ = ff.exdiv_prices(ctx)
    member = fo["member"]
    lot_hist = ff.ffill_lots(np.where(member, fo["lot_hist"], np.nan))
    n50_o, n50_c = idx_open_close("Nifty_50", ctx.dates)
    r1 = ctx.close / np.vstack([np.full((1, N), np.nan), ctx.close[:-1]]) - 1
    rm = np.full(T, np.nan)
    rm[1:] = n50_c[1:] / n50_c[:-1] - 1
    beta = rolling_beta(r1, rm, 60)
    i0 = next(i for i, d in enumerate(ctx.dates) if d.isoformat() >= CLEAN_START)
    fac = {k: v[i0:] for k, v in idx_returns(ctx.dates).items()}
    sl = slice(i0 - 1, T)
    flags = fo["roll_flags"]
    ox_a, cx_a = np.column_stack([ox, n50_o]), np.column_stack([cx, n50_c])
    lot_a = np.column_stack([lot_hist, np.ones(T)])
    print("F&O member names per day (2022+): mean", member[i0:].sum(axis=1).mean().round(1),
          "| lots known share", np.isfinite(lot_hist[i0:][member[i0:]]).mean().round(3), flush=True)
    out_rows = []
    for pf in pred_files:
        pred0 = np.load(pf)
        pred = np.where(member & np.isfinite(pred0), pred0, np.nan)
        for cap, k, keep in ((2_000_000, 10, 20), (10_000_000, 20, 40)):
            def run(W, hedge=False):
                if hedge:
                    reb = np.r_[True, np.any(W[1:] != W[:-1], axis=1)]
                    hc = hedge_column(W, beta, reb)
                    Wa = np.column_stack([W, hc])
                    e = simulate_futures(ox_a[sl], cx_a[sl], Wa[sl], ctx.dates[sl], lot_a[sl], capital=cap,
                                         roll_flags=flags[sl]).equity
                else:
                    e = simulate_futures(ox[sl], cx[sl], W[sl], ctx.dates[sl], lot_hist[sl], capital=cap,
                                         roll_flags=flags[sl]).equity
                return e[1:] / e[:-1] - 1

            def mk(score_fn, hedge=False):
                return lambda ph: run(build_weights(score_fn(), ctx, i0 - ph, 5, "hyst", k=k, keep=keep), hedge)

            def rev():
                s = np.full_like(pred, np.nan)
                s[5:] = -(ctx.close[5:] / ctx.close[:-5] - 1)
                return np.where(np.isfinite(pred), s, np.nan)

            label = f"{pf.split('/')[-1].replace('.npy', '').replace('ml_', '')} | Rs {cap // 100000}L top-{k}"
            rows = [run_phases(f"ML long-only [{label}]", mk(lambda: pred), list(range(5)), fac).summary(),
                    run_phases(f"ML hedged vs Nifty [{label}]", mk(lambda: pred, True), list(range(5)), fac).summary()]
            if pf == pred_files[0]:
                rows.append(run_phases(f"5-day reversal [Rs {cap // 100000}L top-{k}]", mk(rev), list(range(5)), fac).summary())
                rr = []
                for seed in range(5):
                    rng = np.random.default_rng(seed)
                    rs = np.where(np.isfinite(pred), rng.normal(size=pred.shape), np.nan)
                    rr.append(run_phases("r", mk(lambda rs=rs: rs), [0, 2, 4], fac))
                sh = np.concatenate([x.sharpes for x in rr])
                at = np.concatenate([x.alpha_t for x in rr])
                rows.append({"strategy": f"random scores [Rs {cap // 100000}L top-{k}]",
                             "sharpe_mean": round(float(sh.mean()), 2), "sharpe_min": round(float(sh.min()), 2),
                             "sharpe_max": round(float(sh.max()), 2),
                             "cagr_mean": round(float(np.mean([c for x in rr for c in x.cagrs])), 3),
                             "maxdd_worst": round(float(np.min([d for x in rr for d in x.dds])), 3),
                             "alpha_yr": round(float(np.mean([a for x in rr for a in x.alpha])), 3),
                             "alpha_t_mean": round(float(at.mean()), 2), "alpha_t_max": round(float(at.max()), 2)})
            for r in rows:
                print(r, flush=True)
            out_rows += rows
    out_rows += index_rows(ctx.dates, i0)
    df = pl.DataFrame(out_rows, infer_schema_length=None)
    df.write_csv(DATA / "clean" / "compare_v2_results.csv")
    with pl.Config(tbl_rows=40, tbl_width_chars=200, fmt_str_lengths=60):
        print(df)


if __name__ == "__main__":
    main(sys.argv[1:] or [str(DATA / "clean" / "ml_swing_pred_h5.npy")])
