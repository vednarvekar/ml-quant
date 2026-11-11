"""Market-neutral long/short with stock futures: does the cross-sectional skill (rank IC) pay?

Long the top-k and short the bottom-k predictions (equal weights, gross = equity x exposure), bias-free
futures setup (point-in-time F&O, historical lots, ex-dividend prices, roll before expiry). Needs a
large account because every name is a whole lot (hence Rs 5 crore default). Phase-averaged,
vs random and 5-day-reversal L/S baselines.

uv run python -m mlq.compare_ls pred1.npy [pred2.npy ...]
"""
import sys

import numpy as np
import polars as pl

from mlq import fo_features as ff
from mlq.config import DATA
from mlq.evaluate import CLEAN_START, idx_returns, run_phases
from mlq.futures_bt import simulate_futures
from mlq.swing import make_ctx


def ls_weights(score, first, R, k, keep):
    T, N = score.shape
    W = np.zeros((T, N))
    cur, since = np.zeros(N), 10**9
    longs, shorts = set(), set()
    for t in range(max(first - 1, 0), T):
        since += 1
        s = score[t]
        if since >= R and np.isfinite(s).sum() >= 4 * k:
            since = 0
            # order is descending by score
            order = sorted([j for j in range(N) if np.isfinite(s[j])], key=lambda j: -s[j])
            rank = {j: r for r, j in enumerate(order)}
            n = len(order)
            keepL = [j for j in longs if rank.get(j, 10**9) < keep]
            keepS = [j for j in shorts if rank.get(j, -1) >= n - keep]
            L = keepL[:k] + [j for j in order if j not in keepL][:k - len(keepL[:k])]
            S = keepS[:k] + [j for j in reversed(order) if j not in keepS and j not in L][:k - len(keepS[:k])]
            longs, shorts = set(L), set(S)
            cur = np.zeros(N)
            cur[L] = 0.5 / k
            cur[S] = -0.5 / k
        W[t] = cur
    return W


def main(pred_files):
    ctx = make_ctx()
    T, N = ctx.close.shape
    fo = ff.build(ctx)
    ox, cx, _ = ff.exdiv_prices(ctx)
    member = fo["member"]
    lot_hist = ff.ffill_lots(np.where(member, fo["lot_hist"], np.nan))
    i0 = next(i for i, d in enumerate(ctx.dates) if d.isoformat() >= CLEAN_START)
    fac = {k: v[i0:] for k, v in idx_returns(ctx.dates).items()}
    sl = slice(i0 - 1, T)
    flags = fo["roll_flags"]
    rows = []
    for pf in pred_files:
        name = pf.split("/")[-1].replace(".npy", "").replace("ml_", "")
        pred = np.where(member & np.isfinite(np.load(pf)), np.load(pf), np.nan)
        import re
        m_h = re.search(r"_h(\d+)_pred", pf)
        hold = int(m_h.group(1)) if m_h else 5
        cfgs = sorted({(50_000_000, k, R) for k in (10, 15) for R in (hold, max(5, hold // 2))})
        for cap, k, R in cfgs:
            keep = 2 * k

            def run(W, cap=cap):
                e = simulate_futures(ox[sl], cx[sl], W[sl], ctx.dates[sl], lot_hist[sl], capital=cap,
                                     roll_flags=flags[sl]).equity
                return e[1:] / e[:-1] - 1

            def mk(score_fn, k=k, R=R, keep=keep):
                return lambda ph: run(ls_weights(score_fn(), i0 - ph, R, k, keep))

            def rev():
                s = np.full_like(pred, np.nan)
                s[5:] = -(ctx.close[5:] / ctx.close[:-5] - 1)
                return np.where(np.isfinite(pred), s, np.nan)

            tag = f"Rs {cap // 10_000_000}cr k={k} R={R}"
            phases = list(range(0, R, max(1, R // 5)))
            rows.append(run_phases(f"ML L/S [{name} | {tag}]", mk(lambda: pred), phases, fac).summary())
            print(rows[-1], flush=True)
            if pf == pred_files[0]:
                rows.append(run_phases(f"5-day reversal L/S [{tag}]", mk(rev), phases, fac).summary())
                rr = []
                for seed in range(4):
                    rs = np.where(np.isfinite(pred), np.random.default_rng(seed).normal(size=pred.shape), np.nan)
                    rr.append(run_phases("r", mk(lambda rs=rs: rs), phases[:3], fac))
                sh = np.concatenate([x.sharpes for x in rr])
                rows.append({"strategy": f"random L/S [{tag}]", "sharpe_mean": round(float(sh.mean()), 2),
                             "sharpe_min": round(float(sh.min()), 2), "sharpe_max": round(float(sh.max()), 2)})
                print(rows[-2], rows[-1], flush=True)
    df = pl.DataFrame(rows, infer_schema_length=None)
    df.write_csv(DATA / "clean" / "compare_ls_results.csv")
    with pl.Config(tbl_rows=40, tbl_width_chars=200, fmt_str_lengths=60):
        print(df)


if __name__ == "__main__":
    main(sys.argv[1:])
