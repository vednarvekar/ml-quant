"""Holdout test: run the FROZEN candidates on 2013-2021, a period never used to pick settings.

All design choices (model v3, 5-day label, k, rebalance, sizes) were made on 2022-2026. The model's
predictions for 2013-2021 already exist (walk-forward: each day scored by a model trained only on
earlier data). Nothing is re-fitted or re-tuned here.

Caveat printed with the results: before 2022 our price data misses some stocks that later
delisted (survivorship). The coverage table shows how many F&O names each year we actually have.

uv run python -m mlq.holdout
"""
import numpy as np
import polars as pl

from mlq import fo_features as ff
from mlq.compare_ls import ls_weights
from mlq.config import DATA
from mlq.evaluate import idx_returns, run_phases, stats
from mlq.futures_bt import simulate_futures
from mlq.swing import make_ctx
from mlq.swing_portfolio import build_weights

START, END = "2013-01-01", "2021-12-31"


def coverage(ctx) -> pl.DataFrame:
    have = set(ctx.syms)
    rows = []
    for y in range(2013, 2022):
        fs = sorted((DATA / "raw" / "nse_fo").glob(f"{y}-06-*.parquet"))[:1]
        if not fs:
            continue
        d = pl.read_parquet(fs[0])
        stk = set(d.filter(pl.col("kind") == "S")["sym"].unique().to_list())
        rows.append({"year": y, "fo_stocks": len(stk), "we_have": len(stk & have),
                     "pct": round(len(stk & have) / max(len(stk), 1), 2)})
    return pl.DataFrame(rows)


def main():
    ctx = make_ctx()
    T, N = ctx.close.shape
    fo = ff.build(ctx)
    ox, cx, _ = ff.exdiv_prices(ctx)
    member = fo["member"]
    lot_hist = ff.ffill_lots(np.where(member, fo["lot_hist"], np.nan))
    pred0 = np.load(DATA / "clean" / "ml_v3_pred_raw.npy")
    if pred0.shape[0] < T:                      # file saved before newer days were added
        pred0 = np.vstack([pred0, np.full((T - pred0.shape[0], N), np.nan)])
    pred = np.where(member & np.isfinite(pred0), pred0, np.nan)
    iso = [d.isoformat() for d in ctx.dates]
    i0 = next(i for i, d in enumerate(iso) if d >= START)
    i1 = max(i for i, d in enumerate(iso) if d <= END) + 1
    fac = {k: v[i0:i1] for k, v in idx_returns(ctx.dates).items()}
    sl = slice(i0 - 1, i1)
    flags = fo["roll_flags"]

    print(coverage(ctx), flush=True)

    def run(W, cap):
        e = simulate_futures(ox[sl], cx[sl], W[sl], ctx.dates[sl], lot_hist[sl], capital=cap,
                             roll_flags=flags[sl]).equity
        return e[1:] / e[:-1] - 1

    rev = np.full_like(pred, np.nan)
    rev[5:] = -(ctx.close[5:] / ctx.close[:-5] - 1)
    rev = np.where(np.isfinite(pred), rev, np.nan)
    rows, yearly = [], []
    for label, mode, cap, k in (("L/S k=10 R=5, Rs 5cr", "ls", 50_000_000, 10),
                                ("long-only top-20, Rs 1cr", "long", 10_000_000, 20)):
        def W_of(score, ph, mode=mode, k=k):
            if mode == "ls":
                return ls_weights(score, i0 - ph, 5, k, 2 * k)
            return build_weights(score, ctx, i0 - ph, 5, "hyst", k=k, keep=2 * k)

        r = run_phases(f"ML {label}", lambda ph: run(W_of(pred, ph), cap), list(range(5)), fac)
        rows.append(r.summary())
        print(rows[-1], flush=True)
        rows.append(run_phases(f"5-day reversal {label}", lambda ph: run(W_of(rev, ph), cap),
                               list(range(5)), fac).summary())
        print(rows[-1], flush=True)
        sh = []
        for seed in range(4):
            rs = np.where(np.isfinite(pred), np.random.default_rng(seed).normal(size=pred.shape), np.nan)
            sh += run_phases("r", lambda ph, rs=rs: run(W_of(rs, ph), cap), [0, 2], fac).sharpes
        rows.append({"strategy": f"random {label}", "sharpe_mean": round(float(np.mean(sh)), 2),
                     "sharpe_min": round(float(np.min(sh)), 2), "sharpe_max": round(float(np.max(sh)), 2)})
        print(rows[-1], flush=True)
        # per-year Sharpe of the ML strategy (phase 0)
        r0 = run(W_of(pred, 0), cap)
        yrs = np.array([d.year for d in ctx.dates[i0:i1]])
        for y in sorted(set(yrs)):
            s = stats(r0[yrs == y])
            yearly.append({"strategy": label, "year": int(y), "return": round(float(np.prod(1 + r0[yrs == y]) - 1), 3),
                           "sharpe": round(s["sharpe"], 2)})
    for key, nm in (("n50", "Nifty 50"), ("mid", "Midcap 100")):
        if key in fac:
            s = stats(fac[key])
            rows.append({"strategy": f"{nm} buy&hold", "sharpe_mean": round(s["sharpe"], 2),
                         "cagr_mean": round(s["cagr"], 3), "maxdd_worst": round(s["max_dd"], 3)})
    df = pl.DataFrame(rows, infer_schema_length=None)
    df.write_csv(DATA / "clean" / "holdout_results.csv")
    yd = pl.DataFrame(yearly)
    yd.write_csv(DATA / "clean" / "holdout_yearly.csv")
    with pl.Config(tbl_rows=40, tbl_width_chars=200, fmt_str_lengths=60):
        print(df)
        print(yd)


if __name__ == "__main__":
    main()
