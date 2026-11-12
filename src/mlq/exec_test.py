"""Execution robustness (audit #3): does the candidate survive realistic fills instead of the raw 09:15 open?

Re-runs the SAME decisions (v3 5-day model) but with trades filled at
  open | VWAP 09:15-09:30 | VWAP 09:15-09:45 | open of the 10:00 bar
built from the 5-minute data (adjusted to the daily basis). Nothing is re-fitted or re-tuned: this
is a stress test of the existing candidate, reported as-is.

uv run python -m mlq.exec_test
"""
import numpy as np
import polars as pl

from mlq import fo_features as ff
from mlq.compare_ls import ls_weights
from mlq.config import DATA
from mlq.evaluate import CLEAN_START, idx_returns, run_phases
from mlq.futures_bt import simulate_futures
from mlq.intraday import B
from mlq.swing import make_ctx
from mlq.swing_portfolio import build_weights


def fill_matrices(ctx, kinds):
    """Return {kind: (T,N) price matrix on the ex-dividend basis} for the 5m era (NaN before 2022)."""
    T, N = len(ctx.dates), len(ctx.syms)
    ri = {d: i for i, d in enumerate(ctx.dates)}
    out = {k: np.full((T, N), np.nan) for k in kinds}
    for j, s in enumerate(ctx.syms):
        f = DATA / "clean" / "adj_5m" / f"{s}.parquet"
        if not f.exists():
            continue
        g = pl.read_parquet(f).filter(~pl.col("special_session")).with_columns(
            ((pl.col("ts").dt.hour().cast(pl.Int32) * 60 + pl.col("ts").dt.minute().cast(pl.Int32) - 555) // 5).alias("bar"))
        g = g.filter((pl.col("bar") >= 0) & (pl.col("bar") < B))
        for d, gd in g.group_by("date"):
            i = ri.get(d[0])
            if i is None:
                continue
            bars = gd.sort("bar")
            bn, o, c, v = bars["bar"].to_numpy(), bars["open"].to_numpy(), bars["close"].to_numpy(), bars["volume"].to_numpy().astype(float)
            for k in kinds:
                if k == "open":
                    m = bn == 0
                    if m.any():
                        out[k][i, j] = o[m][0]
                elif k.startswith("vwap"):
                    nb = int(k.replace("vwap", ""))              # first nb bars (e.g. 3 = 09:15-09:30)
                    m = bn < nb
                    if m.any() and v[m].sum() > 0:
                        out[k][i, j] = float((c[m] * v[m]).sum() / v[m].sum())
                elif k == "bar9":
                    m = bn == 9
                    if m.any():
                        out[k][i, j] = o[m][0]
    return out


def main():
    ctx = make_ctx()
    T, N = ctx.close.shape
    fo = ff.build(ctx)
    ox, cx, _ = ff.exdiv_prices(ctx)
    member = fo["member"]
    lot_hist = ff.ffill_lots(np.where(member, fo["lot_hist"], np.nan))
    pred0 = np.load(DATA / "clean" / "ml_v3_pred_raw.npy")
    pred = np.where(member & np.isfinite(pred0), pred0, np.nan)
    i0 = next(i for i, d in enumerate(ctx.dates) if d.isoformat() >= CLEAN_START)
    fac = {k: v[i0:] for k, v in idx_returns(ctx.dates).items()}
    sl = slice(i0 - 1, T)
    flags = fo["roll_flags"]
    kinds = ["open", "vwap3", "vwap6", "bar9"]
    print("building fill prices from 5-minute data...", flush=True)
    raw = fill_matrices(ctx, kinds)
    # scale 5m adjusted prices to the ex-dividend daily basis (same factor as exdiv_prices applies to open)
    c_adj = ctx.close
    scale = np.where(np.isfinite(c_adj) & (c_adj > 0), cx / c_adj, np.nan)
    rows = []
    for name, cap, k, R, mode in (("L/S k=10 R=5, Rs 5cr", 50_000_000, 10, 5, "ls"),
                                  ("long-only top-20, Rs 1cr", 10_000_000, 20, 5, "long")):
        for kind in kinds:
            px = raw[kind] * scale
            px = np.where(np.isfinite(px), px, ox)               # fall back to daily open if 5m missing

            def run(W, px=px, cap=cap):
                e = simulate_futures(px[sl], cx[sl], W[sl], ctx.dates[sl], lot_hist[sl], capital=cap,
                                     roll_flags=flags[sl]).equity
                return e[1:] / e[:-1] - 1

            def runner(ph, mode=mode, k=k, R=R, run=run):
                if mode == "ls":
                    return run(ls_weights(pred, i0 - ph, R, k, 2 * k))
                return run(build_weights(pred, ctx, i0 - ph, R, "hyst", k=k, keep=2 * k))

            r = run_phases(f"{name} | fill={kind}", runner, list(range(5)), fac).summary()
            rows.append(r)
            print(r, flush=True)
    df = pl.DataFrame(rows, infer_schema_length=None)
    df.write_csv(DATA / "clean" / "exec_test_results.csv")
    with pl.Config(tbl_rows=20, tbl_width_chars=200, fmt_str_lengths=60):
        print(df)


if __name__ == "__main__":
    main()
