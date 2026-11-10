"""Phase 4, experiment 1: ML trade-ranking for intraday (LightGBM, walk-forward).

At 09:45 (close of bar 5) each stock gets features known by then; the model predicts the stock's
return relative to the day's median from the 09:50 open (bar 6) to the 15:15 open (bar 72).
Each day we go long the top-k and short the bottom-k predictions. The model is retrained
every ~3 months on the past only (purged by 1 day); only out-of-sample days are traded.

uv run python -m mlq.ml_intraday
"""
import numpy as np
import polars as pl
from lightgbm import LGBMRegressor
from scipy.stats import spearmanr

from mlq import registry
from mlq.config import DATA
from mlq.intraday import SQ, load_panel, simulate
from mlq.intraday_strats import pick_symbols
from mlq.validation import purged_walk_forward, sharpe

ENTRY = 5      # decide at close of this bar, trade next open (bar 6)


def last_close(C):
    D, B, N = C.shape
    out = np.full((D, N), np.nan, np.float32)
    for b in range(B - 1, -1, -1):
        m = np.isnan(out) & ~np.isnan(C[:, b])
        out[m] = C[:, b][m]
    return out


def shift(x, k):
    out = np.full_like(x, np.nan)
    out[k:] = x[:-k]
    return out


def vix_series(dates):
    v = pl.read_parquet(DATA / "clean" / "idx" / "1d" / "India_VIX.parquet").select(["date", "close"])
    m = dict(zip(v["date"].to_list(), v["close"].to_list()))
    return np.array([m.get(d, np.nan) for d in dates], float)


def features(a, dates):
    O, H, L, C, V = a["o"], a["h"], a["l"], a["c"], a["v"]
    D, B, N = C.shape
    lc = last_close(C)
    pc, pc2, pc5 = shift(lc, 1), shift(lc, 2), shift(lc, 5)
    pc20 = shift(lc, 20)
    o0, c5 = O[:, 0], C[:, ENTRY]
    hi = np.nanmax(H[:, :ENTRY + 1], axis=1)
    lo = np.nanmin(L[:, :ENTRY + 1], axis=1)
    v30 = np.nansum(V[:, :ENTRY + 1], axis=1).astype(float)
    v30[v30 == 0] = np.nan
    vavg = np.full_like(v30, np.nan)
    for d in range(20, D):
        vavg[d] = np.nanmean(v30[d - 20:d], axis=0)
    # previous day full-day volatility proxy (high-low of 5m bars)
    dh, dl = np.nanmax(H, axis=1), np.nanmin(L, axis=1)
    prange = shift((dh - dl) / lc, 1)
    f = {
        "gap": o0 / pc - 1,
        "r30": c5 / o0 - 1,
        "range30": (hi - lo) / o0,
        "pos30": (c5 - lo) / np.maximum(hi - lo, 1e-9),
        "volratio": v30 / vavg,
        "prev_ret1": pc / pc2 - 1,
        "prev_ret5": pc / pc5 - 1,
        "prev_ret20": pc / pc20 - 1,
        "prev_range": prange,
        "last_bar_ret": c5 / C[:, ENTRY - 1] - 1,
    }
    # cross-sectional / market context
    for k in ("gap", "r30"):
        x = f[k]
        f[k + "_mkt"] = np.repeat(np.nanmedian(x, axis=1, keepdims=True), N, axis=1)
        f[k + "_rel"] = x - f[k + "_mkt"]
        r = np.argsort(np.argsort(np.where(np.isnan(x), np.inf, x), axis=1), axis=1).astype(float)
        f[k + "_rank"] = r / max(N - 1, 1)
    vx = vix_series(dates)
    vprev = np.concatenate([[np.nan], vx[:-1]])
    f["vix"] = np.repeat(vprev[:, None], N, axis=1)
    f["vix_chg"] = np.repeat((vprev / np.concatenate([[np.nan, np.nan], vx[:-2]]) - 1)[:, None], N, axis=1)
    f["dow"] = np.repeat(np.array([d.weekday() for d in dates], float)[:, None], N, axis=1)
    y = O[:, SQ] / O[:, ENTRY + 1] - 1
    yd = y - np.nanmedian(y, axis=1, keepdims=True)
    return f, yd, y


def main():
    syms = pick_symbols(80)
    print(len(syms), "symbols; loading...", flush=True)
    dates, a = load_panel(syms)
    O = a["o"]
    D, B, N = O.shape
    f, yd, y = features(a, dates)
    names = list(f)
    X = np.stack([f[k] for k in names], axis=-1)            # D,N,F
    print("features", names, flush=True)

    pred = np.full((D, N), np.nan)
    oos_days = []
    for tr, te in purged_walk_forward(D, test_len=63, h=1, embargo=1, min_train=252):
        Xtr, ytr = X[tr].reshape(-1, X.shape[-1]), yd[tr].reshape(-1)
        ok = ~np.isnan(ytr)
        mdl = LGBMRegressor(n_estimators=250, learning_rate=0.03, num_leaves=15, min_child_samples=300,
                            subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=5.0,
                            objective="huber", alpha=0.02, verbose=-1, n_jobs=4)
        mdl.fit(Xtr[ok], ytr[ok])
        pred[te] = mdl.predict(X[te].reshape(-1, X.shape[-1])).reshape(len(te), N)
        oos_days.extend(te.tolist())
    oos = np.array(sorted(oos_days))
    imp = dict(zip(names, mdl.feature_importances_))
    print("OOS days:", len(oos), "| top features (last model):",
          sorted(imp, key=imp.get, reverse=True)[:6], flush=True)

    ics = []
    for d in oos:
        m = ~np.isnan(pred[d]) & ~np.isnan(yd[d])
        if m.sum() > 10:
            ics.append(spearmanr(pred[d][m], yd[d][m])[0])
    ics = np.array(ics)
    print(f"daily rank IC mean {ics.mean():.4f}  (t-stat {ics.mean() / ics.std(ddof=1) * np.sqrt(len(ics)):.2f})",
          flush=True)

    rows = []
    for k in (1, 2, 3):
        P = np.zeros((len(oos), B, N), np.float32)
        S = np.zeros((len(oos), B, N), np.float32)
        for i, d in enumerate(oos):
            p = np.where(np.isnan(pred[d]), 0, pred[d])
            ok = ~np.isnan(O[d, ENTRY + 1]) & ~np.isnan(O[d, SQ])
            p = np.where(ok, p, 0)
            order = np.argsort(p)
            for j in order[-k:]:
                if p[j] > 0:
                    P[i, ENTRY:SQ - 1, j] = 1
                    S[i, ENTRY:SQ - 1, j] = abs(p[j])
            for j in order[:k]:
                if p[j] < 0:
                    P[i, ENTRY:SQ - 1, j] = -1
                    S[i, ENTRY:SQ - 1, j] = abs(p[j])
        for sl in (5.0, 2.0, 1.0):
            r = simulate(O[oos], P, S, K=2 * k, slippage_bps=sl)
            m = r.metrics()
            registry.log_trial(f"LGBM intraday k{k} slip{sl}", "intraday-ml", {"k": k, "slip": sl},
                               r.returns(), {"ic": float(ics.mean())})
            rows.append({"k_each_side": k, "slip_bps": sl, "trades": m["trades"],
                         "gross_bps": round((m["gross_pnl"] / max(m["trades"], 1)) / (r.notional / 1e4), 1),
                         "net_bps": round(float(m["net_bps_per_trade"]), 1),
                         "sharpe": round(m["sharpe"], 2), "ret/yr": round(m["ret_per_year"], 3),
                         "max_dd": round(m["max_dd"], 3),
                         "dsr": round(registry.dsr(r.returns(), "intraday"), 2)})
    out = pl.DataFrame(rows)
    out.write_csv(DATA / "clean" / "ml_intraday_results.csv")
    with pl.Config(tbl_rows=20, tbl_width_chars=140):
        print(out)


if __name__ == "__main__":
    main()
