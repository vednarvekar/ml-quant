"""Phase 4, experiment 2: ML cross-sectional ranking for swing (LightGBM, walk-forward).

Every day, for each stock in the point-in-time universe, features known at the close predict the
return from the next open to 6 days later (relative to the universe median). We rebalance weekly:
long the top-k, short the bottom-k (stock futures, F&O names) or long-only top-k (delivery).
Walk-forward: retrain every ~6 months on the past only (purged 6 days + embargo).

uv run python -m mlq.ml_swing
"""
import numpy as np
import polars as pl
from lightgbm import LGBMRegressor
from numpy.lib.stride_tricks import sliding_window_view as swv
from scipy.stats import spearmanr

from mlq import registry
from mlq.backtest import panel, simulate
from mlq.baselines import load_prices
from mlq.config import DATA
from mlq.swing import make_ctx, week_end_idx
from mlq.validation import purged_walk_forward

HOLD = 5          # label: open[t+1] -> open[t+1+HOLD]
UNI = 150         # point-in-time liquidity universe size


def shift(x, k):
    """shift(x, k)[t] = x[t-k]; negative k looks forward (labels only)."""
    out = np.full_like(x, np.nan)
    if k > 0:
        out[k:] = x[:-k]
    elif k < 0:
        out[:k] = x[-k:]
    else:
        out[:] = x
    return out


def roll_std(r, n):
    w = swv(np.vstack([np.full((n - 1, r.shape[1]), np.nan), r]), n, axis=0)
    return np.nanstd(w, axis=-1)


def roll_mean(x, n):
    w = swv(np.vstack([np.full((n - 1, x.shape[1]), np.nan), x]), n, axis=0)
    return np.nanmean(w, axis=-1)


def rank_pct(x, mask):
    xx = np.where(mask & ~np.isnan(x), x, np.inf)
    r = np.argsort(np.argsort(xx, axis=1), axis=1).astype(float)
    n = (mask & ~np.isnan(x)).sum(axis=1, keepdims=True)
    out = r / np.maximum(n - 1, 1)
    out[~(mask & ~np.isnan(x))] = np.nan
    return out


def build(ctx, p):
    _, _, O = panel(p, "open")
    _, _, H = panel(p, "high")
    _, _, L = panel(p, "low")
    _, _, V = panel(p, "volume")
    C = ctx.close
    T, N = C.shape
    M = np.zeros((T, N), bool)
    for t in range(T):
        M[t] = ctx.members(t, UNI)
    r1 = C / shift(C, 1) - 1
    f = {}
    for k in (1, 5, 10, 20, 60):
        f[f"ret{k}"] = C / shift(C, k) - 1
    f["mom12_1"] = shift(C, 21) / shift(C, 252) - 1
    f["vol20"] = roll_std(r1, 20)
    f["vol60"] = roll_std(r1, 60)
    f["range20"] = roll_mean((H - L) / C, 20)
    v = V.astype(float)
    f["volratio"] = v / roll_mean(shift(v, 1), 20)
    f["dist_hi252"] = C / np.fmax.reduce(swv(np.vstack([np.full((251, N), np.nan), C]), 252, axis=0), axis=-1) - 1
    f["gap"] = O / shift(C, 1) - 1
    f["close_pos"] = (C - L) / np.maximum(H - L, 1e-9)
    idx = ctx.idx_close[:, None]
    f["mkt_ret20"] = np.repeat(idx / shift(idx, 20) - 1, N, axis=1)
    for k in ("ret5", "ret20", "ret60", "vol20", "mom12_1"):
        f[k + "_rk"] = rank_pct(f[k], M)
    vx = vix(ctx.dates)
    f["vix"] = np.repeat(vx[:, None], N, axis=1)
    f["vix_chg5"] = np.repeat((vx / shift(vx[:, None], 5)[:, 0] - 1)[:, None], N, axis=1)
    y = shift(O, -(HOLD + 1)) / shift(O, -1) - 1                 # open[t+1] -> open[t+1+HOLD]
    ymed = np.array([np.nanmedian(np.where(M[t], y[t], np.nan)) if M[t].any() else np.nan for t in range(T)])
    yd = y - ymed[:, None]
    return f, yd, M, O


def vix(dates):
    v = pl.read_parquet(DATA / "clean" / "idx" / "1d" / "India_VIX.parquet").select(["date", "close"])
    m = dict(zip(v["date"].to_list(), v["close"].to_list()))
    return np.array([m.get(d, np.nan) for d in dates], float)


def main():
    p = load_prices()
    ctx = make_ctx()
    f, yd, M, O = build(ctx, p)
    names = list(f)
    T, N = ctx.close.shape
    X = np.stack([f[k] for k in names], axis=-1)
    print("features:", len(names), flush=True)
    pred = np.full((T, N), np.nan)
    folds = list(purged_walk_forward(T, test_len=126, h=HOLD + 1, embargo=5, min_train=1500))
    print("folds:", len(folds), "OOS starts", ctx.dates[folds[0][1][0]], flush=True)
    for tr, te in folds:
        m = M[tr] & ~np.isnan(yd[tr])
        Xtr, ytr = X[tr][m], yd[tr][m]
        ok = ~np.isnan(Xtr).all(axis=1)
        mdl = LGBMRegressor(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=500,
                            subsample=0.7, subsample_freq=1, colsample_bytree=0.7, reg_lambda=10.0,
                            objective="huber", alpha=0.05, verbose=-1, n_jobs=4)
        mdl.fit(Xtr[ok], ytr[ok])
        sel = M[te]
        P = np.full((len(te), N), np.nan)
        Xt = X[te]
        P[sel] = mdl.predict(Xt[sel])
        pred[te] = P
    imp = dict(zip(names, mdl.feature_importances_))
    print("top features:", sorted(imp, key=imp.get, reverse=True)[:8], flush=True)

    reb = [t for t in week_end_idx(ctx.dates) if not np.all(np.isnan(pred[t]))]
    ic = np.array([spearmanr(pred[t][~np.isnan(pred[t]) & ~np.isnan(yd[t])],
                             yd[t][~np.isnan(pred[t]) & ~np.isnan(yd[t])])[0] for t in reb])
    split = next(i for i, t in enumerate(reb) if ctx.dates[t].isoformat() >= "2022-01-01")
    for nm, s in (("2016-2021 (survivor-biased)", ic[:split]), ("2022+ (clean)", ic[split:])):
        print(f"weekly rank IC {nm}: mean {s.mean():.4f}  t={s.mean() / s.std(ddof=1) * np.sqrt(len(s)):.2f}  n={len(s)}")

    first = folds[0][1][0]
    fno = ctx.fno
    rows = []
    for mode, k in (("L/S futures", 10), ("L/S futures", 20), ("Long-only delivery", 20)):
        W = np.zeros((T, N))
        cur = np.zeros(N)
        rebs = set(week_end_idx(ctx.dates))
        for t in range(first - 1, T):
            if t in rebs and not np.all(np.isnan(pred[t])):
                sc = np.where(np.isnan(pred[t]), np.nan, pred[t])
                cur = np.zeros(N)
                if mode == "L/S futures":
                    sc2 = np.where(fno, sc, np.nan)
                    idx = [j for j in np.argsort(np.where(np.isnan(sc2), -np.inf, sc2)) if not np.isnan(sc2[j])]
                    if len(idx) >= 2 * k:
                        cur[idx[-k:]] = 0.5 / k
                        cur[idx[:k]] = -0.5 / k
                else:
                    idx = [j for j in np.argsort(np.where(np.isnan(sc), -np.inf, sc)) if not np.isnan(sc[j])]
                    if len(idx) >= k:
                        cur[idx[-k:]] = 1.0 / k
            W[t] = cur
        kind = "futures" if mode.startswith("L/S") else "delivery"
        for lo_, hi_, tag in ((first, T, "all OOS"), (next(i for i, d in enumerate(ctx.dates) if d.isoformat() >= "2022-01-03"), T, "2022+ clean")):
            for sl in (5.0, 2.0):
                sli = slice(max(lo_ - 1, 0), hi_)
                r = simulate(O[sli], ctx.close[sli], W[sli], ctx.dates[sli], kind=kind, slippage_bps=sl)
                m = r.metrics()
                e = r.equity["equity"].to_numpy()
                dly = e[1:] / e[:-1] - 1
                registry.log_trial(f"ML swing {mode} k{k} {tag} slip{sl}", "swing-ml", {"k": k, "mode": mode}, dly)
                rows.append({"mode": mode, "k": k, "period": tag, "slip": sl, "cagr": round(m["cagr"], 3),
                             "sharpe": round(m["sharpe"], 2), "max_dd": round(m["max_dd"], 3),
                             "turn/yr": round(m["turnover_per_year"], 1), "fees_k": round(m["fees"] / 1e3),
                             "dsr": round(registry.dsr(dly, "swing-ml"), 2)})
    out = pl.DataFrame(rows)
    out.write_csv(DATA / "clean" / "ml_swing_results.csv")
    with pl.Config(tbl_rows=30, tbl_width_chars=150):
        print(out)


if __name__ == "__main__":
    main()
