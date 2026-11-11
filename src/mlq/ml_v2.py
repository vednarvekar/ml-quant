"""Model v2 (audit steps 3-4): retargeted ranker.

Changes vs ml_swing:
- universe = point-in-time top-100 (data coverage ~96%, not the holes in rank 101-150)
- label options: 'raw' (relative to cross-sectional median) or 'resid' (market-neutral: the stock's
  5-day return minus beta x Nifty 5-day return, then demeaned) -> skill that is not just beta
- new features from the NSE F&O bhavcopy: open-interest build-up, OI relative to its own history,
  futures volume ratio, annualised futures basis; plus beta, return relative to Nifty, size regime
  (Midcap minus Nifty 20-day return), and the old 21 daily features
- walk-forward retrain every 126 days (purged), same as before

uv run python -m mlq.ml_v2            # trains both label types, saves ml_v2_pred_{raw,resid}.npy
"""
import sys

import numpy as np
import polars as pl
from scipy.stats import spearmanr

from mlq import fo_features as ff
from mlq import ml_swing as ms
from mlq.backtest import panel
from mlq.baselines import load_prices
from mlq.config import DATA
from mlq.evaluate import idx_returns
from mlq.swing import make_ctx
from mlq.validation import purged_walk_forward

HOLD = ms.HOLD
UNI = 100


def idx_open_close(name: str, dates: list):
    d = pl.read_parquet(DATA / "clean" / "idx" / "1d" / f"{name}.parquet").select(["date", "open", "close"])
    mo, mc = dict(zip(d["date"].to_list(), d["open"].to_list())), dict(zip(d["date"].to_list(), d["close"].to_list()))
    return (np.array([mo.get(x, np.nan) for x in dates], float), np.array([mc.get(x, np.nan) for x in dates], float))


def rolling_beta(r, rm, n=60):
    """Trailing n-day beta of each stock's daily return r (T,N) on the market return rm (T,)."""
    T, N = r.shape
    out = np.full((T, N), np.nan)
    for t in range(n, T):
        x = rm[t - n + 1:t + 1]
        if np.isnan(x).sum() > n * 0.2:
            continue
        xc = np.where(np.isnan(x), 0.0, x - np.nanmean(x))
        y = r[t - n + 1:t + 1]
        yc = np.where(np.isnan(y), 0.0, y - np.nanmean(y, axis=0))
        out[t] = (xc @ yc) / max(float(xc @ xc), 1e-12)
    return out


def build_v2(ctx, p, fo, use_mto=False):
    ms.UNI = UNI
    f, yd_raw, M, O = ms.build(ctx, p)
    T, N = ctx.close.shape
    C = ctx.close
    r1 = C / ms.shift(C, 1) - 1
    n50_o, n50_c = idx_open_close("Nifty_50", ctx.dates)
    mid_o, mid_c = idx_open_close("NIFTY_MIDCAP_100", ctx.dates)
    rm = np.full(T, np.nan)
    rm[1:] = n50_c[1:] / n50_c[:-1] - 1
    beta = rolling_beta(r1, rm, 60)
    f["beta60"] = beta
    f["rel_ret20"] = f["ret20"] - (n50_c / ms.shift(n50_c[:, None], 20)[:, 0] - 1)[:, None]
    f["size_regime20"] = np.repeat(((mid_c / ms.shift(mid_c[:, None], 20)[:, 0]) -
                                    (n50_c / ms.shift(n50_c[:, None], 20)[:, 0]))[:, None], N, axis=1)
    # futures-derived features
    oi, fv = fo["oi_tot"], fo["fut_vol"]
    f["oi_chg5"] = oi / ms.shift(oi, 5) - 1
    f["oi_rel20"] = oi / ms.roll_mean(ms.shift(oi, 1), 20)
    f["futvol_ratio"] = fv / ms.roll_mean(ms.shift(fv, 1), 20)
    _, _, spot = ff.exdiv_prices(ctx)
    dte = np.where(fo["near_dte"] >= 1, fo["near_dte"], np.nan)
    f["basis_ann"] = (fo["near_settle"] / spot - 1) * 365.0 / dte
    f["oi_x_ret5"] = f["oi_chg5"] * np.sign(f["ret5"])
    if use_mto:                                            # delivery % features (v3)
        dp = ff.mto_matrix(ctx)
        f["deliv_pct"] = dp
        f["deliv_rel20"] = dp / ms.roll_mean(ms.shift(dp, 1), 20)
        f["deliv_chg5"] = dp - ms.shift(dp, 5)
        f["deliv_x_ret1"] = dp * np.sign(f["ret1"])
    # labels: raw (already demeaned) and market-neutral residual
    y_raw = ms.shift(O, -(HOLD + 1)) / ms.shift(O, -1) - 1
    yn = ms.shift(n50_o[:, None], -(HOLD + 1))[:, 0] / ms.shift(n50_o[:, None], -1)[:, 0] - 1
    resid = y_raw - np.where(np.isfinite(beta), beta, 1.0) * yn[:, None]
    med = np.array([np.nanmedian(np.where(M[t], resid[t], np.nan)) if M[t].any() else np.nan for t in range(T)])
    return f, yd_raw, resid - med[:, None], M


def train_wf(X, y, M, T, tag, prefix="ml_v2"):
    pred = np.full((T, X.shape[1]), np.nan)
    for tr, te in purged_walk_forward(T, test_len=126, h=HOLD + 1, embargo=5, min_train=1500):
        m = M[tr] & ~np.isnan(y[tr])
        Xtr, ytr = X[tr][m], y[tr][m]
        ok = ~np.isnan(Xtr).all(axis=1)
        mdl = ms.make_model()
        mdl.fit(Xtr[ok], ytr[ok])
        sel = M[te]
        P = np.full((len(te), X.shape[1]), np.nan)
        P[sel] = mdl.predict(X[te][sel])
        pred[te] = P
    np.save(DATA / "clean" / f"{prefix}_pred_{tag}.npy", pred)
    return pred, mdl


def main():
    p = load_prices()
    ctx = make_ctx()
    fo = ff.build(ctx)
    v3 = len(sys.argv) > 1 and sys.argv[1] == "v3"
    prefix = "ml_v3" if v3 else "ml_v2"
    f, y_raw, y_res, M = build_v2(ctx, p, fo, use_mto=v3)
    names = list(f)
    X = np.stack([f[k] for k in names], axis=-1)
    T, N = ctx.close.shape
    print("features:", len(names), names[-10:], flush=True)
    first_ic_date = "2022-01-03"
    for tag, y in (("raw", y_raw), ("resid", y_res)):
        pred, mdl = train_wf(X, y, M, T, tag, prefix)
        imp = dict(zip(names, mdl.feature_importances_))
        print(f"[{tag}] top features:", sorted(imp, key=imp.get, reverse=True)[:8], flush=True)
        reb = [t for t in ms.week_end_idx(ctx.dates) if not np.all(np.isnan(pred[t]))]
        for lab_name, yy in (("raw-label", y_raw), ("resid-label", y_res)):
            ics = []
            for t in reb:
                ok = np.isfinite(pred[t]) & np.isfinite(yy[t])
                if ok.sum() > 10:
                    ics.append((ctx.dates[t].isoformat(), spearmanr(pred[t][ok], yy[t][ok])[0]))
            for nm, sub in (("2016-2021", [v for d, v in ics if d < first_ic_date]),
                            ("2022+", [v for d, v in ics if d >= first_ic_date])):
                s = np.array(sub)
                print(f"  [{tag}] IC vs {lab_name} {nm}: mean {s.mean():.4f} t={s.mean() / s.std(ddof=1) * np.sqrt(len(s)):.2f}",
                      flush=True)


if __name__ == "__main__":
    main()
