"""Real-futures fill test on PAST days (no one has to be at the screen).

The backtest and the paper trader fill at the cash-market open + 5 bps. The strategy trades stock
futures, and its result is very sensitive to fill cost (5 bps -> Sharpe ~0.75, 10 bps -> ~0.35,
15 bps -> below 0). This module measures, from archived 1-minute FUTURES candles:

  1. liquidity at the open: how much trades in the first 1 / 5 / 15 minutes vs our order size
  2. spread proxy (Roll estimator on 1-minute closes) and a rough square-root impact estimate
  3. delay cost for the model's actual picks: futures 09:15 open -> VWAP of the first 5 / 15 minutes,
     signed by trade direction (what we lose if we cannot get the opening print)
  4. day-1 edge of the picks measured on futures vs on cash (does the edge exist in the futures?)

Nothing is fitted or tuned here. Needs: python -m mlq.upstox.fut_minutes (run daily by paper.update).

uv run python -m mlq.fut_fill [order_rupees]        # default order = Rs 25 lakh per name
"""
import sys
from datetime import date, time, timedelta

import numpy as np
import polars as pl

from mlq.config import DATA
from mlq.upstox.fut_minutes import ROOT

OPEN_T = time(9, 15)


def day_table(f) -> pl.DataFrame:
    """One row per trading day of one contract file: opening print, early VWAPs, volumes, spread proxy."""
    sym, exp = f.stem.rsplit("_", 1)
    d = pl.read_parquet(f).with_columns(
        pl.col("ts").dt.date().alias("date"),
        ((pl.col("ts").dt.hour().cast(pl.Int32) * 60 + pl.col("ts").dt.minute().cast(pl.Int32)) - 555).alias("m"),
        ((pl.col("high") + pl.col("low") + pl.col("close")) / 3).alias("tp"))
    d = d.filter((pl.col("m") >= 0) & (pl.col("m") < 375)).sort("ts")
    if d.is_empty():
        return pl.DataFrame()

    def vwap(n):
        w = pl.col("m") < n
        return ((pl.col("tp") * pl.col("volume")).filter(w).sum() / pl.col("volume").filter(w).sum()).alias(f"vwap{n}")

    def vol(n):
        return (pl.col("close") * pl.col("volume")).filter(pl.col("m") < n).sum().alias(f"rs{n}")

    early = pl.col("m") < 30
    dp = pl.col("close").filter(early).log().diff()
    g = d.group_by("date", maintain_order=True).agg(
        pl.col("m").first().alias("first_m"), pl.col("open").first().alias("f_open"),
        pl.col("volume").first().alias("vol_first"), pl.col("close").last().alias("f_close"),
        vwap(1), vwap(5), vwap(15), vol(1), vol(5), vol(15),
        (pl.col("close") * pl.col("volume")).sum().alias("rs_day"),
        pl.cov(dp.slice(1), dp.shift(1).slice(1)).alias("roll_cov"),
        dp.std().alias("sd1m"))
    return g.with_columns(pl.lit(sym).alias("sym"), pl.lit(exp).str.to_date().alias("expiry"))


def load_days() -> pl.DataFrame:
    fs = sorted((ROOT / "1m").glob("*.parquet"))
    if not fs:
        sys.exit("no futures minute data: run  uv run python -m mlq.upstox.fut_minutes")
    t = pl.concat([x for x in (day_table(f) for f in fs) if x.height], how="vertical_relaxed")
    # contract role on each day: near = the first monthly expiry on/after that day
    fo_exp = set()
    lo = t["date"].min() - timedelta(days=5)
    for f in sorted((DATA / "raw" / "nse_fo").glob("*.parquet")):
        if f.stem[:10] >= lo.isoformat():
            e = pl.read_parquet(f, columns=["kind", "expiry"]).filter(pl.col("kind") == "S")["expiry"].unique().to_list()
            fo_exp |= {x for x in e if x is not None}
    exps = sorted(fo_exp | set(t["expiry"].unique().to_list()))
    near = {d: next((e for e in exps if e >= d), None) for d in t["date"].unique().to_list()}
    return t.with_columns(pl.col("date").replace_strict(near, return_dtype=pl.Date).alias("near_exp")) \
        .with_columns(pl.when(pl.col("expiry") == pl.col("near_exp")).then(pl.lit("near")).otherwise(pl.lit("next")).alias("role"),
                      (pl.col("expiry") - pl.col("date")).dt.total_days().alias("dte"))


def scores(ctx) -> np.ndarray:
    """Walk-forward v3 predictions; days after the saved file are scored by the paper model
    (trained only on earlier data). Row t = score known at the close of day t."""
    from lightgbm import Booster

    from mlq import fo_features as ff
    from mlq.baselines import load_prices
    from mlq.paper import signals
    T, N = ctx.close.shape
    pred = np.load(DATA / "clean" / "ml_v3_pred_raw.npy")
    if pred.shape[0] < T:
        pred = np.vstack([pred, np.full((T - pred.shape[0], N), np.nan)])
    last = int(np.nonzero(np.isfinite(pred).any(axis=1))[0].max())
    tag = signals.latest_model_tag()
    if tag and last < T - 1:
        X, _, M, names = signals.features(ctx, load_prices())
        booster, meta = signals.load_model(tag)
        if meta["features"] == names:
            for t in range(last + 1, T):
                if str(ctx.dates[t]) <= meta["trained_through"]:
                    continue                                   # only days the model has not seen
                sel = M[t] & ~np.isnan(X[t]).all(axis=1)
                if sel.any():
                    pred[t, sel] = booster.predict(X[t][sel])
    member = ff.build(ctx)["member"]
    return np.where(member, pred, np.nan)


def tstat(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return (x.mean() * 1e4, x.mean() / x.std(ddof=1) * np.sqrt(len(x)), len(x)) if len(x) > 2 else (np.nan, np.nan, len(x))


def main(order_rs: float = 2_500_000.0, k: int = 10):
    from mlq.backtest import panel
    from mlq.baselines import load_prices
    from mlq.swing import make_ctx
    t = load_days()
    print(f"futures minute data: {t['date'].min()} -> {t['date'].max()}, {t['sym'].n_unique()} stocks, "
          f"{t.height} contract-days", flush=True)
    rows = []
    # ---- 1 + 2: liquidity and cost proxies, all stocks, by contract role -----------------------
    t = t.with_columns(
        (2 * (-pl.col("roll_cov")).clip(lower_bound=0).sqrt() / 2 * 1e4).alias("half_spread_bps"),
        (0.5 * pl.col("sd1m") * np.sqrt(5) * (order_rs / pl.col("rs5")).sqrt() * 1e4).alias("impact5_bps"),
        (order_rs / pl.col("rs5")).alias("part5"), (order_rs / pl.col("rs15")).alias("part15"))
    print(f"\n== Liquidity at the open (order = Rs {order_rs / 1e5:.0f} lakh per name) ==")
    for role in ("near", "next"):
        for lab, cond in (("all days", pl.lit(True)), ("last 5 days before expiry", pl.col("dte") <= 7)):
            if role == "next" and lab != "last 5 days before expiry":
                cond = pl.col("dte") > 35                        # an ordinary day for a next-month contract
                lab = "ordinary days"
            s = t.filter((pl.col("role") == role) & cond)
            if s.is_empty():
                continue
            q = lambda c, p: float(s[c].drop_nulls().quantile(p))                    # noqa: E731
            r = {"contract": role, "days": lab, "stock_days": s.height,
                 "opens_at_0915_pct": round(float((s["first_m"] == 0).mean()) * 100, 1),
                 "rs_lakh_first_min_median": round(q("rs1", 0.5) / 1e5, 1),
                 "rs_lakh_first_5min_median": round(q("rs5", 0.5) / 1e5, 1),
                 "rs_lakh_first_5min_p10": round(q("rs5", 0.1) / 1e5, 1),
                 "order_pct_of_5min_median": round(q("part5", 0.5) * 100, 1),
                 "order_pct_of_5min_p90": round(q("part5", 0.9) * 100, 1),
                 "half_spread_bps_median": round(q("half_spread_bps", 0.5), 1),
                 "impact_5min_bps_median": round(q("impact5_bps", 0.5), 1),
                 "impact_5min_bps_p90": round(q("impact5_bps", 0.9), 1)}
            rows.append(r)
            print(r, flush=True)
    # ---- 3 + 4: the model's own picks ------------------------------------------------------------
    ctx = make_ctx()
    _, _, O = panel(load_prices(), "open")
    sc = scores(ctx)
    di = {d: i for i, d in enumerate(ctx.dates)}
    cj = {s: j for j, s in enumerate(ctx.syms)}
    print(f"\n== The model's picks (top-{k} long / bottom-{k} short, decided the evening before) ==")
    for role in ("near", "next"):
        s = t.filter(pl.col("role") == role)
        acc = {x: [] for x in ("d5", "d15", "fut_d1", "cash_d1", "imp", "hs")}
        ndays = 0
        for (d,), g in s.group_by("date"):
            i = di.get(d)
            if i is None or i == 0:
                continue
            have = {r["sym"]: r for r in g.iter_rows(named=True) if r["sym"] in cj and r["first_m"] == 0}
            cand = [(sc[i - 1, cj[x]], x) for x in have if np.isfinite(sc[i - 1, cj[x]])]
            if len(cand) < 4 * k:
                continue
            cand.sort()
            ndays += 1
            for sign, names in ((1, [x for _, x in cand[-k:]]), (-1, [x for _, x in cand[:k]])):
                for x in names:
                    r, j = have[x], cj[x]
                    acc["d5"].append(sign * (r["vwap5"] / r["f_open"] - 1))
                    acc["d15"].append(sign * (r["vwap15"] / r["f_open"] - 1))
                    acc["fut_d1"].append(sign * (r["f_close"] / r["f_open"] - 1))
                    acc["cash_d1"].append(sign * (ctx.close[i, j] / O[i, j] - 1))
                    acc["imp"].append(r["impact5_bps"] / 1e4 if r["impact5_bps"] is not None else np.nan)
                    acc["hs"].append(r["half_spread_bps"] / 1e4 if r["half_spread_bps"] is not None else np.nan)
        if not ndays:
            continue
        out = {"contract": role, "days": f"picks on {ndays} days"}
        for key, lab in (("d5", "delay_cost_open_to_vwap5_bps"), ("d15", "delay_cost_open_to_vwap15_bps"),
                         ("fut_d1", "day1_edge_futures_bps"), ("cash_d1", "day1_edge_cash_bps"),
                         ("hs", "half_spread_bps_picks"), ("imp", "impact_5min_bps_picks")):
            m, tt, n = tstat(acc[key])
            out[lab] = round(m, 1)
            if key in ("d5", "d15", "fut_d1", "cash_d1"):
                out[lab + "_t"] = round(tt, 1)
        out["n_trades"] = len(acc["d5"])
        rows.append(out)
        print(out, flush=True)
    pl.DataFrame(rows, infer_schema_length=None).write_csv(DATA / "clean" / "fut_fill_results.csv")
    print("\nsaved data/clean/fut_fill_results.csv")
    print("Reading: all-in cost per fill ~ half spread + impact (+ delay cost if the open cannot be had).\n"
          "The strategy needs about 5 bps or less per fill; at 10 bps most of the edge is gone.")


if __name__ == "__main__":
    main(float(sys.argv[1]) if len(sys.argv) > 1 else 2_500_000.0)
