"""Intraday baseline families on the 5-minute engine, plus an anchored walk-forward selection test.

Families (all rule-based, a few params each, every variant logged in the registry):
  orb      opening-range breakout, sticky direction until 15:15
  gapfade  fade the overnight gap
  mom30    continue the first-30-minute direction
  vwapfade fade deviations from running VWAP (exit when back near VWAP)
Signals at bar b use bars <= b only; the engine trades at the next bar's open.

uv run python -m mlq.intraday_strats
"""
import numpy as np
import polars as pl

from mlq import registry
from mlq.config import DATA
from mlq.intraday import SQ, load_panel, simulate
from mlq.validation import purged_walk_forward, sharpe

NSYM = 40


def pick_symbols(n=NSYM):
    u = pl.read_parquet(DATA / "clean" / "universe.parquet")
    last = u["month"].max()
    names = u.filter(pl.col("month") == last).sort("rank")["name"].to_list()
    out = []
    for s in names:
        f = DATA / "clean" / "adj_5m" / f"{s}.parquet"
        if f.exists():
            d = pl.read_parquet(f, columns=["date"])["date"]
            if str(d.min()) <= "2022-02-01" and d.n_unique() > 1000:
                out.append(s)
        if len(out) >= n:
            break
    return out


def prev_close(C):
    last = np.full((C.shape[0], C.shape[2]), np.nan, np.float32)
    for d in range(C.shape[0]):
        for b in range(C.shape[1] - 1, -1, -1):
            v = C[d, b]
            m = np.isnan(last[d]) & ~np.isnan(v)
            last[d, m] = v[m]
            if not np.isnan(last[d]).any():
                break
    pc = np.full_like(last, np.nan)
    pc[1:] = last[:-1]
    return pc


def sig_orb(a, k=3, buf_bps=0.0):
    H, L, C = a["h"], a["l"], a["c"]
    D, B, N = C.shape
    hi, lo = np.nanmax(H[:, :k], axis=1), np.nanmin(L[:, :k], axis=1)
    P = np.zeros((D, B, N), np.float32)
    S = np.zeros((D, B, N), np.float32)
    state = np.zeros((D, N), np.float32)
    rng = np.maximum(hi - lo, 1e-9)
    for b in range(k - 1, SQ - 1):
        up = (C[:, b] > hi * (1 + buf_bps / 1e4)) & (state == 0)
        dn = (C[:, b] < lo * (1 - buf_bps / 1e4)) & (state == 0)
        state = np.where(up, 1, np.where(dn, -1, state))
        P[:, b] = state
        S[:, b] = np.abs(C[:, b] - np.where(state > 0, hi, lo)) / rng + 1e-6
    return P, S


def sig_gap(a, thr=0.005, entry_bar=0):
    C, O = a["c"], a["o"]
    D, B, N = C.shape
    gap = O[:, 0] / prev_close(C) - 1
    P = np.zeros((D, B, N), np.float32)
    S = np.zeros((D, B, N), np.float32)
    sig = np.where(gap > thr, -1.0, np.where(gap < -thr, 1.0, 0.0))
    P[:, entry_bar:SQ - 1] = sig[:, None, :]
    S[:, entry_bar:SQ - 1] = np.abs(np.nan_to_num(gap))[:, None, :]
    return P, S


def sig_mom30(a, bars=6, thr=0.004):
    C, O = a["c"], a["o"]
    D, B, N = C.shape
    r = C[:, bars - 1] / O[:, 0] - 1
    P = np.zeros((D, B, N), np.float32)
    S = np.zeros((D, B, N), np.float32)
    sig = np.where(r > thr, 1.0, np.where(r < -thr, -1.0, 0.0))
    P[:, bars - 1:SQ - 1] = sig[:, None, :]
    S[:, bars - 1:SQ - 1] = np.abs(np.nan_to_num(r))[:, None, :]
    return P, S


def sig_vwap(a, thr_bps=40.0, exit_frac=0.25, start=6):
    C, V = a["c"], np.nan_to_num(a["v"])
    D, B, N = C.shape
    pv = np.cumsum(np.nan_to_num(C) * V, axis=1)
    vv = np.maximum(np.cumsum(V, axis=1), 1)
    vwap = pv / vv
    dev = (C / vwap - 1) * 1e4
    P = np.zeros((D, B, N), np.float32)
    S = np.zeros((D, B, N), np.float32)
    state = np.zeros((D, N), np.float32)
    for b in range(start, SQ - 1):
        d = np.nan_to_num(dev[:, b])
        enter = (state == 0) & (np.abs(d) > thr_bps)
        state = np.where(enter, -np.sign(d), state)
        state = np.where((state != 0) & (np.abs(d) < thr_bps * exit_frac), 0, state)
        P[:, b] = state
        S[:, b] = np.abs(d)
    return P, S


GRID = (
    [("orb", dict(k=k, buf_bps=b), sig_orb) for k in (3, 6) for b in (0.0, 10.0)]
    + [("gapfade", dict(thr=t), sig_gap) for t in (0.003, 0.006, 0.01)]
    + [("mom30", dict(bars=n, thr=t), sig_mom30) for n in (6, 12) for t in (0.003, 0.006)]
    + [("vwapfade", dict(thr_bps=t), sig_vwap) for t in (30.0, 60.0, 100.0)]
)


def main():
    syms = pick_symbols()
    print(f"{len(syms)} symbols; loading 5m panel...", flush=True)
    dates, a = load_panel(syms)
    O = a["o"]
    print(f"panel {O.shape} (days x bars x syms)", flush=True)
    rets, rows = {}, []
    for fam, params, fn in GRID:
        P, S = fn(a, **params)
        r = simulate(O, P, S)
        m = r.metrics()
        key = f"{fam} {params}"
        rets[key] = r.returns()
        registry.log_trial(key, "intraday", params | {"family": fam}, r.returns(),
                           {"trades": m["trades"], "net_bps_per_trade": m["net_bps_per_trade"]})
        gross_bps = (m["gross_pnl"] / max(m["trades"], 1)) / (r.notional / 1e4)
        rows.append({"variant": key, "trades": m["trades"], "gross_bps/trade": round(gross_bps, 1),
                     "net_bps/trade": round(m["net_bps_per_trade"], 1), "sharpe": round(m["sharpe"], 2),
                     "ret/yr": round(m["ret_per_year"], 3), "max_dd": round(m["max_dd"], 3)})
        print(rows[-1], flush=True)
    out = pl.DataFrame(rows)
    out.write_csv(DATA / "clean" / "intraday_results.csv")

    # anchored walk-forward: choose best variant on the past, trade it on the next block
    keys = list(rets)
    M = np.vstack([rets[k] for k in keys])
    T = M.shape[1]
    oos, picks = [], []
    for tr, te in purged_walk_forward(T, test_len=126, h=1, embargo=1, min_train=252):
        best = int(np.argmax([sharpe(M[i, tr]) for i in range(len(keys))]))
        oos.append(M[best, te])
        picks.append(keys[best])
    oos = np.concatenate(oos)
    print("\nWalk-forward (pick best variant on past, trade next 6 months):")
    print("  OOS days", len(oos), "annualised Sharpe", round(sharpe(oos) * np.sqrt(252), 2),
          "mean/yr", round(oos.mean() * 252, 3))
    print("  DSR:", round(registry.dsr(oos, "intraday"), 2), "after", registry.n_trials("intraday"), "trials")
    print("  picks:", picks)
    sensitivity(a, O)


def sensitivity(a, O):
    """What would it take? Vary slippage and #positions on the two best-gross rules, and try
    following VWAP deviations instead of fading them."""
    print("\nSensitivity (net bps/trade | Sharpe):")
    cases = {"mom30 b6": sig_mom30(a, bars=6, thr=0.003), "gapfade 1%": sig_gap(a, thr=0.01)}
    pv, sv = sig_vwap(a, thr_bps=60.0)
    cases["vwap FOLLOW 60"] = (-pv, sv)
    rows = []
    for name, (P, S) in cases.items():
        for K in (1, 2, 5):
            for sl in (0.0, 1.0, 2.0, 5.0):
                r = simulate(O, P, S, K=K, slippage_bps=sl)
                m = r.metrics()
                registry.log_trial(f"{name} K{K} slip{sl}", "intraday", {"K": K, "slip": sl}, r.returns())
                rows.append({"rule": name, "K": K, "slip_bps": sl, "trades": m["trades"],
                             "gross_bps": round((m["gross_pnl"] / max(m["trades"], 1)) / (r.notional / 1e4), 1),
                             "net_bps": round(float(m["net_bps_per_trade"]), 1), "sharpe": round(m["sharpe"], 2)})
    out = pl.DataFrame(rows)
    out.write_csv(DATA / "clean" / "intraday_sensitivity.csv")
    with pl.Config(tbl_rows=40, tbl_width_chars=120):
        print(out)


if __name__ == "__main__":
    main()
