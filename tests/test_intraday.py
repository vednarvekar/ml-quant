import numpy as np
import pytest

from mlq.intraday import SQ, simulate


def walk(D=120, N=10, seed=0, vol=0.001):
    rng = np.random.default_rng(seed)
    return 100 * np.exp(np.cumsum(rng.normal(0, vol, (D, 75, N)), axis=1))


def test_executes_next_bar_open_and_closes_at_square_off():
    O = np.full((1, 75, 1), 100.0)
    O[0, 12:, 0] = 101.0                      # price steps up at bar 12
    P = np.zeros((1, 75, 1))
    P[0, 10, 0] = 1                           # signal at close of bar 10 -> buy at open of bar 11 (100)
    P[0, 11:, 0] = 1                          # keep long until forced flat
    r = simulate(O, P, slippage_bps=0)
    assert r.trades == 1
    assert r.gross == pytest.approx(r.notional * 0.01, rel=1e-6)   # 100 -> 101, exit at bar SQ open
    assert r.daily[0] < r.gross               # fees reduce it


def test_short_profits_when_price_falls_and_is_flat_by_close():
    O = np.full((1, 75, 1), 100.0)
    O[0, 12:, 0] = 99.0
    P = np.zeros((1, 75, 1))
    P[0, 10:, 0] = -1
    r = simulate(O, P, slippage_bps=0)
    assert r.gross == pytest.approx(r.notional * 0.01, rel=1e-6)
    # after the square-off bar there is no exposure: a move after SQ must not count
    O2 = np.full((1, 75, 1), 100.0)
    O2[0, SQ + 1:, 0] = 50.0
    r2 = simulate(O2, P, slippage_bps=0)
    assert r2.gross == pytest.approx(0.0, abs=1e-6)


def test_max_concurrent_positions():
    O = np.full((1, 75, 8), 100.0)
    P = np.zeros((1, 75, 8))
    P[0, 10:, :] = 1
    r = simulate(O, P, K=3, slippage_bps=0)
    assert r.trades == 3


def test_null_random_signals_lose_about_costs():
    O = walk()
    rng = np.random.default_rng(1)
    P = rng.choice([-1, 0, 1], size=O.shape, p=[0.02, 0.96, 0.02]).astype(float)
    m = simulate(O, P).metrics()
    assert m["net_pnl"] < 0 and m["fees"] > 0 and m["slippage"] > 0
    assert m["sharpe"] < 0.5


def test_no_lookahead_legit_vs_cheat():
    O = walk(seed=2, vol=0.002)
    D, Bn, N = O.shape
    ret = np.zeros_like(O)
    ret[:, 1:] = O[:, 1:] / O[:, :-1] - 1             # open-to-open return into bar b
    legit = np.where(ret > 0.002, 1.0, 0.0)           # uses info up to open of b (known at close b-1.. fine)
    cheat = np.zeros_like(O)
    cheat[:, :-2] = np.where(ret[:, 2:] > 0.002, 1.0, 0.0)   # peeks 2 bars ahead
    ml = simulate(O, legit, slippage_bps=0).metrics()
    mc = simulate(O, cheat, slippage_bps=0).metrics()
    assert mc["gross_pnl"] > 5 * max(ml["gross_pnl"], 1.0)
