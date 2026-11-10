from datetime import date, timedelta

import numpy as np
import pytest

from mlq.backtest import simulate


def dates(n):
    return [date(2020, 1, 1) + timedelta(days=i) for i in range(n)]


def random_walk(T=1500, N=20, seed=0, vol=0.01):
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0, vol, (T, N)), axis=0))
    gap = rng.normal(0, vol / 2, (T, N))
    open_ = np.vstack([close[:1], close[:-1] * np.exp(gap[1:])])
    return open_, close


def test_known_answer_executes_at_next_open():
    open_ = np.array([[100.0], [100.0], [100.0]])
    close = np.array([[50.0], [110.0], [120.0]])        # close[0]=50 is a trap
    w = np.array([[1.0], [1.0], [1.0]])
    r = simulate(open_, close, w, dates(3), capital=100_000, slippage_bps=0)
    # signal at t=0 -> buy at open[1]=100 (NOT close[0]=50): 1000 shares
    eq = r.equity["equity"].to_numpy()
    assert eq[1] == pytest.approx(100_000 + 1000 * 10, abs=300)   # minus fees only
    assert eq[1] < 110_000
    assert r.trades == 1                                           # holds, no re-trade


def test_delivery_short_rejected():
    o = c = np.full((3, 1), 100.0)
    with pytest.raises(ValueError):
        simulate(o, c, np.array([[-1.0], [0], [0]]), dates(3))


def test_null_random_strategy_loses_about_costs():
    open_, close = random_walk()
    T, N = close.shape
    rng = np.random.default_rng(1)
    w = np.zeros((T, N))
    for t in range(T):                                  # random 5 stocks each day, equal weight
        w[t, rng.choice(N, 5, replace=False)] = 0.2
    m = simulate(open_, close, w, dates(T)).metrics()
    assert m["total_return"] < 0 and m["fees"] > 0
    assert m["sharpe"] < 0.5


def test_no_lookahead_legit_signal_earns_nothing_cheat_earns_a_lot():
    open_, close = random_walk(T=1500, N=20, seed=2)
    T, N = close.shape
    ret = np.vstack([np.zeros((1, N)), close[1:] / close[:-1] - 1])
    legit = np.zeros((T, N))
    legit[ret > 0.01] = 1.0                                  # uses return known at close t
    cheat = np.zeros((T, N))
    nxt = np.vstack([ret[1:], np.zeros((1, N))])             # peeks at close t+1 return
    cheat[nxt > 0.01] = 1.0
    for w in (legit, cheat):
        s = w.sum(axis=1, keepdims=True)
        w /= np.maximum(s, 1)
    ml = simulate(open_, close, legit, dates(T), slippage_bps=0).metrics()
    mc = simulate(open_, close, cheat, dates(T), slippage_bps=0).metrics()
    assert ml["sharpe"] < 1.0
    assert mc["sharpe"] > 3 * max(ml["sharpe"], 0.3)


def test_deterministic():
    open_, close = random_walk(T=300, seed=3)
    w = np.full(close.shape, 1 / close.shape[1])
    a = simulate(open_, close, w, dates(300)).metrics()
    b = simulate(open_, close, w, dates(300)).metrics()
    assert a == b
