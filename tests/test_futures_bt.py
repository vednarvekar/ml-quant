from datetime import date, timedelta

import numpy as np

from mlq.futures_bt import simulate_futures


def dts(n):
    return [date(2024, 1, 2) + timedelta(days=i) for i in range(n)]


def test_known_answer_lots_and_pnl():
    # price 100 -> 110 after entry; lot 100; capital 1,000,000, exposure 1 -> 100 lots? notional 10,000/lot
    o = np.array([[100.0]] * 4)
    c = np.array([[100.0], [100.0], [110.0], [110.0]])
    W = np.array([[1.0], [1.0], [1.0], [1.0]])
    r = simulate_futures(o, c, W, dts(4), lot=np.array([100.0]), capital=1_000_000, slippage_bps=0,
                         roll=False)
    # 100 lots? target = 1e6/ (100*100) = 100 lots -> pnl 100*100*10 = 100,000 minus small fees
    assert 95_000 < r.equity[-1] - 1_000_000 < 100_000
    assert r.avg_positions > 0


def test_lot_too_big_means_no_position():
    o = c = np.full((4, 1), 1000.0)
    W = np.ones((4, 1))
    r = simulate_futures(o, c, W, dts(4), lot=np.array([500.0]), capital=100_000, slippage_bps=0)
    assert r.trades == 0 and r.equity[-1] == 100_000     # one lot = 5 lakh > 0.5 lot rounds to 1? see below


def test_margin_cap_limits_notional():
    o = c = np.full((4, 1), 100.0)
    W = np.ones((4, 1))
    r = simulate_futures(o, c, W, dts(4), lot=np.array([100.0]), capital=100_000, exposure=10.0,
                         slippage_bps=0, roll=False)
    assert r.avg_exposure <= 5.0 + 1e-9                  # 20% margin -> at most 5x equity


def test_short_profits_when_price_falls():
    o = np.array([[100.0]] * 4)
    c = np.array([[100.0], [100.0], [90.0], [90.0]])
    W = -np.ones((4, 1))
    r = simulate_futures(o, c, W, dts(4), lot=np.array([100.0]), capital=1_000_000, slippage_bps=0,
                         roll=False)
    assert r.equity[-1] > 1_000_000 + 90_000


def test_nan_lot_names_ignored_and_margin_still_enforced():
    o = c = np.full((4, 2), 100.0)
    W = np.array([[0.5, 0.5]] * 4)
    r = simulate_futures(o, c, W, dts(4), lot=np.array([100.0, np.nan]), capital=100_000, exposure=10.0,
                         slippage_bps=0, roll=False)
    assert np.isfinite(r.avg_exposure) and r.avg_exposure <= 5.0 + 1e-9
