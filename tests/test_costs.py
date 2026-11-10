import pytest

from mlq.costs import leg_cost, round_trip


def test_intraday_stt_only_on_sell():
    buy, sell = leg_cost(100000, "buy", "intraday"), leg_cost(100000, "sell", "intraday")
    assert sell > buy                      # 0.025% STT = Rs 25 on sell only
    assert sell - buy == pytest.approx(25 - 3, abs=0.5)  # minus the Rs 3 buy stamp duty


def test_delivery_stt_both_legs():
    assert leg_cost(100000, "buy", "delivery") > 100   # 0.1% STT = Rs 100 + fees


def test_flat_brokerage_cap():
    # 10 lakh intraday: 0.1% would be Rs 1000, flat Rs 20 wins
    c = leg_cost(1_000_000, "buy", "intraday")
    assert c < 20 * 1.18 + 1_000_000 * 0.0000297 * 1.18 + 1_000_000 * 0.00003 + 15


def test_no_delivery_short():
    with pytest.raises(ValueError):
        round_trip(100, 99, 10, "short", "delivery")


def test_flat_price_loses_money():
    r = round_trip(100, 100, 100, "long", "intraday")
    assert r["net"] < 0 and r["fees"] > 0 and r["slippage"] > 0


def test_futures_short_allowed_and_cheaper_than_delivery():
    f = round_trip(100, 100, 100, "short", "futures", slippage_bps=0)
    d = round_trip(100, 100, 100, "long", "delivery", slippage_bps=0)
    assert f["fees"] < d["fees"]


def test_short_profits_when_price_falls():
    r = round_trip(100, 90, 100, "short", "intraday", slippage_bps=0)
    assert r["ideal"] == 1000 and r["net"] < 1000 and r["net"] > 900
