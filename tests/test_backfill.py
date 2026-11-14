from datetime import date, timedelta

import polars as pl

from mlq.backfill import adjust_one


def frame(closes, opens=None):
    d0 = date(2015, 1, 1)
    opens = opens or closes
    prev = [closes[0]] + closes[:-1]
    return pl.DataFrame({"date": [d0 + timedelta(days=i) for i in range(len(closes))],
                         "open": opens, "high": closes, "low": closes, "close": closes,
                         "prev_close": prev, "volume": [100] * len(closes)})


def test_bonus_confirmed_by_oi_is_adjusted():
    g = frame([100.0, 102.0, 51.5, 52.0])                  # 1:1 bonus on day 2
    oi = {g["date"][1]: 1000, g["date"][2]: 2000}
    out, ev = adjust_one(g, oi)
    assert [m for _, m in ev] == [2.0]
    assert abs(out["ret"][2] - (51.5 / 51.0 - 1)) < 1e-9     # no fake -50%
    assert out["volume"][0] == 200                          # pre-event volume scaled up


def test_crash_without_oi_scaling_is_not_adjusted():
    g = frame([100.0, 100.0, 50.0, 49.0])
    oi = {g["date"][1]: 1000, g["date"][2]: 1050}           # OI barely moved -> real crash
    out, ev = adjust_one(g, oi)
    assert ev == [] and out["ret"][2] < -0.45 and out["exclude_ret"][2]


def test_bad_open_print_is_ignored():
    g = frame([100.0, 100.0, 101.0], opens=[100.0, 100.0, 50.0])
    out, ev = adjust_one(g, {})
    assert ev == []
