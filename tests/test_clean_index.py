from datetime import date, datetime
from zoneinfo import ZoneInfo

import polars as pl

from mlq.clean import clean_index_daily

IST = ZoneInfo("Asia/Kolkata")


def test_drop_stray_early_rows():
    from mlq.clean import drop_stray_early
    from datetime import timedelta
    early = [date(2015, 1, 1) + timedelta(days=i) for i in range(5)]
    late = [date(2020, 4, 1) + timedelta(days=i) for i in range(80)]
    d = pl.DataFrame({"sym": ["A"] * (5 + 80), "date": early + late})
    out = drop_stray_early(d)
    assert out.height == 80 and out["date"].min() == date(2020, 4, 1)
    # a long-history stock with a mid-history gap is left alone
    long_ = [date(2010, 1, 1) + timedelta(days=i) for i in range(100)] + late
    d2 = pl.DataFrame({"sym": ["B"] * len(long_), "date": long_})
    assert drop_stray_early(d2).height == len(long_)


def test_keeps_real_ohlc_row_and_drops_non_trading_days():
    d = pl.DataFrame({
        "ts": [datetime(2024, 1, 2, 0, 0, tzinfo=IST), datetime(2024, 1, 2, 9, 15, tzinfo=IST),
               datetime(2024, 1, 6, 0, 0, tzinfo=IST)],  # 6 Jan = Saturday, not in calendar
        "open": [100.0, 98.0, 101.0], "high": [100.0, 102.0, 101.0],
        "low": [100.0, 97.0, 101.0], "close": [100.0, 100.0, 101.0], "volume": [0, 0, 0]})
    out = clean_index_daily(d, [date(2024, 1, 2)])
    assert out.height == 1
    assert out["high"][0] == 102.0 and out["low"][0] == 97.0   # real range kept
