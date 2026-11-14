from datetime import datetime
from zoneinfo import ZoneInfo

import polars as pl

from mlq.fut_fill import day_table

IST = ZoneInfo("Asia/Kolkata")


def _file(tmp_path, rows):
    f = tmp_path / "ABC_2026-11-23.parquet"
    pl.DataFrame(rows, schema=["ts", "open", "high", "low", "close", "volume", "oi"], orient="row") \
        .with_columns(pl.col("ts").dt.replace_time_zone("Asia/Kolkata")).write_parquet(f)
    return f


def test_day_table_open_vwap_and_volume(tmp_path):
    rows = [(datetime(2026, 11, 13, 9, 15 + m), 100.0 + m, 100.0 + m, 100.0 + m, 100.0 + m, 1000, 0)
            for m in range(20)]
    rows.append((datetime(2026, 11, 13, 15, 29), 150.0, 150.0, 150.0, 150.0, 10, 0))
    rows.append((datetime(2026, 11, 13, 15, 35), 999.0, 999.0, 999.0, 999.0, 10, 0))   # after hours: ignored
    r = day_table(_file(tmp_path, rows)).row(0, named=True)
    assert r["sym"] == "ABC" and r["expiry"].isoformat() == "2026-11-23"
    assert r["first_m"] == 0 and r["f_open"] == 100.0 and r["f_close"] == 150.0
    assert r["vwap1"] == 100.0
    assert abs(r["vwap5"] - 102.0) < 1e-9            # mean of 100..104, equal volume
    assert abs(r["rs5"] - 1000 * (100 + 101 + 102 + 103 + 104)) < 1e-6


def test_day_table_late_first_trade(tmp_path):
    rows = [(datetime(2026, 11, 13, 9, 22), 50.0, 50.0, 50.0, 50.0, 500, 0),
            (datetime(2026, 11, 13, 9, 40), 51.0, 51.0, 51.0, 51.0, 500, 0)]
    r = day_table(_file(tmp_path, rows)).row(0, named=True)
    assert r["first_m"] == 7                         # no trade at 09:15: the open was not available
    assert r["vwap5"] is None or r["vwap5"] != r["vwap5"]
