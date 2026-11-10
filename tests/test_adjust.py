from datetime import date

import polars as pl

from mlq.adjust import combine, components


def test_components_link_renames_via_isin():
    comp = components([("OLD", "I1"), ("NEW", "I1"), ("OTHER", "I9")])
    assert comp["S:OLD"] == comp["S:NEW"] != comp["S:OTHER"]


def test_split_day_is_flagged_and_adjusted_return_is_small():
    # NSE raw halves on a 2:1 split day; Upstox adjusted series stays smooth
    up = pl.DataFrame({"sym": ["A"] * 3, "date": [date(2024, 1, d) for d in (1, 2, 3)],
                       "close": [50.0, 50.5, 51.0]})
    raw = pl.DataFrame({"sym": ["A"] * 3, "date": [date(2024, 1, d) for d in (1, 2, 3)],
                        "raw_close": [100.0, 101.0, 51.0],
                        "raw_prev_close": [99.0, 100.0, 101.0]})
    d = combine(up, raw).sort("date")
    assert d["adj_step"].to_list() == [False, False, True]   # ex-date step detected
    assert abs(d["ret"][2] - 0.0099) < 1e-3                  # adjusted return is normal
    assert abs(d["ret_raw"][2] - (51 / 101 - 1)) < 1e-9      # raw shows the fake crash


def test_big_move_is_flagged_for_review():
    up = pl.DataFrame({"sym": ["A"] * 2, "date": [date(2024, 1, 1), date(2024, 1, 2)],
                       "close": [100.0, 55.0]})
    raw = pl.DataFrame({"sym": ["A"] * 2, "date": [date(2024, 1, 1), date(2024, 1, 2)],
                        "raw_close": [100.0, 55.0], "raw_prev_close": [99.0, 100.0]})
    d = combine(up, raw).sort("date")
    assert d["review"].to_list() == [False, True]
