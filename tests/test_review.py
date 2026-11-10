from datetime import date, timedelta

import polars as pl

from mlq.review import classify


def _frame(sym, closes, vols, manual=()):
    n = len(closes)
    d0 = date(2024, 1, 1)
    df = pl.DataFrame({"sym": [sym] * n, "date": [d0 + timedelta(days=i) for i in range(n)],
                       "close": closes, "volume": vols})
    df = df.with_columns((pl.col("close") / pl.col("close").shift(1) - 1).alias("ret"))
    df = df.with_columns((pl.col("ret").abs() > 0.25).fill_null(False).alias("review"),
                         pl.Series([i in manual for i in range(n)]).alias("manual_event"))
    return df


def test_low_price_jump_is_tick_quantization():
    d = classify(_frame("A", [3.0, 4.5] + [4.5] * 3, [100] * 5))
    assert d.filter(pl.col("review"))["ret_class"][0] == "tick_quantization"
    assert d["exclude_ret"].sum() == 1


def test_huge_unexplained_drop_is_suspected_unadjusted_and_big_volume_move_is_news():
    a = _frame("A", [100.0] * 25 + [20.0], [1000] * 26)       # -80%, normal volume
    b = _frame("B", [100.0] * 25 + [70.0], [1000] * 25 + [9000])  # -30%, 9x volume
    flat = [_frame(f"F{i}", [100.0] * 26, [1000] * 26) for i in range(5)]  # realistic cross-section
    d = classify(pl.concat([a, b, *flat]))
    cls = dict(d.filter(pl.col("review")).select(["sym", "ret_class"]).iter_rows())
    assert cls["A"] == "suspected_unadjusted" and cls["B"] == "news_vol_spike"


def test_manual_event_wins():
    d = classify(_frame("A", [100.0] * 25 + [50.0], [1000] * 26, manual=(25,)))
    assert d.filter(pl.col("review"))["ret_class"][0] == "demerger_unadjusted"
