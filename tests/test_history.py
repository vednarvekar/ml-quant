from datetime import date

from mlq.upstox.history import chunks, parse_candles, window_days


def test_chunks_cover_range_without_gaps_or_overlap():
    wins = list(chunks(date(2022, 1, 1), date(2022, 12, 31), "minutes", 1))
    assert wins[0][0] == date(2022, 1, 1) and wins[-1][1] == date(2022, 12, 31)
    for (a, b), (c, d) in zip(wins, wins[1:]):
        assert (c - b).days == 1
    assert all((b - a).days + 1 <= 28 for a, b in wins)


def test_window_sizes_follow_limits():
    assert window_days("minutes", 5) < 31
    assert window_days("minutes", 30) <= 90
    assert window_days("hours", 1) <= 90
    assert window_days("days", 1) <= 3650


def test_parse_sorts_oldest_first_and_dedups():
    c = [["2024-01-02T09:16:00+05:30", 2, 3, 1, 2.5, 10, 0],
         ["2024-01-02T09:15:00+05:30", 1, 2, 1, 1.5, 5, 0],
         ["2024-01-02T09:15:00+05:30", 1, 2, 1, 1.5, 5, 0]]
    df = parse_candles(c)
    assert df.height == 2
    assert df["ts"][0].minute == 15


def test_parse_empty():
    assert parse_candles([]).height == 0


def test_missing_ranges():
    from mlq.upstox.history import missing_ranges
    d = date
    assert missing_ranges(None, None, d(2022, 1, 1), d(2022, 2, 1)) == [(d(2022, 1, 1), d(2022, 2, 1))]
    assert missing_ranges(d(2022, 1, 10), d(2022, 1, 20), d(2022, 1, 1), d(2022, 2, 1)) == \
        [(d(2022, 1, 1), d(2022, 1, 10)), (d(2022, 1, 20), d(2022, 2, 1))]
    assert missing_ranges(d(2022, 1, 1), d(2022, 2, 1), d(2022, 1, 1), d(2022, 2, 1)) == []
