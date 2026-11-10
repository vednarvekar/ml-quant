from mlq.futures import lots_for, margin_needed


def test_lots_for():
    assert lots_for(100, 500, 100_000) == 2          # one lot = 50k
    assert lots_for(1400, 500, 300_000) == 0         # one lot = 7 lakh > 2x target
    assert margin_needed(100, 500, 2, pct=0.2) == 20_000
