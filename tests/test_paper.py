from datetime import date

import numpy as np

from mlq.paper.broker import PaperBroker, PaperConfig
from mlq.paper.signals import select_names, weights_for


def test_select_names_hysteresis_and_threshold():
    score = {f"S{i}": 100 - i for i in range(30)}
    assert select_names(score, set(), 5, 10) == ["S0", "S1", "S2", "S3", "S4"]
    # S7 is held and within rank 10 -> kept; the rest fill from the top
    assert "S7" in select_names(score, {"S7"}, 5, 10)
    # S20 is held but outside keep -> dropped
    assert "S20" not in select_names(score, {"S20"}, 5, 10)
    assert select_names({"A": 1.0}, set(), 5, 10) == []          # not enough names
    assert abs(sum(weights_for(["a", "b", "c", "d"]).values()) - 1) < 1e-12


def test_broker_executes_at_open_then_pnl_and_persists(tmp_path):
    cfg = PaperConfig(capital=1_000_000, slippage_bps=0)
    b = PaperBroker.load(tmp_path / "s.json", cfg)
    lot = {"AAA": 100}
    b.pending = {"decided": "2024-01-01", "weights": {"AAA": 1.0}}
    # day: prev close 100, open 100, close 110 -> 100 lots(=10,000 shares?) lot=100 -> 100 lots*100 = notional 1e6
    r = b.process_day(date(2024, 1, 2), {"AAA": 100.0}, {"AAA": 110.0}, {"AAA": 100.0}, lot, False)
    assert b.lots == {"AAA": 100.0}
    assert len(r["trades"]) == 1 and r["trades"][0]["side"] == "buy"
    assert 95_000 < b.equity - 1_000_000 < 100_000               # +10% on 1e6 minus costs
    b.save()
    b2 = PaperBroker.load(tmp_path / "s.json", cfg)
    assert b2.lots == b.lots and np.isclose(b2.equity, b.equity) and b2.pending is None


def test_broker_exits_names_not_in_new_targets_and_rolls(tmp_path):
    cfg = PaperConfig(capital=1_000_000, slippage_bps=0)
    b = PaperBroker.load(tmp_path / "s.json", cfg)
    lot = {"AAA": 100, "BBB": 100}
    b.lots = {"AAA": 50.0}
    b.equity = 1_000_000
    b.pending = {"decided": "x", "weights": {"BBB": 1.0}}
    e0 = b.equity
    b.process_day(date(2024, 2, 1), {"AAA": 100.0, "BBB": 100.0}, {"AAA": 100.0, "BBB": 100.0},
                  {"AAA": 100.0, "BBB": 100.0}, lot, True)        # new month -> roll cost on AAA
    assert "AAA" not in b.lots and b.lots["BBB"] > 0
    assert b.equity < e0                                          # fees were paid, no price move


def test_roll_due_two_days_before_expiry_once(tmp_path):
    from mlq.paper.run import roll_due
    b = PaperBroker.load(tmp_path / "s.json", PaperConfig())
    exp = [date(2026, 11, 24), date(2026, 12, 29)]            # a Tuesday expiry
    assert not roll_due(b, date(2026, 11, 19), exp)           # 3 weekdays before
    assert roll_due(b, date(2026, 11, 20), exp)               # 2 weekdays before -> roll
    assert not roll_due(b, date(2026, 11, 23), exp)           # already rolled this expiry
    assert not roll_due(b, date(2026, 11, 25), exp)           # next cycle not due yet
    assert roll_due(b, date(2026, 12, 25), exp)               # next cycle


def test_select_ls_and_short_pnl(tmp_path):
    from mlq.paper.signals import select_ls, weights_ls
    score = {f"S{i}": 100 - i for i in range(60)}
    longs, shorts = select_ls(score, set(), set(), 5, 10)
    assert longs == [f"S{i}" for i in range(5)] and shorts == [f"S{i}" for i in range(59, 54, -1)]
    assert abs(sum(weights_ls(longs, shorts).values())) < 1e-12          # dollar neutral
    assert select_ls({"A": 1.0, "B": 2.0}, set(), set(), 5, 10) == ([], [])
    # hysteresis: a held short still in the bottom `keep` stays
    _, s2 = select_ls(score, set(), {"S52"}, 5, 10)
    assert "S52" in s2
    # broker: a short position earns when the price falls
    cfg = PaperConfig(capital=1_000_000, slippage_bps=0, mode="ls")
    b = PaperBroker.load(tmp_path / "s.json", cfg)
    b.pending = {"decided": "x", "weights": {"AAA": -1.0}}
    b.process_day(date(2024, 1, 2), {"AAA": 100.0}, {"AAA": 90.0}, {"AAA": 100.0}, {"AAA": 100}, False)
    assert b.lots["AAA"] < 0 and b.equity > 1_000_000 + 90_000
