"""Stock-futures facts from the Upstox instrument master (current month contracts).

Lot sizes (today's) tell us (a) which stocks can be shorted overnight at all (F&O list) and
(b) how much capital one lot needs. Lots change over time; history of lot sizes is NOT in this file,
so backtests use today's lot as an approximation (flagged).
"""
import polars as pl

from mlq.config import INSTRUMENTS_FILE

MARGIN_PCT = 0.20   # rough SPAN+exposure margin as % of notional [verify with Upstox margin calculator]


def fut_table() -> pl.DataFrame:
    d = pl.read_parquet(INSTRUMENTS_FILE).filter((pl.col("segment") == "NSE_FO")
                                                 & (pl.col("instrument_type") == "FUT")
                                                 & (pl.col("underlying_type") == "EQUITY"))
    d = d.sort(["underlying_symbol", "expiry"]).group_by("underlying_symbol", maintain_order=True).first()
    return d.select([pl.col("underlying_symbol").alias("sym"), "lot_size", "tick_size", "expiry"])


def lot_sizes() -> dict[str, int]:
    t = fut_table()
    return dict(zip(t["sym"].to_list(), t["lot_size"].to_list()))


def lots_for(price: float, lot_size: int, notional_target: float) -> int:
    """Whole lots closest to the target notional (at least 0; 0 if one lot is > 2x target)."""
    one = price * lot_size
    n = int(round(notional_target / one))
    return n if one <= 2 * notional_target else 0


def margin_needed(price: float, lot_size: int, lots: int, pct: float = MARGIN_PCT) -> float:
    return price * lot_size * lots * pct
