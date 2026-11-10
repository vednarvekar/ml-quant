"""India cash-equity cost model (NSE). All rates are parameters so they are easy to update.

Rates marked [verify] must be checked against the first Upstox contract note.
Costs are returned in rupees for ONE leg (a buy or a sell). Round trip = buy leg + sell leg.

Rules baked in:
- Delivery (swing, held overnight): can only go LONG. No short selling in delivery.
- Intraday (squared off same day, MIS): long or short; STT only on the sell leg.
- Slippage is separate (bps of price, paid against you on every fill).
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class Rates:
    stt_delivery: float = 0.001        # 0.1% both legs
    stt_intraday_sell: float = 0.00025  # 0.025% sell leg only
    exch_txn: float = 0.0000297        # NSE equity ~0.00297% [verify]
    sebi: float = 0.000001             # Rs 10 per crore
    stamp_delivery_buy: float = 0.00015   # 0.015% buy leg
    stamp_intraday_buy: float = 0.00003   # 0.003% buy leg
    gst: float = 0.18                  # on brokerage + exch txn + SEBI
    brk_delivery_flat: float = 20.0    # Upstox: Rs 20 or 2.5%, lower [verify]
    brk_delivery_pct: float = 0.025
    brk_intraday_flat: float = 20.0    # Rs 20 or 0.1%, lower [verify]
    brk_intraday_pct: float = 0.001
    dp_sell_delivery: float = 20.0     # depository charge per sell scrip incl. GST (approx) [verify]
    # stock futures (allows multi-day shorts)
    stt_fut_sell: float = 0.0005       # 0.05% sell leg only (from 1 Apr 2026)
    exch_txn_fut: float = 0.0000173    # NSE futures ~0.00173% [verify]
    stamp_fut_buy: float = 0.00002     # 0.002% buy leg
    brk_fut_flat: float = 20.0         # Rs 20 or 0.03%, lower [verify]
    brk_fut_pct: float = 0.0003


DEFAULT = Rates()


def leg_cost(value: float, side: str, kind: str, r: Rates = DEFAULT) -> float:
    """Rupee cost of one leg. value = price*qty. side 'buy'|'sell'. kind 'delivery'|'intraday'."""
    if side not in ("buy", "sell") or kind not in ("delivery", "intraday", "futures"):
        raise ValueError("side buy|sell, kind delivery|intraday|futures")
    if value <= 0:
        return 0.0
    if kind == "futures":  # value = contract notional (price * lot size * lots)
        brk = min(r.brk_fut_flat, value * r.brk_fut_pct)
        stt = value * r.stt_fut_sell if side == "sell" else 0.0
        stamp = value * r.stamp_fut_buy if side == "buy" else 0.0
        exch, sebi = value * r.exch_txn_fut, value * r.sebi
        return brk + stt + exch + sebi + stamp + (brk + exch + sebi) * r.gst
    if kind == "delivery":
        brk = min(r.brk_delivery_flat, value * r.brk_delivery_pct)
        stt = value * r.stt_delivery
        stamp = value * r.stamp_delivery_buy if side == "buy" else 0.0
        dp = r.dp_sell_delivery if side == "sell" else 0.0
    else:
        brk = min(r.brk_intraday_flat, value * r.brk_intraday_pct)
        stt = value * r.stt_intraday_sell if side == "sell" else 0.0
        stamp = value * r.stamp_intraday_buy if side == "buy" else 0.0
        dp = 0.0
    exch, sebi = value * r.exch_txn, value * r.sebi
    gst = (brk + exch + sebi) * r.gst
    return brk + stt + exch + sebi + stamp + gst + dp


def round_trip(entry_price: float, exit_price: float, qty: int, direction: str, kind: str,
               slippage_bps: float = 5.0, r: Rates = DEFAULT) -> dict:
    """Net P&L of one trade. direction 'long'|'short'. Slippage hurts on both fills."""
    if direction not in ("long", "short"):
        raise ValueError("direction long|short")
    if direction == "short" and kind == "delivery":
        raise ValueError("cannot short in delivery; shorts must be intraday")
    s = slippage_bps / 1e4
    if direction == "long":
        buy_p, sell_p = entry_price * (1 + s), exit_price * (1 - s)
    else:  # short: sell first at entry, buy back at exit
        sell_p, buy_p = entry_price * (1 - s), exit_price * (1 + s)
    gross = (sell_p - buy_p) * qty
    fees = leg_cost(buy_p * qty, "buy", kind, r) + leg_cost(sell_p * qty, "sell", kind, r)
    ideal = ((exit_price - entry_price) if direction == "long" else (entry_price - exit_price)) * qty
    return {"ideal": ideal, "slippage": ideal - gross, "fees": fees, "net": gross - fees}
