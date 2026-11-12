"""Plain-English health check for paper trading.

uv run python -m mlq.paper.status

Tells you: did the daily job run and finish, how are the pretend accounts doing, what are they
holding, and whether anything needs attention.
"""
import json
import re
from datetime import date, datetime
from pathlib import Path

from mlq.config import DATA

HOME = DATA / "paper"


def read_log(p: Path) -> str:
    return p.read_bytes().replace(b"\x00", b"").decode("utf-8", errors="ignore")


def last_runs(n: int = 5) -> list[dict]:
    out = []
    for p in sorted((HOME / "logs").glob("*.log"))[-n:]:
        t = read_log(p)
        starts = re.findall(r"=== start (\S+) ===", t)
        ends = re.findall(r"=== end (\S+) exit=(\S*) ===", t)
        ok = bool(ends) and ends[-1][1] == "0" and "Traceback" not in t
        out.append({"file": p.name, "started": starts[-1] if starts else None,
                    "finished": ends[-1][0] if ends else None, "ok": ok, "error": "Traceback" in t})
    return out


def show_account(d: Path) -> None:
    s = json.loads((d / "state.json").read_text())
    pnl = s["equity"] / s["start_equity"] - 1
    held = {k: v for k, v in s["lots"].items() if v}
    longs = {k: v for k, v in held.items() if v > 0}
    shorts = {k: v for k, v in held.items() if v < 0}
    print()
    print(f"[{d.name}] Rs {s['equity']:,.0f} (started Rs {s['start_equity']:,.0f} on {s['start_date']}) = {pnl:+.2%}")
    print(f"  data processed through {s['last_date']} | long {len(longs)} names, short {len(shorts)} names")
    if longs:
        print("  Long (lots): " + ", ".join(f"{k} {v:.0f}" for k, v in sorted(longs.items())))
    if shorts:
        print("  Short (lots): " + ", ".join(f"{k} {-v:.0f}" for k, v in sorted(shorts.items())))
    if s.get("pending"):
        w = s["pending"]["weights"]
        buys = ", ".join(k for k, x in w.items() if x > 0)
        sells = ", ".join(k for k, x in w.items() if x < 0)
        print(f"  Next trades at the open (decided {s['pending']['decided']}): buy {buys}"
              + (f"; sell short {sells}" if sells else ""))
    rep = d / "report.md"
    if rep.exists() and "ALERTS" in rep.read_text(encoding="utf-8"):
        print(f"  !! alerts in {rep}")


def main():
    print("PAPER TRADING STATUS (pretend money, no real orders)")
    print()
    runs = last_runs()
    if not runs:
        print("- The daily job has never run. It is scheduled for 7:30 PM on weekdays.")
    for r in runs:
        what = "finished OK" if r["ok"] else ("CRASHED (see log)" if r["error"] else
                                               "did NOT finish (was stopped or still running)")
        print(f"- {r['file'][:-4]}: started {r['started']} -> {what}")
    good = [r for r in runs if r["ok"]]
    if good:
        last = datetime.strptime(good[-1]["file"][:-4], "%Y-%m-%d").date()
        gap = (date.today() - last).days
        if gap > 3:
            print()
            print(f"!! Last successful run was {gap} days ago. Check that the PC was on at 7:30 PM.")
    elif runs:
        print()
        print("!! No successful run yet. Run it by hand: uv run python -m mlq.paper.run --update")
    accts = sorted(d for d in HOME.iterdir() if d.is_dir() and (d / "state.json").exists())
    if not accts:
        print()
        print("No accounts yet.")
        return
    for d in accts:
        show_account(d)
    print()
    print("Reminder: judge the strategy only after 6-8 weeks of fresh data. A few days prove nothing.")


if __name__ == "__main__":
    main()
