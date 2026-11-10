"""One command: rebuild every derived dataset from the raw downloads (offline, no network).

raw -> clean -> adjust (daily) -> adjust_5m -> review -> universe -> calendar check

Downloads are separate (they hit the network / rate limits):
  python -m mlq.upstox.bulk | bulk_universe | indices ; python -m mlq.nse_bhav ; python -m mlq.context
Then:  uv run python -m mlq.rebuild
"""
import time

from mlq import adjust, adjust_5m, calendar_check, clean, review, universe

STEPS = [("clean", clean.main), ("adjust daily", adjust.main), ("adjust 5m", adjust_5m.main),
         ("review/classify", review.main), ("universe", universe.main),
         ("calendar check", calendar_check.main)]


def main():
    for name, fn in STEPS:
        t = time.time()
        print(f"\n=== {name} ===", flush=True)
        fn()
        print(f"--- {name} done in {time.time() - t:.0f}s", flush=True)
    print("\nREBUILD OK")


if __name__ == "__main__":
    main()
