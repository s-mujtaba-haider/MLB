"""Full historical backfill: odds, then results.

Run as:
    python scripts/backfill.py --seasons 2024 2025 2026

Every stage is idempotent and cached on disk, so an interrupted run resumes at
zero credit cost. Closing snapshots are pulled only for the seasons named by
--closing (the out-of-sample season, where CLV carries the verdict), because
they double the per-event price of the pull.
"""
from __future__ import annotations

import argparse
import datetime as dt
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from mlbedge import config as C  # noqa: E402
from mlbedge import ingest_odds as IO  # noqa: E402
from mlbedge import ingest_results as IR  # noqa: E402
from mlbedge.oddsapi import LEDGER, OddsClient  # noqa: E402


def season_window(season: int) -> tuple[str, str]:
    """Clamp the season end to yesterday -- an in-progress season has no data
    past today, and today's games may not have finished."""
    lo, hi = C.SEASONS[season]
    cutoff = (dt.date.today() - dt.timedelta(days=1)).isoformat()
    return lo, min(hi, cutoff)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", type=int, nargs="+", default=[2024, 2025, 2026])
    ap.add_argument("--closing", type=int, nargs="*", default=[2026])
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--skip-odds", action="store_true")
    ap.add_argument("--skip-results", action="store_true")
    args = ap.parse_args()

    client = OddsClient()
    t0 = time.time()

    for season in args.seasons:
        lo, hi = season_window(season)
        print(f"\n{'=' * 70}\nSEASON {season}  ({lo} .. {hi})\n{'=' * 70}",
              flush=True)

        ev_path = C.CURATED / f"events_{season}.parquet"
        if ev_path.exists():
            events = pd.read_parquet(ev_path)
            print(f"[events {season}] {len(events)} cached")
        else:
            events = IO.discover_events(season, client=client,
                                        workers=args.workers)
        events = events[events["game_date"].between(lo, hi)].copy()
        print(f"[events {season}] {len(events)} in window", flush=True)
        if events.empty:
            continue

        if not args.skip_odds:
            IO.fetch_featured(events, client=client, workers=args.workers)
            IO.fetch_props(events, tag="decision", client=client,
                           workers=args.workers)
            if season in (args.closing or []):
                IO.fetch_featured(events, client=client, workers=args.workers,
                                  tag="closing")
                IO.fetch_props(events, tag="closing", client=client,
                               workers=args.workers)

        if not args.skip_results:
            IR.run(lo, hi, workers=args.workers, save_prefix=f"s{season}")

        print(f"[season {season}] done at {time.time() - t0:.0f}s. "
              f"{LEDGER.summary()}", flush=True)

    print(f"\nBACKFILL COMPLETE in {(time.time() - t0) / 60:.1f} min. "
          f"{LEDGER.summary()}", flush=True)


if __name__ == "__main__":
    main()
