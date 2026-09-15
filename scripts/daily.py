"""Daily pick generation. Run --once under cron/Task Scheduler, or --loop."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mlbedge.scheduler import Scheduler


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--loop", action="store_true")
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--interval", type=int, default=15,
                    help="minutes between sweeps in loop mode")
    ap.add_argument("--seasons", type=int, nargs="+", default=[2024, 2025, 2026])
    args = ap.parse_args()

    s = Scheduler(seasons=tuple(args.seasons), interval_min=args.interval)
    if args.loop:
        s.loop()
    else:
        n = s.run_once()
        sys.exit(0 if n >= 0 else 1)


if __name__ == "__main__":
    main()
