"""Run the whole Phase 1 sequence end to end.

    python scripts/run_all.py --seasons 2024 2025 2026

Order matters, and each step invalidates what comes after it:

 1. alternate ladders   new raw snapshots, so every parsed quote table is stale
 2. rebuild quotes      drop the derived parquet so alternates are included
 3. anchor weights      fitted on the earliest season only
 4. ladder table        fitted on the earliest season, from direct lines only
 5. validate            the gate, walk-forward across every season
 6. train production    deployable bundles for whatever the gate authorised

Steps 3 and 4 change the configuration hash, so the cached season frames from
an earlier configuration are never silently reused.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mlbedge import config as C  # noqa: E402

PY = sys.executable


def run(label: str, args: list[str]) -> None:
    print(f"\n{'=' * 78}\n{label}\n{'=' * 78}", flush=True)
    t0 = time.time()
    r = subprocess.run([PY] + args, cwd=ROOT)
    if r.returncode != 0:
        raise SystemExit(f"{label} failed with exit {r.returncode}")
    print(f"[{label}] {time.time() - t0:.0f}s", flush=True)


def drop_derived(patterns=("quotes_*.parquet", "graded_*.parquet",
                           "props_*.parquet", "cand_*.parquet",
                           "close_*.parquet")) -> None:
    n = 0
    for pat in patterns:
        for p in C.CURATED.glob(pat):
            p.unlink()
            n += 1
    print(f"dropped {n} stale derived tables", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", type=int, nargs="+", default=[2024, 2025, 2026])
    ap.add_argument("--fit-season", type=int, default=2024)
    ap.add_argument("--skip-alt", action="store_true")
    ap.add_argument("--burn-days", type=int, default=400)
    ap.add_argument("--step-days", type=int, default=30)
    args = ap.parse_args()
    seasons = [str(s) for s in sorted(args.seasons)]

    if not args.skip_alt:
        run("1/7 alternate ladders",
            ["scripts/backfill.py", "--seasons", *seasons, "--only-alt"])
        drop_derived()

    run("2/7 anchor weights",
        ["scripts/fit_weights.py", "--season", str(args.fit_season)])
    run("3/7 ladder table",
        ["scripts/fit_ladder.py", "--season", str(args.fit_season)])

    run("4/7 hyperparameters",
        ["scripts/tune.py", "--season", str(args.fit_season)])

    run("5/7 validate",
        ["scripts/validate.py", "--seasons", *seasons,
         "--burn-days", str(args.burn_days),
         "--step-days", str(args.step_days)])

    run("6/7 train production",
        ["scripts/train_production.py", "--seasons", *seasons])

    print("\nDONE. See reports/market_report.md and models/manifest.json",
          flush=True)


if __name__ == "__main__":
    main()
