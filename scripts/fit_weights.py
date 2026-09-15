"""Fit bookmaker anchor weights from the earliest season only.

That season sits entirely inside every walk-forward fold's training window, so
the fitted weights are never derived from data a fold is evaluated on. Fitting
them across all seasons would leak: the 2026 consensus would encode which
books turned out to be sharp in 2026.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mlbedge import calibrate as CAL
from mlbedge import pipeline as P


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=2024,
                    help="the earliest season; must precede every test fold")
    args = ap.parse_args()

    graded = P.build_graded(args.season, tags=("decision",))
    w = CAL.fit_anchor_weights(graded)
    if not w:
        raise SystemExit("no weights fitted -- not enough graded book views")
    CAL.save(w)
    print()
    print(CAL.report(w))


if __name__ == "__main__":
    main()
