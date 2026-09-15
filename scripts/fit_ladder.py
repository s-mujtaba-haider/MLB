"""Fit the line-ladder table from the earliest season only.

Bootstrapped deliberately: the propositions used to fit the table are built
with an *empty* ladder, so only directly-quoted consensus lines contribute.
The table is then learned from those, and every later season uses it. Fitting
it on the seasons it is scored against would leak, and fitting it on
ladder-derived probabilities would be circular.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mlbedge import calibrate as CAL  # noqa: E402
from mlbedge import config as C  # noqa: E402
from mlbedge import dataset as D  # noqa: E402
from mlbedge import ladder as LAD  # noqa: E402
from mlbedge import pipeline as P  # noqa: E402
from mlbedge.devig import attach_consensus  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=2024,
                    help="the earliest season; must precede every test fold")
    ap.add_argument("--min-cell", type=int, default=LAD.MIN_CELL)
    args = ap.parse_args()

    print(f"loading season {args.season} ...", flush=True)
    graded = P.build_graded(args.season, tags=("decision",))
    games, players = P.build_results(args.season)
    cons = attach_consensus(graded, min_books=2,
                            self_anchor_markets=C.SELF_ANCHOR_MARKETS,
                            fitted_weights=CAL.load())

    # Empty ladder: only directly-quoted lines contribute to the fit.
    props = D.build_props(graded, games, players, cons=cons, ladder_table={})
    print(f"  {len(props):,} directly-priced propositions", flush=True)

    table = LAD.smooth(LAD.fit(props, min_cell=args.min_cell))
    if not table:
        raise SystemExit("no ladder cells met the minimum support")
    LAD.save(table)

    print()
    for market, by_primary in sorted(table.items()):
        for primary, cells in sorted(by_primary.items(), key=lambda kv: float(kv[0])):
            lines = ", ".join(sorted(cells, key=float))
            print(f"  {market:22s} primary {primary:>5s} -> lines [{lines}]")


if __name__ == "__main__":
    main()
