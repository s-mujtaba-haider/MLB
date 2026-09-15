"""Fit the deployable models and write the manifest the API reads.

Run after scripts/validate.py: the gate decides which markets deploy, this
decides what they deploy with.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from mlbedge import config as C  # noqa: E402
from mlbedge import dataset as D  # noqa: E402
from mlbedge import pipeline as P  # noqa: E402
from mlbedge import production as PR  # noqa: E402
from mlbedge.devig import attach_consensus  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", type=int, nargs="+", default=[2024, 2025, 2026])
    args = ap.parse_args()

    gate_csv = C.REPORTS / "gate_results.csv"
    if not gate_csv.exists():
        raise SystemExit("run scripts/validate.py first -- the gate decides "
                         "which markets are allowed to deploy")
    verdicts = pd.read_csv(gate_csv)

    print("loading ...", flush=True)
    graded, games, players = P.load_all(tuple(sorted(args.seasons)),
                                        tags=("decision",))
    cons = attach_consensus(graded, min_books=2)
    props = D.build_props(graded, games, players, cons=cons)
    cand = D.build_candidates(graded, cons=cons, tag="decision")
    cand = cand.merge(games[["espn_id", "game_date"]], on="espn_id", how="left",
                      suffixes=("", "_g"))
    print(f"props {len(props):,}  candidates {len(cand):,}", flush=True)

    bundles = PR.train_all(props, cand, verdicts)
    PR.save(bundles)

    live = [m for m, b in bundles.items() if b.deployment == PR.LIVE]
    veto = [m for m, b in bundles.items() if b.deployment == PR.VETO_FILTERED]
    kill = [m for m in C.ALL_MARKET_NAMES if m not in live and m not in veto]
    print(f"\nlive           ({len(live)}): {', '.join(sorted(live)) or '-'}")
    print(f"veto-filtered  ({len(veto)}): {', '.join(sorted(veto)) or '-'}")
    print(f"killed         ({len(kill)}): {', '.join(sorted(kill)) or '-'}")


if __name__ == "__main__":
    main()
