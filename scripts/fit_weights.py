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

    fitted = CAL.fit_anchor_weights(graded)
    print(f"measured skill weights for {len(fitted)} markets", flush=True)

    print("comparing weighting schemes by consensus log-loss ...", flush=True)
    cmp = CAL.compare_schemes(graded, fitted=fitted)
    if cmp.empty:
        raise SystemExit("not enough graded book views to compare schemes")
    piv = cmp.pivot(index="market", columns="scheme", values="log_loss")
    print(piv.round(5).to_string())

    choice = CAL.best_scheme_per_market(cmp)
    print()
    for m, s in sorted(choice.items()):
        best = piv.loc[m].min()
        prior = piv.loc[m].get("prior", float("nan"))
        print(f"  {m:22s} -> {s:15s} log-loss {best:.5f} "
              f"(prior {prior:.5f}, gain {prior - best:+.5f})")

    from mlbedge.devig import book_views
    books = sorted(book_views(graded[graded["tag"] == "decision"])["book"].unique())
    w = CAL.weights_from_schemes(choice, books, fitted=fitted)
    CAL.save(w)


if __name__ == "__main__":
    main()
