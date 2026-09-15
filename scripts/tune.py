"""Per-market hyperparameter selection, on the earliest season only.

A handful of deliberately different configurations rather than a large grid:
with a hundred candidates the winner is mostly the luckiest, and the gain is
not recoverable out of sample. Six shapes -- shallow, deep, heavily
regularised, lightly regularised -- cover the real axes of variation.

Scored by grouped out-of-fold log-loss of the *blended* probability, which is
what the pipeline actually uses, not of the raw booster.

Leakage discipline: fitted on `--season` only. That season lies inside every
walk-forward fold's training window (the burn-in is longer than one season), so
no fold is ever evaluated with hyperparameters chosen on its test data.
Choosing them on the full history would be exactly the "best of N tries"
problem the gate corrects for, moved one level up where the gate cannot see it.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from mlbedge import config as C  # noqa: E402
from mlbedge import dataset as D  # noqa: E402
from mlbedge import pipeline as P  # noqa: E402
from mlbedge.model import MarketModel  # noqa: E402

PARAMS_PATH = C.DATA / "model_params.json"

GRID = {
    "shallow":        dict(learning_rate=0.06, max_leaf_nodes=7,
                           min_samples_leaf=300, l2=1.0),
    "default":        dict(learning_rate=0.05, max_leaf_nodes=31,
                           min_samples_leaf=200, l2=1.0),
    "conservative":   dict(learning_rate=0.03, max_leaf_nodes=15,
                           min_samples_leaf=500, l2=2.0),
    "very_regular":   dict(learning_rate=0.02, max_leaf_nodes=15,
                           min_samples_leaf=1000, l2=5.0),
    "deep_regular":   dict(learning_rate=0.04, max_leaf_nodes=63,
                           min_samples_leaf=400, l2=3.0),
    "aggressive":     dict(learning_rate=0.08, max_leaf_nodes=63,
                           min_samples_leaf=100, l2=0.5),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=2024)
    ap.add_argument("--markets", nargs="*", default=list(C.ALL_MARKET_NAMES))
    args = ap.parse_args()

    props, _, _ = P.build_season_frames(args.season, tags=("decision",))
    print(f"tuning on {len(props):,} propositions from {args.season}\n",
          flush=True)

    chosen: dict[str, dict] = {}
    for mkt in args.markets:
        p = props[props["market"] == mkt]
        if len(p) < 1500:
            print(f"  {mkt:22s} too little data ({len(p)}) -- keeping default")
            continue
        feats = D.feature_columns(p, mkt)
        if "logit_cons" not in feats:
            feats.append("logit_cons")

        rows = []
        for name, params in GRID.items():
            t0 = time.time()
            try:
                mm = MarketModel(mkt, **params)
                rep = mm.fit(p, feats)
            except ValueError as e:
                print(f"  {mkt:22s} {name:14s} skipped ({e})")
                continue
            rows.append({"config": name, "ll_final": rep.ll_final,
                         "ll_market": rep.ll_market, "shrink": rep.shrink,
                         "gain": rep.model_gain, "secs": time.time() - t0})
        if not rows:
            continue
        r = pd.DataFrame(rows).sort_values("ll_final")
        best = r.iloc[0]
        chosen[mkt] = {**GRID[best["config"]], "_config": best["config"],
                       "_gain_vs_market": float(best["gain"]),
                       "_shrink": float(best["shrink"])}
        print(f"  {mkt:22s} -> {best['config']:14s} "
              f"ll {best['ll_final']:.5f} (market {best['ll_market']:.5f}, "
              f"gain {best['gain']:+.5f}, shrink {best['shrink']:.2f})",
              flush=True)

    PARAMS_PATH.write_text(json.dumps(chosen, indent=2))
    print(f"\nsaved {len(chosen)} tuned configs -> {PARAMS_PATH}")


if __name__ == "__main__":
    main()
