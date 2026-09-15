"""Settle the picks the platform actually surfaced.

The backtest says what the strategy would have done; this says what it did.
Runs after games finish and writes result/profit back onto the pick ledger, so
/api/performance reports realised returns on the picks that were shown, not a
re-run of the simulation.
"""
from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from mlbedge import grade as G  # noqa: E402
from mlbedge import ingest_results as IR  # noqa: E402
from mlbedge import match as M  # noqa: E402
from mlbedge.picks import PICKS_PATH  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=5,
                    help="how far back to (re)grade")
    args = ap.parse_args()

    if not PICKS_PATH.exists():
        raise SystemExit("no picks ledger yet")
    picks = pd.read_parquet(PICKS_PATH)
    lo = (dt.date.today() - dt.timedelta(days=args.days)).isoformat()
    hi = dt.date.today().isoformat()
    target = picks[picks["game_date"].between(lo, hi)]
    if target.empty:
        print("nothing to grade in window")
        return

    games, players = IR.run(lo, hi, workers=8)
    if games.empty:
        print("no finished games in window")
        return

    # Picks carry Odds API event ids; join them to ESPN games by teams+date.
    ev = target[["event_id", "commence_time"]].drop_duplicates()
    slate = games[["espn_id", "date", "home_team", "away_team"]].copy()
    # The pick ledger does not store team names, so match on start time.
    ev["_dt"] = pd.to_datetime(ev["commence_time"], utc=True, format="ISO8601")
    slate["_dt"] = pd.to_datetime(slate["date"], utc=True, format="ISO8601")
    pairs = ev.merge(slate, on="_dt", how="left")
    e2e = dict(zip(pairs["event_id"], pairs["espn_id"]))

    q = target.copy()
    q["tag"] = "decision"
    q = M.attach_players(q, players, e2e)
    graded = G.grade(q, games, players)
    graded["profit"] = G.settle_profit(graded["result"], graded["price"])

    key = ["game_date", "market", "subject", "line", "side", "book"]
    out = picks.merge(graded[key + ["result", "profit", "stat_value"]],
                      on=key, how="left", suffixes=("", "_new"))
    for c in ("result", "profit", "stat_value"):
        if f"{c}_new" in out:
            out[c] = out[f"{c}_new"].combine_first(out.get(c))
            out = out.drop(columns=[f"{c}_new"])
    out.to_parquet(PICKS_PATH, index=False)

    done = out[out["result"].isin(["win", "loss", "push"])]
    if len(done):
        print(f"graded {len(done)} picks, ROI {done['profit'].mean():+.4f}, "
              f"profit {done['profit'].sum():+.2f}u")
        print(done.groupby("market")
                  .agg(n=("profit", "size"), roi=("profit", "mean"))
                  .to_string())
    else:
        print("no picks graded (games may not have finished)")


if __name__ == "__main__":
    main()
