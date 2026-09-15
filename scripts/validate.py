"""Run every market through the gate and write the reports.

    python scripts/validate.py --seasons 2024 2025 2026

Outputs
    reports/gate_results.csv   one row per market: verdict, cause, lever, metrics
    reports/market_report.md   the readable version
    reports/bets.parquet       every out-of-sample bet the gate judged
    reports/leakage.csv        the as-of audit, per market
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from mlbedge import backtest as B  # noqa: E402
from mlbedge import config as C  # noqa: E402
from mlbedge import dataset as D  # noqa: E402
from mlbedge import gate as GATE  # noqa: E402
from mlbedge import leakage as LK  # noqa: E402
from mlbedge import pipeline as P  # noqa: E402
from mlbedge import report as R  # noqa: E402
from mlbedge.devig import attach_consensus  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", type=int, nargs="+", default=[2024, 2025, 2026])
    ap.add_argument("--markets", nargs="*", default=list(C.ALL_MARKET_NAMES))
    ap.add_argument("--burn-days", type=int, default=400)
    ap.add_argument("--step-days", type=int, default=30)
    ap.add_argument("--min-books", type=int, default=3)
    ap.add_argument("--devig", default="shin")
    ap.add_argument("--rebuild", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    seasons = tuple(sorted(args.seasons))
    print(f"Loading seasons {seasons} ...", flush=True)
    graded, games, players = P.load_all(seasons, tags=("decision", "closing"))
    print(f"  graded quotes {len(graded):,}  games {len(games):,}  "
          f"player-games {len(players):,}", flush=True)

    print("Building consensus ...", flush=True)
    cons = attach_consensus(graded, method=args.devig, min_books=2)

    print("Building proposition frame ...", flush=True)
    props = D.build_props(graded, games, players, cons=cons)
    print(f"  propositions {len(props):,}", flush=True)

    print("Building bet candidates ...", flush=True)
    cand = D.build_candidates(graded, cons=cons, tag="decision")
    cand = cand.merge(games[["espn_id", "game_date"]], on="espn_id",
                      how="left", suffixes=("", "_g"))
    if "game_date_g" in cand:
        cand["game_date"] = cand["game_date"].fillna(cand["game_date_g"])
    closing = cons[cons["tag"] == "closing"]
    print(f"  candidates {len(cand):,}  closing quotes {len(closing):,}",
          flush=True)

    folds = B.make_folds(props["game_date"], n_burn_days=args.burn_days,
                         step_days=args.step_days)
    print(f"  {len(folds)} walk-forward folds "
          f"({folds[0].test_start} .. {folds[-1].test_end})", flush=True)

    all_bets, verdicts, leak_rows = [], [], []
    for mkt in args.markets:
        print(f"\n=== {mkt} ===", flush=True)
        bets, reports = B.run_market(mkt, props, cand, folds,
                                     min_books=args.min_books, verbose=True)
        if bets.empty:
            v = GATE.Verdict(mkt, GATE.FAIL, "insufficient_sample",
                             GATE.LEVERS["insufficient_sample"], GATE.KILLED,
                             B.summarise(bets))
            verdicts.append(v)
            print(f"  no bets produced -> {v.verdict} ({v.cause})", flush=True)
            continue

        # Leakage audit for this market, on the bets actually judged.
        p_m = props[props["market"] == mkt]
        feats = D.feature_columns(p_m, mkt)
        train = p_m[p_m["game_date"] <= folds[0].train_end]
        test = bets.assign(won=(bets["result"] == "win").astype(float))
        rep = LK.audit(bets, train, test, feats, outcome_col="won",
                       market_col="logit_cons")
        for f in rep.findings:
            leak_rows.append({"market": mkt, **f.__dict__})

        bets["market"] = mkt
        all_bets.append(bets)
        v = GATE.judge(mkt, bets, closing, leak_report=rep)
        verdicts.append(v)
        s = v.metrics
        print(f"  bets={s['n_bets']} roi={s['roi']:+.4f} "
              f"[{s['roi_lo']:+.4f},{s['roi_hi']:+.4f}] "
              f"p={s.get('p_value', float('nan')):.4f} "
              f"clv={s.get('clv_mean', float('nan')):+.5f} "
              f"-> {v.verdict} {v.cause}", flush=True)

    verdicts = GATE.apply_family_correction(verdicts)

    res = GATE.to_frame(verdicts)
    res.to_csv(C.REPORTS / "gate_results.csv", index=False)
    if leak_rows:
        pd.DataFrame(leak_rows).to_csv(C.REPORTS / "leakage.csv", index=False)
    if all_bets:
        bets_all = pd.concat(all_bets, ignore_index=True)
        bets_all.to_parquet(C.REPORTS / "bets.parquet", index=False)
    else:
        bets_all = pd.DataFrame()

    R.write_market_report(verdicts, bets_all, C.REPORTS / "market_report.md",
                          seasons=seasons, folds=folds)

    print(f"\n{'=' * 78}")
    show = ["market", "verdict", "deployment", "n_bets", "roi", "roi_lo",
            "p_value", "clv_mean", "cause"]
    show = [c for c in show if c in res.columns]
    with pd.option_context("display.width", 200, "display.max_colwidth", 22):
        print(res[show].to_string(index=False))
    npass = int((res["verdict"] == GATE.PASS).sum())
    print(f"\n{npass}/{len(res)} markets PASS. "
          f"Completed in {(time.time() - t0) / 60:.1f} min.")


if __name__ == "__main__":
    main()
