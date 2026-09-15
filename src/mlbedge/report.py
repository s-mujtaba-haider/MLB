"""Human-readable market report."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import backtest as B
from . import config as C
from .gate import FAIL, PASS, VETO, Verdict


def _fmt(x, spec="+.2%"):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "n/a"
    return format(x, spec)


def write_market_report(verdicts: list[Verdict], bets: pd.DataFrame,
                        path: Path, seasons=(), folds=()) -> None:
    L: list[str] = []
    a = L.append

    npass = sum(1 for v in verdicts if v.verdict == PASS)
    nveto = sum(1 for v in verdicts if v.verdict == VETO)
    a("# MLB market validation — Phase 1\n")
    a(f"Seasons: {', '.join(str(s) for s in seasons)}  ")
    if folds:
        a(f"Walk-forward: {len(folds)} expanding-window folds, "
          f"{folds[0].test_start} → {folds[-1].test_end}  ")
    a(f"Verdict: **{npass}/{len(verdicts)} PASS**"
      + (f", {nveto} VETO (leakage)" if nveto else "") + "\n")

    a("## Summary\n")
    a("| Market | Verdict | Deploy | Bets | ROI | 95% CI | p | CLV | Hit | "
      "Shrink |")
    a("|---|---|---|---:|---:|---|---:|---:|---:|---:|")
    for v in sorted(verdicts, key=lambda x: (x.verdict != PASS, x.market)):
        m = v.metrics
        ci = (f"{_fmt(m.get('roi_lo'))} … {_fmt(m.get('roi_hi'))}"
              if np.isfinite(m.get("roi_lo", np.nan)) else "n/a")
        a(f"| `{v.market}` | **{v.verdict}** | {v.deployment} | "
          f"{m.get('n_bets', 0)} | {_fmt(m.get('roi'))} | {ci} | "
          f"{_fmt(m.get('p_value'), '.4f')} | {_fmt(m.get('clv_mean'), '+.4f')} | "
          f"{_fmt(m.get('hit_rate'), '.1%')} | "
          f"{_fmt(m.get('avg_shrink'), '.2f')} |")
    a("")

    a("### How to read this\n")
    a("- **ROI** is flat-stake return per unit risked on out-of-sample bets "
      "only; every fold's model, calibration and EV threshold were fitted "
      "strictly before that fold began.")
    a("- **95% CI** is a percentile bootstrap over bets. A market passes only "
      "if the lower bound clears zero — a positive point estimate with an "
      "interval straddling zero is not evidence.")
    a("- **p** is a one-sided bootstrap p-value, then corrected across all "
      "eleven markets with Benjamini-Hochberg. Testing eleven things and "
      "reporting the best one turns a 5% false-positive rate into roughly 43%.")
    a("- **CLV** is mean closing-line value in probability terms, matched on "
      "the same book and the same proposition. It is measured on every bet "
      "rather than through outcome noise, so it is the lower-variance "
      "evidence that an edge is real.")
    a("- **Major-book ROI** restricts to DraftKings, FanDuel, BetMGM, "
      "Caesars, ESPN Bet, BetRivers and Fanatics. 'Best price across twenty "
      "books' overstates what is executable; an edge that survives only at "
      "obscure or offshore shops is a different product from one available "
      "at DraftKings.")
    a("- **Shrink** is how far the model was allowed to move from the market "
      "consensus, fitted per fold on training data. Near zero means the edge "
      "is line-shopping rather than projection — a real edge, but a different "
      "one, and worth knowing which.\n")

    # -- per market ---------------------------------------------------------
    a("## Per-market detail\n")
    for v in sorted(verdicts, key=lambda x: (x.verdict != PASS, x.market)):
        m = C.BY_NAME.get(v.market)
        a(f"### `{v.market}` — {m.label if m else ''} — **{v.verdict}**\n")
        met = v.metrics
        a(f"- Bets: **{met.get('n_bets', 0)}** "
          f"({met.get('n_win', 0)}W / {met.get('n_loss', 0)}L / "
          f"{met.get('n_push', 0)}P)")
        a(f"- ROI: **{_fmt(met.get('roi'))}** "
          f"(95% CI {_fmt(met.get('roi_lo'))} … {_fmt(met.get('roi_hi'))}), "
          f"p = {_fmt(met.get('p_value'), '.4f')}")
        a(f"- CLV: {_fmt(met.get('clv_mean'), '+.4f')} on "
          f"{met.get('clv_n', 0)} matched bets; "
          f"beat the close {_fmt(met.get('beat_close'), '.1%')}")
        a(f"- Stability: {met.get('n_folds', 0)} folds, "
          f"{_fmt(met.get('fold_win_rate'), '.0%')} profitable, "
          f"largest fold = {_fmt(met.get('max_fold_share'), '.0%')} of profit")
        a(f"- Calibration error: {_fmt(met.get('cal_error'), '.4f')}; "
          f"average price {met.get('avg_price', float('nan')):+.0f}")
        a(f"- Executable at major US books only: "
          f"{met.get('n_major', 0)} bets at {_fmt(met.get('roi_major'))}")
        if v.verdict != PASS:
            a(f"\n**Cause: `{v.cause}`**\n")
            a(f"**Lever.** {v.lever}\n")
        else:
            a("\nShipped live. Fires daily through the scheduler.\n")

        if not bets.empty and "market" in bets:
            sub = bets[bets["market"] == v.market]
            if len(sub) > 30:
                stab = B.period_stability(sub)
                a("<details><summary>Per-fold breakdown</summary>\n")
                a("| Fold | Bets | ROI |")
                a("|---|---:|---:|")
                for r in stab.itertuples():
                    a(f"| {r.fold} | {r.n} | {_fmt(r.roi)} |")
                a("\n</details>\n")

            if "book" in sub.columns and len(sub) > 30:
                bb = (sub.groupby("book")
                         .agg(bets=("profit", "size"), roi=("profit", "mean"),
                              profit=("profit", "sum"))
                         .sort_values("bets", ascending=False))
                bb = bb[bb["bets"] >= 10]
                if not bb.empty:
                    a("<details><summary>Where the bets landed</summary>\n")
                    a("This is the execution question: an edge concentrated at "
                      "one book is only as good as that book's limits and how "
                      "long it tolerates the action.\n")
                    a("| Book | Bets | ROI | Profit (u) |")
                    a("|---|---:|---:|---:|")
                    for bk, r in bb.iterrows():
                        a(f"| {bk} | {int(r['bets'])} | {_fmt(r['roi'])} | "
                          f"{r['profit']:+.1f} |")
                    a("\n</details>\n")

            if "anchor_src" in sub.columns and len(sub) > 30:
                asrc = (sub.groupby("anchor_src")
                           .agg(bets=("profit", "size"), roi=("profit", "mean")))
                if len(asrc) > 1:
                    a("<details><summary>Direct line vs ladder-priced</summary>\n")
                    a("| Anchor | Bets | ROI |")
                    a("|---|---:|---:|")
                    for k, r in asrc.iterrows():
                        a(f"| {k} | {int(r['bets'])} | {_fmt(r['roi'])} |")
                    a("\n</details>\n")

    path.write_text("\n".join(L), encoding="utf-8")
    print(f"wrote {path}")
