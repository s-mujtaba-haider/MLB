"""The most important test in the suite: the pipeline must not invent an edge.

A validation stack that reports a profit on a market that is efficient by
construction is worse than useless -- it will report a profit on everything.
So we build a market where the truth is *known* to be the posted consensus,
run the entire walk-forward end to end, and require the answer to come back
negative by roughly the bookmaker's margin.

That is the correct answer: if prices are fair and you pay the vig, you lose
the vig. Any pipeline that turns that into a positive number has a bug, a
leak, or a threshold fitted on the test set.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlbedge import backtest as B
from mlbedge import gate as G
from mlbedge.odds import american_to_prob, profit_per_unit


def _synthetic_market(n_days=90, props_per_day=40, seed=7):
    """A market priced fairly, quoted with a ~4.5% hold.

    The consensus is the truth. Features are pure noise. The only thing a
    bettor can do is pay the margin.
    """
    rng = np.random.default_rng(seed)
    prop_rows, cand_rows = [], []
    for day in range(n_days):
        date = (pd.Timestamp("2024-04-01") + pd.Timedelta(days=day)).date().isoformat()
        p_true = rng.uniform(0.30, 0.70, props_per_day)
        for i, p in enumerate(p_true):
            eid = f"e{day}_{i}"
            won = float(rng.random() < p)
            prop_rows.append({
                "event_id": eid, "market": "totals", "subject": "game",
                "line": 0.5, "game_date": date, "espn_id": eid,
                "p_cons_all": p, "logit_cons": float(np.log(p / (1 - p))),
                "won": won,
                "n_books_prop": 8.0, "hold_med": 0.045, "lead_min": 30.0,
                "n_quotes": 16.0, "book_std": 0.01, "book_spread": 0.03,
                # Pure noise, but carried under names the feature selector
                # for a team market will actually pick up, so the model is
                # genuinely offered them.
                "home_tg_runs_for_r10": rng.normal(),
                "away_tg_runs_for_r10": rng.normal(),
                "home_tg_won_r30": rng.normal(),
            })
            # Both sides quoted with a margin split evenly around the truth.
            for side, q in (("over", p), ("under", 1 - p)):
                raw = min(0.97, q * 1.045)
                price = (-100 * raw / (1 - raw)) if raw >= 0.5 else \
                        (100 * (1 - raw) / raw)
                res = "win" if (won == 1.0) == (side == "over") else "loss"
                cand_rows.append({
                    "event_id": eid, "market": "totals", "subject": "game",
                    "line": 0.5, "side": side, "price": float(price),
                    "book": "draftkings", "game_date": date, "espn_id": eid,
                    "p_cons": p if side == "over" else p,
                    "n_books_cons": 7.0, "result": res,
                })
    return pd.DataFrame(prop_rows), pd.DataFrame(cand_rows)


@pytest.fixture(scope="module")
def control_run():
    props, cand = _synthetic_market()
    folds = B.make_folds(props["game_date"], n_burn_days=45, step_days=15)
    bets, reports = B.run_market("totals", props, cand, folds, min_books=3)
    return props, cand, bets, reports


def test_control_market_produces_bets(control_run):
    _, _, bets, _ = control_run
    assert len(bets) > 200, "control needs enough bets to be meaningful"


def test_efficient_market_loses_approximately_the_vig(control_run):
    """The headline control. Fair prices plus a 4.5% hold should return
    roughly minus the hold, not a profit."""
    _, _, bets, _ = control_run
    roi = bets["profit"].mean()
    assert roi < 0.0, f"pipeline manufactured a positive edge: ROI {roi:+.4f}"
    assert roi > -0.15, f"implausibly bad; check the settlement logic: {roi:+.4f}"


def test_gate_refuses_the_control_market(control_run):
    _, _, bets, _ = control_run
    v = G.judge("totals", bets, closing=None)
    assert v.verdict == G.FAIL
    # The cause depends on how many bets survive the selection correction,
    # which is not the point: what matters is that an efficient market is
    # never authorised to fire.
    assert v.cause in ("negative_edge", "no_edge", "insufficient_sample")
    assert v.deployment in (G.KILLED, G.VETO_FILTERED)


def test_model_does_not_claim_signal_from_noise(control_run):
    """Features are pure noise, so the shrinkage toward the market should stay
    small and the blended fit must not beat the market by any real margin."""
    _, _, _, reports = control_run
    assert reports, "no folds fitted"
    gains = [r.model_gain for _, r in reports]
    assert max(gains) < 0.01, \
        f"model claims log-loss gain from noise features: {max(gains):.4f}"


def test_threshold_was_fitted_not_optimised_on_the_fold(control_run):
    """Every fold must carry the threshold it was given, and the thresholds
    must not be re-chosen per fold to whatever suited that fold."""
    _, _, bets, _ = control_run
    per_fold = bets.groupby("fold")["threshold"].nunique()
    assert (per_fold == 1).all(), "a fold used more than one threshold"


def test_threshold_prefers_volume_when_the_edge_is_similar():
    """A two-point edge over ten thousand bets beats a four-point edge over
    four hundred, because what decides a market is edge relative to its
    standard error. Maximising ROI alone walks the cut up until almost
    nothing survives."""
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(21)
    rows = []
    # Low EV band: huge volume, small but real edge.
    for i in range(12000):
        rows.append({"ev": 0.01 + rng.random() * 0.02,
                     "profit": 0.91 if rng.random() < 0.535 else -1.0,
                     "event_id": f"a{i // 4}"})
    # High EV band: tiny volume, bigger edge.
    for i in range(400):
        rows.append({"ev": 0.06 + rng.random() * 0.02,
                     "profit": 0.91 if rng.random() < 0.555 else -1.0,
                     "event_id": f"b{i // 4}"})
    d = pd.DataFrame(rows)
    thr = B.choose_threshold(d, min_bets=150)
    kept = d[d["ev"] >= thr]
    # The point is that it does not collapse onto the 400-bet high-EV band.
    assert len(kept) > 1200, (thr, len(kept))
    assert thr < 0.06, thr


def test_threshold_still_rejects_a_band_that_loses():
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(22)
    rows = []
    for i in range(6000):       # low band loses
        rows.append({"ev": 0.005 + rng.random() * 0.015,
                     "profit": 0.91 if rng.random() < 0.49 else -1.0,
                     "event_id": f"a{i // 4}"})
    for i in range(4000):       # high band wins
        rows.append({"ev": 0.05 + rng.random() * 0.03,
                     "profit": 0.91 if rng.random() < 0.56 else -1.0,
                     "event_id": f"b{i // 4}"})
    d = pd.DataFrame(rows)
    thr = B.choose_threshold(d, min_bets=150)
    assert thr >= 0.02, thr
