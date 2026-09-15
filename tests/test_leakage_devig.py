"""Leakage detection and the leave-one-out consensus.

Point-in-time integrity is the claim the whole project rests on, so the guard
is tested against deliberately planted violations rather than trusted.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlbedge import config as C
from mlbedge import leakage as LK
from mlbedge.devig import attach_consensus, book_views, consensus


# ---------------------------------------------------------------------------
# Quote timing
# ---------------------------------------------------------------------------

def test_quote_after_first_pitch_is_vetoed():
    q = pd.DataFrame({"lead_min": [30.0, 30.0, -5.0]})
    rep = LK.LeakReport()
    LK.check_quote_timing(q, rep)
    assert rep.vetoed
    assert "after first pitch" in rep.findings[0].detail


def test_clean_quote_timing_passes():
    rep = LK.LeakReport()
    LK.check_quote_timing(pd.DataFrame({"lead_min": [30.0, 45.0]}), rep)
    assert not rep.vetoed


def test_train_test_bleed_is_vetoed():
    train = pd.DataFrame({"game_date": ["2025-05-01", "2025-06-15"]})
    test = pd.DataFrame({"game_date": ["2025-06-01", "2025-06-20"]})
    rep = LK.LeakReport()
    LK.check_fold_boundary(train, test, rep)
    assert rep.vetoed


def test_clean_fold_boundary_passes():
    train = pd.DataFrame({"game_date": ["2025-05-01", "2025-05-31"]})
    test = pd.DataFrame({"game_date": ["2025-06-01"]})
    rep = LK.LeakReport()
    LK.check_fold_boundary(train, test, rep)
    assert not rep.vetoed


# ---------------------------------------------------------------------------
# As-of recomputation: the structural check
# ---------------------------------------------------------------------------

def _rolling_frame(include_current: bool):
    """Build a rolling feature either correctly (prior games only) or with the
    current game folded in -- the classic off-by-one that leaks the outcome."""
    rng = np.random.default_rng(3)
    rows = []
    for ent in ("p1", "p2", "p3"):
        vals = rng.normal(0.3, 0.1, 40)
        for i, v in enumerate(vals):
            rows.append({"ent": ent, "d": f"2025-05-{i + 1:02d}", "val": v})
    src = pd.DataFrame(rows)
    src = src.sort_values(["ent", "d"]).reset_index(drop=True)
    g = src.groupby("ent")["val"]
    if include_current:
        src["feat"] = g.transform(lambda s: s.rolling(10, min_periods=2).mean())
    else:
        src["feat"] = g.transform(
            lambda s: s.shift(1).rolling(10, min_periods=2).mean())
    return src


def test_correct_lagged_feature_reproduces_exactly():
    src = _rolling_frame(include_current=False)
    rep = LK.LeakReport()
    LK.check_feature_asof(src, src, "ent", "d", "val", "feat", 10, rep)
    assert not rep.vetoed
    assert rep.findings[-1].severity == "ok"


def test_planted_leak_in_a_rolling_feature_is_caught():
    """A window that includes the current game sees its own outcome. This is
    the single most common way a sports model fools itself, and the guard must
    identify it specifically -- not merely notice a mismatch."""
    src = _rolling_frame(include_current=True)
    rep = LK.LeakReport()
    LK.check_feature_asof(src, src, "ent", "d", "val", "feat", 10, rep)
    assert rep.vetoed
    assert "includes the current game" in rep.findings[-1].detail


# ---------------------------------------------------------------------------
# Conditional signal
# ---------------------------------------------------------------------------

def test_leaked_outcome_feature_is_flagged():
    rng = np.random.default_rng(7)
    n = 4000
    p = rng.uniform(0.35, 0.65, n)
    y = (rng.random(n) < p).astype(float)
    df = pd.DataFrame({
        "won": y,
        "logit_cons": np.log(p / (1 - p)),
        "honest": rng.normal(size=n),
        "leaked": y + rng.normal(0, 0.25, n),   # a feature that saw the result
    })
    rep = LK.LeakReport()
    LK.check_conditional_signal(df, ["honest", "leaked"], "won", "logit_cons", rep)
    assert rep.vetoed
    assert "leaked" in rep.findings[-1].detail


def test_honest_features_do_not_trip_the_alarm():
    rng = np.random.default_rng(7)
    n = 4000
    p = rng.uniform(0.35, 0.65, n)
    y = (rng.random(n) < p).astype(float)
    df = pd.DataFrame({"won": y, "logit_cons": np.log(p / (1 - p)),
                       "a": rng.normal(size=n), "b": rng.normal(size=n)})
    rep = LK.LeakReport()
    LK.check_conditional_signal(df, ["a", "b"], "won", "logit_cons", rep)
    assert not rep.vetoed


# ---------------------------------------------------------------------------
# Leave-one-out consensus
# ---------------------------------------------------------------------------

def _quotes(prices_by_book, market="batter_hits", line=0.5):
    rows = []
    for book, (po, pu) in prices_by_book.items():
        for side, price in (("over", po), ("under", pu)):
            rows.append({"event_id": "E", "market": market, "subject": "X",
                         "line": line, "side": side, "price": price,
                         "book": book, "region": "us", "tag": "decision",
                         "snapshot_ts": "2025-06-10T22:00:00Z",
                         "commence_time": "2025-06-10T22:30:00Z",
                         "lead_min": 30.0})
    return pd.DataFrame(rows)


def test_devig_recovers_a_fair_book():
    q = _quotes({"pinnacle": (-105, -105)})
    bv = book_views(q)
    assert bv["p_over"].iloc[0] == pytest.approx(0.5, abs=1e-9)
    assert bv["hold"].iloc[0] > 0


def test_book_is_excluded_from_its_own_benchmark():
    """The central guard against a circular edge. An outlier book must not be
    allowed to drag the consensus it is judged against toward itself."""
    q = _quotes({"pinnacle": (-110, -110), "draftkings": (-110, -110),
                 "betmgm": (-110, -110), "bovada": (+250, -320)})
    bv = consensus(book_views(q), min_books=2)
    outlier = bv[bv["book"] == "bovada"].iloc[0]
    # Its own de-vigged view is far from even money...
    assert outlier["p_over"] < 0.35
    # ...but the benchmark it is judged against knows nothing about it.
    assert outlier["p_cons_loo"] == pytest.approx(0.5, abs=1e-6)
    # The all-book consensus, by contrast, has been dragged down.
    assert outlier["p_cons_all"] < 0.5


def test_loo_consensus_needs_enough_other_books():
    q = _quotes({"pinnacle": (-110, -110), "draftkings": (-110, -110)})
    bv = consensus(book_views(q), min_books=2)
    # Each book has only one other book behind it, below the minimum of 2.
    assert bv["p_cons_loo"].isna().all()


def test_one_sided_quotes_still_get_a_benchmark():
    """FanDuel posting an Over with no Under is very common on alt lines. It
    contributes nothing to the consensus, so the all-book consensus is already
    leave-one-out for it, and it must still be priceable."""
    q = _quotes({"pinnacle": (-110, -110), "draftkings": (-108, -112),
                 "betmgm": (-112, -108)})
    one_sided = pd.DataFrame([{
        "event_id": "E", "market": "batter_hits", "subject": "X", "line": 0.5,
        "side": "over", "price": 160, "book": "fanduel", "region": "us",
        "tag": "decision", "snapshot_ts": "2025-06-10T22:00:00Z",
        "commence_time": "2025-06-10T22:30:00Z", "lead_min": 30.0}])
    out = attach_consensus(pd.concat([q, one_sided], ignore_index=True),
                           min_books=2)
    fd = out[out["book"] == "fanduel"].iloc[0]
    assert np.isfinite(fd["p_cons"])
    assert fd["p_cons"] == pytest.approx(0.5, abs=0.02)


def test_dfs_operators_do_not_vote_in_the_consensus():
    """A DFS pick'em line is posted at a fixed house price and is not an
    opinion about probability."""
    q = _quotes({"pinnacle": (-110, -110), "draftkings": (-110, -110),
                 "prizepicks": (-119, -119)})
    bv = book_views(q)
    assert bv.loc[bv["book"] == "prizepicks", "w"].iloc[0] == 0.0


def test_absurd_hold_is_not_trusted_as_an_anchor():
    q = _quotes({"pinnacle": (-110, -110), "draftkings": (-110, -110),
                 "junkbook": (-400, -400)})
    bv = book_views(q)
    assert bv.loc[bv["book"] == "junkbook", "w"].iloc[0] == 0.0
