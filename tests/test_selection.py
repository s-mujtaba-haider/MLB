"""The winner's-curse correction.

The synthetic here reproduces what a season of real prices actually shows:
when a shopped price looks good against a cross-book consensus, the consensus
is the thing that is wrong. The price is close to the truth, and the apparent
edge is an artefact of having selected the most favourable of twenty numbers.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from mlbedge.odds import american_to_prob, profit_per_unit
from mlbedge.selection import SelectionCalibrator, add_apparent_edge


def _price_from_prob(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 0.02, 0.97)
    return np.where(p >= 0.5, -100 * p / (1 - p), 100 * (1 - p) / p)


def _shopped(n=20000, margin=0.022, real_edge=0.0, seed=5):
    """A shopped market with no real edge (unless `real_edge` is given).

    p_raw   -- what the best available price implies: the truth plus the vig
    p_model -- the leave-one-out consensus, biased upward by the selection
    """
    rng = np.random.default_rng(seed)
    p_true = rng.uniform(0.35, 0.65, n)
    intensity = rng.uniform(0.0, 1.0, n)
    bias = 0.05 * intensity                      # consensus overstates
    p_raw = np.clip(p_true + margin - real_edge, 0.05, 0.95)
    p_model = np.clip(p_true + bias, 0.05, 0.95)
    won = (rng.random(n) < p_true).astype(float)
    price = _price_from_prob(p_raw)
    d = pd.DataFrame({
        "p_model": p_model, "p_raw": american_to_prob(price), "price": price,
        "won_flag": won, "result": np.where(won == 1, "win", "loss"),
        "p_true": p_true,
    })
    d["payout"] = profit_per_unit(d["price"].to_numpy(dtype=float))
    d["profit"] = np.where(d["won_flag"] == 1, d["payout"], -1.0)
    return add_apparent_edge(d)


def test_setup_apparent_edge_is_large_but_unprofitable():
    d = _shopped()
    # A large minority of bets look clearly +EV against the consensus...
    assert (d["apparent_edge"] > 0.02).mean() > 0.10
    # ...and collectively they are not.
    assert d["profit"].mean() < 0, "setup: but there is none to collect"


def test_uncorrected_selection_loses_money():
    """Betting the apparent edge is exactly the mistake being corrected."""
    d = _shopped()
    bet = d[d["apparent_edge"] > 0.02]
    assert len(bet) > 2000
    assert bet["profit"].mean() < 0


def test_calibrator_recovers_the_truth_from_the_price():
    d = _shopped()
    cal = SelectionCalibrator().fit(d)
    assert cal.fitted
    out = cal.apply(d)
    err_model = float(np.abs(out["p_model"] - out["p_true"]).mean())
    err_cal = float(np.abs(out["p_cal"] - out["p_true"]).mean())
    assert err_cal < err_model / 2, (err_cal, err_model)


def test_corrected_ev_refuses_the_bets():
    """With the bias removed, nothing should clear a sane EV threshold."""
    d = _shopped()
    out = SelectionCalibrator().fit(d).apply(d)
    n_before = int((d["apparent_edge"] > 0.02).sum())
    n_after = int((out["ev_cal"] > 0.02).sum())
    assert n_before > 2000
    assert n_after < n_before * 0.05, (n_before, n_after)


def test_a_genuine_edge_survives_the_correction():
    """The correction must not simply refuse everything. With a real 4-point
    edge in the prices, the bets have to come back."""
    d = _shopped(real_edge=0.04)
    assert d["profit"].mean() > 0, "setup: this one really is profitable"
    out = SelectionCalibrator().fit(d).apply(d)
    kept = out[out["ev_cal"] > 0.02]
    assert len(kept) > 1000, len(kept)
    assert kept["profit"].mean() > 0


def test_monotone_in_apparent_edge():
    cal = SelectionCalibrator().fit(_shopped())
    adj = cal.realised_edge(np.linspace(-0.05, 0.15, 40))
    assert all(a <= b + 1e-9 for a, b in zip(adj, adj[1:]))


def test_unfitted_calibrator_is_a_no_op():
    d = _shopped(n=100)
    cal = SelectionCalibrator().fit(d)
    assert not cal.fitted
    out = cal.apply(d)
    assert np.allclose(out["p_cal"], out["p_raw"])


def test_calibrator_fits_anchor_types_separately():
    """A ladder-anchored price carries estimation error on top of the
    selection effect. Pooling the two regimes lets the better-measured group
    absorb the other's correction."""
    direct = _shopped(n=12000, margin=0.022, seed=1).assign(anchor_src="direct")
    # The ladder group has a much larger apparent edge for the same truth.
    ladder = _shopped(n=12000, margin=0.022, seed=2).assign(anchor_src="ladder")
    ladder["p_model"] = np.clip(ladder["p_model"] + 0.06, 0.05, 0.95)
    ladder = add_apparent_edge(ladder.drop(columns=["apparent_edge"]))
    both = pd.concat([direct, ladder], ignore_index=True)

    cal = SelectionCalibrator().fit(both)
    assert set(cal.by_group) == {"direct", "ladder"}
    out = cal.apply(both)
    # Each group must be corrected back toward its own truth.
    for key in ("direct", "ladder"):
        g = out[out["anchor_src"] == key]
        assert np.abs(g["p_cal"] - g["p_true"]).mean() < \
               np.abs(g["p_model"] - g["p_true"]).mean(), key


def _context_dependent(n=40000, seed=9):
    """Same apparent edge, two regimes.

    Against a deep consensus the edge is real; against a shallow one it is
    pure selection. An isotonic fit on edge alone must average the two and get
    both wrong; a conditional fit should tell them apart.
    """
    rng = np.random.default_rng(seed)
    deep = rng.random(n) < 0.5
    n_books = np.where(deep, rng.integers(10, 18, n), rng.integers(3, 5, n))
    p_true = rng.uniform(0.35, 0.65, n)
    apparent = rng.uniform(0.0, 0.10, n)
    # Deep consensus: the apparent edge is genuine. Shallow: it is noise.
    p_raw = np.where(deep, p_true - apparent, p_true)
    p_model = p_raw + apparent
    won = (rng.random(n) < p_true).astype(float)
    price = _price_from_prob(p_raw)
    d = pd.DataFrame({
        "p_model": p_model, "p_raw": american_to_prob(price), "price": price,
        "won_flag": won, "result": np.where(won == 1, "win", "loss"),
        "p_true": p_true, "n_books_cons": n_books.astype(float),
        "lead_min": 30.0, "book": "draftkings", "anchor_src": "direct",
        "deep": deep,
    })
    d["payout"] = profit_per_unit(d["price"].to_numpy(dtype=float))
    d["profit"] = np.where(d["won_flag"] == 1, d["payout"], -1.0)
    return add_apparent_edge(d)


def test_conditional_model_separates_real_from_selected_edges():
    d = _context_dependent()
    cal = SelectionCalibrator().fit(d)
    assert cal.cond is not None, "conditional model should fit at this size"
    out = cal.apply(d)

    deep = out[out["deep"]]
    shallow = out[~out["deep"]]
    # It should keep the genuine edges and discount the manufactured ones.
    assert deep["p_cal"].mean() - deep["p_raw"].mean() > \
           shallow["p_cal"].mean() - shallow["p_raw"].mean()

    # And betting on the corrected EV should beat betting on raw apparent edge.
    naive = d[d["apparent_edge"] > 0.04]
    smart = out[out["ev_cal"] > 0.02]
    assert len(smart) > 500
    assert smart["profit"].mean() > naive["profit"].mean()
