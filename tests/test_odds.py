"""Odds arithmetic and de-vigging.

These are the foundation: every probability, edge and ROI in the project is
built on them, so they are tested against closed-form answers rather than
against themselves.
"""
from __future__ import annotations

import numpy as np
import pytest

from mlbedge import odds as O
from mlbedge.devig import devig_pair


def test_american_to_prob_known_values():
    assert O.american_to_prob(-110) == pytest.approx(110 / 210)
    assert O.american_to_prob(+100) == pytest.approx(0.5)
    assert O.american_to_prob(+150) == pytest.approx(0.4)
    assert O.american_to_prob(-200) == pytest.approx(2 / 3)


def test_decimal_and_profit():
    assert O.american_to_decimal(+150) == pytest.approx(2.5)
    assert O.american_to_decimal(-200) == pytest.approx(1.5)
    assert O.profit_per_unit(-110) == pytest.approx(100 / 110)
    assert O.profit_per_unit(+250) == pytest.approx(2.5)


def test_price_probability_roundtrip():
    for a in (-1000, -250, -110, -101, 100, 105, 340, 2500):
        assert O.prob_to_american(O.american_to_prob(a)) == pytest.approx(a)


@pytest.mark.parametrize("method",
                         ["multiplicative", "additive", "power", "shin"])
def test_devig_sums_to_one(method):
    for prices in ([-110, -110], [-250, +200], [+450, -700], [-1200, +750]):
        raw = O.american_to_prob(np.array(prices, dtype=float))
        p = O.devig(raw, method=method)
        assert p.sum() == pytest.approx(1.0, abs=1e-8)
        assert np.all(p > 0) and np.all(p < 1)


@pytest.mark.parametrize("method",
                         ["multiplicative", "additive", "power", "shin"])
def test_symmetric_market_is_a_coin_flip(method):
    raw = O.american_to_prob(np.array([-110.0, -110.0]))
    assert O.devig(raw, method=method) == pytest.approx([0.5, 0.5], abs=1e-9)


def test_devig_methods_order_correctly_on_longshots():
    """The methods disagree exactly where props live. Multiplicative assumes
    margin is proportional to price and so leaves the longshot with the most
    probability; power strips the most; Shin sits between. Getting this
    ordering wrong means systematically buying or selling the wrong tail."""
    raw = O.american_to_prob(np.array([450.0, -700.0]))
    mult = O.devig(raw, "multiplicative")[0]
    shin = O.devig(raw, "shin")[0]
    power = O.devig(raw, "power")[0]
    assert power < shin < mult


def test_devig_is_a_strict_reduction_of_the_favourite_side():
    """De-vigging must lower every raw implied probability (the overround is
    being removed, not redistributed upward)."""
    raw = O.american_to_prob(np.array([-140.0, +120.0]))
    assert raw.sum() > 1.0
    for m in ("multiplicative", "additive", "power", "shin"):
        p = O.devig(raw, m)
        assert np.all(p <= raw + 1e-12)


def test_devig_rejects_incomplete_outcome_sets():
    with pytest.raises(ValueError):
        O.devig(np.array([0.5]))
    with pytest.raises(ValueError):
        O.devig(np.array([0.4, 0.4]), method="not_a_method")


def test_vectorised_devig_matches_scalar():
    """devig_pair is used on millions of rows via bisection; it must agree
    with the scalar root-finding implementation."""
    rng = np.random.default_rng(0)
    prices = rng.choice([-400, -250, -160, -110, 100, 130, 220, 600],
                        size=(200, 2)).astype(float)
    raw = O.american_to_prob(prices)
    for method in ("multiplicative", "additive", "power", "shin"):
        vec = devig_pair(raw[:, 0], raw[:, 1], method=method)
        for i in range(0, 200, 17):
            scal = O.devig(raw[i], method=method)[0]
            assert vec[i] == pytest.approx(scal, abs=1e-6), method


def test_ev_is_zero_at_the_break_even_price():
    """The most expensive confusion in betting code: the break-even number is
    the RAW implied probability, not the vig-free one."""
    for a in (-250.0, -110.0, 100.0, 380.0):
        assert O.ev_per_unit(O.american_to_prob(a), a) == pytest.approx(0, abs=1e-12)


def test_ev_accounts_for_pushes():
    """Ignoring push probability overstates the edge on integer lines."""
    no_push = O.ev_per_unit(0.50, 100)
    with_push = O.ev_per_unit(0.50, 100, p_push=0.10)
    assert with_push > no_push
    assert with_push == pytest.approx(0.50 * 1.0 - 0.40)


def test_kelly_is_zero_without_an_edge():
    assert O.kelly_fraction(O.american_to_prob(-110), -110) == pytest.approx(0, abs=1e-12)
    assert O.kelly_fraction(0.40, -110) == 0.0        # never negative
    assert O.kelly_fraction(0.60, -110) == pytest.approx((0.909090 * 0.6 - 0.4) / 0.909090, abs=1e-5)


def test_overround_of_a_fair_book_is_zero():
    assert O.overround(np.array([0.5, 0.5])) == pytest.approx(0.0)
    assert O.overround(O.american_to_prob(np.array([-110.0, -110.0]))) == pytest.approx(0.0476, abs=1e-4)
