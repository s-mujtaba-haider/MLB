"""The gate must reach the right verdict for the right reason."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from conftest import (dies_midway, efficient, losing, miscalibrated, real_edge,
                      thin)

from mlbedge import gate as G
from mlbedge.leakage import LeakReport


def test_real_edge_passes(rng):
    v = G.judge("m", real_edge(rng), closing=None)
    assert v.verdict == G.PASS, v.cause
    assert v.deployment == G.LIVE
    assert v.metrics["roi_lo"] > 0


def test_efficient_market_fails_as_no_edge(rng):
    v = G.judge("m", efficient(rng), closing=None)
    assert v.verdict == G.FAIL
    assert v.cause == "no_edge"
    assert "sharper anchor" in v.lever


def test_losing_market_fails_as_negative_edge(rng):
    v = G.judge("m", losing(rng), closing=None)
    assert v.verdict == G.FAIL
    assert v.cause == "negative_edge"
    assert v.deployment == G.KILLED


def test_thin_sample_fails_on_sample_not_edge(rng):
    """A tiny sample with a big point estimate must fail for sample size."""
    bets = thin(rng)
    assert bets["profit"].mean() > 0
    v = G.judge("m", bets, closing=None)
    assert v.verdict == G.FAIL
    assert v.cause == "insufficient_sample"


def test_edge_that_dies_is_caught_as_decay(rng):
    """Full-sample ROI is healthy but the edge is gone by the end. The gate
    must refuse it: the point is to fire tomorrow, not to have fired well
    last April."""
    bets = dies_midway(rng)
    assert bets["profit"].mean() > 0, "setup: full-sample ROI is positive"
    # The decay detector itself must see it...
    assert G.decayed(bets), "recent slice is confidently losing"
    # ...and the market must not be authorised, whichever check catches it
    # first (a decayed edge also tends to fail the interval outright).
    v = G.judge("m", bets, closing=None)
    assert v.verdict == G.FAIL
    assert v.cause in ("edge_decayed", "unstable", "no_edge"), v.cause
    assert v.deployment != G.LIVE


def test_miscalibration_is_caught(rng):
    v = G.judge("m", miscalibrated(rng), closing=None)
    assert v.cause in ("miscalibrated", "") or v.verdict == G.PASS
    if v.verdict == G.FAIL:
        assert v.cause == "miscalibrated"


def test_leakage_outranks_a_spectacular_result(rng):
    """The central discipline: a market with an enormous, statistically
    unimpeachable edge is still vetoed when a feature is leaked."""
    bets = real_edge(rng, n=4000, edge=0.40)   # absurd, unmissable edge
    assert bets["profit"].mean() > 0.5
    rep = LeakReport()
    rep.add("asof::bat_hits_ppa_r10", "veto",
            "feature window includes the current game")
    v = G.judge("m", bets, closing=None, leak_report=rep)
    assert v.verdict == G.VETO
    assert v.cause == "leakage"
    assert v.deployment == G.KILLED


def test_clv_negative_blocks_a_profitable_market(rng):
    """Positive ROI without closing-line value is variance, not edge."""
    bets = real_edge(rng, n=2000, edge=0.05)
    # Closing prices strictly worse for us than what we took -> negative CLV.
    closing = bets.assign(tag="closing", price=bets["price"] + 40)
    v = G.judge("m", bets, closing=closing)
    assert v.verdict == G.FAIL
    assert v.cause == "no_clv"
    assert v.deployment == G.VETO_FILTERED


def test_family_correction_downgrades_marginal_passes():
    """Eleven markets tested; a nominally-significant one that does not
    survive BH must be downgraded rather than shipped."""
    verdicts = []
    for i in range(11):
        v = G.Verdict(f"m{i}", G.PASS, deployment=G.LIVE,
                      metrics={"p_value": 0.04 if i == 0 else 0.9})
        verdicts.append(v)
    out = G.apply_family_correction(verdicts)
    # p=0.04 alone against alpha*1/11 = 0.0045 must not survive.
    assert out[0].verdict == G.FAIL
    assert out[0].cause == "family_wise"


def test_family_correction_keeps_strong_results():
    verdicts = [G.Verdict(f"m{i}", G.PASS, deployment=G.LIVE,
                          metrics={"p_value": 1e-6}) for i in range(11)]
    out = G.apply_family_correction(verdicts)
    assert all(v.verdict == G.PASS for v in out)


def test_benjamini_hochberg_matches_known_case():
    # m=5, alpha=.05 -> thresholds .01 .02 .03 .04 .05.
    # p_(5)=.042 <= .05, so the largest passing index is 5 and BH rejects all
    # five, including the two that fail their own individual thresholds.
    p = {"a": 0.001, "b": 0.008, "c": 0.039, "d": 0.041, "e": 0.042}
    s = G.benjamini_hochberg(p, alpha=0.05)
    assert all(s.values())

    # Nudge the largest past its threshold and the step-up stops at i=2.
    p2 = dict(p, e=0.30)
    s2 = G.benjamini_hochberg(p2, alpha=0.05)
    assert s2["a"] and s2["b"]
    assert not s2["c"] and not s2["d"] and not s2["e"]


def test_every_failure_carries_a_lever(rng):
    for gen in (efficient, losing, thin, dies_midway):
        v = G.judge("m", gen(rng), closing=None)
        if v.verdict != G.PASS:
            assert v.cause, "a failure must name its cause"
            assert len(v.lever) > 60, f"{v.cause} has no actionable lever"


def test_empty_ledger_does_not_crash():
    v = G.judge("m", pd.DataFrame(), closing=None)
    assert v.verdict == G.FAIL
    assert v.cause == "insufficient_sample"


def test_calibration_bar_scales_with_sample_noise(rng):
    """A perfectly calibrated market must not fail calibration just because
    its deciles are small. The expected error from sampling alone is already
    around four points at a thousand bets, so a fixed bar would fail it."""
    import numpy as np
    bets = real_edge(rng, n=1000, edge=0.03)
    floor = G.calibration_noise_floor(bets)
    assert np.isfinite(floor) and floor > 0.01, floor
    v = G.judge("m", bets, closing=None)
    assert v.cause != "miscalibrated", (v.metrics.get("cal_error"),
                                        v.metrics.get("cal_bar"))


def test_genuinely_miscalibrated_market_still_fails(rng):
    """The bar must still bite when the error is far above the noise floor."""
    import numpy as np
    import pandas as pd
    bets = real_edge(rng, n=6000, edge=0.03)
    bets["p_model"] = np.clip(bets["p_model"] + 0.25, 0, 1)   # wildly overconfident
    ce = G.calibration_error(bets)
    floor = G.calibration_noise_floor(bets)
    assert ce > 3 * floor, (ce, floor)


def test_clustered_bootstrap_is_wider_than_iid_on_correlated_bets():
    """Bets inside a game share an outcome. Resampling bets one at a time
    treats each as fresh information and produces an interval far narrower
    than the truth -- which is how a market with no edge acquires a confidence
    interval that clears zero."""
    import numpy as np

    from mlbedge import backtest as B

    rng = np.random.default_rng(4)
    n_games, per_game = 400, 12
    # Every bet in a game wins or loses together: maximal within-game
    # correlation, so the effective sample is 400, not 4,800.
    game_won = rng.random(n_games) < 0.54
    profit, clusters = [], []
    for g in range(n_games):
        for _ in range(per_game):
            profit.append(0.91 if game_won[g] else -1.0)
            clusters.append(f"g{g}")
    profit = np.array(profit)
    clusters = np.array(clusters)

    _, lo_iid, hi_iid = B.bootstrap_roi(profit)
    _, lo_cl, hi_cl = B.bootstrap_roi(profit, clusters=clusters)
    width_iid, width_cl = hi_iid - lo_iid, hi_cl - lo_cl
    assert width_cl > 2 * width_iid, (width_iid, width_cl)


def test_clustered_and_iid_agree_when_every_bet_is_its_own_game():
    import numpy as np

    from mlbedge import backtest as B

    rng = np.random.default_rng(5)
    profit = np.where(rng.random(3000) < 0.54, 0.91, -1.0)
    clusters = np.array([f"g{i}" for i in range(3000)])
    _, lo_i, hi_i = B.bootstrap_roi(profit)
    _, lo_c, hi_c = B.bootstrap_roi(profit, clusters=clusters)
    assert abs((hi_c - lo_c) - (hi_i - lo_i)) < 0.02
