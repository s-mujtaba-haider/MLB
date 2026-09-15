"""The line ladder must recover a known distribution and stay monotone."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlbedge import ladder as L


def _synthetic(n=30000, seed=3):
    """Players with a known Poisson rate. The primary line (0.5) is quoted
    correctly by many books; the ladder must then recover 1.5, 2.5, 3.5."""
    rng = np.random.default_rng(seed)
    mu = rng.uniform(0.3, 1.8, n)
    y = rng.poisson(mu)
    rows = []
    for i in range(n):
        p05 = 1 - np.exp(-mu[i])              # true P(Y > 0.5)
        for line, nb in ((0.5, 9), (1.5, 2), (2.5, 1)):
            rows.append({
                "event_id": f"e{i}", "market": "batter_hits",
                "subject": f"p{i}", "line": line,
                "p_cons_all": p05 if line == 0.5 else np.nan,
                "n_books_prop": nb, "n_quotes": nb * 2,
                "stat_value": float(y[i]),
            })
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def fitted():
    df = _synthetic()
    tbl = L.smooth(L.fit(df, min_cell=40))
    return df, tbl


def test_primary_line_is_the_best_quoted_one(fitted):
    df, _ = fitted
    prim = L.primary_lines(df)
    assert (prim["primary_line"] == 0.5).all()


def test_table_covers_the_alternate_lines(fitted):
    _, tbl = fitted
    assert "batter_hits" in tbl
    cells = tbl["batter_hits"]["0.5"]
    assert {"0.5", "1.5", "2.5"} <= set(cells)


def test_ladder_recovers_a_known_poisson_tail(fitted):
    """With a true Poisson rate, P(Y>1.5) given P(Y>0.5) is determined. The
    learned table must land close to it -- this is the whole premise."""
    df, tbl = fitted
    out = L.attach(df, tbl)
    alt = out[(out["line"] == 1.5) & out["p_ladder"].notna()]
    assert len(alt) > 1000

    # Invert P(Y>0.5)=1-exp(-mu) for mu, then compute the true P(Y>1.5).
    from scipy.stats import poisson
    p05 = alt["p_primary"].to_numpy()
    mu = -np.log(np.clip(1 - p05, 1e-9, 1))
    truth = 1 - poisson.cdf(1, mu)
    err = np.abs(alt["p_ladder"].to_numpy() - truth)
    assert err.mean() < 0.03, f"mean ladder error {err.mean():.4f}"


def test_probabilities_decrease_as_the_line_rises(fitted):
    _, tbl = fitted
    cells = tbl["batter_hits"]["0.5"]
    for bk in cells["0.5"]:
        seq = [cells[t][bk] for t in ("0.5", "1.5", "2.5") if bk in cells[t]]
        assert all(a >= b - 1e-9 for a, b in zip(seq, seq[1:])), seq


def test_probabilities_increase_with_strength(fitted):
    _, tbl = fitted
    for t, pb in tbl["batter_hits"]["0.5"].items():
        keys = sorted(pb, key=int)
        vals = [pb[k] for k in keys]
        assert all(a <= b + 1e-9 for a, b in zip(vals, vals[1:])), (t, vals)


def test_silent_where_it_has_no_evidence():
    """A market or line the table never saw must return NaN, not a guess."""
    df = _synthetic(n=2000)
    tbl = L.smooth(L.fit(df, min_cell=40))
    probe = pd.DataFrame([{
        "market": "pitcher_outs", "line": 15.5, "p_primary": 0.5,
        "primary_line": 15.5}])
    assert np.isnan(L.apply(probe, tbl)[0])


def test_empty_table_is_safe():
    probe = pd.DataFrame([{"market": "batter_hits", "line": 0.5,
                           "p_primary": 0.4, "primary_line": 0.5}])
    assert np.isnan(L.apply(probe, {})[0])
