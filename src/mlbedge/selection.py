"""Correct the winner's curse in a shopped price.

Shopping twenty books and taking the best number is not a neutral act. Every
book's price is the truth plus an error, and taking the maximum selects on that
error: conditional on a book being the best price available, its price is
longer than the truth *more often than the consensus suggests*. The apparent
edge against a cross-book consensus is therefore biased upward, and the bias
grows with how extreme the selection was.

Measured on a season of real prices, the pattern is unambiguous. Bucketing
bets by apparent edge and comparing the realised win rate to both estimates:

    apparent edge   consensus error   price error
      1 - 2%            -1.7%            -0.3%
      2 - 3%            -2.7%            -0.3%
      3 - 5%            -1.5%            +2.0%

Below about three points the *book* is right and the consensus is wrong —
those bets lose. Above it the price genuinely is long. A model fitted at the
proposition level cannot see this, because it never learns that this
particular price is the best of N.

So the correction is fitted where the selection happens: on training
candidates, mapping apparent edge to *realised* edge over the price, with an
isotonic fit so the relationship stays monotone and cannot invent structure
between the buckets it was given.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

MIN_FIT = 500


class SelectionCalibrator:
    """Maps apparent edge over the offered price to realised edge."""

    def __init__(self, min_fit: int = MIN_FIT):
        self.min_fit = min_fit
        self.iso: IsotonicRegression | None = None
        self.n_fit = 0
        self.fallback = 0.0

    def fit(self, bets: pd.DataFrame, edge_col: str = "apparent_edge",
            outcome_col: str = "won_flag", price_prob_col: str = "p_raw"
            ) -> "SelectionCalibrator":
        d = bets.dropna(subset=[edge_col, outcome_col, price_prob_col])
        self.n_fit = len(d)
        if len(d) < self.min_fit:
            return self
        x = d[edge_col].to_numpy(dtype=float)
        # Realised edge over the price actually taken.
        y = d[outcome_col].to_numpy(dtype=float) - d[price_prob_col].to_numpy(dtype=float)
        # Monotone: a larger apparent edge should never imply a smaller real
        # one. Without that constraint the fit chases noise in the sparse tail,
        # which is precisely where the bets are.
        self.iso = IsotonicRegression(increasing=True, out_of_bounds="clip")
        self.iso.fit(x, y)
        return self

    def realised_edge(self, apparent: np.ndarray) -> np.ndarray:
        if self.iso is None:
            return np.zeros(len(apparent), dtype=float)
        return self.iso.predict(np.asarray(apparent, dtype=float))

    def apply(self, bets: pd.DataFrame, edge_col: str = "apparent_edge",
              price_prob_col: str = "p_raw") -> pd.DataFrame:
        """Attach the bias-corrected win probability and EV."""
        from .odds import ev_per_unit

        d = bets.copy()
        if d.empty:
            return d
        adj = self.realised_edge(d[edge_col].to_numpy(dtype=float))
        d["realised_edge"] = adj
        d["p_cal"] = np.clip(d[price_prob_col].to_numpy(dtype=float) + adj,
                             1e-4, 1 - 1e-4)
        d["ev_cal"] = ev_per_unit(d["p_cal"].to_numpy(),
                                  d["price"].to_numpy(dtype=float))
        return d

    @property
    def fitted(self) -> bool:
        return self.iso is not None


def add_apparent_edge(bets: pd.DataFrame) -> pd.DataFrame:
    """Apparent edge of the model's probability over the offered price."""
    d = bets
    if "p_raw" not in d.columns:
        from .odds import american_to_prob
        d = d.assign(p_raw=american_to_prob(d["price"].to_numpy(dtype=float)))
    d = d.assign(apparent_edge=d["p_model"].to_numpy(dtype=float)
                 - d["p_raw"].to_numpy(dtype=float))
    if "won_flag" not in d.columns and "result" in d.columns:
        d = d.assign(won_flag=(d["result"] == "win").astype(float))
    return d
