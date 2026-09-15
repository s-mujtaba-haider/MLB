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
    """Maps apparent edge over the offered price to realised edge.

    Fitted separately per anchor type. A price judged against books quoting
    that exact line and a price judged against the ladder are two different
    bias regimes -- the ladder anchor carries its own estimation error on top
    of the selection effect -- and pooling them lets the better-measured group
    absorb the other's correction.
    """

    def __init__(self, min_fit: int = MIN_FIT, group_col: str = "anchor_src"):
        self.min_fit = min_fit
        self.group_col = group_col
        self.iso: IsotonicRegression | None = None
        self.by_group: dict[str, IsotonicRegression] = {}
        self.cond = None          # conditional model, when there is data for it
        self.n_fit = 0

    @staticmethod
    def _fit_one(x: np.ndarray, y: np.ndarray) -> IsotonicRegression:
        # Monotone: a larger apparent edge should never imply a smaller real
        # one. Without that constraint the fit chases noise in the sparse tail,
        # which is precisely where the bets are.
        iso = IsotonicRegression(increasing=True, out_of_bounds="clip")
        iso.fit(x, y)
        return iso

    def fit(self, bets: pd.DataFrame, edge_col: str = "apparent_edge",
            outcome_col: str = "won_flag", price_prob_col: str = "p_raw"
            ) -> "SelectionCalibrator":
        d = bets.dropna(subset=[edge_col, outcome_col, price_prob_col])
        self.n_fit = len(d)
        if len(d) < self.min_fit:
            return self
        x = d[edge_col].to_numpy(dtype=float)
        # Realised edge over the price actually taken.
        y = (d[outcome_col].to_numpy(dtype=float)
             - d[price_prob_col].to_numpy(dtype=float))
        self.iso = self._fit_one(x, y)

        if self.group_col in d.columns:
            for key, sub in d.groupby(self.group_col, observed=True):
                if len(sub) < self.min_fit:
                    continue
                self.by_group[str(key)] = self._fit_one(
                    sub[edge_col].to_numpy(dtype=float),
                    sub[outcome_col].to_numpy(dtype=float)
                    - sub[price_prob_col].to_numpy(dtype=float))

        self._fit_conditional(d, edge_col, outcome_col)
        return self

    # -- conditional model -------------------------------------------------
    # Whether an apparent edge is real is not a function of its size alone. An
    # edge of four points against three books is a different proposition from
    # the same four points against fifteen, and a stale number an hour out is
    # different from one taken at the bell. This learns P(win) from the price
    # and that context together, constrained to stay increasing in the apparent
    # edge so it cannot invent a non-monotone story out of noise.
    COND_FEATURES = ("apparent_edge", "logit_price", "n_books_cons",
                     "lead_min", "book_tier", "is_ladder")
    MIN_COND_FIT = 4000

    def _cond_matrix(self, d: pd.DataFrame, edge_col: str) -> np.ndarray | None:
        from .config import book_tier
        from .devig import logit

        if "p_raw" not in d.columns:
            return None
        cols = {
            "apparent_edge": d[edge_col].to_numpy(dtype=float),
            "logit_price": logit(d["p_raw"].to_numpy(dtype=float)),
            "n_books_cons": pd.to_numeric(
                d.get("n_books_cons", pd.Series(np.nan, index=d.index)),
                errors="coerce").to_numpy(dtype=float),
            "lead_min": pd.to_numeric(
                d.get("lead_min", pd.Series(np.nan, index=d.index)),
                errors="coerce").to_numpy(dtype=float),
            "book_tier": (d["book"].map(book_tier).to_numpy(dtype=float)
                          if "book" in d.columns
                          else np.full(len(d), np.nan)),
            "is_ladder": ((d[self.group_col].astype(str) == "ladder")
                          .to_numpy(dtype=float)
                          if self.group_col in d.columns
                          else np.zeros(len(d))),
        }
        return np.column_stack([cols[c] for c in self.COND_FEATURES])

    def _fit_conditional(self, d: pd.DataFrame, edge_col: str,
                         outcome_col: str) -> None:
        if len(d) < self.MIN_COND_FIT:
            return
        X = self._cond_matrix(d, edge_col)
        if X is None:
            return
        from sklearn.ensemble import HistGradientBoostingClassifier

        # Monotone increasing in apparent edge; everything else unconstrained.
        mono = [1 if c == "apparent_edge" else 0 for c in self.COND_FEATURES]
        m = HistGradientBoostingClassifier(
            max_iter=200, learning_rate=0.05, max_leaf_nodes=15,
            min_samples_leaf=400, l2_regularization=1.0,
            monotonic_cst=mono, early_stopping=True, validation_fraction=0.2,
            n_iter_no_change=20, random_state=0)
        try:
            m.fit(X, d[outcome_col].to_numpy(dtype=float))
        except (ValueError, TypeError):
            return
        self.cond = m

    def realised_edge(self, apparent: np.ndarray,
                      groups: np.ndarray | None = None) -> np.ndarray:
        if self.iso is None:
            return np.zeros(len(apparent), dtype=float)
        apparent = np.asarray(apparent, dtype=float)
        out = self.iso.predict(apparent)
        if groups is not None and self.by_group:
            g = np.asarray(groups).astype(str)
            for key, iso in self.by_group.items():
                m = g == key
                if m.any():
                    out[m] = iso.predict(apparent[m])
        return out

    def apply(self, bets: pd.DataFrame, edge_col: str = "apparent_edge",
              price_prob_col: str = "p_raw") -> pd.DataFrame:
        """Attach the bias-corrected win probability and EV."""
        from .odds import ev_per_unit

        d = bets.copy()
        if d.empty:
            return d
        groups = (d[self.group_col].to_numpy()
                  if self.group_col in d.columns else None)
        adj = self.realised_edge(d[edge_col].to_numpy(dtype=float), groups)
        d["realised_edge"] = adj
        p_iso = np.clip(d[price_prob_col].to_numpy(dtype=float) + adj,
                        1e-4, 1 - 1e-4)

        if self.cond is not None:
            X = self._cond_matrix(d, edge_col)
            if X is not None:
                p_cond = self.cond.predict_proba(X)[:, 1]
                # Average the two in probability space. The isotonic fit is
                # robust but coarse; the conditional model is sharper but can
                # drift where a context slice is thin. Blending keeps most of
                # the sharpening without betting the farm on it.
                p_iso = np.clip(0.5 * p_iso + 0.5 * p_cond, 1e-4, 1 - 1e-4)
        d["p_cal"] = p_iso
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
