"""Cross-book consensus: turn many quoted prices into one probability.

The single most important detail in this file is **leave-one-out**. To judge
whether DraftKings' price is good we compare it to what *every other* book
thinks. Including DraftKings in its own benchmark drags the benchmark toward
the price being judged and manufactures an edge that does not exist. Every
consensus here is computed twice: once from all books, and once per book with
that book removed. Only the leave-one-out number is ever allowed to price a
bet.

Aggregation happens in log-odds space, weighted by each book's prior precision
as a probability anchor (config.ANCHOR_WEIGHTS). Averaging probabilities
directly is wrong near 0 and 1, which is exactly where prop markets live.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C
from .odds import american_to_prob

EPS = 1e-9
PROP_KEY = ["event_id", "market", "subject", "line", "tag"]


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def expit(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40, 40)))


# ---------------------------------------------------------------------------
# Vectorised two-outcome de-vig
# ---------------------------------------------------------------------------

def devig_pair(raw_a: np.ndarray, raw_b: np.ndarray,
               method: str = "shin") -> np.ndarray:
    """De-vig a two-outcome market elementwise; returns the fair probability
    of side A.

    Vectorised over millions of rows: Shin and power are solved by fixed-
    iteration bisection rather than a per-row root finder.
    """
    a = np.clip(np.asarray(raw_a, dtype=float), EPS, 1 - EPS)
    b = np.clip(np.asarray(raw_b, dtype=float), EPS, 1 - EPS)
    s = a + b

    if method == "multiplicative":
        return a / s
    if method == "additive":
        return np.clip(a - (s - 1.0) / 2.0, EPS, 1 - EPS)
    if method == "power":
        lo = np.full_like(a, 0.05)
        hi = np.full_like(a, 10.0)
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            # a,b < 1, so a**k + b**k is DECREASING in k: f > 0 means the root
            # lies to the right and the lower bound is what moves.
            f = a ** mid + b ** mid - 1.0
            lo = np.where(f > 0, mid, lo)
            hi = np.where(f > 0, hi, mid)
        k = 0.5 * (lo + hi)
        pa = a ** k
        return np.clip(pa / (pa + b ** k), EPS, 1 - EPS)
    if method == "shin":
        def p_of_z(z, x):
            inner = z * z + 4.0 * (1.0 - z) * x * x / s
            return (np.sqrt(np.maximum(inner, 0.0)) - z) / (2.0 * (1.0 - z))

        lo = np.full_like(a, 1e-9)
        hi = np.full_like(a, 0.9999)
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            f = p_of_z(mid, a) + p_of_z(mid, b) - 1.0
            # sum is decreasing in z
            hi = np.where(f < 0, mid, hi)
            lo = np.where(f < 0, lo, mid)
        z = 0.5 * (lo + hi)
        pa, pb = p_of_z(z, a), p_of_z(z, b)
        tot = pa + pb
        out = np.where(tot > 0, pa / tot, a / s)
        # Where there is no overround to strip, Shin is undefined; fall back.
        return np.clip(np.where(s <= 1.0 + 1e-9, a / s, out), EPS, 1 - EPS)
    raise ValueError(f"unknown devig method {method!r}")


# ---------------------------------------------------------------------------
# Book-level fair probabilities
# ---------------------------------------------------------------------------

def book_views(quotes: pd.DataFrame, method: str = "shin") -> pd.DataFrame:
    """One row per (proposition, book) that quoted *both* sides.

    Returns the book's vig-free probability of the over/home side plus the
    margin it was charging.
    """
    q = quotes
    over_side = np.where(q["market"].isin(["h2h", "spreads"]), "home", "over")
    under_side = np.where(q["market"].isin(["h2h", "spreads"]), "away", "under")
    q = q.assign(_is_over=(q["side"].to_numpy() == over_side),
                 _is_under=(q["side"].to_numpy() == under_side))

    keys = PROP_KEY + ["book"]
    ov = (q[q["_is_over"]].groupby(keys, dropna=False, observed=True)["price"]
            .max().rename("price_over"))
    un = (q[q["_is_under"]].groupby(keys, dropna=False, observed=True)["price"]
            .max().rename("price_under"))
    bv = pd.concat([ov, un], axis=1).dropna().reset_index()
    if bv.empty:
        return bv.assign(p_over=[], hold=[])

    raw_o = american_to_prob(bv["price_over"].to_numpy(dtype=float))
    raw_u = american_to_prob(bv["price_under"].to_numpy(dtype=float))
    bv["p_over"] = devig_pair(raw_o, raw_u, method=method)
    bv["hold"] = raw_o + raw_u - 1.0
    bv["w"] = bv["book"].map(C.ANCHOR_WEIGHTS).fillna(C.DEFAULT_ANCHOR_WEIGHT)
    # A book quoting an absurd margin is not expressing an opinion worth
    # weighting; DFS operators post a line at a fixed house price.
    bv.loc[bv["book"].isin(C.DFS_BOOKS), "w"] = 0.0
    bv.loc[bv["hold"] > 0.35, "w"] = 0.0
    return bv


def consensus(book_view: pd.DataFrame, min_books: int = 2) -> pd.DataFrame:
    """Weighted log-odds consensus per proposition, plus a leave-one-out
    consensus for every book that contributed."""
    bv = book_view
    if bv.empty:
        return bv
    bv = bv.copy()
    bv["_l"] = logit(bv["p_over"].to_numpy())
    bv["_wl"] = bv["w"] * bv["_l"]

    g = bv.groupby(PROP_KEY, dropna=False, observed=True)
    sum_w = g["w"].transform("sum")
    sum_wl = g["_wl"].transform("sum")
    n = g["w"].transform("size")
    n_eff = g["w"].transform(lambda s: (s > 0).sum())

    bv["n_books"] = n.astype(int)
    bv["n_books_w"] = n_eff.astype(int)
    with np.errstate(invalid="ignore", divide="ignore"):
        bv["p_cons_all"] = expit(np.where(sum_w > 0, sum_wl / sum_w, np.nan))
        # leave-one-out
        loo_w = sum_w - bv["w"]
        loo_wl = sum_wl - bv["_wl"]
        bv["p_cons_loo"] = expit(np.where(loo_w > 0, loo_wl / loo_w, np.nan))
    bv["n_books_loo"] = (n_eff - (bv["w"] > 0).astype(int)).astype(int)
    bv.loc[bv["n_books_loo"] < min_books, "p_cons_loo"] = np.nan
    bv.loc[bv["n_books_w"] < min_books, "p_cons_all"] = np.nan
    return bv.drop(columns=["_l", "_wl"])


def attach_consensus(quotes: pd.DataFrame, method: str = "shin",
                     min_books: int = 2,
                     self_anchor_markets: frozenset[str] = frozenset()
                     ) -> pd.DataFrame:
    """Attach the consensus a quote should be judged against.

    For a book that quoted both sides, that is the leave-one-out consensus.
    For a book that quoted only one side (very common on alt lines, where
    FanDuel posts an Over and no Under), the book contributed nothing to the
    consensus, so the all-book consensus is already leave-one-out for it.

    `self_anchor_markets` covers the degenerate case where exactly one book in
    the world quotes a market -- batter strikeouts, for most of its history.
    There is no cross-book benchmark to build, so the book's own vig-free price
    becomes the anchor and the edge has to come from the model rather than from
    line shopping. That is not circular: with the book's own opinion as the
    starting point, the model has to disagree with it by more than the vig
    before any bet fires, and if the model adds nothing the shrinkage collapses
    to zero and the market produces no bets at all.
    """
    bv = consensus(book_views(quotes, method=method), min_books=min_books)
    q = quotes.copy()
    if bv.empty:
        q["p_cons"] = np.nan
        q["n_books_cons"] = 0
        q["hold_book"] = np.nan
        return q

    prop_all = (bv.drop_duplicates(PROP_KEY)
                  .set_index(PROP_KEY)[["p_cons_all", "n_books_w"]])
    per_book = bv.set_index(PROP_KEY + ["book"])[
        ["p_cons_loo", "n_books_loo", "hold", "p_over"]]

    q = q.join(prop_all, on=PROP_KEY)
    q = q.join(per_book, on=PROP_KEY + ["book"])

    contributed = q["p_cons_loo"].notna() | q["p_over"].notna()
    q["p_cons"] = np.where(contributed, q["p_cons_loo"], q["p_cons_all"])
    q["n_books_cons"] = np.where(contributed, q["n_books_loo"],
                                 q["n_books_w"]).astype("float")
    q = q.rename(columns={"hold": "hold_book", "p_over": "p_book_fair"})

    q["self_anchored"] = False
    if self_anchor_markets:
        need = (q["p_cons"].isna() & q["p_book_fair"].notna()
                & q["market"].isin(self_anchor_markets))
        q.loc[need, "p_cons"] = q.loc[need, "p_book_fair"]
        q.loc[need, "n_books_cons"] = 1.0
        q.loc[need, "self_anchored"] = True

    return q.drop(columns=["p_cons_loo", "n_books_loo", "p_cons_all",
                           "n_books_w"])


def side_probability(p_over: np.ndarray, side: np.ndarray,
                     market: np.ndarray) -> np.ndarray:
    """Flip a consensus stated for the over/home side onto the quoted side."""
    is_over_side = np.isin(side, ["over", "home"])
    return np.where(is_over_side, p_over, 1.0 - p_over)
