"""Fit each bookmaker's weight as a probability anchor, from data.

`config.ANCHOR_WEIGHTS` are priors -- reasonable ones (Pinnacle and the
zero-vig exchanges above retail, retail above the soft shops), but still
guesses. This module replaces them with measured skill: for each book, how
well does its own de-vigged price predict what actually happened?

The measurement is a log-loss skill score against the base rate, computed per
market because a book that prices moneylines sharply may be lazy about batter
props, and the consensus should know the difference.

**Leakage discipline.** Weights are fitted on the *earliest season only*, which
sits entirely inside every walk-forward fold's training window (the burn-in is
longer than one season). They are therefore never fitted on data any fold is
evaluated on. Fitting them on all seasons would be a subtle, whole-pipeline
leak: the consensus for 2026 would encode which books turned out to be sharp
in 2026.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import config as C
from .dataset import PROP_KEY, proposition_outcome
from .devig import book_views

WEIGHTS_PATH = C.DATA / "anchor_weights.json"

MIN_OBS = 400          # per (market, book) before a weight is trusted
MAX_WEIGHT = 1.0
SHRINK_N = 2000.0      # how fast a measured weight overrides the prior


def _log_loss(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def fit_anchor_weights(graded: pd.DataFrame, method: str = "shin"
                       ) -> dict[str, dict[str, float]]:
    """Return {market: {book: weight}} measured on this frame."""
    dec = graded[graded["tag"] == "decision"]
    bv = book_views(dec, method=method)
    if bv.empty:
        return {}
    out_prop = proposition_outcome(dec)
    if out_prop.empty:
        return {}
    m = bv.merge(out_prop, on=PROP_KEY, how="inner")
    if m.empty:
        return {}

    result: dict[str, dict[str, float]] = {}
    for market, sub in m.groupby("market", observed=True):
        base_rate = float(sub["won"].mean())
        base_ll = _log_loss(sub["won"].to_numpy(dtype=float),
                            np.full(len(sub), base_rate))
        weights: dict[str, float] = {}
        for book, bs in sub.groupby("book", observed=True):
            if len(bs) < MIN_OBS:
                continue
            y = bs["won"].to_numpy(dtype=float)
            p = bs["p_over"].to_numpy(dtype=float)
            # Skill relative to the base rate, on this book's own subset.
            local_base = _log_loss(y, np.full(len(y), base_rate))
            skill = local_base - _log_loss(y, p)
            if skill <= 0:
                weights[book] = 0.0
                continue
            prior = C.ANCHOR_WEIGHTS.get(book, C.DEFAULT_ANCHOR_WEIGHT)
            # Shrink the measurement toward the prior by sample size, so a book
            # with 500 observations does not outrank Pinnacle on noise.
            lam = len(bs) / (len(bs) + SHRINK_N)
            weights[book] = float(lam * skill + (1 - lam) * prior)

        if weights:
            hi = max(weights.values())
            if hi > 0:
                weights = {k: round(min(MAX_WEIGHT, v / hi * MAX_WEIGHT), 4)
                           for k, v in weights.items()}
            result[market] = dict(sorted(weights.items(),
                                         key=lambda kv: -kv[1]))
        _ = base_ll
    return result


def save(weights: dict, path=WEIGHTS_PATH) -> None:
    path.write_text(json.dumps(weights, indent=2))
    print(f"saved fitted anchor weights for {len(weights)} markets -> {path}")


def load(path=WEIGHTS_PATH) -> dict[str, dict[str, float]]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return {}


def report(weights: dict[str, dict[str, float]], top: int = 8) -> str:
    lines = []
    for market, w in weights.items():
        head = ", ".join(f"{b}={v:.2f}" for b, v in list(w.items())[:top])
        lines.append(f"{market:22s} {head}")
    return "\n".join(lines)
