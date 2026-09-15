"""Synthetic market generators with known pathologies.

Each generator builds a bet ledger whose defect is known by construction, so a
test can assert that the gate reaches the right verdict *for the right reason*
rather than merely reaching a verdict.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mlbedge.odds import american_to_prob, profit_per_unit  # noqa: E402


def _ledger(p_true: np.ndarray, price: np.ndarray, rng, fold_size: int = 60,
            p_model: np.ndarray | None = None) -> pd.DataFrame:
    """Realise outcomes from known true probabilities."""
    n = len(p_true)
    won = rng.random(n) < p_true
    payout = profit_per_unit(price)
    profit = np.where(won, payout, -1.0)
    p_model = p_model if p_model is not None else p_true
    return pd.DataFrame({
        "result": np.where(won, "win", "loss"),
        "profit": profit,
        "price": price,
        "p_model": p_model,
        "p_cons": p_true,
        "ev": p_model * payout - (1 - p_model),
        "shrink": 0.2,
        "fold": [f"f{i // fold_size:03d}" for i in range(n)],
        "event_id": [f"e{i}" for i in range(n)],
        "market": "synthetic",
        "subject": "game",
        "line": np.nan,
        "side": "over",
        "book": "draftkings",
        "game_date": pd.date_range("2025-04-01", periods=n,
                                   freq="h").date.astype(str),
    })


@pytest.fixture
def rng():
    return np.random.default_rng(12345)


# A realistic spread of prices. Everything at a single number makes every bet
# carry the same probability, which collapses the calibration deciles and makes
# calibration untestable.
_PRICES = np.array([-260.0, -180.0, -145.0, -120.0, -110.0, 100.0, 115.0,
                    140.0, 185.0, 240.0])


def _prices(rng, n):
    return rng.choice(_PRICES, size=n)


def real_edge(rng, n=4000, edge=0.035):
    """A genuine, persistent edge: true probability exceeds break-even."""
    price = _prices(rng, n)
    return _ledger(american_to_prob(price) + edge, price, rng)


def efficient(rng, n=4000):
    """Fairly priced: true probability equals break-even. No edge to find."""
    price = _prices(rng, n)
    return _ledger(american_to_prob(price), price, rng)


def losing(rng, n=4000, edge=-0.03):
    price = _prices(rng, n)
    return _ledger(american_to_prob(price) + edge, price, rng)


def thin(rng, n=80, edge=0.06):
    price = _prices(rng, n)
    return _ledger(american_to_prob(price) + edge, price, rng, fold_size=20)


def dies_midway(rng, n=4000, edge=0.06, late=-0.035):
    """Edge real in the first half, gone in the second."""
    price = _prices(rng, n)
    p_be = american_to_prob(price)
    p = np.where(np.arange(n) < n // 2, p_be + edge, p_be + late)
    return _ledger(p, price, rng)


def miscalibrated(rng, n=4000, edge=0.03):
    """Profitable, but the stated probabilities are systematically wrong."""
    price = _prices(rng, n)
    df = _ledger(american_to_prob(price) + edge, price, rng)
    df["p_model"] = np.clip(df["p_model"] + 0.12, 0, 1)   # overconfident
    return df
