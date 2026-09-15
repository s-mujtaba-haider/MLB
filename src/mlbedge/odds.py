"""Odds arithmetic and de-vigging.

Two numbers must never be confused, and conflating them is the most expensive
bug in betting code:

* **raw implied probability** -- what the posted price costs you. This is the
  break-even number: bet only if your probability exceeds it.
* **vig-free probability** -- the market's actual opinion, recovered by
  stripping the bookmaker's margin. This is an *estimate of truth*, and is
  never the break-even threshold.

Four margin models are provided because they disagree most exactly where props
live -- on longshots. Multiplicative assumes margin is proportional to price,
which systematically understates the true probability of favourites and
overstates longshots. Shin assumes the margin exists to protect against
insider money, which loads proportionally more onto longshots and empirically
fits US sportsbook pricing better. Shin is the default for that reason.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import brentq

EPS = 1e-12


# ---------------------------------------------------------------------------
# Price conversions
# ---------------------------------------------------------------------------

def american_to_decimal(a: np.ndarray | float) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    return np.where(a > 0, 1.0 + a / 100.0, 1.0 + 100.0 / np.abs(a))


def american_to_prob(a: np.ndarray | float) -> np.ndarray:
    """Raw implied probability -- the break-even number, vig included."""
    a = np.asarray(a, dtype=float)
    # np.where evaluates both branches, so a price of exactly -100 would divide
    # by zero in the positive branch before being discarded. Guard the
    # denominator rather than filtering the warning.
    pos = 100.0 / np.where(a > 0, a + 100.0, 1.0)
    neg = np.abs(a) / (np.abs(a) + 100.0)
    return np.where(a > 0, pos, neg)


def decimal_to_american(d: np.ndarray | float) -> np.ndarray:
    d = np.asarray(d, dtype=float)
    return np.where(d >= 2.0, (d - 1.0) * 100.0, -100.0 / (d - 1.0))


def prob_to_american(p: np.ndarray | float) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    return decimal_to_american(1.0 / p)


def profit_per_unit(a: np.ndarray | float) -> np.ndarray:
    """Profit on a winning 1-unit stake (stake excluded)."""
    a = np.asarray(a, dtype=float)
    return np.where(a > 0, a / 100.0, 100.0 / np.abs(a))


# ---------------------------------------------------------------------------
# De-vigging
# ---------------------------------------------------------------------------

def devig_multiplicative(raw: np.ndarray) -> np.ndarray:
    raw = np.asarray(raw, dtype=float)
    return raw / raw.sum()


def devig_additive(raw: np.ndarray) -> np.ndarray:
    raw = np.asarray(raw, dtype=float)
    p = raw - (raw.sum() - 1.0) / len(raw)
    return np.clip(p, EPS, 1 - EPS)


def devig_power(raw: np.ndarray) -> np.ndarray:
    """Find k with sum(raw_i ** k) == 1."""
    raw = np.clip(np.asarray(raw, dtype=float), EPS, 1 - EPS)
    if abs(raw.sum() - 1.0) < 1e-9:
        return raw

    def f(k: float) -> float:
        return float(np.sum(raw ** k) - 1.0)

    try:
        k = brentq(f, 0.2, 8.0, xtol=1e-10, maxiter=200)
    except (ValueError, RuntimeError):
        return devig_multiplicative(raw)
    return raw ** k


def devig_shin(raw: np.ndarray) -> np.ndarray:
    """Shin (1993): margin as protection against informed money.

    Solves for the insider fraction z that makes the recovered probabilities
    sum to one, then inverts. Falls back to multiplicative when the overround
    is degenerate.
    """
    raw = np.clip(np.asarray(raw, dtype=float), EPS, 1 - EPS)
    total = raw.sum()
    if total <= 1.0 + 1e-9:
        return devig_multiplicative(raw)

    def p_of_z(z: float) -> np.ndarray:
        inner = z * z + 4.0 * (1.0 - z) * raw * raw / total
        return (np.sqrt(np.maximum(inner, 0.0)) - z) / (2.0 * (1.0 - z))

    def f(z: float) -> float:
        return float(p_of_z(z).sum() - 1.0)

    try:
        z = brentq(f, 1e-9, 0.9999, xtol=1e-12, maxiter=200)
    except (ValueError, RuntimeError):
        return devig_multiplicative(raw)
    p = p_of_z(z)
    s = p.sum()
    return p / s if s > 0 else devig_multiplicative(raw)


_METHODS = {
    "multiplicative": devig_multiplicative,
    "additive": devig_additive,
    "power": devig_power,
    "shin": devig_shin,
}


def devig(raw: np.ndarray, method: str = "shin") -> np.ndarray:
    """Strip margin from a complete set of mutually exclusive outcomes."""
    raw = np.asarray(raw, dtype=float)
    if raw.ndim != 1 or len(raw) < 2:
        raise ValueError("devig needs a complete outcome set (>=2 prices)")
    try:
        fn = _METHODS[method]
    except KeyError:
        raise ValueError(f"unknown devig method {method!r}; "
                         f"expected one of {sorted(_METHODS)}") from None
    p = fn(raw)
    return np.clip(p, EPS, 1 - EPS)


def overround(raw: np.ndarray) -> float:
    """Bookmaker margin as a fraction, e.g. 0.045 for a 4.5% hold."""
    return float(np.sum(raw) - 1.0)


def devig_two_sided(price_a: float, price_b: float,
                    method: str = "shin") -> tuple[float, float]:
    """Convenience wrapper for the two-outcome case (over/under, home/away)."""
    raw = american_to_prob(np.array([price_a, price_b], dtype=float))
    p = devig(raw, method=method)
    return float(p[0]), float(p[1])


# ---------------------------------------------------------------------------
# Expected value
# ---------------------------------------------------------------------------

def ev_per_unit(p_win: np.ndarray | float, price: np.ndarray | float,
                p_push: np.ndarray | float = 0.0) -> np.ndarray:
    """Expected profit per unit staked.

    A push returns the stake, so it contributes zero profit but does consume
    probability mass -- ignoring it overstates the edge on integer lines.
    """
    p_win = np.asarray(p_win, dtype=float)
    p_push = np.asarray(p_push, dtype=float)
    win = profit_per_unit(price)
    p_lose = np.clip(1.0 - p_win - p_push, 0.0, 1.0)
    return p_win * win - p_lose


def kelly_fraction(p_win: np.ndarray | float, price: np.ndarray | float,
                   p_push: np.ndarray | float = 0.0) -> np.ndarray:
    """Full-Kelly stake fraction, floored at zero."""
    b = profit_per_unit(price)
    p_win = np.asarray(p_win, dtype=float)
    p_push = np.asarray(p_push, dtype=float)
    q = np.clip(1.0 - p_win - p_push, 0.0, 1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        f = (b * p_win - q) / b
    return np.clip(np.nan_to_num(f, nan=0.0, posinf=0.0, neginf=0.0), 0.0, 1.0)
