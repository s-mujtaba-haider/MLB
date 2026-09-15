"""Per-market probability model.

The model does not predict a probability from nothing. It predicts an
*adjustment to the market*:

    logit(p_final) = logit(p_consensus) + s * (logit(p_model) - logit(p_consensus))

where `s` is a shrinkage coefficient fitted on the training fold by log-loss.
This structure is chosen for honesty as much as accuracy:

* If the market is efficient for a market, the fit drives `s` toward zero, the
  model collapses to the consensus, and any measured edge is line-shopping --
  which is a real edge, but a different one, and the report says so.
* If features genuinely add information, `s` rises and the report can quantify
  exactly how much the model contributed over the market alone.

A gradient booster free to output any probability can always find spurious
structure in 1.5M rows; anchored to the market and shrunk, it cannot drift far
without earning it on data it never saw.

Calibration is isotonic, fitted on out-of-fold predictions *within the training
window only* -- never on the evaluation fold, which is the quiet leak that
flatters most backtests.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import KFold

from .devig import expit, logit

EPS = 1e-6


def log_loss(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


@dataclass
class FitReport:
    market: str
    n_train: int
    shrink: float
    ll_market: float
    ll_model: float
    ll_final: float
    features: list[str] = field(default_factory=list)
    top_features: list[tuple[str, float]] = field(default_factory=list)

    @property
    def model_gain(self) -> float:
        """Log-loss improvement of the blended model over the market alone.
        Positive means the features added something."""
        return self.ll_market - self.ll_final


class MarketModel:
    """One market's probability model."""

    def __init__(self, market: str, seed: int = 0,
                 max_iter: int = 300, learning_rate: float = 0.05,
                 max_leaf_nodes: int = 31, min_samples_leaf: int = 200,
                 l2: float = 1.0, n_calib_folds: int = 4):
        self.market = market
        self.seed = seed
        self.params = dict(max_iter=max_iter, learning_rate=learning_rate,
                           max_leaf_nodes=max_leaf_nodes,
                           min_samples_leaf=min_samples_leaf,
                           l2_regularization=l2,
                           early_stopping=True, validation_fraction=0.15,
                           n_iter_no_change=25, random_state=seed)
        self.n_calib_folds = n_calib_folds
        self.features: list[str] = []
        self.clf: HistGradientBoostingClassifier | None = None
        self.iso: IsotonicRegression | None = None
        self.shrink: float = 0.0
        self.report: FitReport | None = None
        self.oof_frame: pd.DataFrame | None = None

    # -- fitting ---------------------------------------------------------
    def fit(self, train: pd.DataFrame, features: list[str],
            target: str = "won") -> FitReport:
        d = train.dropna(subset=[target, "logit_cons"])
        if len(d) < 400:
            raise ValueError(f"{self.market}: only {len(d)} training rows")
        self.features = list(features)
        X = d[self.features].to_numpy(dtype=float)
        y = d[target].to_numpy(dtype=float)
        lc = d["logit_cons"].to_numpy(dtype=float)

        # Out-of-fold predictions inside the training window, used for both
        # calibration and the shrinkage search. Nothing here sees the test fold.
        oof = np.full(len(d), np.nan)
        kf = KFold(n_splits=self.n_calib_folds, shuffle=True,
                   random_state=self.seed)
        for tr, va in kf.split(X):
            m = HistGradientBoostingClassifier(**self.params)
            m.fit(X[tr], y[tr])
            oof[va] = m.predict_proba(X[va])[:, 1]

        self.iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        self.iso.fit(oof, y)
        oof_cal = self.iso.predict(oof)

        self.shrink = self._fit_shrink(lc, logit(oof_cal), y)

        # Refit on the whole training window for deployment.
        self.clf = HistGradientBoostingClassifier(**self.params)
        self.clf.fit(X, y)

        p_market = expit(lc)
        p_final = expit(lc + self.shrink * (logit(oof_cal) - lc))

        # Keep the out-of-fold probabilities. The EV threshold is chosen from
        # these rather than from the refitted model's in-sample predictions:
        # in-sample EVs are optimistic, and a threshold tuned against them is
        # tuned against the model's own overfit rather than against the edge.
        key = [c for c in ("event_id", "market", "subject", "line")
               if c in d.columns]
        self.oof_frame = d[key].copy() if key else None
        if self.oof_frame is not None:
            self.oof_frame["p_over_oof"] = p_final
        self.report = FitReport(
            market=self.market, n_train=len(d), shrink=self.shrink,
            ll_market=log_loss(y, p_market), ll_model=log_loss(y, oof_cal),
            ll_final=log_loss(y, p_final), features=self.features,
            top_features=self._importances(d, X, y))
        return self.report

    @staticmethod
    def _fit_shrink(logit_cons: np.ndarray, logit_model: np.ndarray,
                    y: np.ndarray) -> float:
        """Grid-search the shrinkage that minimises out-of-fold log-loss."""
        best_s, best_ll = 0.0, np.inf
        delta = logit_model - logit_cons
        for s in np.linspace(0.0, 1.0, 41):
            ll = log_loss(y, expit(logit_cons + s * delta))
            if ll < best_ll:
                best_s, best_ll = float(s), ll
        return best_s

    def _importances(self, d: pd.DataFrame, X: np.ndarray, y: np.ndarray,
                     n: int = 10) -> list[tuple[str, float]]:
        """Cheap permutation importance on a subsample, for the report only."""
        if self.clf is None or len(d) < 2000:
            return []
        rng = np.random.default_rng(self.seed)
        idx = rng.choice(len(d), size=min(20000, len(d)), replace=False)
        Xs, ys = X[idx], y[idx]
        base = log_loss(ys, self.clf.predict_proba(Xs)[:, 1])
        out = []
        for j, name in enumerate(self.features):
            Xp = Xs.copy()
            rng.shuffle(Xp[:, j])
            out.append((name, log_loss(ys, self.clf.predict_proba(Xp)[:, 1]) - base))
        return sorted(out, key=lambda t: -t[1])[:n]

    # -- inference -------------------------------------------------------
    def predict_over(self, df: pd.DataFrame,
                     logit_cons_col: str = "logit_cons") -> np.ndarray:
        """Probability that the over/home side wins."""
        if self.clf is None or self.iso is None:
            raise RuntimeError(f"{self.market}: model not fitted")
        X = df[self.features].to_numpy(dtype=float)
        raw = self.clf.predict_proba(X)[:, 1]
        cal = self.iso.predict(raw)
        lc = df[logit_cons_col].to_numpy(dtype=float)
        out = expit(lc + self.shrink * (logit(cal) - lc))
        # Where the market is unknown there is nothing to adjust.
        return np.where(np.isfinite(lc), out, np.nan)


def side_probability(p_over: np.ndarray, side: pd.Series) -> np.ndarray:
    is_over = side.isin(["over", "home"]).to_numpy()
    return np.where(is_over, p_over, 1.0 - p_over)
