"""Walk-forward evaluation.

Expanding-window folds over the full history. For each fold the model, the
isotonic calibration *and the EV threshold* are fitted on data strictly before
the fold's first game date, then applied unchanged to the fold. Choosing the
threshold on the evaluation data is the most common way a backtest lies about
itself, so the threshold search lives inside the training window with
everything else.

Bets are one per (proposition, side) at the best bettable price, sized flat.
Flat staking, not Kelly, is what the ROI confidence interval is built for --
Kelly sizing makes the realised return depend on bankroll path and turns a
clean binomial question into a messy one.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import config as C
from .dataset import PROP_KEY, feature_columns
from .devig import logit
from .grade import LOSS, PUSH, WIN
from .model import MarketModel, side_probability
from .odds import american_to_prob, ev_per_unit, profit_per_unit
from .selection import SelectionCalibrator, add_apparent_edge


# Stakeable price band. Outside it a quote is either a four-figure favourite
# nobody funds or a lottery ticket whose de-vigged probability is guesswork.
MIN_PRICE, MAX_PRICE = -2000.0, 2000.0


@dataclass
class Fold:
    train_end: str
    test_start: str
    test_end: str


def make_folds(dates: pd.Series, n_burn_days: int = 400,
               step_days: int = 30) -> list[Fold]:
    """Expanding-window folds after a burn-in long enough to fit a model."""
    d = pd.to_datetime(sorted(pd.unique(dates.dropna())))
    if len(d) == 0:
        return []
    start = d.min() + pd.Timedelta(days=n_burn_days)
    folds: list[Fold] = []
    cur = start
    end = d.max()
    while cur <= end:
        nxt = cur + pd.Timedelta(days=step_days)
        folds.append(Fold(
            train_end=(cur - pd.Timedelta(days=1)).date().isoformat(),
            test_start=cur.date().isoformat(),
            test_end=min(nxt - pd.Timedelta(days=1), end).date().isoformat()))
        cur = nxt
    return folds


# ---------------------------------------------------------------------------
# Threshold selection (train-fold only)
# ---------------------------------------------------------------------------

def choose_threshold(train_bets: pd.DataFrame, grid: np.ndarray | None = None,
                     min_bets: int = 150, ev_col: str = "ev") -> float:
    """Pick the EV cut that maximises a shrunk training ROI.

    The raw argmax of training ROI overfits toward thresholds so high that only
    a handful of bets survive, so each candidate is shrunk toward zero by its
    own sample size. This is a decision rule fitted on training data, not a
    reported result.
    """
    if train_bets.empty:
        return 0.02
    grid = grid if grid is not None else np.arange(0.0, 0.16, 0.005)
    if ev_col not in train_bets.columns:
        ev_col = "ev"
    best_t, best_score = 0.03, -np.inf
    for t in grid:
        sel = train_bets[train_bets[ev_col] >= t]
        n = len(sel)
        if n < min_bets:
            continue
        roi = sel["profit"].sum() / n
        # Shrink toward zero: a 200-bet ROI is worth far less than a 5,000-bet one.
        score = roi * n / (n + 1500.0)
        if score > best_score:
            best_t, best_score = float(t), score
    return best_t


# ---------------------------------------------------------------------------
# Core walk-forward
# ---------------------------------------------------------------------------

def run_market(market: str, props: pd.DataFrame, candidates: pd.DataFrame,
               folds: list[Fold], min_books: int | None = None,
               verbose: bool = False) -> tuple[pd.DataFrame, list]:
    """Walk-forward one market. Returns (bets, per-fold fit reports)."""
    if min_books is None:
        min_books = C.min_books_for(market)
    p = props[props["market"] == market].copy()
    c = candidates[candidates["market"] == market].copy()
    if p.empty or c.empty:
        return pd.DataFrame(), []

    feats = feature_columns(p, market)
    if "logit_cons" not in feats:
        feats.append("logit_cons")

    # Candidates inherit the proposition's features; their own consensus is the
    # leave-one-out value for the book being priced.
    # `line` is both a join key and a model feature, so it must not be listed
    # twice in the projection -- pandas rejects a duplicate label on merge.
    keep = [c_ for c_ in p.columns
            if (c_ in feats or c_ in ("game_date", "espn_id"))
            and c_ not in PROP_KEY]
    c = c.merge(p[PROP_KEY + list(dict.fromkeys(keep))], on=PROP_KEY,
                how="inner", suffixes=("", "_prop"))
    c["logit_cons"] = logit(c["p_cons"].to_numpy(dtype=float))
    c = c[c["n_books_cons"] >= min_books]
    c = c[c["result"].isin([WIN, LOSS, PUSH])]
    if c.empty:
        return pd.DataFrame(), []

    # Prices beyond these bounds are not realistically stakeable at size, and
    # at the extremes the de-vig is least trustworthy.
    c = c[(c["price"] >= MIN_PRICE) & (c["price"] <= MAX_PRICE)]
    if c.empty:
        return pd.DataFrame(), []
    c["p_raw"] = american_to_prob(c["price"].to_numpy(dtype=float))
    c["payout"] = profit_per_unit(c["price"].to_numpy(dtype=float))

    all_bets, reports = [], []
    for f in folds:
        tr_p = p[p["game_date"] <= f.train_end]
        te_c = c[(c["game_date"] >= f.test_start) & (c["game_date"] <= f.test_end)]
        if len(tr_p) < 800 or te_c.empty:
            continue
        try:
            mm = MarketModel(market)
            rep = mm.fit(tr_p, feats)
        except ValueError:
            continue
        reports.append((f.test_start, rep))

        # The winner's-curse correction and the EV threshold are both fitted
        # on training-window candidates only, using out-of-fold model
        # probabilities so the correction is not fitted against the model's
        # own overfit.
        tr_c = c[c["game_date"] <= f.train_end]
        tr_bets = add_apparent_edge(_score_oof(tr_c, mm))
        cal = SelectionCalibrator().fit(tr_bets)
        tr_bets = cal.apply(tr_bets)
        thr = choose_threshold(tr_bets, ev_col="ev_cal" if cal.fitted else "ev")

        te_bets = cal.apply(add_apparent_edge(_score(te_c, mm)))
        ev_col = "ev_cal" if cal.fitted else "ev"
        sel = te_bets[te_bets[ev_col] >= thr].copy()
        # One bet per proposition. Both sides can clear a threshold at the
        # margin, and taking both is a hedge that locks in the hold -- not
        # something a bettor would ever do, and it quietly halves the measured
        # edge while doubling the bet count.
        if not sel.empty:
            sel = (sel.sort_values(ev_col, ascending=False)
                      .drop_duplicates(subset=PROP_KEY, keep="first"))
        sel["fold"] = f.test_start
        sel["threshold"] = thr
        sel["shrink"] = rep.shrink
        sel["sel_cal_n"] = cal.n_fit
        all_bets.append(sel)
        if verbose:
            print(f"  [{market}] fold {f.test_start} train={len(tr_p)} "
                  f"thr={thr:.3f} shrink={rep.shrink:.2f} "
                  f"gain={rep.model_gain:+.5f} bets={len(sel)}",
                  flush=True)

    bets = pd.concat(all_bets, ignore_index=True) if all_bets else pd.DataFrame()
    return bets, reports


def _score_oof(cand: pd.DataFrame, mm: MarketModel) -> pd.DataFrame:
    """Score training candidates with the model's out-of-fold probabilities.

    Using the refitted model here would price training candidates with a model
    that has seen them, inflating their EVs and tuning the threshold against
    the model's own overfit instead of against the edge.
    """
    if cand.empty or mm.oof_frame is None:
        return _score(cand, mm)
    d = cand.merge(mm.oof_frame, on=[c for c in PROP_KEY
                                     if c in mm.oof_frame.columns],
                   how="inner")
    d = _ensure_payout(d)
    if d.empty:
        return _score(cand, mm)
    d["p_model"] = side_probability(d["p_over_oof"].to_numpy(dtype=float),
                                    d["side"])
    price = d["price"].to_numpy(dtype=float)
    d["ev"] = ev_per_unit(d["p_model"].to_numpy(), price)
    d["profit"] = np.where(d["result"] == WIN, d["payout"],
                           np.where(d["result"] == LOSS, -1.0, 0.0))
    return d.dropna(subset=["ev"])


def _ensure_payout(d: pd.DataFrame) -> pd.DataFrame:
    """Profit-on-win per unit. run_market precomputes it; callers that score a
    candidate frame directly (train_production) do not."""
    if "payout" not in d.columns:
        d = d.copy()
        d["payout"] = profit_per_unit(d["price"].to_numpy(dtype=float))
    return d


def _score(cand: pd.DataFrame, mm: MarketModel) -> pd.DataFrame:
    if cand.empty:
        return cand.assign(ev=[], p_model=[], profit=[])
    d = _ensure_payout(cand.copy())
    p_over = mm.predict_over(d)
    d["p_model"] = side_probability(p_over, d["side"])
    d["p_market"] = side_probability(d["p_cons"].to_numpy(dtype=float), d["side"])
    d["ev"] = ev_per_unit(d["p_model"].to_numpy(), d["price"].to_numpy(dtype=float))
    d["ev_market"] = ev_per_unit(d["p_market"].to_numpy(),
                                 d["price"].to_numpy(dtype=float))
    prof = np.where(d["result"] == WIN, d["payout"],
                    np.where(d["result"] == LOSS, -1.0, 0.0))
    d["profit"] = prof
    return d.dropna(subset=["ev"])


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def bootstrap_roi(profit: np.ndarray, n_boot: int = 5000,
                  seed: int = 0) -> tuple[float, float, float]:
    """Percentile bootstrap CI for ROI, resampling bets."""
    if len(profit) == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(profit), size=(n_boot, len(profit)))
    rois = profit[idx].mean(axis=1)
    return (float(profit.mean()), float(np.percentile(rois, 2.5)),
            float(np.percentile(rois, 97.5)))


def clv_metrics(bets: pd.DataFrame, closing: pd.DataFrame) -> dict:
    """Closing-line value, the lowest-variance evidence that an edge is real.

    Matched on (proposition, side, book): what did the price we took do by the
    time the market closed?
    """
    if bets.empty or closing is None or closing.empty:
        return {"clv_n": 0, "clv_mean": np.nan, "beat_close": np.nan}
    cl = (closing[closing["tag"] == "closing"]
          .groupby(PROP_KEY + ["side", "book"], dropna=False, observed=True)
          ["price"].max().rename("close_price").reset_index())
    m = bets.merge(cl, on=PROP_KEY + ["side", "book"], how="inner")
    if m.empty:
        return {"clv_n": 0, "clv_mean": np.nan, "beat_close": np.nan}
    p_bet = american_to_prob(m["price"].to_numpy(dtype=float))
    p_close = american_to_prob(m["close_price"].to_numpy(dtype=float))
    # We took a longer price than the close when our implied probability is
    # lower than the closing implied probability.
    clv = p_close - p_bet
    return {"clv_n": int(len(m)), "clv_mean": float(clv.mean()),
            "beat_close": float((clv > 0).mean())}


def summarise(bets: pd.DataFrame) -> dict:
    if bets.empty:
        return {"n_bets": 0, "roi": np.nan, "roi_lo": np.nan, "roi_hi": np.nan,
                "profit": 0.0, "hit_rate": np.nan, "avg_price": np.nan,
                "avg_ev": np.nan, "n_win": 0, "n_loss": 0, "n_push": 0}
    prof = bets["profit"].to_numpy(dtype=float)
    roi, lo, hi = bootstrap_roi(prof)
    dec = bets["result"]
    nw, nl = int((dec == WIN).sum()), int((dec == LOSS).sum())
    return {
        "n_bets": len(bets), "roi": roi, "roi_lo": lo, "roi_hi": hi,
        "profit": float(prof.sum()),
        "hit_rate": nw / (nw + nl) if (nw + nl) else np.nan,
        "avg_price": float(bets["price"].mean()),
        "avg_ev": float(bets["ev"].mean()),
        "n_win": nw, "n_loss": nl, "n_push": int((dec == PUSH).sum()),
    }


def period_stability(bets: pd.DataFrame, by: str = "fold") -> pd.DataFrame:
    if bets.empty:
        return pd.DataFrame()
    g = bets.groupby(by).agg(n=("profit", "size"), roi=("profit", "mean"),
                             profit=("profit", "sum")).reset_index()
    return g


def calibration(bets: pd.DataFrame, bins: int = 10) -> pd.DataFrame:
    """Predicted vs realised, on graded (non-push) bets."""
    d = bets[bets["result"].isin([WIN, LOSS])]
    if d.empty:
        return pd.DataFrame()
    q = pd.qcut(d["p_model"], bins, duplicates="drop")
    out = (d.groupby(q, observed=True)
             .agg(n=("p_model", "size"), pred=("p_model", "mean"),
                  actual=("result", lambda s: (s == WIN).mean()))
             .reset_index(drop=True))
    return out
